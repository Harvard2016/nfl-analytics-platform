"""Coverage experiment B: a small temporal interaction model over defender-receiver pairs.

Input per play: frames x defenders x receivers x pair features, with validity masks. A shared MLP embeds each
pair, masked pooling over receivers then defenders gives one vector per frame (no player-order encoding), and a
causal sequence model (unidirectional GRU, or a causal temporal convolution for comparison) reads the frames.
The output at frame t depends only on frames 0..t, so one forward pass gives the prediction for every prefix.

Target: agreement with the released play-level man/zone label. It is not a claim about who covered whom at any instant.
Same selected-player research setting and the same excluded fields as `relational.py`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from ..shared import config
from . import relational, schema

SOURCE = "bdb2026"
HORIZON_FRAMES = list(relational.HORIZONS.values())            # 0, 5, 10, 15
PAIR_FEATS = ["rel_x", "rel_y", "sep", "rel_vx", "rel_vy", "closing", "dircos", "facing",
              "d_depth", "d_lateral", "d_vx", "d_vy", "d_speed", "r_depth", "r_lateral", "r_vx", "r_vy", "r_speed"]
# Fixed unit scales (yards / 10, yards per second / 5). Constants, not fitted on any data.
POS, VEL = 10.0, 5.0


def pair_features(d: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """d: (B, T, D, 6), r: (B, T, R, 6) -> (B, T, D, R, F). Mirrors relational.pair_arrays."""
    dp, rp = d[..., None, :], r[..., None, :, :]
    rel = dp[..., :2] - rp[..., :2]
    sep = rel.norm(dim=-1, keepdim=True)
    rv = dp[..., 2:4] - rp[..., 2:4]
    closing = -(rel * rv).sum(-1, keepdim=True) / sep.clamp(min=1e-3)
    sd, sr = dp[..., 2:4].norm(dim=-1, keepdim=True), rp[..., 2:4].norm(dim=-1, keepdim=True)
    moving = (sd > 0.5) & (sr > 0.5)
    dircos = torch.where(moving, (dp[..., 2:4] * rp[..., 2:4]).sum(-1, keepdim=True) / (sd * sr).clamp(min=1e-6), torch.zeros_like(sep))
    facing = (dp[..., 4:6] * (-rel / sep.clamp(min=1e-3))).sum(-1, keepdim=True)
    ex = lambda t: t.expand(*sep.shape[:-1], t.shape[-1])
    return torch.cat([rel / POS, sep / POS, rv / VEL, closing / VEL, dircos, facing,
                      ex(dp[..., :2]) / POS, ex(dp[..., 2:4]) / VEL, ex(sd) / VEL,
                      ex(rp[..., :2]) / POS, ex(rp[..., 2:4]) / VEL, ex(sr) / VEL], dim=-1)


def masked_pool(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Mean and max over `dim` using only valid entries. mask broadcasts to x without the feature axis."""
    m = mask.unsqueeze(-1)
    mean = (x * m).sum(dim) / m.sum(dim).clamp(min=1)
    mx = x.masked_fill(~m, float("-inf")).amax(dim)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], dim=-1)


class CausalTCN(nn.Module):
    """Three causal convolution blocks (kernel 3, dilations 1, 2, 4). Left padding only: no future frames."""

    def __init__(self, dim_in: int, hidden: int, dropout: float):
        super().__init__()
        self.pads, self.convs = [], nn.ModuleList()
        for i, dil in enumerate((1, 2, 4)):
            self.pads.append(2 * dil)
            self.convs.append(nn.Conv1d(dim_in if i == 0 else hidden, hidden, 3, dilation=dil))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, T, C)
        h = x.transpose(1, 2)
        for pad, conv in zip(self.pads, self.convs):
            h = self.drop(torch.relu(conv(nn.functional.pad(h, (pad, 0)))))
        return h.transpose(1, 2)


class CoverageNet(nn.Module):
    def __init__(self, temporal: str = "gru", embed: int = 64, hidden: int = 96, dropout: float = 0.1, n_family: tuple[int, int] | None = None):
        super().__init__()
        f = len(PAIR_FEATS)
        self.pair = nn.Sequential(nn.Linear(f, embed), nn.ReLU(), nn.Linear(embed, embed), nn.ReLU())
        self.defender = nn.Sequential(nn.Linear(2 * embed, embed), nn.ReLU())
        self.temporal_kind = temporal
        self.temporal = (nn.GRU(2 * embed, hidden, num_layers=2, dropout=dropout, batch_first=True) if temporal == "gru"
                         else CausalTCN(2 * embed, hidden, dropout))
        self.head = nn.Linear(hidden, 1)
        self.pair_edit = None
        # optional hierarchical family heads: P(family) = P(group) * P(family | group); group truth is never an input
        self.family = None if n_family is None else nn.ModuleList([nn.Linear(hidden, n_family[0]), nn.Linear(hidden, n_family[1])])

    def trunk(self, d, r, dmask, rmask) -> torch.Tensor:
        feats = pair_features(d, r)
        if self.pair_edit is not None:                                              # used only for input-ablation explanations
            feats = self.pair_edit(feats)
        x = self.pair(feats)                                                        # (B, T, D, R, E)
        pm = dmask[:, None, :, None] & rmask[:, None, None, :]                     # valid pairs
        x = masked_pool(x, pm.expand(x.shape[:-1]), dim=3)                          # pool receivers -> (B, T, D, 2E)
        x = self.defender(x)
        x = masked_pool(x, dmask[:, None, :].expand(x.shape[:-1]), dim=2)           # pool defenders -> (B, T, 2E)
        out = self.temporal(x)
        return out[0] if isinstance(out, tuple) else out                            # (B, T, H)

    def forward(self, d, r, dmask, rmask) -> torch.Tensor:
        """Man logit at every frame: (B, T). Entry t uses frames 0..t only."""
        return self.head(self.trunk(d, r, dmask, rmask)).squeeze(-1)

    def forward_family(self, d, r, dmask, rmask):
        h = self.trunk(d, r, dmask, rmask)
        return self.head(h).squeeze(-1), self.family[0](h), self.family[1](h)


def load_arrays(features_dir: Path | None = None) -> dict:
    fd = Path(features_dir or config.FEATURES / SOURCE)
    z = np.load(fd / "tensors_v2.npz")
    idx = pl.read_parquet(fd / "tensors_v2_index.parquet")
    lab = idx[schema.TARGET].to_list()
    return {"defs": z["defs"], "recs": z["recs"], "dmask": z["dmask"], "rmask": z["rmask"], "nframes": z["nframes"].astype(int),
            "y": np.array([1.0 if v == "Man" else 0.0 if v == "Zone" else np.nan for v in lab], np.float32),
            "week": idx["week"].to_numpy(), "game": idx["gameId"].to_numpy(), "team": np.array(idx["defensiveTeam"].to_list()),
            "family": np.array([v or "" for v in idx["coverage_type"].to_list()]),
            "key": np.array([f"{g}:{p}" for g, p in zip(idx["gameId"].to_list(), idx["playId"].to_list())])}


def to_batch(A: dict, ix: np.ndarray, device: str, reflect: np.ndarray | None = None):
    d, r = A["defs"][ix], A["recs"][ix]
    if reflect is not None and reflect.any():
        d, r = d.copy(), r.copy()
        d[reflect], r[reflect] = relational.reflect(d[reflect]), relational.reflect(r[reflect])
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    return t(d), t(r), t(A["dmask"][ix], torch.bool), t(A["rmask"][ix], torch.bool)


@torch.no_grad()
def predict_logits(model: CoverageNet, A: dict, ix: np.ndarray, device: str = "cpu", batch: int = 256) -> np.ndarray:
    """(n, T) man logits. Entries past a play's last frame are computed on zero padding and must not be used."""
    model.eval()
    out = [model(*to_batch(A, ix[i:i + batch], device)).cpu().numpy() for i in range(0, len(ix), batch)]
    return np.concatenate(out)


def horizon_loss(logits: torch.Tensor, y: torch.Tensor, nframes: torch.Tensor, rng: torch.Generator | None) -> torch.Tensor:
    """Binary cross-entropy at one permitted horizon per play (random when training, all horizons averaged when rng is None)."""
    hz = torch.tensor(HORIZON_FRAMES, device=logits.device)
    ok = nframes[:, None] > hz[None, :]                                           # prefixes that actually exist
    if rng is not None:
        w = ok.float() + 1e-9
        pick = torch.multinomial(w.cpu(), 1, generator=rng).to(logits.device)
        lg = logits.gather(1, hz[pick.squeeze(1)][:, None]).squeeze(1)
        return nn.functional.binary_cross_entropy_with_logits(lg, y)
    lg = logits[:, hz]
    loss = nn.functional.binary_cross_entropy_with_logits(lg, y[:, None].expand_as(lg), reduction="none")
    return (loss * ok).sum() / ok.sum()


def train_model(A: dict, train_ix: np.ndarray, val_ix: np.ndarray | None, *, temporal: str = "gru", seed: int = 42, augment: bool = False,
                max_epochs: int = 50, patience: int = 8, fixed_epochs: int | None = None, device: str = "cpu", batch: int = 64,
                lr: float = 1e-3, weight_decay: float = 1e-4, log=print) -> dict:
    torch.manual_seed(seed)
    rng_np, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    model = CoverageNet(temporal).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_all, nf_all = torch.as_tensor(A["y"]), torch.as_tensor(A["nframes"])
    best, best_state, best_epoch, history, t0 = float("inf"), None, 0, [], time.time()
    epochs = fixed_epochs or max_epochs
    for epoch in range(1, epochs + 1):
        model.train()
        perm, tot = rng_np.permutation(train_ix), 0.0
        for i in range(0, len(perm), batch):
            ix = perm[i:i + batch]
            refl = rng_np.random(len(ix)) < 0.5 if augment else None
            logits = model(*to_batch(A, ix, device, refl))
            loss = horizon_loss(logits, y_all[ix].to(device), nf_all[ix].to(device), gen)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(ix)
        row = {"epoch": epoch, "train_loss": tot / len(perm), "seconds": round(time.time() - t0, 1)}
        if val_ix is not None:
            lg = torch.as_tensor(predict_logits(model, A, val_ix, device))
            row["val_loss"] = float(horizon_loss(lg, y_all[val_ix], nf_all[val_ix], None))
            if row["val_loss"] < best - 1e-4:
                best, best_epoch = row["val_loss"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if fixed_epochs is None and epoch - best_epoch >= patience:
                history.append(row)
                break
        history.append(row)
        log(f"  {temporal} seed {seed} epoch {epoch} train {row['train_loss']:.4f} val {row.get('val_loss', float('nan')):.4f} {row['seconds']}s")
    if best_state is not None and fixed_epochs is None:
        model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters())
    return {"model": model, "history": history, "best_epoch": best_epoch or epochs, "best_val_loss": None if best == float("inf") else best,
            "seconds": round(time.time() - t0, 1), "parameters": n_params, "seconds_per_epoch": round((time.time() - t0) / len(history), 2),
            "plays_per_second": round(len(train_ix) * len(history) / max(time.time() - t0, 1e-6))}


def structural_checks(device: str = "cpu") -> dict:
    """Mechanical guarantees, tested on random inputs: order invariance, future isolation, padding isolation, reflection."""
    torch.manual_seed(0)
    out = {}
    for kind in ("gru", "tcn"):
        m = CoverageNet(kind).to(device).eval()
        B, T, D, R = 4, 16, 11, 6
        d, r = torch.randn(B, T, D, 6), torch.randn(B, T, R, 6)
        dm, rm = torch.zeros(B, D, dtype=torch.bool), torch.zeros(B, R, dtype=torch.bool)
        dm[:, :7], rm[:, :5] = True, True
        with torch.no_grad():
            base = m(d, r, dm, rm)
            pd_, pr = torch.randperm(7), torch.randperm(5)
            d2, r2 = d.clone(), r.clone()
            d2[:, :, :7], r2[:, :, :5] = d[:, :, pd_], r[:, :, pr]
            order = float((m(d2, r2, dm, rm) - base).abs().max())
            d3, r3 = d.clone(), r.clone()
            d3[:, 11:], r3[:, 11:] = torch.randn(B, 5, D, 6) * 50, torch.randn(B, 5, R, 6) * 50     # rewrite frames 11..15
            future = float((m(d3, r3, dm, rm)[:, :11] - base[:, :11]).abs().max())
            future_changed = float((m(d3, r3, dm, rm)[:, 11:] - base[:, 11:]).abs().max())
            d4, r4 = d.clone(), r.clone()
            d4[:, :, 7:], r4[:, :, 5:] = 1e3, -1e3                                                   # garbage in masked slots
            padding = float((m(d4, r4, dm, rm) - base).abs().max())
        out[kind] = {"player_order_max_abs_diff": order, "future_frames_max_abs_diff_on_earlier_outputs": future,
                     "later_outputs_do_change": future_changed > 1e-3, "masked_slots_max_abs_diff": padding}
    # reflection: mirroring raw lateral coordinates and re-deriving features equals reflect() on the tensor
    rng = np.random.default_rng(0)
    y, ref, s, ang, o = rng.uniform(5, 48, 20), 26.0, rng.uniform(0, 8, 20), rng.uniform(0, 360, 20), rng.uniform(0, 360, 20)
    def feats(y, ref, ang, o):
        v = relational.unit(ang) * s[:, None]
        return np.stack([np.zeros(20), y - ref, v[:, 0], v[:, 1], relational.unit(o)[:, 0], relational.unit(o)[:, 1]], -1)
    W = config.FIELD_WIDTH
    mirrored = feats(W - y, W - ref, (180 - ang) % 360, (180 - o) % 360)           # mirror left-right: lateral component flips, so angle -> 180 - angle
    out["reflection_max_abs_diff"] = float(np.abs(mirrored - relational.reflect(feats(y, ref, ang, o))).max())
    return out


def save_checks(path: Path | None = None) -> dict:
    res = structural_checks()
    path = Path(path or config.REPORTS / "v2" / "coverage_neural_checks.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    return res
