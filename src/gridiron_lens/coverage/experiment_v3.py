"""Coverage v3 experiments E0-E4. Registered in docs/experiments/coverage_v3.md before the first run.

Development folds only (weeks 1-12). Weeks 15-18 are never read for scoring here. Out-of-fold logits are saved so the
error table, ensembles and calibration comparisons all use predictions made for plays the model did not train on.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from torch import nn

from ..shared import config
from ..shared.provenance import now_utc, write_json
from ..shared.runs import Run
from . import evalkit, neural, relational
from .experiment_a import FOLDS

OUT = config.REPORTS / os.environ.get("GL_OUT", "v3")
HZ = relational.HORIZONS
MAN = ["COVER_0_MAN", "COVER_1_MAN", "COVER_2_MAN"]
ZONE = ["COVER_2_ZONE", "COVER_3_ZONE", "COVER_4_ZONE", "COVER_6_ZONE"]
CLASSES = MAN + ZONE
NOISE = {"pos_sd": 0.15, "vel_sd": 0.3, "frame_hold": 0.10, "drop_def": 0.10, "drop_rec": 0.05, "min_def": 3, "min_rec": 1}
MAX_EPOCHS, PATIENCE, BATCH = int(os.environ.get("GL_MAX_EPOCHS", "50")), 8, 64


def sig(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- models

def attn_pool(x: torch.Tensor, mask: torch.Tensor, dim: int, scorer: nn.Module) -> torch.Tensor:
    """Masked attention-weighted sum and masked max over `dim`. Order-invariant; fully masked sets give zeros."""
    m = mask.unsqueeze(-1)
    w = scorer(x).masked_fill(~m, float("-inf"))
    w = torch.nan_to_num(torch.softmax(w, dim=dim), nan=0.0)
    mx = x.masked_fill(~m, float("-inf")).amax(dim)
    return torch.cat([(x * w).sum(dim), torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))], dim=-1)


class AttentionNet(neural.CoverageNet):
    """Control architecture with attention pooling over receivers and defenders instead of the mean."""

    def __init__(self, **kw):
        super().__init__("gru", **kw)
        e = self.pair[0].out_features
        self.rec_score, self.def_score = nn.Linear(e, 1), nn.Linear(e, 1)

    def trunk(self, d, r, dmask, rmask):
        x = self.pair(neural.pair_features(d, r))
        pm = dmask[:, None, :, None] & rmask[:, None, None, :]
        x = attn_pool(x, pm.expand(x.shape[:-1]), 3, self.rec_score)
        x = self.defender(x)
        x = attn_pool(x, dmask[:, None, :].expand(x.shape[:-1]), 2, self.def_score)
        return self.temporal(x)[0]


class FlatFamilyNet(neural.CoverageNet):
    def __init__(self):
        super().__init__("gru")
        self.flat = nn.Linear(self.head.in_features, len(CLASSES))


def degrade(d: np.ndarray, r: np.ndarray, dm: np.ndarray, rm: np.ndarray, rng: np.random.Generator):
    """Sensor-style corruption. Frame t may be replaced by frame t-1 (never by a later frame). Orientation is not altered."""
    d, r, dm, rm = d.copy(), r.copy(), dm.copy(), rm.copy()
    for a in (d, r):
        a[..., 0:2] += rng.normal(0, NOISE["pos_sd"], a[..., 0:2].shape).astype(np.float32)
        a[..., 2:4] += rng.normal(0, NOISE["vel_sd"], a[..., 2:4].shape).astype(np.float32)
        hold = rng.random(a.shape[:2]) < NOISE["frame_hold"]
        hold[:, 0] = False
        for t in range(1, a.shape[1]):
            a[hold[:, t], t] = a[hold[:, t], t - 1]
    for m, p, keep in ((dm, NOISE["drop_def"], NOISE["min_def"]), (rm, NOISE["drop_rec"], NOISE["min_rec"])):
        drop = (rng.random(m.shape) < p) & m
        ok = (m & ~drop).sum(1) >= keep
        m[ok] &= ~drop[ok]
    return d, r, dm, rm


def batch(A: dict, ix: np.ndarray, rng: np.random.Generator | None, reflect: bool, robust: bool):
    d, r, dm, rm = A["defs"][ix], A["recs"][ix], A["dmask"][ix], A["rmask"][ix]
    if rng is not None and reflect:
        f = rng.random(len(ix)) < 0.5
        d, r = d.copy(), r.copy()
        d[f], r[f] = relational.reflect(d[f]), relational.reflect(r[f])
    if rng is not None and robust:
        d, r, dm, rm = degrade(d, r, dm, rm, rng)
    t = torch.as_tensor
    return t(d), t(r), t(dm), t(rm)


@torch.no_grad()
def logits_of(model, A: dict, ix: np.ndarray, degraded_seed: int | None = None, family: str | None = None) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(ix), 256):
        b = ix[i:i + 256]
        d, r, dm, rm = A["defs"][b], A["recs"][b], A["dmask"][b], A["rmask"][b]
        if degraded_seed is not None:
            d, r, dm, rm = degrade(d, r, dm, rm, np.random.default_rng(degraded_seed + i))
        args = tuple(torch.as_tensor(a) for a in (d, r, dm, rm))
        if family == "flat":
            out.append(torch.log_softmax(model.flat(model.trunk(*args)), -1).numpy())
        elif family == "hier":
            g, fm, fz = model.forward_family(*args)
            lm, lz = nn.functional.logsigmoid(g)[..., None] + torch.log_softmax(fm, -1), nn.functional.logsigmoid(-g)[..., None] + torch.log_softmax(fz, -1)
            out.append(torch.cat([lm, lz], -1).numpy())
        else:
            out.append(model(*args).numpy())
    return np.concatenate(out)


def family_loss(model, kind: str, args, y_fam: torch.Tensor, y_man: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(len(frame))
    if kind == "flat":
        return nn.functional.cross_entropy(model.flat(model.trunk(*args))[rows, frame], y_fam)
    g, fm, fz = model.forward_family(*args)
    g, fm, fz = g[rows, frame], fm[rows, frame], fz[rows, frame]
    is_man = y_man > 0.5
    loss = nn.functional.binary_cross_entropy_with_logits(g, y_man)
    if is_man.any():
        loss = loss + nn.functional.cross_entropy(fm[is_man], y_fam[is_man]) * is_man.float().mean()
    if (~is_man).any():
        loss = loss + nn.functional.cross_entropy(fz[~is_man], y_fam[~is_man] - len(MAN)) * (~is_man).float().mean()
    return loss


def fit(A: dict, tr: np.ndarray, va: np.ndarray, make, seed: int, robust: bool = False, family: str | None = None, yfam: np.ndarray | None = None) -> dict:
    torch.manual_seed(seed)
    rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    model = make()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    y_all, nf_all = torch.as_tensor(A["y"]), torch.as_tensor(A["nframes"])
    hz = torch.tensor(list(HZ.values()))
    best, state, best_epoch, t0 = float("inf"), None, 0, time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), BATCH):
            ix = perm[i:i + BATCH]
            args = batch(A, ix, rng, True, robust)
            if family is None:
                loss = neural.horizon_loss(model(*args), y_all[ix], nf_all[ix], gen)
            else:
                ok = (nf_all[ix][:, None] > hz[None, :]).float() + 1e-9
                frame = hz[torch.multinomial(ok, 1, generator=gen).squeeze(1)]
                loss = family_loss(model, family, args, torch.as_tensor(yfam[ix]), y_all[ix], frame)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if family is None:
            val = float(neural.horizon_loss(torch.as_tensor(logits_of(model, A, va)), y_all[va], nf_all[va], None))
        else:
            lp = logits_of(model, A, va, family=family)
            okv = A["nframes"][va][:, None] > np.array(list(HZ.values()))[None, :]
            val = float(np.sum([-(lp[okv[:, j], f][np.arange(okv[:, j].sum()), yfam[va][okv[:, j]]]).sum() for j, f in enumerate(HZ.values())]) / okv.sum())
        if val < best - 1e-4:
            best, best_epoch, state = val, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(state)
    return {"model": model, "best_epoch": best_epoch, "val_loss": best, "seconds": round(time.time() - t0, 1), "parameters": sum(p.numel() for p in model.parameters())}


def horizon_table(y: np.ndarray, nframes: np.ndarray, logits: np.ndarray) -> dict:
    out = {}
    for h, f in HZ.items():
        ok = nframes > f
        m = evalkit.binary_metrics(y[ok].astype(int), sig(logits[ok, f]))
        m["man_pr_auc"] = float(average_precision_score(y[ok], logits[ok, f]))
        out[h] = {k: m[k] for k in ("n", "log_loss", "brier", "accuracy", "balanced_accuracy", "macro_f1", "man_precision", "man_recall", "man_pr_auc")}
    return out


def paired_game_boot(d: np.ndarray, games: np.ndarray, n: int = 2000, seed: int = 20261004) -> dict:
    ug = np.unique(games)
    by = [d[games == g] for g in ug]
    rng = np.random.default_rng(seed)
    means = [np.concatenate([by[j] for j in rng.integers(0, len(ug), len(ug))]).mean() for _ in range(n)]
    return {"mean_difference": float(d.mean()), "interval_95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))], "games": len(ug)}


def ll_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


CONFIGS = {
    "E0 control (gru+reflect)": {"make": lambda: neural.CoverageNet("gru"), "robust": False, "seeds": [42, 7, 2026]},
    "E1 attention pooling": {"make": AttentionNet, "robust": False, "seeds": [42]},
    "E2 robustness training": {"make": lambda: neural.CoverageNet("gru"), "robust": True, "seeds": [42]},
}


def run(only: list[str] | None = None, log=print) -> dict:
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    A = neural.load_arrays()
    lab = ~np.isnan(A["y"])
    ck = OUT / "coverage_v3_partial.json"
    saved = json.loads(ck.read_text()) if ck.exists() else {}
    oof_path = OUT / "coverage_v3_oof.npz"
    oof = dict(np.load(oof_path)) if oof_path.exists() else {}
    val_all = np.concatenate([np.where(lab & np.isin(A["week"], va))[0] for _, va in FOLDS])
    oof["index"], oof["fold"] = val_all, np.concatenate([np.full((lab & np.isin(A["week"], va)).sum(), i) for i, (_, va) in enumerate(FOLDS)])

    for name, cfg in CONFIGS.items():
        if only and not any(name.startswith(o) for o in only):
            continue
        for seed in cfg["seeds"]:
            key = f"{name}|seed{seed}"
            if key in saved:
                continue
            folds, lg_clean, lg_deg = [], [], []
            for fi, (tr_w, va_w) in enumerate(FOLDS):
                tr, va = np.where(lab & np.isin(A["week"], tr_w))[0], np.where(lab & np.isin(A["week"], va_w))[0]
                r = fit(A, tr, va, cfg["make"], seed, cfg["robust"])
                lc, ld = logits_of(r["model"], A, va), logits_of(r["model"], A, va, degraded_seed=1000 + fi)
                lg_clean.append(lc), lg_deg.append(ld)
                folds.append({"fold": fi, "train_plays": len(tr), "validate_plays": len(va), "best_epoch": r["best_epoch"], "val_loss": r["val_loss"], "seconds": r["seconds"],
                              "parameters": r["parameters"], "clean": horizon_table(A["y"][va], A["nframes"][va], lc), "degraded": horizon_table(A["y"][va], A["nframes"][va], ld)})
                log(f"{key} fold {fi}: epoch {r['best_epoch']} val {r['val_loss']:.4f} +1.5s ll {folds[-1]['clean']['post_1_5s']['log_loss']:.4f} degraded {folds[-1]['degraded']['post_1_5s']['log_loss']:.4f} ({r['seconds']}s)")
            oof[f"{key}|clean"], oof[f"{key}|degraded"] = np.concatenate(lg_clean), np.concatenate(lg_deg)
            saved[key] = {"folds": folds, "mean_val_loss": float(np.mean([f["val_loss"] for f in folds])), "parameters": folds[0]["parameters"],
                          "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)}
            np.savez_compressed(oof_path, **oof)
            write_json(ck, saved)

    # E4: coverage family, flat vs hierarchical, same folds
    fam = np.array([CLASSES.index(c) if c in CLASSES else -1 for c in A["family"]])
    for kind, make in (("flat", FlatFamilyNet), ("hier", lambda: neural.CoverageNet("gru", n_family=(len(MAN), len(ZONE))))):
        key = f"E4 family {kind}|seed42"
        if key in saved or (only and not any("E4".startswith(o) for o in only)):
            continue
        folds = []
        for fi, (tr_w, va_w) in enumerate(FOLDS):
            tr, va = np.where(lab & (fam >= 0) & np.isin(A["week"], tr_w))[0], np.where(lab & (fam >= 0) & np.isin(A["week"], va_w))[0]
            r = fit(A, tr, va, make, 42, family=kind, yfam=fam)
            lp = logits_of(r["model"], A, va, family=kind)
            f15 = A["nframes"][va] > HZ["post_1_5s"]
            y, pr = fam[va][f15], np.exp(lp[f15, HZ["post_1_5s"]])
            pred = pr.argmax(1)
            prc, rc, _, sup = precision_recall_fscore_support(y, pred, labels=list(range(len(CLASSES))), zero_division=0)
            folds.append({"fold": fi, "plays": int(f15.sum()), "best_epoch": r["best_epoch"], "seconds": r["seconds"], "accuracy": float(accuracy_score(y, pred)),
                          "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(len(CLASSES))), zero_division=0)),
                          "log_loss": float(log_loss(y, np.clip(pr, 1e-6, 1), labels=list(range(len(CLASSES))))),
                          "per_class": {c: {"precision": float(prc[i]), "recall": float(rc[i]), "support": int(sup[i])} for i, c in enumerate(CLASSES)}})
            log(f"{key} fold {fi}: acc {folds[-1]['accuracy']:.4f} macro F1 {folds[-1]['macro_f1']:.4f} log loss {folds[-1]['log_loss']:.4f} ({r['seconds']}s)")
        saved[key] = {"folds": folds, "cutoff": "post_1_5s", "excluded": ["PREVENT"]}
        write_json(ck, saved)
    return summarize(A, saved, oof, log)


def nested(val_fold: np.ndarray, fn) -> np.ndarray:
    """Apply `fn(train_mask) -> predictor` fitted on the other folds to each fold."""
    out = None
    for k in np.unique(val_fold):
        pred = fn(val_fold != k)
        res = pred(val_fold == k)
        out = np.full((len(val_fold),) + res.shape[1:], np.nan) if out is None else out
        out[val_fold == k] = res
    return out


def summarize(A: dict, saved: dict, oof: dict, log=print) -> dict:
    ix, fold = oof["index"], oof["fold"]
    y, nf, games = A["y"][ix], A["nframes"][ix], A["game"][ix]
    f15 = HZ["post_1_5s"]
    ok15 = nf > f15
    ctrl_key = "E0 control (gru+reflect)|seed42"
    rep = {"module": "coverage", "version": "coverage-v3-experiments", "created_at_utc": now_utc(), "registered_in": "docs/experiments/coverage_v3.md",
           "split_role": "development folds within weeks 1-12 (out-of-fold). Weeks 15-18 were not scored.", "configurations_run": len(saved), "noise": NOISE, "models": {}}
    for key, s in saved.items():
        if key.startswith("E4"):
            continue
        entry = {"mean_val_loss": s["mean_val_loss"], "fold_val_loss": [f["val_loss"] for f in s["folds"]], "parameters": s["parameters"], "seconds": sum(f["seconds"] for f in s["folds"]),
                 "peak_memory_mb": s["peak_memory_mb"], "pooled_clean": horizon_table(y, nf, oof[f"{key}|clean"]), "pooled_degraded": horizon_table(y, nf, oof[f"{key}|degraded"])}
        if key != ctrl_key and ctrl_key in saved:
            d = ll_rows(y[ok15], sig(oof[f"{key}|clean"][ok15, f15])) - ll_rows(y[ok15], sig(oof[f"{ctrl_key}|clean"][ok15, f15]))
            entry["vs_control_post_1_5s"] = paired_game_boot(d, games[ok15])
            entry["lower_in_every_fold"] = bool(all(a < b for a, b in zip(entry["fold_val_loss"], [f["val_loss"] for f in saved[ctrl_key]["folds"]])))
            c15, m15 = rep["models"][ctrl_key]["pooled_clean"]["post_1_5s"], entry["pooled_clean"]["post_1_5s"]
            entry["guardrails_ok"] = bool(all(m15[k] >= c15[k] - 0.01 for k in ("macro_f1", "man_precision", "man_recall")))
            if key.startswith("E0"):                                   # another seed of the control is not a candidate: it measures seed-to-seed variation
                entry["role"] = "control, different seed (not a candidate)"
            else:
                entry["passes_rule"] = bool(entry["lower_in_every_fold"] and entry["vs_control_post_1_5s"]["interval_95"][1] < 0 and entry["guardrails_ok"])
        rep["models"][key] = entry

    # E3: ensembles and calibration, all fitted on the other folds
    seeds = [k for k in saved if k.startswith("E0") and f"{k}|clean" in oof]
    if len(seeds) == 3:
        ens = np.mean([sig(oof[f"{k}|clean"]) for k in seeds], axis=0)
        e3 = {"three_seed_average": {h: evalkit.binary_metrics(y[nf > f].astype(int), ens[nf > f, f]) for h, f in HZ.items()}}
        d = ll_rows(y[ok15], ens[ok15, f15]) - ll_rows(y[ok15], sig(oof[f"{ctrl_key}|clean"][ok15, f15]))
        e3["three_seed_average_vs_control_post_1_5s"] = paired_game_boot(d, games[ok15])
        # relational boosted trees out-of-fold at +1.5 s, then a blend weight fitted on the other folds
        from .benchmark_v2 import table
        t, g, r = table("post_1_5s")
        keys = [f"{a}:{b}" for a, b in zip(t["gameId"].to_list(), t["playId"].to_list())]
        pos = {k: i for i, k in enumerate(keys)}
        x = t.select(g + r).to_numpy().astype(np.float32)
        tree = np.full(len(ix), np.nan)
        wk, key_all = A["week"], A["key"]
        for fi, (tr_w, va_w) in enumerate(FOLDS):
            tr_rows = [pos[k] for k, w, lb, n in zip(key_all, wk, ~np.isnan(A["y"]), A["nframes"]) if w in tr_w and lb and n > f15 and k in pos]
            lab_map = dict(zip(key_all, A["y"]))
            ytr = np.array([lab_map[keys[i]] for i in tr_rows])
            m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=300, l2_regularization=1.0, random_state=20261004).fit(x[tr_rows], ytr)
            sel = np.where((fold == fi) & ok15)[0]
            rows = [pos.get(key_all[ix[j]]) for j in sel]
            have = [j for j, rr in zip(sel, rows) if rr is not None]
            tree[have] = m.decision_function(x[[rr for rr in rows if rr is not None]])
        both = ok15 & ~np.isnan(tree)
        tl, cl = tree[both], oof[f"{ctrl_key}|clean"][both, f15]
        yb, fb, gb = y[both], fold[both], games[both]

        def blend(trm):
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.c_[cl[trm], tl[trm]], yb[trm])
            return lambda tem: lr.predict_proba(np.c_[cl[tem], tl[tem]])[:, 1]

        def platt(trm):
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(cl[trm, None], yb[trm])
            return lambda tem: lr.predict_proba(cl[tem, None])[:, 1]

        def temperature(trm):
            ts = np.linspace(0.5, 3.0, 126)
            tbest = ts[int(np.argmin([ll_rows(yb[trm], sig(cl[trm] / tt)).mean() for tt in ts]))]
            return lambda tem: sig(cl[tem] / tbest)

        for nm, fn in (("temporal + relational trees blend (nested)", blend), ("Platt on the control (nested)", platt), ("temperature on the control (nested)", temperature)):
            p = nested(fb, fn)
            e3[nm] = evalkit.binary_metrics(yb.astype(int), p) | {"vs_uncalibrated_control": paired_game_boot(ll_rows(yb, p) - ll_rows(yb, sig(cl)), gb),
                                                                 "reliability": evalkit.reliability(yb, p)}
        e3["uncalibrated control, same plays"] = evalkit.binary_metrics(yb.astype(int), sig(cl))
        e3["tree only, same plays"] = evalkit.binary_metrics(yb.astype(int), sig(tl))
        rep["E3_ensembles_and_calibration"] = e3
    fam = {k: v for k, v in saved.items() if k.startswith("E4")}
    if fam:
        rep["E4_family"] = {k: {"mean": {m: float(np.mean([f[m] for f in v["folds"]])) for m in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss")},
                                "per_class_recall_mean": {c: float(np.mean([f["per_class"][c]["recall"] for f in v["folds"]])) for c in CLASSES},
                                "per_class_support_total": {c: int(sum(f["per_class"][c]["support"] for f in v["folds"])) for c in CLASSES}, "folds": v["folds"]} for k, v in fam.items()}
        rep["E4_note"] = "Development only, +1.5 s cutoff, uncalibrated. PREVENT excluded for too few plays. The group in the hierarchical head is predicted, never supplied."
    seeds_ll = [v["pooled_clean"]["post_1_5s"]["log_loss"] for k, v in rep["models"].items() if k.startswith("E0")]
    rep["seed_variation"] = {"control_log_loss_post_1_5s_by_seed": seeds_ll, "range": float(max(seeds_ll) - min(seeds_ll)),
                             "note": "Three seeds of the same control differ by this much. A candidate's gain has to be read against it; one control seed differs from another by more than either candidate differs from the control."}
    passing = [k for k, v in rep["models"].items() if v.get("passes_rule")]
    rep["decision"] = {"candidates_passing_rule": passing, "champion": "v2 temporal (unchanged)" if not passing else "see finalists",
                       "statement": "No candidate met the registered rule; the v2 temporal model stays the champion." if not passing else "A candidate met the rule on seed 42; finalist seeds are required before promotion."}
    write_json(OUT / "coverage_v3_experiments.json", rep)
    run = Run("coverage", "v3-experiments", "v3", config_={"folds": FOLDS, "noise": NOISE}, seed=42)
    run.set(target="released man/zone label", population="labelled plays in weeks 1-12 with the prefix available", input_cutoff=HZ, split={"folds": FOLDS},
            metrics={k: {"mean_val_loss": v["mean_val_loss"], "post_1_5s_log_loss": v["pooled_clean"]["post_1_5s"]["log_loss"]} for k, v in rep["models"].items()})
    run.output("report", OUT / "coverage_v3_experiments.json")
    run.finish()
    log(json.dumps({k: {"val": round(v["mean_val_loss"], 4), "ll15": round(v["pooled_clean"]["post_1_5s"]["log_loss"], 4), "deg15": round(v["pooled_degraded"]["post_1_5s"]["log_loss"], 4),
                        "pass": v.get("passes_rule")} for k, v in rep["models"].items()}, indent=1))
    return rep


if __name__ == "__main__":
    run(sys.argv[1:] or None)
