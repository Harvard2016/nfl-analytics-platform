"""Highlight ranking experiments H0-H3 and their evaluation. See pipeline.py for the target and its limits.

H0  loudness above a local background (no training).
H1  commentary words: TF-IDF + regularized logistic regression, vocabulary fitted on training games only.
H2  released audio/visual embeddings reduced to 64 components per modality (fitted on training games), then
    logistic regression or a bounded boosted-tree model; single modalities and simple concatenation.
H3  small temporal-convolution model over 32-clip windows with explicit modality masks.

Scores are ranking scores for agreement with editorial selection. They are not calibrated probabilities: with 6
validation games there are too few held-out groups to fit and check a calibrator.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn

from ..shared import config
from ..shared.provenance import write_json
from ..shared.runs import Run
from . import pipeline as P

N_COMP = 64
BACKGROUND_CLIPS = 151          # 5-minute centred window for the loudness background (offline processing)
WINDOW, STRIDE = 32, 16
BUDGETS = {"1 minute": 30, "3 minutes": 90, "5 minutes": 150}
SEED = 20261004
STREAMS = ["vid_clip", "vid_slowfast", "aud_pann", "loudness"]


# ------------------------------------------------------------------ features

def loudness_features(vol: np.ndarray) -> np.ndarray:
    """[level above local median, change from the previous clip, raw level standardized within the game]."""
    finite = np.isfinite(vol)                      # silent clips come through as -inf or NaN; treat them as the quietest observed level
    vol = np.where(finite, vol, vol[finite].min() if finite.any() else 0.0).astype(np.float32)
    pad = BACKGROUND_CLIPS // 2
    padded = np.pad(vol, pad, mode="edge")
    med = np.array([np.median(padded[i:i + BACKGROUND_CLIPS]) for i in range(len(vol))], np.float32)
    return np.stack([vol - med, np.diff(vol, prepend=vol[0]), (vol - vol.mean()) / (vol.std() + 1e-6)], 1).astype(np.float32)


def clip_documents(vid: int, n: int) -> list[str]:
    """Commentary around each clip: words spoken from 2 s before to 6 s after ("n_"), and 6-20 s after ("a_").

    Commentators usually describe a play after it happens, so later words are informative. Offline processing only.
    """
    words = json.loads((P.PROC / "whisper" / f"american_football_{vid}.json").read_text())
    now, after = [[] for _ in range(n)], [[] for _ in range(n)]
    for w in words:
        tok = "".join(ch for ch in w["word"].lower() if ch.isalnum() or ch == "'")
        if not tok or "start" not in w:
            continue
        t = float(w["start"])
        for i in range(max(0, int((t - 6) // 2)), min(n, int((t + 2) // 2) + 1)):
            if 2 * i - 2 <= t < 2 * i + 6:
                now[i].append("n_" + tok)
        for i in range(max(0, int((t - 20) // 2)), min(n, int((t - 6) // 2) + 1)):
            if 2 * i + 6 <= t < 2 * i + 20:
                after[i].append("a_" + tok)
    return [" ".join(a + b) for a, b in zip(now, after)]


def build_features(split: dict, log=print) -> None:
    """Reduce each embedding to 64 components with transforms fitted on training games only; cache per game."""
    P.FEAT.mkdir(parents=True, exist_ok=True)
    vol = P.load_volume()
    nclips = {v: len(y) for v, y in P.load_labels().items()}
    comps = {}
    for name in P.MODALITIES:
        t0 = time.time()
        ipca = IncrementalPCA(n_components=N_COMP)
        for v in split["train"]:
            x = P.load_modality(v, name, nclips[v])
            for i in range(0, len(x), 4096):
                if len(x) - i >= N_COMP:
                    ipca.partial_fit(x[i:i + 4096])
        tr = np.concatenate([ipca.transform(P.load_modality(v, name, nclips[v])) for v in split["train"]])
        comps[name] = (ipca, tr.mean(0), tr.std(0) + 1e-6)
        log(f"{name}: PCA to {N_COMP} fitted on {len(split['train'])} training games, explained variance {ipca.explained_variance_ratio_.sum():.3f} ({time.time() - t0:.0f}s)")
    info = {"components": N_COMP, "explained_variance": {k: float(v[0].explained_variance_ratio_.sum()) for k, v in comps.items()},
            "fitted_on": "training games only"}
    for v in P.video_ids():
        out = {name: ((ipca.transform(P.load_modality(v, name, nclips[v])) - mu) / sd).astype(np.float32) for name, (ipca, mu, sd) in comps.items()}
        np.savez_compressed(P.FEAT / f"game_{v}.npz", loudness=loudness_features(vol[v]), **out)
    write_json(P.FEAT / "features_report.json", info)


def load_game(v: int) -> dict:
    with np.load(P.FEAT / f"game_{v}.npz") as z:
        return {k: np.nan_to_num(z[k], nan=0.0, posinf=0.0, neginf=0.0) for k in z.files}


def stack(games: list[int], labels: dict, streams: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, gs = [], [], []
    for v in games:
        g = load_game(v)
        xs.append(np.concatenate([g[s] for s in streams], 1)), ys.append(labels[v]), gs.append(np.full(len(labels[v]), v))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)


# ------------------------------------------------------------------ metrics

def runs_of_ones(y: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], y, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def select_budget(score: np.ndarray, k: int, block: int = 5) -> np.ndarray:
    """Greedy pick of the highest-scoring 10-second blocks without overlap until k clips are selected. No labels involved."""
    sm = np.convolve(score, np.ones(block) / block, mode="same")
    chosen = np.zeros(len(score), bool)
    for i in np.argsort(-sm):
        lo, hi = max(0, i - block // 2), min(len(score), i + block // 2 + 1)
        if chosen[lo:hi].any():
            continue
        chosen[lo:hi] = True
        if chosen.sum() >= k:
            break
    return chosen


def game_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    out = {"clips": len(y), "positive": int(y.sum()), "average_precision": float(average_precision_score(y, score)),
           "hit_at_1": bool(y[int(np.argmax(score))] == 1), "chance_average_precision": float(y.mean())}
    segs = runs_of_ones(y)
    for name, k in BUDGETS.items():
        sel = select_budget(score, k)
        inter = int((sel & (y == 1)).sum())
        out[name] = {"selected_clips": int(sel.sum()), "precision": inter / max(1, int(sel.sum())), "recall_of_highlight_time": inter / max(1, int(y.sum())),
                     "temporal_iou": inter / max(1, int((sel | (y == 1)).sum())),
                     "highlight_segments": len(segs), "segments_missed_entirely": int(sum(not sel[a:b].any() for a, b in segs))}
    return out


def summarize(per_game: dict[int, dict]) -> dict:
    g = list(per_game.values())
    out = {"games": len(g), "mean_average_precision": float(np.mean([x["average_precision"] for x in g])),
           "average_precision_range": [float(min(x["average_precision"] for x in g)), float(max(x["average_precision"] for x in g))],
           "hit_at_1": float(np.mean([x["hit_at_1"] for x in g])), "chance_average_precision": float(np.mean([x["chance_average_precision"] for x in g]))}
    for name in BUDGETS:
        out[name] = {k: float(np.mean([x[name][k] for x in g])) for k in ("precision", "recall_of_highlight_time", "temporal_iou")}
        out[name]["segments_missed_share"] = float(sum(x[name]["segments_missed_entirely"] for x in g) / max(1, sum(x[name]["highlight_segments"] for x in g)))
    return out


def evaluate(scores: dict[int, np.ndarray], labels: dict, games: list[int]) -> dict:
    per = {v: game_metrics(labels[v], scores[v]) for v in games}
    return {"summary": summarize(per), "per_game": {str(v): per[v] for v in games}}


# ------------------------------------------------------------------ H3 temporal model

class TemporalFusion(nn.Module):
    """Each stream is projected to 64 dims, multiplied by its availability mask, concatenated with the mask bits, then
    three 1-D convolution blocks (kernel 5, symmetric padding: the model sees clips on both sides) give one score per clip."""

    def __init__(self, dims: dict[str, int], hidden: int = 128, dropout: float = 0.1, blocks: int = 3):
        super().__init__()
        self.names = list(dims)
        self.proj = nn.ModuleDict({k: nn.Linear(d, 64) for k, d in dims.items()})
        layers, c = [], 64 * len(dims) + len(dims)
        for _ in range(blocks):
            layers += [nn.Conv1d(c, hidden, 5, padding=2), nn.ReLU(), nn.Dropout(dropout)]
            c = hidden
        self.conv = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, 1, 1)

    def forward(self, x: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:      # x[k]: (B, T, d), mask: (B, S)
        parts = [torch.relu(self.proj[k](x[k])) * mask[:, i, None, None] for i, k in enumerate(self.names)]
        h = torch.cat(parts + [mask[:, None, :].expand(-1, parts[0].shape[1], -1)], dim=-1).transpose(1, 2)
        return self.head(self.conv(h)).squeeze(1)


def score_game(model: TemporalFusion, g: dict, streams: list[str], drop: list[str] = ()) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = {k: torch.as_tensor(g[k])[None] for k in streams}
        mask = torch.tensor([[0.0 if k in drop else 1.0 for k in streams]])
        return model(x, mask)[0].numpy()


def train_temporal(split: dict, labels: dict, streams: list[str], seed: int = 42, max_epochs: int = 30, patience: int = 5, modality_dropout: float = 0.15,
                   log=print) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    games = {v: load_game(v) for v in split["train"] + split["validation"]}
    dims = {k: games[split["train"][0]][k].shape[1] for k in streams}
    model = TemporalFusion(dims)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    windows = [(v, s) for v in split["train"] for s in range(0, len(labels[v]) - WINDOW + 1, STRIDE)]
    best, best_state, best_epoch, hist, t0 = -1.0, None, 0, [], time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order, tot = rng.permutation(len(windows)), 0.0
        for i in range(0, len(order), 64):
            b = [windows[j] for j in order[i:i + 64]]
            x = {k: torch.as_tensor(np.stack([games[v][k][s:s + WINDOW] for v, s in b])) for k in streams}
            y = torch.as_tensor(np.stack([labels[v][s:s + WINDOW] for v, s in b]), dtype=torch.float32)
            mask = torch.as_tensor((rng.random((len(b), len(streams))) > modality_dropout).astype(np.float32))
            mask[mask.sum(1) == 0] = 1.0
            loss = nn.functional.binary_cross_entropy_with_logits(model(x, mask), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        val = float(np.mean([average_precision_score(labels[v], score_game(model, games[v], streams)) for v in split["validation"]]))
        hist.append({"epoch": epoch, "train_loss": tot / len(windows), "validation_map": val, "seconds": round(time.time() - t0, 1)})
        log(f"  H3 epoch {epoch} loss {tot / len(windows):.4f} val mAP {val:.4f}")
        if val > best + 1e-4:
            best, best_epoch, best_state = val, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return {"model": model, "history": hist, "best_epoch": best_epoch, "validation_map": best, "seconds": round(time.time() - t0, 1),
            "parameters": sum(p.numel() for p in model.parameters()), "windows": len(windows), "streams": streams,
            "peak_note": "trained on CPU; one game is scored in a single forward pass"}


# ------------------------------------------------------------------ experiment driver

PROBES = [
    ("plain call", "touchdown"), ("negated", "no touchdown"), ("ruled out", "it is not a touchdown the pass is incomplete"),
    ("historical", "he had a touchdown last week against dallas"), ("hypothetical", "that would have been a touchdown if he catches it"),
    ("replay talk", "let's take another look at that touchdown on the replay"), ("interception call", "intercepted"), ("routine", "second down and seven"),
]


def run(log=print) -> dict:
    run_rec = Run("highlights", "ranking-v1", seed=SEED)
    aud = P.audit()
    split = P.make_split()
    labels, vol = P.load_labels(), P.load_volume()
    if not (P.FEAT / "features_report.json").exists():
        build_features(split, log)
    tr, va, te = split["train"], split["validation"], split["test"]
    scores: dict[str, dict[int, np.ndarray]] = {}
    info: dict[str, dict] = {}

    # H0: loudness above local background (no training)
    scores["H0 loudness"] = {v: loudness_features(vol[v])[:, 0] for v in P.video_ids()}
    info["H0 loudness"] = {"inputs": "audio loudness per 2-second clip", "method": "level minus the median of the surrounding 5 minutes", "trained": False}

    # H1: commentary TF-IDF + logistic regression
    docs = {v: clip_documents(v, len(labels[v])) for v in P.video_ids()}
    vec = TfidfVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 2), min_df=5, max_features=30000, sublinear_tf=True)
    xtr = vec.fit_transform([d for v in tr for d in docs[v]])
    lr_text = LogisticRegression(C=1.0, max_iter=2000).fit(xtr, np.concatenate([labels[v] for v in tr]))
    scores["H1 commentary"] = {v: lr_text.decision_function(vec.transform(docs[v])) for v in P.video_ids()}
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(lr_text.coef_[0])
    probe = [{"case": c, "text": t, "score": round(float(lr_text.decision_function(vec.transform([" ".join("n_" + w for w in t.split())]))[0]), 3)} for c, t in PROBES]
    info["H1 commentary"] = {"inputs": "WhisperX word timestamps: words from 2 s before to 6 s after the clip, and 6-20 s after", "vocabulary": len(names),
                             "vocabulary_fitted_on": "training games only", "top_positive_terms": names[order[-25:]][::-1].tolist(),
                             "top_negative_terms": names[order[:15]].tolist(), "probes": probe,
                             "probe_note": "Scores for short phrases placed at the clip. A word is evidence about what the announcers said, not proof of what happened."}

    # H2: reduced embeddings, single modalities and concatenation
    for label_, streams in (("H2 visual CLIP", ["vid_clip"]), ("H2 visual SlowFast", ["vid_slowfast"]), ("H2 audio PANN", ["aud_pann"]),
                            ("H2 all embeddings + loudness", STREAMS)):
        x, y, _ = stack(tr, labels, streams)
        m = LogisticRegression(C=0.5, max_iter=1000).fit(x, y)
        scores[f"{label_}, logistic"] = {v: m.decision_function(stack([v], labels, streams)[0]) for v in P.video_ids()}
        info[f"{label_}, logistic"] = {"inputs": streams, "method": "logistic regression on 64 PCA components per modality"}
    x, y, _ = stack(tr, labels, STREAMS)
    gbm = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, l2_regularization=1.0, random_state=SEED).fit(x, y)
    scores["H2 all embeddings + loudness, boosted trees"] = {v: gbm.decision_function(stack([v], labels, STREAMS)[0]) for v in P.video_ids()}
    info["H2 all embeddings + loudness, boosted trees"] = {"inputs": STREAMS, "method": "boosted trees, depth 4, 200 rounds"}

    # H3: temporal fusion
    t3 = train_temporal(split, labels, STREAMS, log=log)
    scores["H3 temporal fusion"] = {v: score_game(t3["model"], load_game(v), STREAMS) for v in P.video_ids()}
    info["H3 temporal fusion"] = {"inputs": STREAMS, "method": "64-d projection per stream with modality masks, three convolution blocks over 32-clip windows (sees clips on both sides)",
                                  **{k: t3[k] for k in ("best_epoch", "validation_map", "seconds", "parameters", "windows")}, "history": t3["history"]}
    for d in ("aud_pann", "vid_clip", "vid_slowfast", "loudness"):                     # masked-stream ablation of the trained model
        scores[f"H3 without {d}"] = {v: score_game(t3["model"], load_game(v), STREAMS, drop=[d]) for v in P.video_ids()}
        info[f"H3 without {d}"] = {"inputs": [s for s in STREAMS if s != d], "method": "same trained H3 model with one stream masked at scoring time"}
    # late fusion of the commentary score with H3, weight fixed at 0.5 on within-game z-scores (no tuning)
    z = lambda a: (a - a.mean()) / (a.std() + 1e-6)
    scores["H3 + commentary (equal-weight z-score sum)"] = {v: z(scores["H3 temporal fusion"][v]) + z(scores["H1 commentary"][v]) for v in P.video_ids()}
    info["H3 + commentary (equal-weight z-score sum)"] = {"inputs": STREAMS + ["commentary"], "method": "sum of within-game z-scores of H3 and H1; weights fixed in advance"}

    results = {name: {"info": info[name], "validation": evaluate(s, labels, va), "test": evaluate(s, labels, te)} for name, s in scores.items()}
    main = [n for n in results if not n.startswith("H3 without")]
    shipped = max(main, key=lambda n: results[n]["validation"]["summary"]["mean_average_precision"])
    rep = {
        "module": "highlights", "task": "ranking agreement with editorial highlight selection", "run_id": run_rec.record["run_id"],
        "source": P.SOURCE, "revision": P.REVISION, "scope": f"{aud['videos']} NFL games, seasons {aud['seasons_in_titles'][0]}-{aud['seasons_in_titles'][-1]}, all from the NFL channel",
        "split": {k: split[k] for k in ("version", "train", "validation", "test", "rule")}, "clips": aud["clips"], "positive_share": aud["positive_share"],
        "label": aud["label_definition"], "score_meaning": "Ranking score for agreement with editorial selection. Not a calibrated probability.",
        "calibration": "Not fitted: 6 validation games are too few held-out groups to fit and check a calibrator.",
        "captions": "Model-generated segment captions were not used by any model here.",
        "budget_note": "Budget metrics pick the top-scoring 10-second blocks up to 1, 3 or 5 minutes without looking at labels. They differ from benchmark top-K metrics that use the number of true highlight clips.",
        "selection_rule": "shipped model = highest mean average precision on the 6 validation games", "shipped": shipped,
        "results": results,
    }
    write_json(P.OUT / "highlights_ranking.json", rep)
    np.savez_compressed(P.OUT / "highlights_scores.npz", **{f"{n}|{v}": s.astype(np.float32) for n, d in scores.items() for v, s in d.items() if n in (shipped, "H0 loudness", "H1 commentary", "H3 temporal fusion")})
    mdir = config.MODELS / "highlights"
    mdir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": t3["model"].state_dict(), "streams": STREAMS}, mdir / "temporal_fusion.pt")
    joblib.dump({"vectorizer": vec, "model": lr_text}, mdir / "commentary_tfidf_logistic.joblib")
    run_rec.set(target=rep["task"], population=f"{aud['clips']} two-second clips in {aud['videos']} games", input_cutoff="offline: context on both sides of a clip",
                split=rep["split"], features={"components_per_modality": N_COMP}, calibration=rep["calibration"],
                metrics={n: {"validation_map": r["validation"]["summary"]["mean_average_precision"], "test_map": r["test"]["summary"]["mean_average_precision"]} for n, r in results.items()})
    run_rec.output("report", P.OUT / "highlights_ranking.json")
    run_rec.finish()
    return rep
