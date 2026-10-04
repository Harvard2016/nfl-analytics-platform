"""Highlights v3 experiments on the grouped development folds. Registered in docs/experiments/highlights_v3.md first.

Every learned transform is fitted inside the fold. Validation and test games are never read here.
"""
from __future__ import annotations

import json
import re
import resource
import sys
import time

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn

from ..shared import config
from ..shared.provenance import now_utc, write_json
from ..shared.runs import Run
from . import decode as D
from . import models as M
from . import pipeline as P

OUT = config.REPORTS / "v3"
EPOCHS, SEED = 5, 42
LEX = {
    "negation": {"no", "not", "never", "incomplete", "can't", "couldn't", "doesn't", "didn't", "won't", "isn't", "wasn't", "short"},
    "reversal": {"overturned", "reversed", "nullified", "penalty", "flag", "holding", "offside", "ruled", "review", "challenge", "back"},
    "hypothetical": {"would", "could", "if", "almost", "nearly", "should"},
    "replay": {"replay", "look", "again", "watch", "angle", "see"},
    "past": {"last", "earlier", "week", "season", "year", "ago", "career"},
}


def fold_features(train: list[int], games: list[int], nclips: dict, vol: dict, log) -> dict[int, dict]:
    """Per-fold PCA and scaling fitted on `train`, applied to `games`. Kept in memory; the v2 cache is not touched."""
    out = {v: {"loudness": M.loudness_features(vol[v])} for v in games}
    for name in P.MODALITIES:
        t0 = time.time()
        ipca = IncrementalPCA(n_components=M.N_COMP)
        for v in train:
            x = P.load_modality(v, name, nclips[v])
            for i in range(0, len(x), 4096):
                if len(x) - i >= M.N_COMP:
                    ipca.partial_fit(x[i:i + 4096])
        tr = np.concatenate([ipca.transform(P.load_modality(v, name, nclips[v])) for v in train])
        mu, sd = tr.mean(0), tr.std(0) + 1e-6
        for v in games:
            out[v][name] = np.nan_to_num(((ipca.transform(P.load_modality(v, name, nclips[v])) - mu) / sd).astype(np.float32))
        log(f"  {name}: fold PCA fitted on {len(train)} games ({time.time() - t0:.0f}s)")
    for v in games:
        out[v]["loudness"] = np.nan_to_num(out[v]["loudness"], nan=0.0, posinf=0.0, neginf=0.0)
    return out


def train(feats: dict[int, dict], labels: dict, games: list[int], variant: str) -> M.TemporalFusion:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = M.TemporalFusion({k: feats[games[0]][k].shape[1] for k in M.STREAMS})
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    per_game = {v: list(range(0, len(labels[v]) - M.WINDOW + 1, M.STRIDE)) for v in games}
    for _ in range(EPOCHS):
        if variant == "H-W":                                                     # every game contributes the same number of windows
            k = min(len(s) for s in per_game.values())
            windows = [(v, s) for v in games for s in rng.choice(per_game[v], k, replace=False)]
        else:
            windows = [(v, s) for v in games for s in per_game[v]]
        order = rng.permutation(len(windows))
        model.train()
        for i in range(0, len(order), 64):
            b = [windows[j] for j in order[i:i + 64]]
            x = {k: torch.as_tensor(np.stack([feats[v][k][s:s + M.WINDOW] for v, s in b])) for k in M.STREAMS}
            y = torch.as_tensor(np.stack([labels[v][s:s + M.WINDOW] for v, s in b]), dtype=torch.float32)
            mask = torch.as_tensor((rng.random((len(b), len(M.STREAMS))) > 0.15).astype(np.float32))
            mask[mask.sum(1) == 0] = 1.0
            out = model(x, mask)
            loss = nn.functional.binary_cross_entropy_with_logits(out, y, pos_weight=torch.tensor(3.0) if variant == "H-W" else None)
            if variant == "H-R":                                                 # pairwise term within each window: positives should outrank negatives
                pos, neg = y > 0.5, y < 0.5
                has = pos.any(1) & neg.any(1)
                if has.any():
                    pm = (out.masked_fill(~pos, 0).sum(1) / pos.sum(1).clamp(min=1))[has]
                    nm = (out.masked_fill(~neg, 0).sum(1) / neg.sum(1).clamp(min=1))[has]
                    loss = loss + 0.5 * nn.functional.softplus(nm - pm).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def lexicon_features(vid: int, n: int) -> np.ndarray:
    """Counts per clip in the window from 2 s before to 20 s after (the v2 commentary window), plus speech rate."""
    words = json.loads((P.PROC / "whisper" / f"american_football_{vid}.json").read_text())
    f = np.zeros((n, len(LEX) + 1), np.float32)
    for w in words:
        if "start" not in w:
            continue
        tok = re.sub(r"[^a-z']", "", w["word"].lower())
        t = float(w["start"])
        lo, hi = max(0, int((t - 20) // 2)), min(n, int((t + 2) // 2) + 1)
        f[lo:hi, -1] += 1
        for j, name in enumerate(LEX):
            if tok in LEX[name] or (name == "negation" and tok.endswith("n't")):
                f[lo:hi, j] += 1
    f[:, :-1] = np.log1p(f[:, :-1])
    f[:, -1] = np.log1p(f[:, -1])
    return f


def shot_segments(vid: int, n: int) -> list[tuple[int, int]]:
    rows = [tuple(int(x) for x in ln.split()) for ln in (P.PROC / "shots" / f"american_football_{vid}.mp4.scenes.txt").read_text().splitlines() if ln.strip()]
    fps = rows[-1][1] / (n * P.CLIP_SECONDS)                                      # frames per second implied by the last shot frame and the labelled duration
    return clip_spans([(a / fps, (b + 1) / fps) for a, b in rows], n)


def sentence_segments(vid: int, n: int) -> list[tuple[int, int]]:
    seg = json.loads((P.PROC / "segments" / f"american_football_{vid}.json").read_text())
    return clip_spans([(float(s["start"]), float(s["end"])) for s in seg], n)


def clip_spans(spans: list[tuple[float, float]], n: int, max_clips: int = 10) -> list[tuple[int, int]]:
    """Seconds -> clip index ranges; long spans are split into pieces of at most 20 s so one segment cannot swallow the budget."""
    out = []
    for a, b in spans:
        lo, hi = max(0, int(a // 2)), min(n, max(int(a // 2) + 1, int(np.ceil(b / 2))))
        for s in range(lo, hi, max_clips):
            if min(hi, s + max_clips) > s:
                out.append((s, min(hi, s + max_clips)))
    return out


def decode_segments(score: np.ndarray, segs: list[tuple[int, int]], how: str, budget_s: float) -> list[D.Segment]:
    val = {"peak": np.max, "mean": np.mean, "q75": lambda a: np.quantile(a, 0.75)}[how]
    ranked = sorted(((float(val(score[a:b])), a, b) for a, b in segs if b > a), key=lambda t: (-t[0], t[1]))
    taken, chosen, left = np.zeros(len(score), bool), [], budget_s
    for s, a, b in ranked:
        if taken[a:b].any():
            continue
        dur = (b - a) * 2.0
        if dur > left:
            if left < 4.0:
                break
            b, dur = a + int(left // 2), float(int(left // 2) * 2)
            if b <= a:
                break
        taken[a:b] = True
        left -= dur
        chosen.append(D.Segment(a * 2.0, b * 2.0, (a + int(np.argmax(score[a:b])) + 0.5) * 2.0, s, len(chosen) + 1))
    return chosen


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 4000) -> dict:
    rng, d = np.random.default_rng(20261004), a - b
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return {"mean_difference": float(d.mean()), "interval_95_over_games": [float(np.quantile(m, .025)), float(np.quantile(m, .975))], "games": len(d)}


def run(log=print) -> dict:
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    split, labels, vol = P.make_split(), P.load_labels(), P.load_volume()
    nclips = {v: len(y) for v, y in labels.items()}
    folds = split["development_folds"]
    ck = OUT / "highlights_v3_oof.npz"
    oof = dict(np.load(ck)) if ck.exists() else {}
    docs = {v: M.clip_documents(v, nclips[v]) for v in split["train"]}
    lex = {v: lexicon_features(v, nclips[v]) for v in split["train"]}
    for fi, held in enumerate(folds):
        if all(f"H-W|{v}" in oof for v in held):
            continue
        tr = [v for v in split["train"] if v not in held]
        log(f"fold {fi}: train {len(tr)} games, held out {held}")
        feats = fold_features(tr, split["train"], nclips, vol, log)
        for variant in ("H-C", "H-R", "H-W"):
            t0 = time.time()
            m = train(feats, labels, tr, variant)
            for v in held:
                oof[f"{variant}|{v}"] = M.score_game(m, feats[v], M.STREAMS)
            log(f"  {variant}: {EPOCHS} epochs in {time.time() - t0:.0f}s, held-out mAP {np.mean([average_precision_score(labels[v], oof[f'{variant}|{v}']) for v in held]):.4f}")
        vec = TfidfVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 2), min_df=5, max_features=30000, sublinear_tf=True)
        xtr, ytr = vec.fit_transform([d for v in tr for d in docs[v]]), np.concatenate([labels[v] for v in tr])
        h1 = LogisticRegression(C=1.0, max_iter=2000).fit(xtr, ytr)
        s_tr = h1.decision_function(xtr)
        ltr = np.concatenate([lex[v] for v in tr])
        mu, sd = ltr.mean(0), ltr.std(0) + 1e-6
        zn = lambda a, mu=mu, sd=sd: (a - mu) / sd
        hn = LogisticRegression(C=1.0, max_iter=2000).fit(np.c_[s_tr, zn(ltr), s_tr[:, None] * zn(ltr)], ytr)      # word score, lexicon counts, and their interactions
        for v in held:
            s = h1.decision_function(vec.transform(docs[v]))
            oof[f"H1|{v}"] = s
            oof[f"H-N|{v}"] = hn.decision_function(np.c_[s, zn(lex[v]), s[:, None] * zn(lex[v])])
        np.savez_compressed(ck, **oof)
    games = split["train"]
    fold_of = {v: i for i, f in enumerate(folds) for v in f}
    z = lambda a: (a - a.mean()) / (a.std() + 1e-6)
    y_all = {v: labels[v] for v in games}
    # learned late fusion, stacked: weights fitted on the other folds' out-of-fold scores
    for name, text in (("H-F H3 + H1 (learned)", "H1"), ("H-F H3 + H-N (learned)", "H-N")):
        for k in range(len(folds)):
            tr = [v for v in games if fold_of[v] != k]
            lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.concatenate([np.c_[z(oof[f"H-C|{v}"]), z(oof[f"{text}|{v}"])] for v in tr]), np.concatenate([y_all[v] for v in tr]))
            for v in folds[k]:
                oof[f"{name}|{v}"] = lr.decision_function(np.c_[z(oof[f"H-C|{v}"]), z(oof[f"{text}|{v}"])])
    for v in games:
        oof[f"H3 + H1 equal-weight sum (v2 rule)|{v}"] = z(oof[f"H-C|{v}"]) + z(oof[f"H1|{v}"])
    names = {"H-C": "H-C control (H3, cross-entropy)", "H-R": "H-R ranking-aware", "H-W": "H-W positive weight 3, game-balanced", "H1": "H1 commentary words (v2 model)", "H-N": "H-N negation-aware commentary",
             "H-F H3 + H1 (learned)": "H-F learned fusion: H3 + commentary", "H-F H3 + H-N (learned)": "H-F learned fusion: H3 + negation-aware commentary",
             "H3 + H1 equal-weight sum (v2 rule)": "H3 + commentary, equal-weight sum (v2 rule)"}
    ap = {k: np.array([average_precision_score(y_all[v], oof[f"{k}|{v}"]) for v in games]) for k in names}
    rankers = {}
    for k, label in names.items():
        strict = [D.metrics(y_all[v], D.decode(oof[f"{k}|{v}"], 180, block=3, lead_s=2, tail_s=2), 180) for v in games]
        base = "H1" if k == "H-N" else "H-C"
        rankers[label] = {"mean_average_precision": float(ap[k].mean()), "by_fold": [float(np.mean([ap[k][games.index(v)] for v in f])) for f in folds],
                          "strict_180s": {m: float(np.mean([r[m] for r in strict])) for m in ("precision", "recall_of_labelled_time", "output_s")},
                          **({} if k == base else {f"vs_{names[base].split(' ')[0]}": boot_diff(ap[k], ap[base])})}
    seg = {}
    for src, fn in (("shots", shot_segments), ("sentences", sentence_segments)):
        for how in ("peak", "mean", "q75"):
            rows = [D.metrics(y_all[v], decode_segments(oof[f"H-C|{v}"], fn(v, nclips[v]), how, 180), 180) for v in games]
            seg[f"{src} / {how}"] = {m: float(np.mean([r[m] for r in rows])) for m in ("recall_of_labelled_time", "precision", "output_s", "segments")} | {
                "all_within_budget": all(r["within_budget"] for r in rows), "segments_half_covered_share": float(sum(r["segments_half_covered"] for r in rows) / sum(r["labelled_segments"] for r in rows))}
    blk = [D.metrics(y_all[v], D.decode(oof[f"H-C|{v}"], 180, block=3, lead_s=2, tail_s=2), 180) for v in games]
    seg["block decoder (v3 default)"] = {m: float(np.mean([r[m] for r in blk])) for m in ("recall_of_labelled_time", "precision", "output_s", "segments")} | {
        "all_within_budget": True, "segments_half_covered_share": float(sum(r["segments_half_covered"] for r in blk) / sum(r["labelled_segments"] for r in blk))}
    probes = {}
    for case, text in M.PROBES:
        toks = text.split()
        probes[case] = {"text": text, **{name: int(sum(t in LEX[name] or (name == "negation" and t.endswith("n't")) for t in toks)) for name in LEX}}
    passing = [k for k, v in rankers.items() if any(isinstance(x, dict) and "interval_95_over_games" in x and x["interval_95_over_games"][0] > 0 for x in v.values()) and not k.startswith(("H-N", "H1"))]
    rep = {"module": "highlights", "version": "highlights-v3-experiments", "created_at_utc": now_utc(), "registered_in": "docs/experiments/highlights_v3.md",
           "split_role": "28 training games in 4 grouped development folds, out-of-fold. Validation and test games were not scored.", "epochs": EPOCHS, "seed": SEED,
           "transforms": "PCA, scaling and text vocabulary fitted inside each fold", "configurations": len(names) + len(seg) - 1,
           "rankers": rankers, "segment_decoding_at_strict_180s": seg, "shot_time_base": "shot frames converted with the frame rate implied by the last shot frame and the labelled duration",
           "lexicon": {k: sorted(v) for k, v in LEX.items()}, "lexicon_probes": probes,
           "probe_note": "Probes show which lexicon counters fire on short phrases. They check mechanics; they are not an event evaluation.",
           "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
           "decision": {"rankers_passing_rule": passing, "statement": ("No ranker met the registered rule; H3 stays the shipped ranker." if not passing else
                        "A ranker met the rule on development folds. It must be retrained on all training games as a new version before any validation or benchmark comparison.")}}
    write_json(OUT / "highlights_experiments_v3.json", rep)
    r = Run("highlights", "v3-experiments", "v3", seed=SEED)
    r.set(target="editorial highlight selection (ranking)", population="28 training games, grouped 4-fold", input_cutoff="offline: context on both sides of a clip",
          split={"folds": folds}, metrics={k: v["mean_average_precision"] for k, v in rankers.items()})
    r.output("report", OUT / "highlights_experiments_v3.json")
    r.finish()
    log(json.dumps({"mAP": {k: round(v["mean_average_precision"], 4) for k, v in rankers.items()}, "segments": {k: round(v["recall_of_labelled_time"], 4) for k, v in seg.items()}, "decision": rep["decision"]}, indent=1))
    return rep


if __name__ == "__main__":
    run(lambda s: (print(s), sys.stdout.flush()))
