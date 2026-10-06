"""Inference bundle for the H3 highlight ranker (v3).

`temporal_fusion.pt` holds weights and stream names only. To score media that is not in the cache, inference also needs the
exact transforms that produced the cached model inputs: one IncrementalPCA per embedding stream plus the training mean and
standard deviation of its components. v2 fitted those and discarded them.

`build()` refits them deterministically on the same 28 training games with the same settings, checks the transformed arrays
against the existing caches, and only then writes the bundle. If parity fails, nothing is written: a new feature version would
have to be trained and evaluated instead of reusing the v2 metrics.

The bundle does not contain the upstream feature extractors (CLIP, SlowFast, PANN). `score_raw` takes their per-clip outputs.
"""
from __future__ import annotations

import json
import platform
import time

import joblib
import numpy as np
import sklearn
import torch
from sklearn.decomposition import IncrementalPCA

from ..shared import config
from ..shared.provenance import now_utc, sha256_file, write_json
from . import models as M
from . import pipeline as P

DIR = config.MODELS / "highlights" / "v3"
OUT = config.REPORTS / "v3"
VERSION = "highlights-h3-bundle-v3"
PARITY_TOL = 1e-3            # reduced features are float32; inputs to the model are standardized (unit scale)
EXTRACTORS = {
    "vid_clip": {"dim": 512, "source": "released SVHighlights feature files (HERO_Video_Feature_Extractor CLIP stream)", "bundled": False},
    "vid_slowfast": {"dim": 2304, "source": "released SVHighlights feature files (HERO_Video_Feature_Extractor SlowFast stream)", "bundled": False},
    "aud_pann": {"dim": 2048, "source": "released SVHighlights feature files (PANN audio embedding)", "bundled": False},
    "loudness": {"dim": 1, "source": "mean volume in dB per 2-second clip (ffmpeg volumedetect in the release; reproducible from audio)", "bundled": True},
}


def build(log=print) -> dict:
    split = P.make_split()
    labels = P.load_labels()
    nclips = {v: len(y) for v, y in labels.items()}
    transforms, parity, t0 = {}, {}, time.time()
    for name in P.MODALITIES:
        ipca = IncrementalPCA(n_components=M.N_COMP)
        for v in split["train"]:                                             # identical order and batching to models.build_features
            x = P.load_modality(v, name, nclips[v])
            for i in range(0, len(x), 4096):
                if len(x) - i >= M.N_COMP:
                    ipca.partial_fit(x[i:i + 4096])
        tr = np.concatenate([ipca.transform(P.load_modality(v, name, nclips[v])) for v in split["train"]])
        mu, sd = tr.mean(0), tr.std(0) + 1e-6
        transforms[name] = {"components": ipca.components_.astype(np.float64), "pca_mean": ipca.mean_.astype(np.float64), "mean": mu, "std": sd,
                            "input_dim": int(ipca.components_.shape[1]), "explained_variance": float(ipca.explained_variance_ratio_.sum())}
        worst = 0.0
        for v in P.video_ids():
            z = reduce(transforms[name], P.load_modality(v, name, nclips[v]))
            worst = max(worst, float(np.abs(z - M.load_game(v)[name]).max()))
        parity[name] = worst
        log(f"{name}: refit on {len(split['train'])} training games, max abs difference from the cached features {worst:.2e} ({time.time() - t0:.0f}s)")
    ok = all(v <= PARITY_TOL for v in parity.values())
    rep = {"version": VERSION, "created_at_utc": now_utc(), "parity_tolerance": PARITY_TOL, "feature_parity_max_abs_diff": parity, "feature_parity": ok}
    if not ok:
        write_json(OUT / "highlights_bundle_parity.json", rep | {"bundle_written": False, "consequence": "The v2 transforms could not be reproduced. A new feature version must be trained and evaluated; v2 metrics do not carry over."})
        return rep
    DIR.mkdir(parents=True, exist_ok=True)
    w = torch.load(config.MODELS / "highlights" / "temporal_fusion.pt", weights_only=False)
    dims = {k: (M.N_COMP if k in transforms else 3) for k in w["streams"]}
    joblib.dump(transforms, DIR / "transforms.joblib")
    torch.save({"state_dict": w["state_dict"], "streams": w["streams"], "dims": dims}, DIR / "weights.pt")
    model = load_model()
    sc, score_par = np.load(config.REPORTS / "v2" / "highlights_scores.npz"), {}
    vol = P.load_volume()
    for v in P.video_ids():                                                 # raw embeddings + loudness through the bundle vs the saved v2 scores
        s = score_raw(model, transforms, {k: P.load_modality(v, k, nclips[v]) for k in P.MODALITIES}, vol[v])
        score_par[str(v)] = float(np.abs(s - sc[f"H3 temporal fusion|{v}"]).max())
    man = {
        "version": VERSION, "created_at_utc": now_utc(), "task": "ranking agreement with editorial highlight selection (not event detection, not a probability)",
        "architecture": {"class": "gridiron_lens.highlights.models.TemporalFusion", "hidden": 128, "blocks": 3, "kernel": 5, "projection": 64,
                         "receptive_field_clips": 13, "context": "symmetric: uses 6 clips (12 s) on each side; offline analysis, not live detection"},
        "streams": w["streams"], "stream_dims_after_transform": dims, "extractors": EXTRACTORS,
        "sampling": {"clip_seconds": P.CLIP_SECONDS, "time_origin": "clip i covers [2i, 2i+2) seconds from the trimmed start; source time = trim_start_s + clip time"},
        "transforms": {"kind": "IncrementalPCA(64) then (x - mean) / std, fitted on the 28 training games only", "file": "transforms.joblib",
                       "loudness": "three features from dB level: level minus 5-minute centred median, change from the previous clip, within-video z-score"},
        "mask_policy": "A missing stream is passed as zeros with its mask bit set to 0. The model was trained with 15% stream dropout. Benchmark figures apply only when all four streams are present; masked-stream results are reported separately.",
        "libraries": {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
        "files": {f.name: {"bytes": f.stat().st_size, "sha256": sha256_file(f)} for f in (DIR / "transforms.joblib", DIR / "weights.pt")},
        "source_weights_sha256": sha256_file(config.MODELS / "highlights" / "temporal_fusion.pt"), "training_split": "svh-football-split-v1 (28 training games)",
        "parity": {"features_max_abs_diff": parity, "scores_max_abs_diff_over_40_games": max(score_par.values()), "tolerance": PARITY_TOL},
    }
    write_json(DIR / "manifest.json", man)
    write_json(OUT / "highlights_bundle_parity.json", rep | {"bundle_written": True, "score_parity_max_abs_diff": max(score_par.values()), "score_parity_by_game": score_par,
                                                             "chunked_inference_max_abs_diff": chunk_check(model, transforms, nclips, vol), "manifest": man})
    log(f"bundle written to {config.rel(DIR)}; score parity {max(score_par.values()):.2e}")
    return man


def reduce(t: dict, x: np.ndarray) -> np.ndarray:
    if x.shape[1] != t["input_dim"]:
        raise ValueError(f"expected {t['input_dim']} input dimensions, got {x.shape[1]}: features from a different extractor are not compatible")
    return (((x.astype(np.float64) - t["pca_mean"]) @ t["components"].T - t["mean"]) / t["std"]).astype(np.float32)


def load_model() -> M.TemporalFusion:
    w = torch.load(DIR / "weights.pt", weights_only=False)
    m = M.TemporalFusion(w["dims"])
    m.load_state_dict(w["state_dict"])
    return m.eval()


def load() -> tuple[M.TemporalFusion, dict, dict]:
    man = json.loads((DIR / "manifest.json").read_text())
    for name, rec in man["files"].items():
        if sha256_file(DIR / name) != rec["sha256"]:
            raise RuntimeError(f"bundle file {name} does not match its manifest checksum")
    return load_model(), joblib.load(DIR / "transforms.joblib"), man


CHUNK, OVERLAP = 2048, 16            # overlap >= the 6-clip one-sided receptive field, so chunked output equals a single pass


def score_raw(model: M.TemporalFusion, transforms: dict, raw: dict[str, np.ndarray | None], loudness_db: np.ndarray | None, chunk: int | None = None) -> np.ndarray:
    """Scores per 2-second clip from raw extractor outputs. A stream given as None is masked (zeros + mask bit 0)."""
    n = next(len(v) for v in [*raw.values(), loudness_db] if v is not None)
    feats, present = {}, {}
    for k in model.names:
        src = loudness_db if k == "loudness" else raw.get(k)
        present[k] = src is not None
        if src is None:
            feats[k] = np.zeros((n, 3 if k == "loudness" else M.N_COMP), np.float32)
        else:
            f = M.loudness_features(np.asarray(src, np.float32)[:n]) if k == "loudness" else reduce(transforms[k], src[:n])
            feats[k] = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    mask = torch.tensor([[1.0 if present[k] else 0.0 for k in model.names]])
    out = np.empty(n, np.float32)
    step = chunk or n
    with torch.no_grad():
        for a in range(0, n, step):
            lo, hi = max(0, a - OVERLAP), min(n, a + step + OVERLAP)
            s = model({k: torch.as_tensor(feats[k][lo:hi])[None] for k in model.names}, mask)[0].numpy()
            out[a:min(n, a + step)] = s[a - lo:a - lo + min(n, a + step) - a]
    return out


def chunk_check(model, transforms, nclips, vol) -> float:
    v = P.video_ids()[0]
    raw = {k: P.load_modality(v, k, nclips[v]) for k in P.MODALITIES}
    return float(np.abs(score_raw(model, transforms, raw, vol[v]) - score_raw(model, transforms, raw, vol[v], chunk=CHUNK)).max())


if __name__ == "__main__":
    build()
