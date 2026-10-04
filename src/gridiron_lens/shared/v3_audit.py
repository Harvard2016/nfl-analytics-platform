"""V3 stage 0: inventory of the local machine and artifacts, and reproduction of saved-model predictions on real local data.

Writes reports/v3/inventory.json and reports/v3/reproduction.json. Nothing is retrained and nothing saved is overwritten.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys

import numpy as np

from . import config
from .provenance import now_utc, write_json
from .provenance import sha256_file as sha256_path

OUT = config.REPORTS / "v3"


def _sh(*a: str) -> str:
    try:
        return subprocess.run(a, capture_output=True, text=True, cwd=config.ROOT, timeout=30).stdout.strip()
    except Exception as e:                                                   # tool missing or timed out
        return f"unavailable ({type(e).__name__})"


def inventory() -> dict:
    du = shutil.disk_usage(config.ROOT)
    models = {config.rel(p): {"bytes": p.stat().st_size, "sha256": sha256_path(p)} for p in sorted(config.MODELS.rglob("*")) if p.is_file() and "synthetic" not in str(p)}
    raw = {p.name: {"files": sum(1 for f in p.rglob("*") if f.is_file()), "bytes": sum(f.stat().st_size for f in p.rglob("*") if f.is_file())}
           for p in sorted(config.RAW.iterdir()) if p.is_dir()}
    inv = {
        "created_at_utc": now_utc(),
        "git": {"branch": _sh("git", "branch", "--show-current"), "commit": _sh("git", "rev-parse", "HEAD"), "dirty_files": len(_sh("git", "status", "--short").splitlines())},
        "python": sys.version.split()[0], "node": _sh("/opt/homebrew/bin/node", "--version"), "machine": platform.machine(), "cpu": _sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "cpu_count": int(_sh("sysctl", "-n", "hw.ncpu") or 0), "ram_gb": round(int(_sh("sysctl", "-n", "hw.memsize") or 0) / 2**30, 1),
        "disk_free_gb": round(du.free / 2**30, 1), "disk_total_gb": round(du.total / 2**30, 1),
        "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe"), "kaggle_cli": shutil.which("kaggle"),
        "raw_sources": raw, "local_media_dir_exists": (config.ROOT / "data" / "local_media").exists(),
        "models": models,
        "schemas": {"coverage_site_export": "coverage-demo-v3", "run_records": "reports/v2/runs/*.json"},
    }
    write_json(OUT / "inventory.json", inv)
    return inv


def reproduce() -> dict:
    """Recompute predictions from the saved weights on real cached inputs and compare with the saved prediction files."""
    import torch

    from ..coverage import neural
    from ..highlights import models as HM
    from ..highlights import pipeline as HP
    out: dict = {"created_at_utc": now_utc(), "note": "Saved weights applied to real cached local inputs. No retraining."}

    # coverage: temporal GRU, three seeds, Platt calibrators stored in the bundle
    A = neural.load_arrays()
    saved = np.load(config.REPORTS / "v2" / "coverage_expB_predictions.npz")
    assert (saved["key"] == A["key"]).all()
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(len(A["y"]), 500, replace=False))
    cov = {}
    for seed in (42, 7, 2026):
        b = torch.load(config.MODELS / "coverage_v2" / f"temporal_gru_seed{seed}.pt", weights_only=False)
        m = neural.CoverageNet(b["config"]["temporal"])
        m.load_state_dict(b["state_dict"])
        lg = neural.predict_logits(m, A, sample)
        diffs = []
        for j, (h, f) in enumerate(zip(saved["horizons"], neural.HORIZON_FRAMES)):
            ok = A["nframes"][sample] > f
            p = 1 / (1 + np.exp(-(b["platt"][str(h)]["a"] * lg[ok, f] + b["platt"][str(h)]["b"])))
            diffs.append(float(np.abs(p - saved[f"seed{seed}"][sample][ok, j]).max()))
            assert np.isnan(saved[f"seed{seed}"][sample][~ok, j]).all()        # no saved prediction where the prefix does not exist
        cov[f"seed{seed}"] = {"plays": len(sample), "max_abs_probability_difference": max(diffs)}
    out["coverage_temporal"] = cov

    # highlights: H3 temporal fusion on the cached reduced features
    b = torch.load(config.MODELS / "highlights" / "temporal_fusion.pt", weights_only=False)
    g0 = HM.load_game(HP.video_ids()[0])
    model = HM.TemporalFusion({k: g0[k].shape[1] for k in b["streams"]})
    model.load_state_dict(b["state_dict"])
    sc = np.load(config.REPORTS / "v2" / "highlights_scores.npz")
    hl = {}
    for v in (23, 8, 1):                                                        # one test, one validation, one training game
        s = HM.score_game(model, HM.load_game(v), b["streams"])
        hl[str(v)] = {"clips": len(s), "max_abs_score_difference": float(np.abs(s - sc[f"H3 temporal fusion|{v}"]).max())}
    out["highlights_h3"] = hl
    out["highlights_h3_bundle_gap"] = "temporal_fusion.pt holds weights and stream names only. The PCA transforms and mean/std used to build the cached features are not saved, so the bundle cannot score new media."

    rep = json.loads((config.REPORTS / "v2" / "highlights_ranking.json").read_text())
    pg = rep["results"]["H3 temporal fusion"]["test"]["per_game"]
    out["highlights_budget_overshoot"] = {g: {"selected_clips": v["3 minutes"]["selected_clips"], "seconds": 2 * v["3 minutes"]["selected_clips"]} for g, v in pg.items()}
    out["pregame"] = "No saved model file exists: backtest and forecast refit from the data on disk each time, so there is no bundle to reproduce from. Addressed in v3 by saving a fitted bundle with its training cohort hash."
    write_json(OUT / "reproduction.json", out)
    return out


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in inventory().items() if k != "models"}, indent=1)[:1500])
    print(json.dumps(reproduce(), indent=1))
