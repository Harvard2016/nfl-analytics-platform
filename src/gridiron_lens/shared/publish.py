"""Frozen research export for the website: run records, rights manifest and the v1 baseline summary.

The site reads these files only. It never talks to a live experiment tracker.
"""
from __future__ import annotations

import json

from . import config
from .provenance import now_utc, write_json
from .runs import load_runs

WEB = config.WEB_DEMO / "research"
V3_EXPORTS = {"coverage_v3_experiments.json": "real/coverage/experiments_v3.json", "highlights_eval_v3.json": "highlights/eval_v3.json",
              "highlights_experiments_v3.json": "highlights/experiments_v3.json", "pregame_experiments_v3.json": "pregame/experiments_v3.json"}
KEEP = ("module", "run_id", "name", "results_version", "started_at", "finished_at", "git_commit", "seed", "environment", "target", "population", "exclusions",
        "input_cutoff", "split", "calibration", "hyperparameters", "metrics", "ablations", "outputs", "notes", "train_seconds", "peak_memory_mb", "evidence_kinds", "data")


def run() -> dict:
    runs = [{k: r.get(k) for k in KEEP} for v in ("v2", "v3") for r in load_runs(v)]
    latest: dict[tuple, dict] = {}
    for r in runs:                                   # keep every run, mark the most recent of each kind
        latest[(r["module"], r["name"])] = r
    for r in runs:
        r["latest"] = latest[(r["module"], r["name"])]["run_id"] == r["run_id"]
    base = json.loads((config.REPORTS / "v1" / "baseline_manifest.json").read_text())
    rights = json.loads((config.MANIFESTS / "rights.json").read_text())
    write_json(WEB / "runs.json", {"generated_at": now_utc(), "runs": runs})
    write_json(WEB / "rights.json", rights)
    write_json(WEB / "baseline_v1.json", {
        "frozen_at": base["frozen_at"], "git_commit": base["git_commit"], "note": base["note"],
        "coverage": {"counts": base["coverage"]["data"]["confirmed_counts"], "split": {k: base["coverage"]["split"][k] for k in ("version", "weeks", "class_counts")},
                     "models": base["coverage"]["models"], "metrics": base["coverage"]["metrics"]["post_1_5s"]["geometry_gbm"], "ui_defaults": base["coverage"]["ui_defaults"]},
        "pregame": base["pregame"]["metrics"],
    })
    copied = []
    for src, dest in V3_EXPORTS.items():                 # aggregate results only: no per-play tracking, no media, no transcript text
        f = config.REPORTS / "v3" / src
        if f.exists():
            write_json(config.WEB_DEMO / dest, json.loads(f.read_text()))
            copied.append(dest)
    return {"runs": len(runs), "out": config.rel(WEB), "v3_exports": copied}


if __name__ == "__main__":
    print(run())
