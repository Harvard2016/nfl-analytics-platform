"""Structured run records: one JSON per experiment run, the same contract for every module.

A local stand-in for an experiment tracker. The website reads frozen exports built from these files,
never a live tracking server. Fields follow the project run contract: identity, data, split, target,
population, cutoff, cost, settings, metrics, uncertainty, ablations and links to saved outputs.
"""
from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from . import config
from .provenance import now_utc, write_json

PACKAGES = ("numpy", "polars", "scikit-learn", "torch", "pandas", "duckdb", "pyarrow")


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.ROOT, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=config.ROOT, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL, timeout=20).stdout.strip()
        return out + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment(device: str = "cpu") -> dict:
    vers = {}
    for p in PACKAGES:
        try:
            vers[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "device": device, "packages": vers}


def peak_memory_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


class Run:
    """Collects one run's record and writes reports/<version>/runs/<run_id>.json on finish()."""

    def __init__(self, module: str, name: str, version: str = "v2", *, config_: dict | None = None, seed: int | None = None,
                 device: str = "cpu"):
        self.t0 = time.time()
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        self.record: dict = {
            "module": module, "run_id": f"{module}-{name}-{stamp}", "name": name, "results_version": version,
            "started_at": now_utc(), "git_commit": git_commit(), "seed": seed, "config": config_ or {},
            "environment": environment(device), "data": {}, "split": {}, "target": None, "population": None,
            "exclusions": None, "input_cutoff": None, "features": None, "preprocessing": None, "calibration": None,
            "hyperparameters": None, "metrics": {}, "uncertainty": {}, "ablations": {}, "outputs": {}, "notes": [],
            "evidence_kinds": {"inputs": "observed", "labels": "released", "outputs": "model prediction",
                               "explanations": "model-generated interpretation", "summaries": "statistical association"},
        }
        self.dir = config.REPORTS / version / "runs"

    def set(self, **kw) -> Run:
        self.record.update(kw)
        return self

    def data(self, name: str, path: Path, **extra) -> None:
        path = Path(path)
        self.record["data"][name] = {"path": config.rel(path), "sha256": sha256_path(path) if path.is_file() else None, **extra}

    def output(self, name: str, path: Path) -> None:
        path = Path(path)
        self.record["outputs"][name] = {"path": config.rel(path), "sha256": sha256_path(path) if path.is_file() else None}

    def note(self, text: str) -> None:
        self.record["notes"].append(text)

    def finish(self) -> Path:
        self.record["finished_at"] = now_utc()
        self.record["train_seconds"] = round(time.time() - self.t0, 1)
        self.record["peak_memory_mb"] = peak_memory_mb()
        return write_json(self.dir / f"{self.record['run_id']}.json", self.record)


def load_runs(version: str = "v2", module: str | None = None) -> list[dict]:
    d = config.REPORTS / version / "runs"
    runs = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []
    return [r for r in runs if module is None or r["module"] == module]
