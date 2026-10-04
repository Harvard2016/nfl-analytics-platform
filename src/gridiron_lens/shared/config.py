"""Project paths. Override the root with GRIDIRON_ROOT; nothing here is laptop-specific."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("GRIDIRON_ROOT", Path(__file__).resolve().parents[3]))

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "data" / "features"
MANIFESTS = ROOT / "data" / "manifests"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
WEB_DEMO = ROOT / "apps" / "web" / "public" / "demo"

FIELD_LENGTH = 120.0  # yards, including both end zones
FIELD_WIDTH = 53.3


def rel(path: Path) -> str:
    """Path relative to the project root, so manifests never leak absolute local paths."""
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return Path(path).name
