"""File-level provenance: size and SHA-256, streamed so large CSVs never load into RAM."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def file_record(path: Path, with_hash: bool = True) -> dict:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path) if with_hash else None,
    }


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    return path
