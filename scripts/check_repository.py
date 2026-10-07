"""Check tracked paths, public JSON and documentation before publication.

This is a publication guard, not a license checker or a replacement for Gitleaks.
It reads the Git index, so stage intended files before running it locally.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = (
    "data/raw/", "data/processed/", "data/features/", "data/snapshots/",
    "data/local_media/", "data/jobs/", "models/", "artifacts/", "mlruns/",
)
PRIVATE_NAMES = {"kaggle.json", "id_rsa", "id_ed25519", ".npmrc", ".pypirc"}
PRIVATE_SUFFIXES = {
    ".pem", ".p12", ".pfx", ".key", ".mp4", ".mov", ".mkv", ".webm",
    ".wav", ".mp3", ".m4a", ".zip", ".tar", ".tgz", ".gz", ".7z",
    ".sqlite", ".sqlite3", ".db", ".pt", ".joblib", ".pkl", ".pickle",
    ".safetensors", ".onnx", ".npy", ".npz", ".parquet", ".duckdb",
}
SECRET_FIELDS = {
    "access_token", "refresh_token", "password", "secret_key", "api_key",
    "private_key", "authorization", "client_secret",
}


def path_problem(name: str) -> str | None:
    p = PurePosixPath(name)
    lower = name.lower()
    if lower.startswith(PRIVATE):
        return "private data/model directory"
    if p.name.lower() in PRIVATE_NAMES:
        return "credential filename"
    if p.name.lower().startswith(".env") and p.name != ".env.example":
        return "environment file (only .env.example is publishable)"
    if p.suffix.lower() in PRIVATE_SUFFIXES:
        return "bulk data, media, credential or model file"
    if "/test-results/" in name or "/playwright-report/" in name:
        return "temporary browser output"
    return None


def credential_fields(value, pointer: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            at = f"{pointer}/{key}"
            if key.lower() in SECRET_FIELDS:
                found.append(at)
            found.extend(credential_fields(child, at))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            found.extend(credential_fields(child, f"{pointer}/{i}"))
    return found


def markdown_links(path: Path) -> list[str]:
    text = path.read_text()
    links = re.findall(r"\]\(([^)]+)\)", text)
    links += re.findall(r'(?:src|href)="([^"]+)"', text)
    broken = []
    for link in links:
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", link) or link.startswith("#"):
            continue
        target = link.split("#", 1)[0].split("?", 1)[0]
        if target and not (path.parent / target).exists():
            broken.append(target)
    return broken


def main() -> int:
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")[:-1]
    errors = []
    for name in names:
        problem = path_problem(name)
        if problem:
            errors.append(f"{name}: {problem}")
        path = ROOT / name
        if name.startswith("apps/web/public/") and path.suffix == ".json":
            try:
                fields = credential_fields(json.loads(path.read_text()))
            except (OSError, ValueError):
                errors.append(f"{name}: invalid or unreadable public JSON")
            else:
                errors.extend(f"{name}: forbidden public field {field}" for field in fields)
    catalog = json.loads((ROOT / "data/catalog.json").read_text())
    rights = json.loads((ROOT / "data/manifests/rights.json").read_text())
    ids = [d["id"] for d in catalog["datasets"]]
    if len(ids) != len(set(ids)) or set(ids) != {d["id"] for d in rights["sources"]}:
        errors.append("data/catalog.json: source IDs must match the rights manifest exactly")
    for name in (*PRIVATE, "data/raw/kaggle.json", "data/local_media/game.mp4", "models/test.pt"):
        check = subprocess.run(["git", "check-ignore", "--no-index", "-q", name], cwd=ROOT, check=False)
        if check.returncode:
            errors.append(f"{name}: must be covered by .gitignore")
    docs = ["README.md", "CONTRIBUTING.md", "SECURITY.md", "data/README.md", "reports/README.md",
            "docs/README.md", "docs/GETTING_STARTED.md", "docs/README_RESEARCH.md", "apps/web/README.md"]
    for name in docs:
        errors.extend(f"{name}: broken local link {link}" for link in markdown_links(ROOT / name))
    if errors:
        print("Repository publication check failed:")
        for error in errors:
            print("-", error)  # names/pointers only; never print values or credentials
        return 1
    print(f"Publication check passed: {len(names)} tracked files; catalog, private paths, public JSON and local documentation links checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
