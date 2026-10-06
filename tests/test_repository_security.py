"""Publication guards: private paths and credentials cannot silently enter exports."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "repository_check", Path(__file__).resolve().parents[1] / "scripts/check_repository.py"
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


@pytest.mark.parametrize("name", [
    "data/raw/another-source/new.csv", "data/local_media/game.mp4", "models/model.pt",
    "secrets/kaggle.json", "apps/web/.env.local", "archive/source.zip",
    "apps/web/test-results/trace.txt", "elsewhere/game.MOV",
])
def test_private_files_are_blocked_even_outside_expected_directories(name):
    assert guard.path_problem(name)


def test_public_assets_and_nonsecret_forecast_hashes_are_allowed():
    for name in [".env.example", "docs/assets/readme/home.jpg", "docs/assets/readme/coverage-playback.gif",
                 "data/catalog.json", "apps/web/public/demo/pregame/forecasts.json"]:
        assert guard.path_problem(name) is None
    assert guard.credential_fields({"records": [{"idempotency_key": "f" * 24}]}) == []


def test_nested_public_credentials_are_rejected_without_printing_values():
    payload = {"games": [{"private": {"access_token": "synthetic-not-a-token"}}]}
    assert guard.credential_fields(payload) == ["/games/0/private/access_token"]
