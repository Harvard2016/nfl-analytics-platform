"""Forecast protocol v3: timing is derived from persisted times, kickoffs are converted explicitly, records are idempotent."""
import datetime as dt
import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from gridiron_lens.pregame import forecast_v3 as V


def test_kickoff_conversion_handles_dst_and_missing_time():
    assert V.kickoff_utc(dt.date(2026, 10, 11), "13:00") == datetime(2026, 10, 11, 17, 0, tzinfo=UTC)      # EDT, UTC-4
    assert V.kickoff_utc(dt.date(2026, 11, 8), "13:00") == datetime(2026, 11, 8, 18, 0, tzinfo=UTC)        # EST, UTC-5
    assert V.kickoff_utc(dt.date(2026, 11, 1), "13:00") == datetime(2026, 11, 1, 18, 0, tzinfo=UTC)        # the day the clocks change
    assert V.kickoff_utc(dt.date(2026, 10, 11), None) is None                                               # never an invented 13:00


def test_run_that_crosses_the_cutoff_while_fitting_is_late():
    kick = datetime(2026, 10, 11, 17, 0, tzinfo=UTC)
    cutoff = kick - timedelta(hours=24)
    started, snap = cutoff - timedelta(minutes=2), cutoff - timedelta(minutes=30)
    assert V.timing(started, snap, kick)["timing"] == "before_cutoff"
    persisted = cutoff + timedelta(seconds=40)                       # fitting ran past the cutoff: the persisted time decides
    t = V.timing(persisted, snap, kick)
    assert t["timing"] == "late" and not t["eligible_for_official_cohort"]


def test_snapshot_fetched_after_cutoff_or_missing_kickoff_is_not_official():
    kick = datetime(2026, 10, 11, 17, 0, tzinfo=UTC)
    cutoff = kick - timedelta(hours=24)
    assert V.timing(cutoff - timedelta(hours=1), cutoff + timedelta(minutes=1), kick)["eligible_for_official_cohort"] is False
    assert V.timing(cutoff, cutoff, kick)["eligible_for_official_cohort"] is True                       # exactly at the cutoff counts
    assert V.timing(kick + timedelta(minutes=1), cutoff, kick)["timing"] == "after_kickoff"
    t = V.timing(cutoff, cutoff, None)
    assert t["timing"] == "ineligible" and t["cutoff_utc"] is None


def test_policy_is_declared_before_outcomes():
    for k in ("update", "official_record", "ties", "postponed_or_cancelled", "missing_kickoff", "primary_model", "control"):
        assert V.POLICY[k]


@pytest.fixture
def snapshot_inputs(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    pbp = raw / "pbp"
    pbp.mkdir(parents=True)
    games = raw / "games.csv"
    games.write_text("synthetic original schedule")
    for season in (2025, 2026):
        pl.DataFrame({"game_id": ["synthetic"], "week": [1], "game_date": [dt.date(season, 9, 1)]}).write_parquet(pbp / f"play_by_play_{season}.parquet")
    monkeypatch.setattr(V, "SNAPSHOTS", tmp_path / "snapshots")
    monkeypatch.setattr(V.pipeline, "RAW", games)
    monkeypatch.setattr(V.F, "PBP_DIR", pbp)
    monkeypatch.setattr(V.pipeline, "load_games", lambda _: pl.DataFrame({"game_id": ["synthetic"], "result": [7.0], "gameday": [dt.date(2026, 9, 1)], "season": [2026]}))
    meta = V.snapshot(2026, fetch=False, clock=lambda: datetime(2026, 10, 1, tzinfo=UTC), log=lambda _: None)
    return games, pbp, meta


def test_snapshot_survives_in_place_raw_file_rewrites(snapshot_inputs):
    games, pbp, meta = snapshot_inputs
    directory = V.verify_snapshot(meta)
    original = {r["name"]: (directory / r["name"]).read_bytes() for r in meta["files"]}
    games.write_text("updated schedule")
    for f in pbp.glob("*.parquet"):
        f.write_bytes(b"updated upstream input")
    assert V.verify_snapshot(meta) == directory
    assert all((directory / name).read_bytes() == content for name, content in original.items())


@pytest.mark.parametrize("change", ["rewrite", "delete"])
def test_forecast_rejects_changed_snapshot_before_loading_or_fitting(snapshot_inputs, change):
    _, _, meta = snapshot_inputs
    p = V.SNAPSHOTS / meta["snapshot_id"] / "games.csv"
    if change == "rewrite":
        p.write_text("tampered input")
    else:
        p.unlink()
    with pytest.raises(ValueError, match="Snapshot input changed or is missing"):
        V.forecast(meta["snapshot_id"])
    with pytest.raises(ValueError, match="Snapshot input changed or is missing"):
        V.publish()


def test_snapshot_rejects_manifest_hash_changes(snapshot_inputs):
    _, _, meta = snapshot_inputs
    with pytest.raises(ValueError, match="content hash"):
        V.verify_snapshot(meta | {"content_sha256": "wrong"})


@pytest.mark.parametrize("change", ["input", "missing_manifest", "record_hash"])
def test_publish_checks_older_record_snapshots_without_rewriting_records(snapshot_inputs, tmp_path, monkeypatch, change):
    _, _, old = snapshot_inputs
    V.snapshot(2026, fetch=False, clock=lambda: datetime(2026, 10, 2, tzinfo=UTC), log=lambda _: None)
    monkeypatch.setattr(V, "FORECASTS", tmp_path / "forecasts")
    monkeypatch.setattr(V, "LEGACY", tmp_path / "legacy")
    monkeypatch.setattr(V, "WEB", tmp_path / "web")
    V.FORECASTS.mkdir()
    source = {"snapshot_id": old["snapshot_id"], "content_sha256": old["content_sha256"]}
    if change == "input":
        (V.SNAPSHOTS / old["snapshot_id"] / "games.csv").write_text("tampered historical source")
    elif change == "missing_manifest":
        (V.SNAPSHOTS / old["snapshot_id"] / "snapshot.json").unlink()
    else:
        source["content_sha256"] = "wrong record hash"
    record = V.FORECASTS / "forecasts_synthetic.json"
    body = json.dumps({"records": [{"source_snapshot": source}]})
    record.write_text(body)
    with pytest.raises(ValueError, match="Snapshot input changed|preserved snapshot.*missing|no longer matches"):
        V.publish()
    assert record.read_text() == body
    assert not (V.WEB / "forecasts.json").exists()
