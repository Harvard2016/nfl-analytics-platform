"""Forecast protocol v3: timing is derived from persisted times, kickoffs are converted explicitly, records are idempotent."""
import datetime as dt
from datetime import UTC, datetime, timedelta

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
