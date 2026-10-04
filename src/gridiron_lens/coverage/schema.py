"""Sources, labels and leakage rules for the coverage pipeline.

Two raw layouts are understood:
- bdb2026: NFL Big Data Bowl 2026 Analytics (2023 season). The real data in use.
- bdb2025: the Big Data Bowl 2025 layout. Kaggle removed that dataset at the host's request, so this
  layout is now only used by the synthetic test fixture.
Both are normalized to one internal table shape, so features, training and export are shared.
"""
from __future__ import annotations

NULLS = ["NA", ""]
SOURCES = {
    "bdb2026": {
        "layout": "bdb2026",
        "url": "https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data",
        "label_source": "BDB 2026 supplementary file; who charted it is not stated in the files",
        "split_weeks": {"train": list(range(1, 13)), "dev": [13, 14], "test": [15, 16, 17, 18]},
        "split_version": "weeks-1-12_13-14_15-18-v1",
    },
    "synthetic_bdb": {
        "layout": "bdb2025",
        "url": None,
        "label_source": "synthetic fixture",
        "split_weeks": {"train": [1, 2, 3, 4, 5, 6], "dev": [7], "test": [8, 9]},
        "split_version": "weeks-1-6_7_8-9-v1",
    },
}

WEEKS_2025 = tuple(range(1, 10))


def tracking_file(week: int) -> str:
    return f"tracking_week_{week}.csv"


# Charted targets after normalization. Never model inputs.
TARGET = "man_zone"
LABEL_COLUMNS = ("man_zone", "coverage_type")
TARGET_CLASSES = ("Man", "Zone")  # positive class for binary metrics is "Man"
RAW_LABELS = {
    "bdb2025": {"pff_manZone": "man_zone", "pff_passCoverage": "coverage_type"},
    "bdb2026": {"team_coverage_man_zone": "man_zone", "team_coverage_type": "coverage_type"},
}
MAN_ZONE_VALUES = {"MAN_COVERAGE": "Man", "ZONE_COVERAGE": "Zone", "Man": "Man", "Zone": "Zone"}

# Pre-snap play context that may enter a model. Every other play-level column is excluded by default:
# most of the rest describes the outcome (pass result, yards gained, EPA, route of the targeted receiver...).
CONTEXT_WHITELIST = ("quarter", "down", "yardsToGo", "yards_to_goal", "def_score_diff")

# Never features. In the 2026 tracking file, ball_land_x/y, num_frames_output, player_to_predict and
# the "Targeted Receiver" role all reveal what happened after the throw.
FORBIDDEN_SUBSTRINGS = ("pff_", "man_zone", "coverage", "pass_result", "passResult", "yardsGained", "yards_gained",
                        "expected_points", "expectedPoints", "WinProb", "win_prob", "target", "Target", "timeTo",
                        "ball_land", "num_frames_output", "player_to_predict", "route", "penalt", "interception")
