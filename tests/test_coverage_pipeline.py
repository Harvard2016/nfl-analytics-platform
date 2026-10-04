"""Data-integrity and leakage checks for the coverage pipeline, run on the SYNTHETIC fixture.

These prove the transforms and guards behave; they say nothing about model quality on real data.
"""
import json

import numpy as np
import polars as pl
import pytest

from gridiron_lens.coverage import export, features, fixture, normalize, schema, studies, train

SRC = "synthetic_bdb"


@pytest.fixture(scope="session")
def pipe(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    d = {k: root / k for k in ("raw", "processed", "features", "models", "reports", "demo")}
    fixture.generate(d["raw"], plays_per_game=30, games_per_week=1, seed=3)
    d["normalize"] = normalize.run(SRC, d["raw"], d["processed"])
    features.run(SRC, d["processed"], d["features"])
    d["splits"] = root / "splits.json"
    d["report"] = train.run(SRC, d["features"], d["models"], d["reports"], d["splits"])
    return d


def test_direction_normalization_puts_offense_behind_the_line(pipe):
    for week in pipe["normalize"]["weeks"]:
        assert week["offense_behind_los_at_snap"] == {"left": 1.0, "right": 1.0}
        assert week["defense_beyond_los_at_snap"] == {"left": 1.0, "right": 1.0}
        assert week["direction_vs_displacement_cosine"]["median"] > 0.95
        assert week["ball_to_los_abs_yards_at_snap"]["p95"] < 1.5
        assert week["dropped_no_snap_frame"] == 0


def test_left_plays_are_rotated_and_raw_coordinates_kept(pipe):
    t = pl.read_parquet(pipe["processed"] / "tracking_week_1.parquet")
    left = t.filter(pl.col("flipped") & (pl.col("side") != "ball"))
    assert left.height > 0
    assert np.allclose(left["x"], 120.0 - left["x_raw"]) and np.allclose(left["y"], 53.3 - left["y_raw"])
    right = t.filter(~pl.col("flipped"))
    assert np.allclose(right["x"], right["x_raw"])
    assert set(t["side"].unique()) == {"offense", "defense", "ball"}
    assert t.filter(pl.col("side") == "ball")["nflId"].null_count() == t.filter(pl.col("side") == "ball").height


def test_feature_builder_refuses_frames_after_the_cutoff(pipe):
    t = pl.read_parquet(pipe["processed"] / "tracking_week_1.parquet")
    g, p = t.select(["gameId", "playId"]).row(0)
    play = t.filter((pl.col("gameId") == g) & (pl.col("playId") == p))
    with pytest.raises(AssertionError):
        features.play_features(play, "at_snap")
    ok, reason = features.play_features(play.filter(pl.col("rel_frame") <= 0), "at_snap")
    assert reason == "ok" and set(ok) == set(features.feature_names("at_snap"))


def test_pre_snap_features_ignore_everything_after_the_snap(pipe, tmp_path):
    """Scramble every post-snap position: pre-snap features must not move."""
    src = pl.read_parquet(pipe["processed"] / "tracking_week_1.parquet")
    post = pl.col("rel_frame") > 0
    scrambled = src.with_columns(pl.when(post).then(pl.col("x_rel") * -3 + 7).otherwise(pl.col("x_rel")).alias("x_rel"),
                                 pl.when(post).then(pl.lit(1.0)).otherwise(pl.col("y")).alias("y"))
    scrambled.write_parquet(tmp_path / "tracking_week_1.parquet")
    plays = pl.read_parquet(pipe["processed"] / "plays.parquet")
    a, _ = features.build_window(tmp_path, "at_snap", plays)
    b = pl.read_parquet(pipe["features"] / "features_at_snap.parquet").filter(pl.col("week") == 1)
    cols = features.feature_names("at_snap")
    assert np.allclose(a.sort(["gameId", "playId"]).select(cols).to_numpy(),
                       b.sort(["gameId", "playId"]).select(cols).to_numpy(), equal_nan=True)


def test_no_label_or_outcome_column_can_be_a_feature():
    for window in features.WINDOWS:
        used = features.feature_names(window) + train.CONTEXT
        assert not set(used) & set(schema.LABEL_COLUMNS)
        assert not [c for c in used for bad in schema.FORBIDDEN_SUBSTRINGS if bad in c]


def test_splits_keep_whole_games_together_and_are_frozen(pipe):
    s = json.loads(pipe["splits"].read_text())
    games = {k: set(v["games"]) for k, v in s["splits"].items()}
    assert not games["train"] & games["dev"] and not games["train"] & games["test"] and not games["dev"] & games["test"]
    plays = [p for v in s["splits"].values() for p in v["plays"]]
    assert len(plays) == len(set(plays))
    assert train.freeze_splits(SRC, pipe["features"], pipe["splits"])["frozen_at"] == s["frozen_at"]


def test_report_is_marked_synthetic_and_probabilities_are_valid(pipe):
    r = pipe["report"]
    assert r["synthetic"] is True
    for res in r["windows"].values():
        for m in res["models"].values():
            t = m["test"]
            assert t["n"] == sum(r["class_counts"]["test"].values())
            assert 0 <= t["brier"] <= 1 and np.isfinite(t["log_loss"])
            assert np.array(t["confusion"]["rows_charted_cols_predicted"]).sum() == t["n"]


def test_export_sample_is_prediction_blind_and_error_sets_are_real_errors(pipe):
    out = export.run(SRC, 6, pipe["processed"], pipe["features"], pipe["models"], pipe["demo"], pipe["splits"],
                     pipe["reports"] / "coverage_man_zone_SYNTHETIC.json", n_errors=3)
    index = json.loads((pipe["demo"] / "index.json").read_text())
    test = sorted(json.loads(pipe["splits"].read_text())["splits"]["test"]["plays"])
    # the sample must be reproducible from the split manifest and the seed alone: no model output involved
    expected = sorted(np.random.default_rng(export.SAMPLE_SEED).choice(test, size=6, replace=False).tolist())
    assert [i.split(":", 1)[1] for i in index["sample"]["ids"]] == expected
    assert out["sample"] == 6 and index["synthetic"] is True and index["default_model"] == "geometry_gbm"
    by_id = {p["id"]: p for p in index["plays"]}
    for model, per_window in index["errors"]["ids"].items():
        for window, ids in per_window.items():
            assert len(ids) <= 3
            for i in ids:  # every error-set play really is a lean against the released label, for that model and window
                assert by_id[i]["predictions"][model][window]["predicted"] != by_id[i]["released"]["manZone"]
    for entry in index["plays"]:
        text = (pipe["demo"] / entry["file"]).read_text()
        play = json.loads(text)
        assert play["synthetic"] is True and entry["split"] == "test"
        assert "/Users/" not in text and str(pipe["raw"]) not in text
        assert len(play["entities"]) == 23 and sum(e["side"] == "ball" for e in play["entities"]) == 1
        for name, w in play["windows"].items():
            assert set(w["explanations"]) == set(export.SELECTABLE)
            for model, pred in w["predictions"].items():
                assert abs(pred["p_man"] + pred["p_zone"] - 1) < 1e-3
                assert pred["accepted"] == (pred["confidence"] >= pred["confidence_threshold"])
                if model in export.SELECTABLE:
                    assert entry["predictions"][model][name]["p_man"] == pred["p_man"]
                    assert index["cutoffs"][model][name] == pred["confidence_threshold"]


def test_real_layout_leak_columns_never_reach_the_normalized_table():
    """ball_land_x/y, num_frames_output and player_to_predict describe what happened after the throw."""
    assert not {"ball_land_x", "ball_land_y", "num_frames_output", "player_to_predict"} & set(normalize.OUT_COLS)


def test_funnel_accounts_for_every_play_and_matches_the_split_manifest(pipe):
    ctx = {"source": SRC, "processed": pipe["processed"], "splits": json.loads(pipe["splits"].read_text()),
           "tables": {w: pl.read_parquet(pipe["features"] / f"features_{w}.parquet").with_columns(studies.K) for w in features.WINDOWS}}
    f = studies.funnel(ctx)
    assert f["matches_split_manifest"] is True
    assert f["steps"][0]["plays"] - sum(s.get("removed", 0) for s in f["steps"]) == f["evaluated_plays"]
    tests = {w: set(t.filter(pl.col("_k").is_in(ctx["splits"]["splits"]["test"]["plays"]))["_k"]) for w, t in ctx["tables"].items()}
    assert len({frozenset(v) for v in tests.values()}) == 1  # every window is scored on the same test plays
