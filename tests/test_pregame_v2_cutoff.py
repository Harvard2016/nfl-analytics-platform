"""Pregame v2 cutoff tests on a small synthetic schedule and play-by-play (mechanics only, not model quality)."""
import datetime as dt

import numpy as np
import polars as pl

from gridiron_lens.pregame import features_v2 as F

TEAMS = ["AAA", "BBB", "CCC", "DDD"]


def _games(weeks: int = 10, played: int = 10) -> pl.DataFrame:
    rows, day = [], dt.date(2030, 9, 8)
    for w in range(1, weeks + 1):
        pairs = [(TEAMS[0], TEAMS[1 + w % 3]), tuple(t for t in TEAMS[1:] if t != TEAMS[1 + w % 3])]
        for i, (h, a) in enumerate(pairs):
            done = w <= played
            rows.append({"game_id": f"2030_{w:02d}_{a}_{h}", "season": 2030, "game_type": "REG", "week": w, "gameday": day, "gametime": "13:00",
                         "home_team": h, "away_team": a, "home": h, "away": a, "home_score": 20 + w + i if done else None, "away_score": 17 if done else None,
                         "result": 3 + w + i if done else None, "location": "Home", "home_rest": 7, "away_rest": 7, "div_game": 0})
        day += dt.timedelta(days=7)
    return pl.DataFrame(rows, schema_overrides={"home_score": pl.Int64, "away_score": pl.Int64, "result": pl.Int64})


def _pbp(games: pl.DataFrame, seed: int = 0, starter: dict | None = None) -> pl.DataFrame:
    rng, rows = np.random.default_rng(seed), []
    for g in games.filter(pl.col("result").is_not_null()).iter_rows(named=True):
        for off, deff in ((g["home"], g["away"]), (g["away"], g["home"])):
            qb = (starter or {}).get((g["game_id"], off), f"QB_{off}")
            for _ in range(40):
                is_pass = rng.random() < 0.6
                rows.append({"game_id": g["game_id"], "season": 2030, "posteam": off, "defteam": deff, "pass": int(is_pass), "rush": int(not is_pass),
                             "epa": float(rng.normal(0.05, 1)), "success": int(rng.random() < 0.45), "qb_dropback": int(is_pass), "sack": int(is_pass and rng.random() < 0.06),
                             "interception": int(is_pass and rng.random() < 0.03), "fumble_lost": 0, "yards_gained": float(rng.normal(5, 8)), "two_point_attempt": 0,
                             "id": qb if is_pass else None, "name": qb if is_pass else None, "qb_epa": float(rng.normal(0.05, 1)), "cpoe": float(rng.normal(0, 10))})
    return pl.DataFrame(rows, schema_overrides={"id": pl.String, "name": pl.String})


FEATURE_COLS = lambda f: [c for c in f.columns if c.startswith(("d_", "home_qb", "away_qb", "elo_", "form_", "win_pct", "rest_"))]


def _build(games, pbp):
    return F.build(games, *F.team_game_stats(pbp)).sort("game_id")


def test_target_game_result_box_score_and_starter_do_not_change_its_features():
    games = _games()
    base = _build(games, _pbp(games))
    target = "2030_06_BBB_AAA" if "2030_06_BBB_AAA" in games["game_id"].to_list() else games.filter(pl.col("week") == 6)["game_id"][0]
    tampered_games = games.with_columns(pl.when(pl.col("game_id") == target).then(pl.lit(-45)).otherwise(pl.col("result")).alias("result"))
    home = games.filter(pl.col("game_id") == target)["home"][0]
    pbp2 = _pbp(tampered_games, seed=0, starter={(target, home): "SOMEONE_ELSE"})
    pbp2 = pbp2.with_columns(pl.when(pl.col("game_id") == target).then(pl.col("epa") + 5).otherwise(pl.col("epa")).alias("epa"))
    after = _build(tampered_games, pbp2)
    cols = FEATURE_COLS(base)
    assert base.filter(pl.col("game_id") == target).select(cols).equals(after.filter(pl.col("game_id") == target).select(cols))


def test_changing_an_earlier_game_changes_later_features():
    games = _games()
    base = _build(games, _pbp(games))
    early = games.filter(pl.col("week") == 3)["game_id"][0]
    pbp2 = _pbp(games).with_columns(pl.when(pl.col("game_id") == early).then(pl.col("epa") + 5).otherwise(pl.col("epa")).alias("epa"))
    after = _build(games, pbp2)
    later = games.filter(pl.col("week") >= 4)["game_id"].to_list()
    cols = [c for c in base.columns if c.startswith("d_off_epa")]
    assert not base.filter(pl.col("game_id").is_in(later)).select(cols).equals(after.filter(pl.col("game_id").is_in(later)).select(cols))
    before = games.filter(pl.col("week") <= 3)["game_id"].to_list()
    assert base.filter(pl.col("game_id").is_in(before)).select(FEATURE_COLS(base)).equals(after.filter(pl.col("game_id").is_in(before)).select(FEATURE_COLS(base)))


def test_appending_future_games_does_not_change_past_rows():
    short, long = _games(weeks=6, played=6), _games(weeks=10, played=10)
    a, b = _build(short, _pbp(short)), _build(long, _pbp(long))
    ids = short["game_id"].to_list()
    cols = FEATURE_COLS(a)
    assert a.select(cols).equals(b.filter(pl.col("game_id").is_in(ids)).select(cols))


def test_unplayed_game_path_matches_training_path():
    """Features for a game built before it is played equal the features stored for it after it is played."""
    played = _games(weeks=8, played=8)
    unplayed = _games(weeks=8, played=7)
    a = _build(played, _pbp(played))
    b = _build(unplayed, _pbp(unplayed))
    ids = played.filter(pl.col("week") == 8)["game_id"].to_list()
    cols = FEATURE_COLS(a)
    assert a.filter(pl.col("game_id").is_in(ids)).select(cols).equals(b.filter(pl.col("game_id").is_in(ids)).select(cols))


def test_projected_quarterback_is_the_previous_game_starter_not_the_target_game_starter():
    games = _games()
    target = games.filter(pl.col("week") == 5)["game_id"][0]
    home = games.filter(pl.col("game_id") == target)["home"][0]
    f = _build(games, _pbp(games, starter={(target, home): "SURPRISE_STARTER"}))
    assert f.filter(pl.col("game_id") == target)["home_qb_id"][0] == f"QB_{home}"


def test_cutoff_rule_counts_games_inside_28_hours():
    games = _games()
    f = _build(games, _pbp(games))
    assert F.cutoff_violations(f) == 0
    squeezed = f.with_columns(pl.when(pl.col("week") == 4).then(pl.lit(20.0)).otherwise(pl.col("home_hours_since_prev_kickoff")).alias("home_hours_since_prev_kickoff"))
    assert F.cutoff_violations(squeezed) == 2
