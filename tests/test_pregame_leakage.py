"""Pregame features must not see the game being forecast, same-day games, or anything later."""
import datetime as dt

import polars as pl

from gridiron_lens.pregame import pipeline


def _schedule() -> pl.DataFrame:
    teams = ["AAA", "BBB", "CCC", "DDD"]
    rows, day = [], dt.date(2030, 9, 1)
    for week in range(1, 9):
        for i, (h, a) in enumerate([(teams[0], teams[week % 3 + 1]), (teams[(week % 3 + 1) % 3 + 1], teams[(week + 1) % 3 + 1])]):
            if h == a:
                continue
            rows.append({"game_id": f"2030_{week:02d}_{a}_{h}", "season": 2030, "game_type": "REG", "week": week, "gameday": day,
                         "home_team": h, "away_team": a, "home": h, "away": a, "home_score": 20 + week + i, "away_score": 17,
                         "result": 3 + week + i, "location": "Home", "home_rest": 7, "away_rest": 7, "div_game": 0})
        day += dt.timedelta(days=7)
    return pl.DataFrame(rows)


def test_features_ignore_the_game_itself_same_day_games_and_the_future():
    games = _schedule()
    base = pipeline.build_snapshots(games)
    cut = games["gameday"].unique().sort()[4]
    # deliberate leak fixture: rewrite every result on or after the cutoff date to something absurd
    tampered = games.with_columns(pl.when(pl.col("gameday") >= cut).then(pl.lit(-60)).otherwise(pl.col("result")).alias("result"))
    after = pipeline.build_snapshots(tampered)
    cols = ["game_id", *pipeline.FEATURES, "elo_home", "elo_away"]
    assert base.filter(pl.col("gameday") <= cut).select(cols).equals(after.filter(pl.col("gameday") <= cut).select(cols))
    assert not base.filter(pl.col("gameday") > cut).select(cols).equals(after.filter(pl.col("gameday") > cut).select(cols))


def test_no_result_or_market_column_is_a_feature():
    banned = ("score", "result", "total", "moneyline", "spread", "odds", "qb", "overtime")
    assert not [f for f in pipeline.FEATURES for b in banned if b in f]
