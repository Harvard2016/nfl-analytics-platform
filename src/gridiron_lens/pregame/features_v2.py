"""Pregame feature construction, version 2. Independent of the coverage and highlights modules.

Cutoff: 24 hours before scheduled kickoff. A completed game may feed a forecast only if it kicked off at least
28 hours before the target kickoff (24 hours plus an assumed 4-hour game length). `cutoff_violations` checks that
rule for both teams' most recent games; the count is reported with every run.

Everything is computed from games completed before the cutoff and then joined to the target game. The target
game's own score, box score and starting quarterback are never read when its features are built.

Reconstruction caveat: play-by-play and schedules are today's nflverse releases, not archives of what was
published at the time. nflverse EPA comes from models fitted on data that includes later seasons, so EPA-based
features describe past games with a small amount of hindsight. They are still built only from earlier games.

Quarterback: the projected starter is the team's primary passer in its previous game. That is a stated
assumption, not knowledge of who started.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from ..shared import config
from . import pipeline

PBP_DIR = config.RAW / "nflverse" / "pbp"
GAME_HOURS, CUTOFF_HOURS = 4, 24
VARIANTS = ("w4", "w8", "ewm")
EWM_HALFLIFE = 4.0
TEAM_STATS = ["off_epa", "pass_epa", "rush_epa", "success", "sack_rate", "turnover_rate", "explosive"]
QB_WINDOW, QB_PRIOR_DROPBACKS = 8, 150
PBP_COLS = ["game_id", "season", "posteam", "defteam", "pass", "rush", "epa", "success", "qb_dropback", "sack", "interception",
            "fumble_lost", "yards_gained", "two_point_attempt", "id", "name", "qb_epa", "cpoe"]


def team_game_stats(pbp: pl.LazyFrame | pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per team-game offensive stats (the opponent's row is that team's defense allowed) and per quarterback-game stats."""
    lf = pbp.lazy().filter(((pl.col("pass") == 1) | (pl.col("rush") == 1)) & pl.col("epa").is_not_null() & pl.col("posteam").is_not_null()
                           & (pl.col("two_point_attempt").fill_null(0) == 0))
    lf = lf.with_columns(pl.col("posteam").replace(pipeline.FRANCHISE), pl.col("defteam").replace(pipeline.FRANCHISE))
    db = pl.col("qb_dropback") == 1
    team = lf.group_by(["game_id", "posteam", "defteam"]).agg(
        pl.len().alias("plays"), pl.col("epa").mean().alias("off_epa"), pl.col("success").mean().alias("success"),
        pl.col("epa").filter(db).mean().alias("pass_epa"), pl.col("epa").filter(pl.col("rush") == 1).mean().alias("rush_epa"),
        db.sum().alias("dropbacks"), (pl.col("sack").fill_null(0).sum() / db.sum().clip(lower_bound=1)).alias("sack_rate"),
        ((pl.col("interception").fill_null(0) + pl.col("fumble_lost").fill_null(0)).sum() / pl.len()).alias("turnover_rate"),
        (((pl.col("pass") == 1) & (pl.col("yards_gained") >= 20)) | ((pl.col("rush") == 1) & (pl.col("yards_gained") >= 10))).mean().alias("explosive"),
    ).collect()
    qb = (lf.filter(db & pl.col("id").is_not_null()).group_by(["game_id", "posteam", "id"])
          .agg(pl.col("name").first(), pl.len().alias("dropbacks"), pl.col("qb_epa").fill_null(0).sum().alias("qb_epa_sum"),
               pl.col("cpoe").mean().alias("cpoe"), pl.col("sack").fill_null(0).sum().alias("sacks"),
               pl.col("interception").fill_null(0).sum().alias("ints")).collect()
          .sort(["game_id", "posteam", "dropbacks", "id"], descending=[False, False, True, False]))
    primary = qb.group_by(["game_id", "posteam"], maintain_order=True).first()
    return team, primary


def load_stats(pbp_dir: Path = PBP_DIR) -> tuple[pl.DataFrame, pl.DataFrame]:
    teams, qbs = [], []
    for f in sorted(pbp_dir.glob("play_by_play_*.parquet")):
        t, q = team_game_stats(pl.scan_parquet(f).select(PBP_COLS))
        teams.append(t), qbs.append(q)
    return pl.concat(teams), pl.concat(qbs)


def kickoff(games: pl.DataFrame) -> pl.DataFrame:
    """Kickoff as a naive US-Eastern timestamp. Games with no listed time are placed at 13:00."""
    return games.with_columns((pl.col("gameday").cast(pl.String) + " " + pl.col("gametime").fill_null("13:00")).str.to_datetime("%Y-%m-%d %H:%M").alias("kickoff"))


def _window(vals: list[float], variant: str) -> float:
    v = [x for x in vals if x is not None and not np.isnan(x)]
    if not v:
        return np.nan
    if variant == "w4":
        return float(np.mean(v[-4:]))
    if variant == "w8":
        return float(np.mean(v[-8:]))
    w = 0.5 ** (np.arange(len(v))[::-1] / EWM_HALFLIFE)
    return float(np.dot(w, v) / w.sum())


def build(games: pl.DataFrame, team: pl.DataFrame, qb: pl.DataFrame) -> pl.DataFrame:
    """One row per scheduled game. Adds v2 features to the v1 snapshot (Elo, form, record, rest)."""
    snap = pipeline.build_snapshots(games)
    g = kickoff(games)
    tstat = {(r["game_id"], r["posteam"]): r for r in team.iter_rows(named=True)}
    qstat = {(r["game_id"], r["posteam"]): r for r in qb.iter_rows(named=True)}
    hist: dict[str, list[dict]] = defaultdict(list)          # per team: completed games, in order
    qb_hist: dict[str, list[dict]] = defaultdict(list)       # per quarterback id: games as primary passer
    last_kick: dict[str, object] = {}
    rows = []
    for (_day,), day in g.group_by(["gameday"], maintain_order=True):
        todays = list(day.iter_rows(named=True))
        for r in todays:                                     # features first, from state before this date
            row = {"game_id": r["game_id"], "kickoff": r["kickoff"], "home_rest": r["home_rest"], "away_rest": r["away_rest"]}
            for side in ("home", "away"):
                t = r[side]
                h = hist[t]
                prev = last_kick.get(t)
                row[f"{side}_hours_since_prev_kickoff"] = None if prev is None else (r["kickoff"] - prev).total_seconds() / 3600
                for v in VARIANTS:
                    for s in TEAM_STATS:
                        row[f"{side}_{s}_{v}"] = _window([x["off"].get(s) for x in h], v)
                        row[f"{side}_def_{s}_{v}"] = _window([x["deff"].get(s) for x in h], v)
                proj = next((x["qb"] for x in reversed(h) if x["qb"]), None)       # previous-game primary passer
                row[f"{side}_qb_id"], row[f"{side}_qb_name"] = (proj["id"], proj["name"]) if proj else (None, None)
                recent = qb_hist[proj["id"]][-QB_WINDOW:] if proj else []
                row[f"{side}_qb_dropbacks"] = float(sum(x["dropbacks"] for x in recent))
                row[f"{side}_qb_epa_sum"] = float(sum(x["qb_epa_sum"] for x in recent))
                row[f"{side}_qb_games"] = len(recent)
            rows.append(row)
        for r in todays:                                     # then record the date's completed games
            if r["result"] is None:
                continue
            for side, opp in (("home", "away"), ("away", "home")):
                t, o = r[side], r[opp]
                off, deff, q = tstat.get((r["game_id"], t)), tstat.get((r["game_id"], o)), qstat.get((r["game_id"], t))
                hist[t].append({"off": off or {}, "deff": deff or {}, "qb": {"id": q["id"], "name": q["name"]} if q else None})
                if q:
                    qb_hist[q["id"]].append({"dropbacks": q["dropbacks"], "qb_epa_sum": q["qb_epa_sum"]})
                last_kick[t] = r["kickoff"]
    f = pl.DataFrame(rows, infer_schema_length=None)
    out = snap.join(f, on="game_id", how="left")
    diffs = []
    for v in VARIANTS:
        for s in TEAM_STATS:
            diffs.append((pl.col(f"home_{s}_{v}") - pl.col(f"away_{s}_{v}")).alias(f"d_{s}_{v}"))
            diffs.append((pl.col(f"home_def_{s}_{v}") - pl.col(f"away_def_{s}_{v}")).alias(f"d_def_{s}_{v}"))
    return out.with_columns(
        *diffs,
        (pl.col("home_rest") <= 5).cast(pl.Float64).alias("home_short_week"), (pl.col("away_rest") <= 5).cast(pl.Float64).alias("away_short_week"),
        (pl.col("home_rest") >= 13).cast(pl.Float64).alias("home_off_bye"), (pl.col("away_rest") >= 13).cast(pl.Float64).alias("away_off_bye"),
    )


def cutoff_violations(feat: pl.DataFrame) -> int:
    """Games where a team's previous kickoff was less than 28 hours before this kickoff (should be zero)."""
    need = CUTOFF_HOURS + GAME_HOURS
    return feat.filter((pl.col("home_hours_since_prev_kickoff") < need) | (pl.col("away_hours_since_prev_kickoff") < need)).height


GROUPS = {
    "rest": ["rest_diff", "home_short_week", "away_short_week", "home_off_bye", "away_off_bye", "div_game"],
    "form": ["form_diff", "win_pct_diff"],
    "efficiency": lambda v: [f"d_off_epa_{v}", f"d_def_off_epa_{v}", f"d_pass_epa_{v}", f"d_def_pass_epa_{v}", f"d_rush_epa_{v}", f"d_def_rush_epa_{v}"],
    "rates": lambda v: [f"d_success_{v}", f"d_def_success_{v}", f"d_sack_rate_{v}", f"d_def_sack_rate_{v}", f"d_turnover_rate_{v}",
                        f"d_def_turnover_rate_{v}", f"d_explosive_{v}", f"d_def_explosive_{v}"],
    "qb": ["d_qb_epa_shrunk", "d_qb_log_dropbacks"],
}


def group_columns(groups: list[str], variant: str) -> list[str]:
    cols: list[str] = []
    for g in groups:
        c = GROUPS[g]
        cols += c(variant) if callable(c) else c
    return cols


DESCRIPTIONS = {
    "rest_diff": "Home minus away days of rest", "home_short_week": "Home team on 5 or fewer days of rest", "away_short_week": "Away team on 5 or fewer days of rest",
    "home_off_bye": "Home team coming off a bye (13+ days)", "away_off_bye": "Away team coming off a bye (13+ days)", "div_game": "Division game",
    "form_diff": "Home minus away average point margin, last 8 games", "win_pct_diff": "Home minus away win share this season",
    "d_off_epa": "Offense: expected points added per play", "d_def_off_epa": "Defense: expected points allowed per play",
    "d_pass_epa": "Offense: expected points added per dropback", "d_def_pass_epa": "Defense: expected points allowed per dropback",
    "d_rush_epa": "Offense: expected points added per rush", "d_def_rush_epa": "Defense: expected points allowed per rush",
    "d_success": "Offense: success rate", "d_def_success": "Defense: success rate allowed", "d_sack_rate": "Offense: sacks taken per dropback",
    "d_def_sack_rate": "Defense: sacks made per dropback", "d_turnover_rate": "Offense: giveaways per play", "d_def_turnover_rate": "Defense: takeaways per play",
    "d_explosive": "Offense: explosive-play rate (20+ yard pass, 10+ yard rush)", "d_def_explosive": "Defense: explosive-play rate allowed",
    "d_qb_epa_shrunk": "Projected quarterbacks: expected points added per dropback over each one's last 8 starts, pulled toward the league average when he has few dropbacks",
    "d_qb_log_dropbacks": "Projected quarterbacks: difference in recent experience (log dropbacks, last 8 starts)",
}


def describe(col: str) -> str:
    base = col
    for v in VARIANTS:
        base = base.removesuffix(f"_{v}")
    text = DESCRIPTIONS.get(base, col)
    return text if base == col or base in ("d_qb_epa_shrunk", "d_qb_log_dropbacks") else f"{text} (home minus away)"
