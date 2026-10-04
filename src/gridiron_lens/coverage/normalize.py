"""Coordinate and snap normalization: raw tracking CSV -> one Parquet file per week.

Canonical view: offense always moves toward +x. Plays with play direction "left" are rotated
180 degrees: x' = 120 - x, y' = 53.3 - y, and both angles turn by 180. The line of scrimmage goes
through the same transform before it is subtracted. Raw coordinates are kept next to normalized ones.

Time is snap-relative: rel_frame 0 is the snap, 10 frames per second.
- bdb2025 layout: the frame marked SNAP. Plays without that marker are dropped and counted.
- bdb2026 layout: the file has no snap marker and starts at frame 1. The audit found players
  nearly still and the passer about 5 yards behind the line at frame 1, so frame 1 is treated as
  the snap. That is an inference from the data, checked again in each week's report.

Both layouts produce the same columns, so everything downstream is shared.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ..shared import config
from ..shared.provenance import now_utc, write_json
from . import schema

L, W = config.FIELD_LENGTH, config.FIELD_WIDTH
KEY = ["gameId", "playId"]
OUT_COLS = [*KEY, "nflId", "frameId", "rel_frame", "side", "role", "jerseyNumber", "x_raw", "y_raw",
            "x", "y", "x_rel", "los_x", "ref_y", "s", "a", "o", "dir", "flipped"]


def _canonical(lf: pl.LazyFrame, direction: str, los: str) -> pl.LazyFrame:
    left = pl.col(direction) == "left"

    def turn(c: str) -> pl.Expr:
        return pl.when(left).then((pl.col(c) + 180.0) % 360.0).otherwise(pl.col(c))

    return lf.with_columns(
        left.alias("flipped"), pl.col("x").alias("x_raw"), pl.col("y").alias("y_raw"),
        pl.when(left).then(L - pl.col("x")).otherwise(pl.col("x")).alias("x"),
        pl.when(left).then(W - pl.col("y")).otherwise(pl.col("y")).alias("y"),
        turn("o").alias("o"), turn("dir").alias("dir"),
        pl.when(left).then(L - pl.col(los)).otherwise(pl.col(los)).alias("los_x"),
    ).with_columns((pl.col("x") - pl.col("los_x")).alias("x_rel"))


def _checks(out: Path) -> dict:
    """Read the written file back and test the conventions against the data itself."""
    done = pl.scan_parquet(out)
    snap = done.filter(pl.col("rel_frame") == 0)
    side_x = snap.group_by([*KEY, "flipped", "side"]).agg(pl.col("x_rel").mean().alias("m")).collect()

    def share(side: str, flipped: bool, sign: int) -> float | None:
        m = side_x.filter((pl.col("side") == side) & (pl.col("flipped") == flipped))["m"]
        return float(((m * sign) > 0).mean()) if m.len() else None

    # Angle convention: degrees clockwise from +y, so the unit vector is (sin, cos). Compare the
    # reported direction of travel with the actual frame-to-frame displacement of fast players.
    mv = (
        done.filter(pl.col("side") != "ball").sort([*KEY, "nflId", "rel_frame"])
        .with_columns((pl.col("x").shift(-1).over([*KEY, "nflId"]) - pl.col("x")).alias("dx"),
                      (pl.col("y").shift(-1).over([*KEY, "nflId"]) - pl.col("y")).alias("dy"))
        .filter((pl.col("s") > 3) & pl.col("dx").is_not_null() & pl.col("dir").is_not_null())
        .select(((pl.col("dir").radians().sin() * pl.col("dx") + pl.col("dir").radians().cos() * pl.col("dy"))
                 / (pl.col("dx").pow(2) + pl.col("dy").pow(2)).sqrt()).alias("cos"))
        .filter(pl.col("cos").is_finite()).collect()
    )
    ball = snap.filter(pl.col("side") == "ball").select((pl.col("x") - pl.col("los_x")).abs().alias("d")).collect()
    speed = snap.filter(pl.col("side") != "ball").select(pl.col("s").median()).collect().item()
    return {
        "plays_written": done.select(KEY).unique().collect().height,
        "rows_by_side": {s: int(n) for s, n in done.group_by("side").agg(pl.len()).collect().iter_rows()},
        "offense_behind_los_at_snap": {"left": share("offense", True, -1), "right": share("offense", False, -1)},
        "defense_beyond_los_at_snap": {"left": share("defense", True, 1), "right": share("defense", False, 1)},
        "median_player_speed_at_snap": speed,
        "direction_vs_displacement_cosine": {"median": float(mv["cos"].median()) if mv.height else None, "n": mv.height},
        "ball_to_los_abs_yards_at_snap": {"median": float(ball["d"].median()), "p95": float(ball["d"].quantile(0.95))} if ball.height else None,
    }


# ---------------------------------------------------------------- bdb2025 layout (synthetic fixture)

def _plays_2025(raw_dir: Path) -> pl.DataFrame:
    plays = pl.read_csv(raw_dir / "plays.csv", null_values=schema.NULLS, infer_schema_length=20000)
    games = pl.read_csv(raw_dir / "games.csv", null_values=schema.NULLS, infer_schema_length=20000)
    df = plays.join(games.select(["gameId", "season", "week", "homeTeamAbbr", "visitorTeamAbbr"]), on="gameId")
    diff = pl.col("preSnapHomeScore") - pl.col("preSnapVisitorScore")
    return df.rename(schema.RAW_LABELS["bdb2025"]).select(
        *KEY, "season", "week", "quarter", "down", "yardsToGo", "possessionTeam", "defensiveTeam",
        "absoluteYardlineNumber", pl.col("man_zone").replace_strict(schema.MAN_ZONE_VALUES, default=None), "coverage_type",
        pl.col("playDescription") if "playDescription" in df.columns else pl.lit(None).alias("playDescription"),
        pl.when(pl.col("defensiveTeam") == pl.col("homeTeamAbbr")).then(diff).otherwise(-diff).alias("def_score_diff"))


def _week_2025(raw_dir: Path, out: Path, week: int, plays: pl.DataFrame) -> dict:
    lf = pl.scan_csv(raw_dir / schema.tracking_file(week), null_values=schema.NULLS, infer_schema_length=20000,
                     schema_overrides={"nflId": pl.Int64, "jerseyNumber": pl.Int64, "event": pl.String, "o": pl.Float64, "dir": pl.Float64})
    all_plays = lf.select(KEY).unique().collect(engine="streaming")
    snap = lf.filter(pl.col("frameType") == "SNAP").group_by(KEY).agg(pl.col("frameId").min().alias("snap_frame")).collect(engine="streaming")
    ref = (lf.filter((pl.col("frameType") == "SNAP") & pl.col("nflId").is_null())
           .select(*KEY, pl.when(pl.col("playDirection") == "left").then(W - pl.col("y")).otherwise(pl.col("y")).alias("ref_y"))
           .unique(KEY).collect(engine="streaming"))
    meta = plays.select([*KEY, "possessionTeam", "defensiveTeam", "absoluteYardlineNumber"])
    norm = _canonical(lf.join(snap.lazy(), on=KEY).join(meta.lazy(), on=KEY).join(ref.lazy(), on=KEY, how="left"),
                      "playDirection", "absoluteYardlineNumber").with_columns(
        (pl.col("frameId") - pl.col("snap_frame")).alias("rel_frame"), pl.lit(None, dtype=pl.String).alias("role"),
        pl.when(pl.col("nflId").is_null()).then(pl.lit("ball"))
        .when(pl.col("club") == pl.col("possessionTeam")).then(pl.lit("offense"))
        .when(pl.col("club") == pl.col("defensiveTeam")).then(pl.lit("defense")).otherwise(pl.lit("unknown")).alias("side"))
    norm.select(OUT_COLS).sort([*KEY, "frameId", "side", "nflId"]).sink_parquet(out, compression="zstd")
    return {"week": week, "plays_in_tracking": all_plays.height, "dropped_no_snap_frame": all_plays.height - snap.height}


# ---------------------------------------------------------------- bdb2026 layout (real data)

def _plays_2026(raw_dir: Path) -> pl.DataFrame:
    s = pl.read_csv(raw_dir / "supplementary_data.csv", null_values=schema.NULLS, infer_schema_length=20000)
    diff = pl.col("pre_snap_home_score") - pl.col("pre_snap_visitor_score")
    return s.rename(schema.RAW_LABELS["bdb2026"]).select(
        pl.col("game_id").alias("gameId"), pl.col("play_id").alias("playId"), "season", "week", "quarter", "down",
        pl.col("yards_to_go").alias("yardsToGo"), pl.col("possession_team").alias("possessionTeam"),
        pl.col("defensive_team").alias("defensiveTeam"),
        pl.col("man_zone").replace_strict(schema.MAN_ZONE_VALUES, default=None), "coverage_type",
        pl.col("play_description").alias("playDescription"),
        pl.when(pl.col("defensive_team") == pl.col("home_team_abbr")).then(diff).otherwise(-diff).alias("def_score_diff"))


def _week_2026(src: Path, out: Path, week: int) -> tuple[dict, pl.DataFrame]:
    lf = pl.scan_csv(src, null_values=schema.NULLS, infer_schema_length=20000).rename(
        {"game_id": "gameId", "play_id": "playId", "nfl_id": "nflId", "frame_id": "frameId"})
    people = lf.select("nflId", pl.col("player_name").alias("displayName"), pl.col("player_position").alias("position")).unique("nflId").collect()
    at1 = lf.filter(pl.col("frameId") == 1).select(
        *KEY, "player_role", "player_side", pl.when(pl.col("play_direction") == "left").then(W - pl.col("y")).otherwise(pl.col("y")).alias("yn")).collect()
    ref = at1.group_by(KEY).agg(  # lateral centre of the formation: the passer's spot at the snap
        pl.col("yn").filter(pl.col("player_role") == "Passer").first().alias("qb_y"),
        pl.col("yn").filter(pl.col("player_side") == "Offense").mean().alias("off_y"),
    ).select(*KEY, pl.coalesce("qb_y", "off_y").alias("ref_y"))
    norm = _canonical(lf.join(ref.lazy(), on=KEY, how="left"), "play_direction", "absolute_yardline_number").with_columns(
        (pl.col("frameId") - 1).alias("rel_frame"), pl.col("player_side").str.to_lowercase().alias("side"),
        pl.col("player_role").alias("role"), pl.lit(None, dtype=pl.Int64).alias("jerseyNumber"))
    norm.select(OUT_COLS).sort([*KEY, "frameId", "side", "nflId"]).sink_parquet(out, compression="zstd")
    no_qb = at1.group_by(KEY).agg((pl.col("player_role") == "Passer").any().alias("q")).filter(~pl.col("q")).height
    return {"week": week, "plays_in_tracking": ref.height, "plays_without_passer": no_qb}, people


def run(source: str, raw_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    layout = schema.SOURCES[source]["layout"]
    raw_dir = Path(raw_dir or config.RAW / source)
    out_dir = Path(out_dir or config.PROCESSED / source)
    out_dir.mkdir(parents=True, exist_ok=True)
    synthetic = (raw_dir / "SYNTHETIC").exists()
    weeks, people = [], []
    if layout == "bdb2025":
        plays = _plays_2025(raw_dir)
        for w in [w for w in schema.WEEKS_2025 if (raw_dir / schema.tracking_file(w)).exists()]:
            out = out_dir / f"tracking_week_{w}.parquet"
            weeks.append(_week_2025(raw_dir, out, w, plays) | {"output": config.rel(out)} | _checks(out))
        players = pl.read_csv(raw_dir / "players.csv", null_values=schema.NULLS).select(["nflId", "displayName", "position"])
        plays = plays.drop("absoluteYardlineNumber")
    else:
        plays = _plays_2026(raw_dir)
        for src in sorted(raw_dir.glob("input_*_w*.csv")):
            w = int(src.stem.split("_w")[-1])
            out = out_dir / f"tracking_week_{w}.parquet"
            rep, ppl = _week_2026(src, out, w)
            people.append(ppl)
            weeks.append(rep | {"input": src.name, "output": config.rel(out)} | _checks(out))
        players = pl.concat(people).unique("nflId")
    plays.write_parquet(out_dir / "plays.parquet")
    players.write_parquet(out_dir / "players.parquet")
    report = {
        "source": source, "layout": layout, "normalized_at": now_utc(), "synthetic": synthetic,
        "convention": "offense moves +x; left plays rotated 180 deg; x_rel = x - los_x; rel_frame 0 = snap; 10 frames per second",
        "snap_definition": "frame marked SNAP" if layout == "bdb2025" else "first frame of the file (inferred, see median_player_speed_at_snap)",
        "weeks": weeks,
    }
    if synthetic:
        (out_dir / "SYNTHETIC").write_text("Derived from the synthetic fixture.\n")
    write_json(out_dir / "normalize_report.json", report)
    return report
