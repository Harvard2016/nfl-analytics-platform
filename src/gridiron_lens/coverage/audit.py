"""Data and label audit of the raw files: what is actually on disk, not what a schema page says.

Writes data/manifests/<source>_audit.json with file hashes, columns, seasons, key duplicates,
label counts by week, frames per play and tracked players per play.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ..shared import config
from ..shared.provenance import file_record, now_utc, write_json
from . import schema


def _counts(df: pl.DataFrame, col: str) -> dict:
    return {("<missing>" if k is None else str(k)): int(n) for k, n in df.group_by(col).len().sort("len", descending=True).iter_rows()}


def _audit_2026(raw: Path) -> dict:
    s = pl.read_csv(raw / "supplementary_data.csv", null_values=schema.NULLS, infer_schema_length=20000)
    key = ["game_id", "play_id"]
    weeks, tracked = [], []
    for f in sorted(raw.glob("input_*_w*.csv")):
        lf = pl.scan_csv(f, null_values=schema.NULLS, infer_schema_length=20000)
        pp = lf.group_by(key).agg(
            pl.len().alias("rows"), pl.col("frame_id").min().alias("first"), pl.col("frame_id").max().alias("last"),
            pl.col("frame_id").n_unique().alias("frames"), pl.col("play_direction").n_unique().alias("dirs"),
            pl.col("nfl_id").filter(pl.col("player_side") == "Defense").n_unique().alias("defenders"),
            pl.col("nfl_id").filter(pl.col("player_side") == "Offense").n_unique().alias("offense"),
            pl.col("x").null_count().alias("null_x"),
        ).collect(engine="streaming")
        dup = lf.group_by(key + ["nfl_id", "frame_id"]).len().filter(pl.col("len") > 1).select(pl.len()).collect(engine="streaming").item()
        tracked.append(pp.select(key))
        weeks.append({
            "file": f.name, "rows": int(pp["rows"].sum()), "plays": pp.height, "games": pp["game_id"].n_unique(),
            "first_frame_values": sorted(pp["first"].unique().to_list()),
            "frames_per_play": {"min": int(pp["frames"].min()), "median": float(pp["frames"].median()), "max": int(pp["frames"].max())},
            "plays_with_frame_gaps": pp.filter(pl.col("frames") != pl.col("last") - pl.col("first") + 1).height,
            "plays_with_mixed_direction": pp.filter(pl.col("dirs") > 1).height,
            "defenders_per_play": {"min": int(pp["defenders"].min()), "median": float(pp["defenders"].median()), "max": int(pp["defenders"].max())},
            "offense_per_play": {"min": int(pp["offense"].min()), "median": float(pp["offense"].median()), "max": int(pp["offense"].max())},
            "null_position_rows": int(pp["null_x"].sum()), "duplicate_player_frame_rows": int(dup),
            "plays_missing_from_supplementary": pp.join(s, on=key, how="anti").height,
        })
    trk = pl.concat(tracked).unique()
    joined = s.join(trk, on=key, how="inner")
    cols = pl.scan_csv(min(raw.glob("input_*_w*.csv"))).collect_schema().names()
    return {
        "supplementary": {
            "columns": s.columns, "rows": s.height, "duplicate_game_play": s.height - s.select(key).n_unique(),
            "play_id_reused_across_games": s.height - s["play_id"].n_unique(),
            "seasons": _counts(s, "season"), "games": s["game_id"].n_unique(),
            "labels_all_rows": {c: _counts(s, c) for c in schema.RAW_LABELS["bdb2026"]},
            "rows_without_tracking": s.height - joined.height,
        },
        "tracked_plays": {
            "plays": trk.height, "seasons": _counts(joined, "season"),
            "labels": {c: _counts(joined, c) for c in schema.RAW_LABELS["bdb2026"]},
            "man_zone_by_week": [{"week": w, "label": l or "<missing>", "plays": n} for w, l, n in
                                 joined.group_by(["week", "team_coverage_man_zone"]).len().sort(["week", "team_coverage_man_zone"]).iter_rows()],
        },
        "tracking_columns": cols,
        "tracking_columns_never_used_as_features": ["player_to_predict", "num_frames_output", "ball_land_x", "ball_land_y",
                                                    "player_role value 'Targeted Receiver'"],
        "known_limits": ["Only the passer, route runners and coverage defenders are tracked: no linemen, no pass rushers, no ball.",
                         "Tracking covers the snap to the throw only. There are no pre-snap frames.",
                         "Only pass plays that reached a throw are included. This is not every defensive snap.",
                         "output_*.csv holds positions after the throw for some players; the coverage models do not read it."],
        "weeks": weeks,
    }


def _audit_2025(raw: Path) -> dict:
    p = pl.read_csv(raw / "plays.csv", null_values=schema.NULLS, infer_schema_length=20000)
    g = pl.read_csv(raw / "games.csv", null_values=schema.NULLS, infer_schema_length=20000)
    return {"plays": {"rows": p.height, "duplicate_game_play": p.height - p.select(["gameId", "playId"]).n_unique(),
                      "labels": {c: _counts(p, c) for c in schema.RAW_LABELS["bdb2025"]}},
            "games": {"rows": g.height, "seasons": _counts(g, "season")}}


def run(source: str, raw_dir: Path | None = None, out: Path | None = None, with_hash: bool = True) -> dict:
    raw = Path(raw_dir or config.RAW / source)
    out = Path(out or config.MANIFESTS / f"{source}_audit.json")
    info = schema.SOURCES[source]
    files = sorted(f for f in raw.glob("*.csv"))
    report = {
        "source": source, "source_url": info["url"], "audited_at": now_utc(), "raw_dir": config.rel(raw),
        "synthetic": (raw / "SYNTHETIC").exists(),
        "files": [file_record(f, with_hash) for f in files], "total_bytes": sum(f.stat().st_size for f in files),
        "rights": "Synthetic, no restriction." if (raw / "SYNTHETIC").exists() else
                  "UNVERIFIED by this project. Downloaded by the owner after accepting the competition rules. "
                  "Public display or redistribution has not been checked against those rules.",
    }
    report |= _audit_2026(raw) if info["layout"] == "bdb2026" else _audit_2025(raw)
    write_json(out, report)
    return report
