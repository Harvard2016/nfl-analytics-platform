"""Cutoff-safe geometric features, one row per play per observation window.

A window names the latest frame the model may see. Only frames at or before that cutoff are read
from disk, so later frames cannot leak in. Charted labels are attached as the target column only.
Frames are not independent samples: the unit is the play.

The feature builder never reads which receiver was targeted, where the ball landed or how long the
play lasted. It does use who the passer is, which is known at the snap.

v3 dropped n_def (how many coverage defenders the file tracks). The dataset only tracks defenders
it labels "Defensive Coverage", a label that depends on who rushed, which is not known at the snap.
Ablation on the locked split: at-snap boosted accuracy 0.848 with it, 0.832 without. The remaining
features are still computed over that selected set of defenders, so some of that information
remains; this is a stated limit of the source, not something the features can remove.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ..shared import config
from ..shared.provenance import now_utc, write_json
from . import schema

# window -> (latest frame the model sees, frame motion is measured from). Frame 0 is the snap.
WINDOWS = {"at_snap": (0, None), "post_1s": (10, 0), "post_1_5s": (15, 0)}
MIN_DEFENSE, MIN_OFFENSE = 4, 3
N_SKILL = 5
FEATURE_SCHEMA_VERSION = "cov-geo-v3"

STATIC = ["depth_max", "depth_2nd", "depth_mean", "depth_std", "n_deep_10", "n_deep_7",
          "deep2_lateral_sep", "width_std", "y_offset", "n_box", "cov_dist_mean", "cov_dist_min",
          "cov_dist_max", "press_count", "cover_facing_receiver", "face_backfield", "speed_mean", "speed_max"]
MOTION = ["d_depth_abs", "d_lateral_abs", "follow_cos", "cov_dist_change", "off_max_lateral", "motion_follow"]

DESCRIPTIONS = {
    "depth_max": "Depth of the deepest defender past the line of scrimmage (yards)",
    "depth_2nd": "Depth of the second-deepest defender (yards)",
    "depth_mean": "Mean defender depth (yards)",
    "depth_std": "Spread of defender depths (yards)",
    "n_deep_10": "Defenders 10+ yards deep",
    "n_deep_7": "Defenders 7+ yards deep",
    "deep2_lateral_sep": "Sideline-to-sideline gap between the two deepest defenders (yards)",
    "width_std": "Lateral spread of the defense (yards)",
    "y_offset": "Lateral shift of the defense's centre from the middle of the formation (yards)",
    "n_box": "Defenders within 3 yards of the line and 5 yards of the middle of the formation",
    "cov_dist_mean": "Mean distance from each route runner to his nearest defender (yards)",
    "cov_dist_min": "Smallest such distance (yards)",
    "cov_dist_max": "Largest such distance (yards)",
    "press_count": "Route runners with a defender within 3 yards",
    "cover_facing_receiver": "How squarely nearest defenders face their route runner (1 = straight at him)",
    "face_backfield": "How much defenders face the offensive backfield (1 = straight at it)",
    "speed_mean": "Mean defender speed (yards/second)",
    "speed_max": "Fastest defender speed (yards/second)",
    "d_depth_abs": "Mean absolute change in defender depth since the snap (yards)",
    "d_lateral_abs": "Mean absolute lateral defender movement since the snap (yards)",
    "follow_cos": "Direction agreement between nearest defenders and their route runner's movement",
    "cov_dist_change": "Change in nearest-defender distance since the snap (yards)",
    "off_max_lateral": "Largest lateral move by any offensive player since the snap (yards)",
    "motion_follow": "Lateral move of that player's nearest defender in the same direction (yards)",
}


def feature_names(window: str) -> list[str]:
    if WINDOWS[window][1] is None:
        return list(STATIC)
    return STATIC + MOTION + [f"snap_{n}" for n in STATIC]  # later windows also see the shell at the snap


def _facing(deg: np.ndarray) -> np.ndarray:
    """Angles are degrees clockwise from +y (checked against displacement in the normalize report)."""
    r = np.deg2rad(deg)
    return np.stack([np.sin(r), np.cos(r)], axis=-1)


def _frame(p: pl.DataFrame, frame: int) -> dict | None:
    f = p.filter(pl.col("rel_frame") == frame)
    d = f.filter(pl.col("side") == "defense")
    o = f.filter((pl.col("side") == "offense") & (pl.col("role").is_null() | (pl.col("role") != "Passer")))
    if d.height < MIN_DEFENSE or o.height < MIN_OFFENSE or f["ref_y"][0] is None:
        return None

    def arr(t: pl.DataFrame, c: str) -> np.ndarray:
        return t[c].to_numpy().astype(float)

    return {
        "ref_y": float(f["ref_y"][0]),
        "d_id": d["nflId"].to_numpy(), "d_xy": np.stack([arr(d, "x_rel"), arr(d, "y")], -1),
        "d_s": arr(d, "s"), "d_o": arr(d, "o"),
        "o_id": o["nflId"].to_numpy(), "o_xy": np.stack([arr(o, "x_rel"), arr(o, "y")], -1),
    }


def _skill_pairs(fr: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Up to N_SKILL non-passer offensive players farthest from the middle of the formation, each with his nearest defender."""
    away = np.hypot(fr["o_xy"][:, 0], fr["o_xy"][:, 1] - fr["ref_y"])
    skill = np.argsort(-away)[:N_SKILL]
    dist = np.linalg.norm(fr["o_xy"][skill, None, :] - fr["d_xy"][None, :, :], axis=-1)
    near = dist.argmin(axis=1)
    return skill, near, dist[np.arange(len(skill)), near]


def _static(fr: dict) -> dict:
    order = np.argsort(-fr["d_xy"][:, 0])
    depth = fr["d_xy"][order, 0]
    y = fr["d_xy"][:, 1]
    skill, near, cov = _skill_pairs(fr)
    to_rcv = fr["o_xy"][skill] - fr["d_xy"][near]
    to_rcv = to_rcv / np.maximum(np.linalg.norm(to_rcv, axis=1, keepdims=True), 1e-6)
    face = _facing(fr["d_o"])
    with np.errstate(invalid="ignore"):
        cover_facing = np.nanmean((face[near] * to_rcv).sum(1)) if np.isfinite(face[near]).any() else np.nan
        backfield = np.nanmean(-face[:, 0]) if np.isfinite(face).any() else np.nan
    return {
        "depth_max": depth[0], "depth_2nd": depth[1], "depth_mean": depth.mean(), "depth_std": depth.std(),
        "n_deep_10": float((depth >= 10).sum()), "n_deep_7": float((depth >= 7).sum()),
        "deep2_lateral_sep": abs(y[order[0]] - y[order[1]]),
        "width_std": y.std(), "y_offset": y.mean() - fr["ref_y"],
        "n_box": float(((fr["d_xy"][:, 0] < 3) & (np.abs(y - fr["ref_y"]) < 5)).sum()),
        "cov_dist_mean": cov.mean(), "cov_dist_min": cov.min(), "cov_dist_max": cov.max(),
        "press_count": float((cov <= 3).sum()),
        "cover_facing_receiver": cover_facing, "face_backfield": backfield,
        "speed_mean": np.nanmean(fr["d_s"]), "speed_max": np.nanmax(fr["d_s"]),
    }


def _motion(f0: dict, f1: dict) -> dict:
    d0 = dict(zip(f0["d_id"].tolist(), f0["d_xy"]))
    o0 = dict(zip(f0["o_id"].tolist(), f0["o_xy"]))
    dd = np.array([f1["d_xy"][i] - d0[k] for i, k in enumerate(f1["d_id"].tolist()) if k in d0])
    od = {k: f1["o_xy"][i] - o0[k] for i, k in enumerate(f1["o_id"].tolist()) if k in o0}
    if len(dd) == 0 or not od:
        return dict.fromkeys(MOTION, np.nan)
    skill, near, cov1 = _skill_pairs(f1)
    cos, dchange = [], []
    for s, n, c1 in zip(skill, near, cov1):
        ok, dk = f1["o_id"][s], f1["d_id"][n]
        if ok not in o0 or dk not in d0:
            continue
        mo, md = f1["o_xy"][s] - o0[ok], f1["d_xy"][n] - d0[dk]
        dchange.append(c1 - np.linalg.norm(o0[ok] - d0[dk]))
        if np.linalg.norm(mo) > 0.3 and np.linalg.norm(md) > 0.3:
            cos.append(float(mo @ md / (np.linalg.norm(mo) * np.linalg.norm(md))))
    mover = max(od, key=lambda k: abs(od[k][1]))
    mi = int(np.where(f1["o_id"] == mover)[0][0])
    nd = f1["d_id"][np.linalg.norm(f1["d_xy"] - f1["o_xy"][mi], axis=1).argmin()]
    follow = (f1["d_xy"][f1["d_id"] == nd][0][1] - d0[nd][1]) * np.sign(od[mover][1]) if nd in d0 else np.nan
    return {
        "d_depth_abs": np.abs(dd[:, 0]).mean(), "d_lateral_abs": np.abs(dd[:, 1]).mean(),
        "follow_cos": float(np.mean(cos)) if cos else np.nan,
        "cov_dist_change": float(np.mean(dchange)) if dchange else np.nan,
        "off_max_lateral": abs(od[mover][1]), "motion_follow": follow,
    }


def play_features(p: pl.DataFrame, window: str) -> tuple[dict | None, str]:
    """Features for one play. `p` must already be limited to frames at or before the cutoff."""
    cutoff, start = WINDOWS[window]
    assert p["rel_frame"].max() <= cutoff, "frames after the cutoff reached the feature builder"
    f1 = _frame(p, cutoff)
    if f1 is None:
        return None, "too_few_players_or_no_frame_at_cutoff"
    out = _static(f1)
    if start is not None:
        snap = _frame(p, start)
        if snap is None:
            return None, "too_few_players_at_snap"
        out |= _motion(snap, f1) | {f"snap_{k}": v for k, v in _static(snap).items()}
    return out, "ok"


def build_window(processed_dir: Path, window: str, plays: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    cutoff, start = WINDOWS[window]
    frames = [cutoff] if start is None else [start, cutoff]
    rows, reasons, los = [], {}, []
    for path in sorted(processed_dir.glob("tracking_week_*.parquet")):
        df = (
            pl.scan_parquet(path)
            .filter(pl.col("rel_frame").is_in(frames) & (pl.col("rel_frame") <= cutoff))
            .select(["gameId", "playId", "nflId", "rel_frame", "side", "role", "x_rel", "y", "s", "o", "los_x", "ref_y"])
            .collect()
        )
        los.append(df.group_by(["gameId", "playId"]).agg(pl.col("los_x").first()))
        for (gid, pid), p in df.partition_by(["gameId", "playId"], as_dict=True).items():
            feats, reason = play_features(p, window)
            reasons[reason] = reasons.get(reason, 0) + 1
            if feats is not None:
                rows.append({"gameId": gid, "playId": pid} | feats)
    names = feature_names(window)
    feats = pl.DataFrame(rows, schema={"gameId": pl.Int64, "playId": pl.Int64} | dict.fromkeys(names, pl.Float64))
    # yards_to_goal uses the direction-normalized line of scrimmage: the goal line is at x = 110
    los_x = pl.concat(los).with_columns((110.0 - pl.col("los_x")).alias("yards_to_goal")).drop("los_x")
    table = plays.join(los_x, on=["gameId", "playId"], how="inner").join(feats, on=["gameId", "playId"], how="inner")
    return table, reasons


def run(source: str, processed_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    processed_dir = Path(processed_dir or config.PROCESSED / source)
    out_dir = Path(out_dir or config.FEATURES / source)
    out_dir.mkdir(parents=True, exist_ok=True)
    plays = pl.read_parquet(processed_dir / "plays.parquet")
    report = {"source": source, "feature_schema": FEATURE_SCHEMA_VERSION, "built_at": now_utc(), "windows": {},
              "descriptions": DESCRIPTIONS, "target": schema.TARGET,
              "synthetic": (processed_dir / "SYNTHETIC").exists()}
    for window, (cutoff, start) in WINDOWS.items():
        table, reasons = build_window(processed_dir, window, plays)
        table.write_parquet(out_dir / f"features_{window}.parquet")
        report["windows"][window] = {
            "latest_frame": cutoff, "motion_from_frame": start, "features": feature_names(window),
            "plays_with_features": table.height,
            "plays_with_man_zone_label": table.filter(pl.col(schema.TARGET).is_in(schema.TARGET_CLASSES)).height,
            "tracking_play_status": reasons,
        }
    if report["synthetic"]:
        (out_dir / "SYNTHETIC").write_text("Derived from the synthetic fixture.\n")
    write_json(out_dir / "feature_report.json", report)
    return report
