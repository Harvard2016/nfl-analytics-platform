"""Player tensors and relational defender-receiver features (experiment A, feature schema cov-rel-v1).

Research setting, stated plainly: the release tracks only the passer, route runners and the defenders it
marks as coverage players, and that choice was made with knowledge of the play. Everything here is
computed over that selected set. Masks and aggregates can still reveal the set, so nothing built from
this module is a full-field, pre-snap or leakage-free system.

Used from the tracking file: position, speed, direction of travel, orientation, side, and whether a
player is the passer (known at the snap). Never used: ball landing spot, requested output length,
player_to_predict, the "Targeted Receiver" role, later frames, or any coverage label.

"Receiver" below means a non-passer offensive player in the release. Pair measurements are geometry,
not confirmed matchups or coverage responsibilities.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

from ..shared import config
from ..shared.provenance import now_utc, write_json
from . import schema

T_MAX = 16                      # frames 0..15: snap to +1.5 s
HORIZONS = {"at_snap": 0, "post_0_5s": 5, "post_1s": 10, "post_1_5s": 15}
D_MAX, R_MAX = 11, 6
PLAYER_FEATS = ["x_rel", "y_c", "vx", "vy", "ox", "oy"]   # y_c = lateral position from the formation centre
KEY = ["gameId", "playId"]
SCHEMA = "cov-rel-v1"


def unit(deg: np.ndarray) -> np.ndarray:
    """Tracking angles are degrees clockwise from +y, so the unit vector is (sin, cos) in (x, y)."""
    r = np.deg2rad(deg)
    return np.stack([np.sin(r), np.cos(r)], axis=-1)


def reflect(players: np.ndarray) -> np.ndarray:
    """Mirror left/right across the formation centre: lateral position, lateral velocity and lateral facing change sign."""
    out = players.copy()
    out[..., [1, 3, 5]] *= -1
    return out


def build_tensors(source: str = "bdb2026", processed: Path | None = None, out: Path | None = None) -> dict:
    processed = Path(processed or config.PROCESSED / source)
    out = Path(out or config.FEATURES / source / "tensors_v2.npz")
    plays = pl.read_parquet(processed / "plays.parquet")
    meta, defs, recs, dmask, rmask, nframes = [], [], [], [], [], []
    dropped = {"more_players_than_slots": 0, "no_formation_centre": 0}
    for path in sorted(processed.glob("tracking_week_*.parquet")):
        df = (pl.scan_parquet(path).filter((pl.col("rel_frame") < T_MAX) & (pl.col("side") != "ball"))
              .filter(pl.col("role").is_null() | (pl.col("role") != "Passer"))
              .select([*KEY, "nflId", "rel_frame", "side", "x_rel", "y", "ref_y", "s", "dir", "o"]).collect())
        for (gid, pid), p in df.partition_by(KEY, as_dict=True).items():
            if p["ref_y"][0] is None:
                dropped["no_formation_centre"] += 1
                continue
            n = int(p["rel_frame"].max()) + 1
            arrs, masks = [], []
            for side, cap in (("defense", D_MAX), ("offense", R_MAX)):
                q = p.filter(pl.col("side") == side).sort(["nflId", "rel_frame"])
                ids = q["nflId"].unique(maintain_order=True).to_list()
                if len(ids) > cap:
                    break
                a = np.zeros((T_MAX, cap, len(PLAYER_FEATS)), np.float32)
                for i, pid_ in enumerate(ids):
                    r = q.filter(pl.col("nflId") == pid_)
                    f = r["rel_frame"].to_numpy()
                    v = unit(r["dir"].to_numpy()) * r["s"].to_numpy()[:, None]
                    o = unit(r["o"].to_numpy())
                    a[f, i] = np.stack([r["x_rel"].to_numpy(), r["y"].to_numpy() - p["ref_y"][0], v[:, 0], v[:, 1], o[:, 0], o[:, 1]], -1)
                m = np.zeros(cap, bool)
                m[: len(ids)] = True
                arrs.append(a), masks.append(m)
            if len(arrs) < 2:
                dropped["more_players_than_slots"] += 1
                continue
            meta.append((gid, pid)), defs.append(arrs[0]), recs.append(arrs[1]), dmask.append(masks[0]), rmask.append(masks[1]), nframes.append(n)
    keys = pl.DataFrame(meta, schema=KEY, orient="row").join(plays, on=KEY, how="left", maintain_order="left")
    np.savez_compressed(out, defs=np.stack(defs), recs=np.stack(recs), dmask=np.stack(dmask), rmask=np.stack(rmask),
                        nframes=np.array(nframes, np.int16), gameId=keys["gameId"].to_numpy(), playId=keys["playId"].to_numpy())
    keys.select([*KEY, "week", "defensiveTeam", "possessionTeam", schema.TARGET, "coverage_type"]).with_columns(
        pl.Series("n_frames", nframes), pl.Series("n_def", np.stack(dmask).sum(1)), pl.Series("n_rec", np.stack(rmask).sum(1))
    ).write_parquet(out.with_name("tensors_v2_index.parquet"))
    rep = {"schema": SCHEMA, "built_at": now_utc(), "plays": len(meta), "dropped": dropped, "frames": T_MAX, "player_features": PLAYER_FEATS,
           "defender_slots": D_MAX, "receiver_slots": R_MAX, "horizons": HORIZONS,
           "prefix_available": {h: int((np.array(nframes) > f).sum()) for h, f in HORIZONS.items()}}
    write_json(out.with_name("tensors_v2_report.json"), rep)
    return rep


def _agg(prefix: str, v: np.ndarray) -> dict:
    """Order-invariant summary over players. Empty input gives NaN."""
    if v.size == 0 or not np.isfinite(v).any():
        return {f"{prefix}_{s}": np.nan for s in ("mean", "min", "max", "std")}
    return {f"{prefix}_mean": float(np.nanmean(v)), f"{prefix}_min": float(np.nanmin(v)),
            f"{prefix}_max": float(np.nanmax(v)), f"{prefix}_std": float(np.nanstd(v))}


def pair_arrays(d: np.ndarray, r: np.ndarray) -> dict:
    """Pair measurements for frames x defenders x receivers. d: (T, D, 6), r: (T, R, 6)."""
    rel = d[:, :, None, :2] - r[:, None, :, :2]                 # defender minus receiver position
    sep = np.linalg.norm(rel, axis=-1)
    rv = d[:, :, None, 2:4] - r[:, None, :, 2:4]
    closing = -(rel * rv).sum(-1) / np.maximum(sep, 1e-3)        # positive when the gap is shrinking
    sd, sr = np.linalg.norm(d[..., 2:4], axis=-1), np.linalg.norm(r[..., 2:4], axis=-1)
    dot = (d[:, :, None, 2:4] * r[:, None, :, 2:4]).sum(-1)
    both = (sd[:, :, None] > 0.5) & (sr[:, None, :] > 0.5)       # direction agreement is undefined for players standing still
    dircos = np.where(both, dot / np.maximum(sd[:, :, None] * sr[:, None, :], 1e-6), np.nan)
    to_rec = -rel / np.maximum(sep[..., None], 1e-3)
    facing = (d[:, :, None, 4:6] * to_rec).sum(-1)               # 1 when the defender faces the receiver
    return {"rel": rel, "sep": sep, "rv": rv, "closing": closing, "dircos": dircos, "facing": facing}


def play_relational(d: np.ndarray, r: np.ndarray, frame: int) -> dict:
    """Relational features from frames 0..frame only. d, r hold the valid players, all frames."""
    assert d.shape[0] == frame + 1 and r.shape[0] == frame + 1, "caller must pass exactly the observed prefix"
    P = pair_arrays(d, r)
    sep, T = P["sep"], frame + 1
    f: dict = {}
    near_r = sep.argmin(2)                                       # each defender's nearest receiver, per frame
    near_d = sep.argmin(1)                                       # each receiver's nearest defender, per frame
    di = np.arange(d.shape[1])
    now = near_r[-1]
    f |= _agg("d_near_sep", sep[-1, di, now])
    f |= _agg("d_near_relx", P["rel"][-1, di, now, 0])
    f |= _agg("d_near_rely_abs", np.abs(P["rel"][-1, di, now, 1]))
    f |= _agg("d_near_closing", P["closing"][-1, di, now])
    f |= _agg("d_near_facing", P["facing"][-1, di, now])
    f |= _agg("r_near_sep", sep[-1].min(0))
    srt = np.sort(sep[-1], axis=1)
    f |= _agg("d_near_margin", srt[:, 1] - srt[:, 0] if srt.shape[1] > 1 else np.full(len(di), np.nan))   # small = ambiguous nearest receiver
    # one-to-one pairing that minimizes total separation at the cutoff frame: geometry, not a responsibility label
    rows, cols = linear_sum_assignment(sep[-1])
    f |= _agg("match_sep", sep[-1, rows, cols])
    f |= _agg("match_facing", P["facing"][-1, rows, cols])
    f["match_within_3"] = float((sep[-1, rows, cols] <= 3).mean())
    f["match_within_5"] = float((sep[-1, rows, cols] <= 5).mean())
    # defense as a group
    f["def_width"] = float(np.ptp(d[-1, :, 1]))
    f["def_depth_range"] = float(np.ptp(d[-1, :, 0]))
    f["def_depth_std"] = float(d[-1, :, 0].std())
    f["def_speed_mean"] = float(np.linalg.norm(d[-1, :, 2:4], axis=1).mean())
    spd = np.linalg.norm(d[-1, :, 2:4], axis=1)
    mv = spd > 0.5
    f["def_move_coherence"] = float(np.linalg.norm((d[-1, mv, 2:4] / spd[mv, None]).mean(0))) if mv.sum() >= 2 else np.nan
    if frame > 0:
        tm = sep[:, rows, cols]                                  # separation history of the cutoff-frame pairing
        f |= _agg("match_sep_mean_t", tm.mean(0))
        f |= _agg("match_sep_min_t", tm.min(0))
        f |= _agg("match_sep_std_t", tm.std(0))
        f |= _agg("match_sep_change", tm[-1] - tm[0])
        f |= _agg("match_closing_mean_t", P["closing"][:, rows, cols].mean(0))
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                f |= _agg("match_dircos_mean_t", np.nanmean(P["dircos"][:, rows, cols], axis=0))
                f |= _agg("d_follow_dircos", np.nanmean(np.take_along_axis(P["dircos"], near_r[..., None], 2)[..., 0], axis=0))
        # how steadily each defender stays nearest to one receiver
        modal = np.array([np.bincount(near_r[:, i]).max() for i in di]) / T
        switches = (np.diff(near_r, axis=0) != 0).sum(0)
        f |= _agg("d_near_persist", modal)
        f["d_near_switch_mean"] = float(switches.mean())
        f["d_near_switch_any"] = float((switches > 0).mean())
        f["r_near_def_changes"] = float((np.diff(near_d, axis=0) != 0).sum(0).mean())
        sw = np.argwhere(np.diff(near_r, axis=0) != 0)           # movement agreement with the new nearest receiver at a switch
        if len(sw):
            vals = [P["dircos"][t + 1, i, near_r[t + 1, i]] for t, i in sw]
            f["switch_dircos_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        else:
            f["switch_dircos_mean"] = np.nan
        disp = d[-1, :, :2] - d[0, :, :2]                        # movement since the snap, relative to the line of scrimmage
        f |= _agg("d_disp_depth", disp[:, 0])
        f |= _agg("d_disp_lateral_abs", np.abs(disp[:, 1]))
        f["def_width_change"] = float(np.ptp(d[-1, :, 1]) - np.ptp(d[0, :, 1]))
        f["def_depth_mean_change"] = float(d[-1, :, 0].mean() - d[0, :, 0].mean())
    return f


def build_features(source: str = "bdb2026", features_dir: Path | None = None) -> dict:
    fd = Path(features_dir or config.FEATURES / source)
    with np.load(fd / "tensors_v2.npz") as npz:      # materialize once: indexing an NpzFile re-reads the whole array
        z = {k: npz[k] for k in npz.files}
    index = pl.read_parquet(fd / "tensors_v2_index.parquet")
    rep = {"schema": SCHEMA, "built_at": now_utc(), "horizons": {}}
    for name, frame in HORIZONS.items():
        rows = []
        for i in range(len(z["nframes"])):
            if z["nframes"][i] <= frame or z["dmask"][i].sum() < 4 or z["rmask"][i].sum() < 3:
                continue
            d = z["defs"][i][: frame + 1][:, z["dmask"][i]].astype(np.float64)
            r = z["recs"][i][: frame + 1][:, z["rmask"][i]].astype(np.float64)
            rows.append({"gameId": int(z["gameId"][i]), "playId": int(z["playId"][i])} | play_relational(d, r, frame))
        t = pl.DataFrame(rows).join(index, on=KEY, how="left")
        t.write_parquet(fd / f"relational_{name}.parquet")
        cols = [c for c in t.columns if c not in index.columns]
        rep["horizons"][name] = {"latest_frame": frame, "plays": t.height, "features": cols}
    write_json(fd / "relational_report.json", rep)
    return rep
