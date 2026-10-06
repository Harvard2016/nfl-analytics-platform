"""Calibration measured on real paired video and tracking: NFL Helmet Assignment (60 plays, sideline and end-zone views).

The release gives, per video frame, helmet boxes labelled with player ids, and 10 Hz tracking for all 22 players. That lets
the field-calibration code be tested on real footage without building a detector:

1. For a video frame, match each labelled helmet to that player's tracked field position at the same moment.
2. Fit a homography on some players, measure the error on the players left out (yards).
3. Fit at the snap and reuse it later in the play, to measure how fast one calibration goes stale as the camera moves.

Image points are helmet centres, not feet. Helmets sit roughly on one plane above the field, so a homography still fits, but
crouched or falling players break that. This measures calibration geometry only. The release has no coverage labels and only
2 of its 60 plays are passes, so nothing here evaluates coverage, and the helmet ids come from labels, not from a detector.

Timing: the release documents that the snap is at video frame 10 and video runs at 59.94 frames per second. The offset sweep
below checks that against the data instead of assuming it.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import polars as pl

from ..shared import config
from ..shared.provenance import now_utc, sha256_file, write_json
from ..shared.runs import Run
from . import geometry as G

RAW = config.RAW / "helmet_assignment"
OUT = config.REPORTS / "v3"
FPS, SNAP_FRAME, SEED = 59.94, 10, 20261006
OFFSETS = (-12, -6, -3, 0, 3, 6, 12)
STALE_S = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


def load() -> tuple[dict, dict]:
    lab = pl.read_csv(RAW / "train_labels.csv").filter(~pl.col("isSidelinePlayer")).with_columns(
        (pl.col("left") + pl.col("width") / 2).alias("u"), (pl.col("top") + pl.col("height") / 2).alias("v"))
    trk = pl.read_csv(RAW / "train_player_tracking.csv")
    frames: dict = {}
    for (video, frame), g in lab.group_by(["video", "frame"]):
        frames[(video, int(frame))] = {r["label"]: (r["u"], r["v"]) for r in g.iter_rows(named=True)}
    tracks: dict = {}
    for (gk, pid), g in trk.group_by(["gameKey", "playID"]):
        t = np.array([datetime.fromisoformat(s).timestamp() for s in g["time"].to_list()])
        snap = t[np.array([e == "ball_snap" for e in g["event"].to_list()])].min()
        per = {}
        for p, x, y, tt in zip(g["player"].to_list(), g["x"].to_list(), g["y"].to_list(), t):
            per.setdefault(p, []).append((tt - snap, x, y))
        tracks[(int(gk), int(pid))] = {p: np.array(sorted(v)) for p, v in per.items()}
    return frames, tracks


def field_at(track: np.ndarray, t: float) -> np.ndarray | None:
    """Tracked position at time t (seconds from the snap), linearly interpolated between 10 Hz samples."""
    if t < track[0, 0] or t > track[-1, 0]:
        return None
    return np.array([np.interp(t, track[:, 0], track[:, 1]), np.interp(t, track[:, 0], track[:, 2])])


def pairs(frames: dict, tracks: dict, video: str, frame: int, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    gk, pid = int(video.split("_")[0]), int(video.split("_")[1])
    t = (frame + offset - SNAP_FRAME) / FPS
    img, fld = [], []
    for p, uv in frames.get((video, frame), {}).items():
        tr = tracks[(gk, pid)].get(p)
        xy = None if tr is None else field_at(tr, t)
        if xy is not None:
            img.append(uv), fld.append(xy)
    return np.array(img, float), np.array(fld, float)


def holdout_error(img: np.ndarray, fld: np.ndarray, rng: np.random.Generator, folds: int = 4) -> np.ndarray | None:
    """Per-player error in yards when that player is left out of the fit."""
    n = len(img)
    if n < 10:
        return None
    order, err = rng.permutation(n), np.full(n, np.nan)
    for k in range(folds):
        te = order[k::folds]
        tr = np.setdiff1d(order, te)
        try:
            H = G.fit_homography(img[tr], fld[tr])
        except (ValueError, np.linalg.LinAlgError):
            return None
        err[te] = np.linalg.norm(G.project(H, img[te]) - fld[te], axis=1)
    return err


def run(log=print) -> dict:
    frames, tracks = load()
    rng = np.random.default_rng(SEED)
    videos = sorted({v for v, _ in frames})
    per_view: dict = {"Sideline": [], "Endzone": []}
    frame_rms: dict = {"Sideline": [], "Endzone": []}
    sweep = {o: [] for o in OFFSETS}
    stale: dict = {v: {s: [] for s in STALE_S} for v in per_view}
    used = 0
    for video in videos:
        view = "Sideline" if "Sideline" in video else "Endzone"
        last = max(f for v, f in frames if v == video)
        for frame in range(SNAP_FRAME, min(last, SNAP_FRAME + int(3 * FPS)) + 1, 6):            # every tenth of a second for 3 s after the snap
            img, fld = pairs(frames, tracks, video, frame)
            e = holdout_error(img, fld, rng)
            if e is None:
                continue
            used += 1
            per_view[view].append(e)
            frame_rms[view].append(float(np.sqrt(np.nanmean(e ** 2))))
            if frame % 30 == SNAP_FRAME % 30:                                                  # offset sweep on a subsample
                for o in OFFSETS:
                    io, fo = pairs(frames, tracks, video, frame, o)
                    eo = holdout_error(io, fo, np.random.default_rng(SEED))
                    if eo is not None:
                        sweep[o].append(float(np.sqrt(np.nanmean(eo ** 2))))
        img0, fld0 = pairs(frames, tracks, video, SNAP_FRAME)                                  # one calibration at the snap, reused later
        if len(img0) >= 10:
            try:
                H0 = G.fit_homography(img0, fld0)
            except (ValueError, np.linalg.LinAlgError):
                continue
            for s in STALE_S:
                f = SNAP_FRAME + round(s * FPS)
                im, fl = pairs(frames, tracks, video, f)
                if len(im) >= 10:
                    stale[view][s].append(float(np.sqrt(np.mean(np.linalg.norm(G.project(H0, im) - fl, axis=1) ** 2))))

    def dist(a: list[np.ndarray]) -> dict:
        x = np.concatenate(a)
        x = x[np.isfinite(x)]
        return {"player_errors": len(x), "median_yards": float(np.median(x)), "mean_yards": float(x.mean()), "p90_yards": float(np.quantile(x, 0.9)), "share_within_1_yard": float((x <= 1.0).mean()),
                "share_within_2_yards": float((x <= 2.0).mean())}

    rep = {
        "module": "video-coverage (research)", "version": "videocov-helmet-eval-v1", "created_at_utc": now_utc(),
        "source": "NFL Health & Safety - Helmet Assignment (Kaggle), training split: 60 plays x 2 views, labelled helmet boxes and 10 Hz tracking",
        "files": {f: sha256_file(RAW / f) for f in ("train_labels.csv", "train_player_tracking.csv")},
        "what_is_measured": "Homography from helmet centres to tracked field positions, fitted on three quarters of the visible players in a frame and scored on the rest. Frames every 0.1 s for 3 s after the snap.",
        "what_is_not_measured": ["coverage accuracy (the release has no coverage labels; 2 of 60 plays are passes)", "player detection or identity (helmet ids come from the released labels)",
                                 "foot-point localisation (helmet centres are used)", "broadcast footage (these are fixed all-22 style sideline and end-zone cameras)"],
        "frames_used": used, "videos": len(videos),
        "held_out_player_error": {v: dist(per_view[v]) for v in per_view},
        "frame_gate": {v: {"frames": len(frame_rms[v]), "median_frame_rms_yards": float(np.median(frame_rms[v])), "share_of_frames_passing_1_yard_gate": float((np.array(frame_rms[v]) <= G.MAX_HOLDOUT_ERROR_YD).mean()),
                           "share_of_frames_under_2_yards": float((np.array(frame_rms[v]) <= 2.0).mean())} for v in frame_rms},
        "timing_check": {"note": "Held-out frame RMS when the video is shifted against the tracking by this many frames. The documented alignment (0) should be at or near the minimum.",
                         "median_frame_rms_yards_by_offset": {str(o): float(np.median(v)) for o, v in sweep.items() if v}},
        "stale_calibration": {v: {f"{s:.2f}s after the snap": {"plays": len(x), "median_rms_yards": float(np.median(x)), "p90_rms_yards": float(np.quantile(x, 0.9))} for s, x in stale[v].items() if x} for v in stale},
        "stale_note": "One homography fitted at the snap and reused later in the same play. Error growth is camera motion (pan, zoom) plus players leaving the helmet plane.",
    }
    write_json(OUT / "videocov_helmet_eval.json", rep)
    r = Run("coverage", "videocov-helmet-eval", "v3", seed=SEED)
    r.set(target="field position of held-out players (yards)", population="60 plays x 2 views, frames 0-3 s after the snap", metrics={"held_out": rep["held_out_player_error"], "gate": rep["frame_gate"]})
    r.output("report", OUT / "videocov_helmet_eval.json")
    r.finish()
    log(json.dumps({k: rep[k] for k in ("frames_used", "held_out_player_error", "frame_gate", "timing_check", "stale_calibration")}, indent=1))
    return rep


if __name__ == "__main__":
    run()
