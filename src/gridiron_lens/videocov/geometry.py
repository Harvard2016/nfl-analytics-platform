"""Field calibration and trajectory mapping for one fixed camera view.

A single homography maps image points on the ground plane to field coordinates for ONE camera pose. A pan, zoom or cut
needs a new calibration; `gates` rejects a clip whose calibration is stale. Image points must be ground contact points
(feet). A helmet centre is above the ground plane and maps to the wrong place; no correction for that is implemented.

Field coordinates: x 0-120 yards along the length (end zones included), y 0-53.3 yards across.
"""
from __future__ import annotations

import numpy as np

MIN_FIT_POINTS, MIN_HOLDOUT_POINTS = 4, 2
MAX_HOLDOUT_ERROR_YD = 1.0
MIN_TRACK_COVERAGE = 0.8
MIN_DEFENDERS, MIN_ROUTE_RUNNERS = 3, 1


def _normalize(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c, s = p.mean(0), np.sqrt(2) / max(np.linalg.norm(p - p.mean(0), axis=1).mean(), 1e-9)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return (np.c_[p, np.ones(len(p))] @ T.T)[:, :2], T


def fit_homography(image_pts: np.ndarray, field_pts: np.ndarray) -> np.ndarray:
    """Direct linear transform with point normalization. Needs at least four non-collinear correspondences."""
    image_pts, field_pts = np.asarray(image_pts, float), np.asarray(field_pts, float)
    if len(image_pts) < MIN_FIT_POINTS or len(image_pts) != len(field_pts):
        raise ValueError(f"need at least {MIN_FIT_POINTS} matching image and field points")
    a, Ta = _normalize(image_pts)
    b, Tb = _normalize(field_pts)
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows += [[-x, -y, -1, 0, 0, 0, u * x, u * y, u], [0, 0, 0, -x, -y, -1, v * x, v * y, v]]
    _, sv, vt = np.linalg.svd(np.array(rows))
    if sv[-2] < 1e-8:
        raise ValueError("the landmarks are collinear or repeated: a homography cannot be fitted")
    H = np.linalg.inv(Tb) @ vt[-1].reshape(3, 3) @ Ta
    return H / H[2, 2]


def project(H: np.ndarray, image_pts: np.ndarray) -> np.ndarray:
    p = np.c_[np.asarray(image_pts, float), np.ones(len(image_pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def calibrate(landmarks: list[dict], holdout: list[dict]) -> dict:
    """Fit on `landmarks`, measure on `holdout` (landmarks the fit never saw). Each item: {"image": [px, py], "field": [x, y]}."""
    H = fit_homography([m["image"] for m in landmarks], [m["field"] for m in landmarks])
    fit_err = np.linalg.norm(project(H, [m["image"] for m in landmarks]) - np.array([m["field"] for m in landmarks]), axis=1)
    out = {"H": H, "fit_points": len(landmarks), "fit_rms_yards": float(np.sqrt((fit_err ** 2).mean())), "holdout_points": len(holdout), "holdout_rms_yards": None, "holdout_max_yards": None}
    if holdout:
        e = np.linalg.norm(project(H, [m["image"] for m in holdout]) - np.array([m["field"] for m in holdout]), axis=1)
        out |= {"holdout_rms_yards": float(np.sqrt((e ** 2).mean())), "holdout_max_yards": float(e.max())}
    return out


def map_tracks(H: np.ndarray, tracks: dict[str, np.ndarray], visible: dict[str, np.ndarray]) -> dict[str, dict]:
    """Reviewed image tracks (T x 2 foot points) -> field coordinates with a visibility mask. Hidden frames stay masked, never interpolated."""
    out = {}
    for pid, px in tracks.items():
        vis = np.asarray(visible[pid], bool)
        xy = np.full((len(px), 2), np.nan)
        if vis.any():
            xy[vis] = project(H, np.asarray(px, float)[vis])
        on_field = vis & (xy[:, 0] >= 0) & (xy[:, 0] <= 120) & (xy[:, 1] >= 0) & (xy[:, 1] <= 53.3)
        out[pid] = {"xy": xy, "visible": on_field, "off_field_frames": int((vis & ~on_field).sum())}
    return out


def causal_velocity(xy: np.ndarray, visible: np.ndarray, fps: float) -> np.ndarray:
    """Velocity at frame t from frames t and t-1 only (yards/second). Unknown where either frame is hidden. This is direction of travel, not body orientation."""
    v = np.full_like(xy, np.nan)
    ok = visible[1:] & visible[:-1]
    v[1:][ok] = (xy[1:][ok] - xy[:-1][ok]) * fps
    return v


def gates(cal: dict, mapped: dict[str, dict], sides: dict[str, str], snap_frame: int, fps: float, shot_cuts: list[int]) -> dict:
    """Quality gates before any experimental coverage estimate. Returns every failed gate with a reason; never fills in missing players."""
    need = round(1.5 * fps) + 1
    window = slice(snap_frame, snap_frame + need)
    reasons = []
    if cal["holdout_points"] < MIN_HOLDOUT_POINTS:
        reasons.append(f"calibration has {cal['holdout_points']} held-out landmark(s); at least {MIN_HOLDOUT_POINTS} are needed to check it")
    elif cal["holdout_rms_yards"] > MAX_HOLDOUT_ERROR_YD:
        reasons.append(f"calibration error on held-out landmarks is {cal['holdout_rms_yards']:.2f} yards (limit {MAX_HOLDOUT_ERROR_YD})")
    if any(snap_frame < c < snap_frame + need for c in shot_cuts):
        reasons.append("a camera cut falls between the snap and +1.5 s: one calibration cannot cover both shots")
    cover = {pid: float(m["visible"][window].mean()) if len(m["visible"][window]) == need else 0.0 for pid, m in mapped.items()}
    kept = {s: [p for p, c in cover.items() if sides.get(p) == s and c >= MIN_TRACK_COVERAGE] for s in ("defense", "offense")}
    if len(kept["defense"]) < MIN_DEFENDERS:
        reasons.append(f"only {len(kept['defense'])} defender(s) are visible for at least {MIN_TRACK_COVERAGE:.0%} of the first 1.5 seconds; off-screen defenders are not invented")
    if len(kept["offense"]) < MIN_ROUTE_RUNNERS:
        reasons.append("no route runner is visible through the first 1.5 seconds")
    return {"passed": not reasons, "result": "inputs usable for an experimental estimate" if not reasons else "insufficient evidence", "reasons": reasons,
            "track_coverage": cover, "usable_players": kept, "frames_needed": need,
            "note": "Passing these gates does not validate coverage accuracy on video. No labelled video evaluation exists."}
