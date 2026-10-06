"""Calibration and gating mechanics on an invented camera. No footage is involved and nothing here measures accuracy on video."""
import numpy as np
import pytest

from gridiron_lens.videocov import geometry as G

H_TRUE = np.array([[0.09, 0.012, 20.0], [-0.004, 0.11, 5.0], [0.00002, 0.0004, 1.0]])


def _img(field_pts):
    return G.project(np.linalg.inv(H_TRUE), np.asarray(field_pts, float))


def _marks(pts, noise=0.0, seed=0):
    img = _img(pts) + np.random.default_rng(seed).normal(0, noise, (len(pts), 2))
    return [{"image": i.tolist(), "field": list(p)} for i, p in zip(img, pts)]


FIT = [(30, 0), (30, 53.3), (50, 0), (50, 53.3), (40, 18.4), (40, 34.9)]
HOLD = [(35, 18.4), (45, 34.9), (45, 0)]


def test_homography_recovers_held_out_landmarks():
    cal = G.calibrate(_marks(FIT), _marks(HOLD))
    assert cal["holdout_rms_yards"] < 1e-6 and cal["fit_points"] == 6


def test_collinear_or_too_few_landmarks_are_rejected():
    with pytest.raises(ValueError):
        G.fit_homography(_img([(30, 0), (40, 0), (50, 0), (60, 0)]), [(30, 0), (40, 0), (50, 0), (60, 0)])
    with pytest.raises(ValueError):
        G.fit_homography(_img(FIT[:3]), FIT[:3])


def test_velocity_is_causal_and_hidden_frames_stay_hidden():
    xy = np.c_[np.linspace(40, 44, 9), np.full(9, 20.0)]
    vis = np.ones(9, bool)
    vis[4] = False
    v = G.causal_velocity(xy, vis, fps=10)
    assert np.allclose(v[1], [5.0, 0.0]) and np.isnan(v[0]).all() and np.isnan(v[4]).all() and np.isnan(v[5]).all()
    xy2 = xy.copy()
    xy2[6:] += 50                                                           # rewriting later frames cannot change earlier velocities
    assert np.allclose(G.causal_velocity(xy2, vis, 10)[:6], v[:6], equal_nan=True)


def _clip(n_def=4, hide=None, frames=30):
    tracks, visible, sides = {}, {}, {}
    for i in range(n_def + 2):
        pid = f"p{i}"
        field = np.c_[np.linspace(40, 44, frames) + i, np.full(frames, 8.0 + 6 * i)]
        tracks[pid], visible[pid], sides[pid] = _img(field), np.ones(frames, bool), "defense" if i < n_def else "offense"
    if hide:
        visible[hide][3:14] = False
    return tracks, visible, sides


def test_gates_pass_only_with_checked_calibration_and_enough_visible_players():
    cal = G.calibrate(_marks(FIT), _marks(HOLD))
    tr, vis, sides = _clip()
    g = G.gates(cal, G.map_tracks(cal["H"], tr, vis), sides, snap_frame=2, fps=10, shot_cuts=[])
    assert g["passed"] and "does not validate" in g["note"]
    assert not G.gates(G.calibrate(_marks(FIT), []), G.map_tracks(cal["H"], tr, vis), sides, 2, 10, [])["passed"]            # nothing to check the calibration against
    bad = G.calibrate(_marks(FIT, noise=40.0), _marks(HOLD))
    assert "calibration error" in " ".join(G.gates(bad, G.map_tracks(bad["H"], tr, vis), sides, 2, 10, [])["reasons"])
    assert "camera cut" in " ".join(G.gates(cal, G.map_tracks(cal["H"], tr, vis), sides, 2, 10, shot_cuts=[9])["reasons"])
    tr2, vis2, sides2 = _clip(n_def=3, hide="p0")
    g2 = G.gates(cal, G.map_tracks(cal["H"], tr2, vis2), sides2, 2, 10, [])
    assert g2["result"] == "insufficient evidence" and "not invented" in " ".join(g2["reasons"])
