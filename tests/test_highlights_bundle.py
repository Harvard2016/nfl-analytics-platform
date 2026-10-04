"""Highlight inference bundle: parity with the cached model inputs and saved scores on real local data.

These tests need the local SVHighlights features and the built bundle; they skip in CI, where neither exists.
"""
import numpy as np
import pytest

from gridiron_lens.highlights import bundle as B
from gridiron_lens.highlights import models as M
from gridiron_lens.highlights import pipeline as P
from gridiron_lens.shared import config

needs_bundle = pytest.mark.skipif(not (B.DIR / "manifest.json").exists() or not (P.PROC / "american_football").exists(), reason="local bundle and features not present")


@needs_bundle
def test_bundle_reproduces_cached_features_and_saved_scores():
    model, tr, man = B.load()
    v = P.make_split()["test"][0]
    n = len(P.load_labels()[v])
    raw = {k: P.load_modality(v, k, n) for k in P.MODALITIES}
    for k in P.MODALITIES:
        assert np.abs(B.reduce(tr[k], raw[k]) - M.load_game(v)[k]).max() < B.PARITY_TOL
    saved = np.load(config.REPORTS / "v2" / "highlights_scores.npz")[f"H3 temporal fusion|{v}"]
    full = B.score_raw(model, tr, raw, P.load_volume()[v])
    assert np.abs(full - saved).max() < 1e-4
    assert np.abs(B.score_raw(model, tr, raw, P.load_volume()[v], chunk=512) - full).max() < 1e-4          # bounded chunks with overlap equal a single pass
    assert man["parity"]["scores_max_abs_diff_over_40_games"] < 1e-4


@needs_bundle
def test_masked_stream_is_explicit_and_wrong_extractor_is_rejected():
    model, tr, _ = B.load()
    v = P.make_split()["test"][0]
    n = len(P.load_labels()[v])
    raw = {k: P.load_modality(v, k, n) for k in P.MODALITIES}
    masked = B.score_raw(model, tr, {**raw, "vid_clip": None}, P.load_volume()[v])
    cached = M.score_game(M.TemporalFusion({k: (3 if k == "loudness" else 64) for k in model.names}), M.load_game(v), model.names)       # shape only
    assert masked.shape == cached.shape and np.isfinite(masked).all()
    assert np.abs(masked - B.score_raw(model, tr, raw, P.load_volume()[v])).max() > 1e-3                     # masking changes the score: never silently equal to full H3
    with pytest.raises(ValueError):
        B.reduce(tr["vid_clip"], np.zeros((4, 512), np.float32))                                             # a different CLIP width must not pass
