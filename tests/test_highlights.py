"""Highlight pipeline mechanics. Tests that need the downloaded dataset skip when it is absent."""
import json

import numpy as np
import pytest

from gridiron_lens.highlights import models as M
from gridiron_lens.highlights import pipeline as P
from gridiron_lens.shared import config

SPLIT = config.MANIFESTS / "svhighlights_split.json"


def test_budget_selection_never_sees_labels_and_respects_the_budget():
    rng = np.random.default_rng(0)
    score = rng.normal(size=4000)
    sel = M.select_budget(score, 90)
    assert 90 <= sel.sum() <= 94                       # whole 5-clip blocks, so at most one block over
    assert sel[int(np.argmax(np.convolve(score, np.ones(5) / 5, mode="same")))]


def test_highlight_runs_and_game_metrics():
    y = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0] * 40)
    assert M.runs_of_ones(y)[:2] == [(1, 3), (5, 6)]
    perfect = M.game_metrics(y, y.astype(float))
    assert perfect["average_precision"] == 1.0 and perfect["hit_at_1"]
    assert abs(perfect["chance_average_precision"] - 0.3) < 1e-9


def test_loudness_features_handle_silence():
    vol = np.array([-20.0, -21.0, -np.inf, np.nan, -19.0] * 60, np.float32)
    assert np.isfinite(M.loudness_features(vol)).all()


@pytest.mark.skipif(not SPLIT.exists(), reason="SVHighlights split manifest not created on this machine")
def test_split_is_by_game_and_disjoint():
    s = json.loads(SPLIT.read_text())
    tr, va, te = set(s["train"]), set(s["validation"]), set(s["test"])
    assert not tr & va and not tr & te and not va & te and len(tr | va | te) == 40
    assert all(set(f) <= tr for f in s["development_folds"])


@pytest.mark.skipif(not (config.MANIFESTS / "svhighlights_audit.json").exists(), reason="SVHighlights audit not run on this machine")
def test_audit_found_one_league_and_no_duplicate_sources():
    a = json.loads((config.MANIFESTS / "svhighlights_audit.json").read_text())
    assert a["leagues"] == ["NFL"] and a["duplicate_source_links"] == 0 and a["videos"] == 40
    assert set(P.MODALITIES) == set(a["feature_dims"])
