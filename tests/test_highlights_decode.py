"""Strict reel decoder: the final output never exceeds the budget, whatever the scores, padding or media length."""
import numpy as np
import pytest

from gridiron_lens.highlights import decode as D
from gridiron_lens.highlights import models as M


def _check(score, budget, **kw):
    segs = D.decode(np.asarray(score, float), budget, **kw)
    dur = kw.get("duration_s", len(score) * D.CLIP_S)
    assert D.output_seconds(segs) <= budget + 1e-6
    assert all(0 <= s.start_s < s.end_s <= dur + 1e-9 for s in segs)
    m = D.merge([(s.start_s, s.end_s) for s in segs])
    assert all(a2 > b1 for (_, b1), (a2, _) in zip(m, m[1:]))                 # no second appears twice
    return segs


@pytest.mark.parametrize("pad", [(0.0, 0.0), (2.0, 2.0), (6.0, 4.0)])
@pytest.mark.parametrize("budget", [1, 5, 60, 180, 300])
def test_never_exceeds_budget(pad, budget):
    rng = np.random.default_rng(budget)
    _check(rng.normal(size=4000), budget, lead_s=pad[0], tail_s=pad[1])


def test_known_overshoot_case_is_fixed():
    rng = np.random.default_rng(40)
    s = rng.normal(size=4500)
    assert M.select_budget(s, 90).sum() * 2 >= 180                           # v1 may run over; preserved as is
    assert D.output_seconds(_check(s, 180)) <= 180


def test_short_video_edges_silence_and_ties():
    assert D.decode(np.array([]), 60) == []
    assert _check([1.0, 2.0, 3.0], 60) and D.output_seconds(_check([1.0, 2.0, 3.0], 60)) <= 6          # shorter than one block
    segs = _check(np.r_[9.0, np.zeros(200), 9.0], 30, lead_s=4, tail_s=4)                              # peaks at both ends: padding is clamped
    assert segs[0].start_s == 0.0
    assert _check(np.full(500, -np.inf), 60) == []                                                     # silence only: nothing finite to rank
    assert len(_check(np.r_[np.full(50, np.nan), np.ones(50)], 20)) >= 1
    a, b = _check(np.ones(300), 40), _check(np.ones(300), 40)
    assert a == b and a[0].start_s == 0.0                                                              # equal scores: deterministic, earliest first
    assert _check(np.ones(100), 60, duration_s=47.3)[-1].end_s <= 47.3                                 # end of file


def test_overlapping_padding_is_not_double_counted():
    s = np.zeros(400)
    s[100:105], s[106:111] = 5, 4                                                                       # two adjacent blocks whose padding overlaps
    segs = _check(s, 60, lead_s=4, tail_s=4)
    assert D.output_seconds(segs[:2]) < sum(x.end_s - x.start_s for x in segs[:2])


def test_metrics_report_strict_duration_and_oracle():
    y = np.zeros(1000, np.int8)
    y[100:140] = 1
    s = y + np.random.default_rng(0).normal(scale=0.01, size=1000)
    m = D.metrics(y, D.decode(s, 60), 60)
    assert m["within_budget"] and m["oracle_recall_upper_bound"] == 60 / 80 and 0 < m["recall_of_labelled_time"] <= m["oracle_recall_upper_bound"] + 1e-9
