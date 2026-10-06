"""Reel decoder v2: a strict budget in final output seconds, and the metrics that go with it.

The v1 selector (`models.select_budget`) stops after the block that crosses the budget, so a nominal 3-minute reel can run
188 seconds, and padded clips push it further. v1 and its report are preserved. This decoder:

- measures the budget on the final output: after lead-in/tail padding, after merging overlaps, clamped to the media duration;
- never exceeds it (the last segment is trimmed to fit, or dropped if what remains is shorter than `min_segment_s`);
- breaks score ties by time (earlier first), ignores non-finite scores, and never emits a second twice.

Labels are never read here. Oracle and recall figures in `metrics` are evaluation only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLIP_S = 2.0


@dataclass(frozen=True)
class Segment:
    start_s: float          # padded clip start
    end_s: float            # padded clip end
    moment_s: float         # centre of the highest-scoring clip in the block
    score: float            # block score used for ranking (not a probability)
    rank: int


def smooth(score: np.ndarray, block: int) -> np.ndarray:
    s = np.where(np.isfinite(score), score, -np.inf).astype(np.float64)
    if not np.isfinite(s).any():
        return s
    filled = np.where(np.isfinite(s), s, np.nanmin(np.where(np.isfinite(s), s, np.nan)))
    sm = np.convolve(filled, np.ones(block) / block, mode="same")
    return np.where(np.isfinite(s), sm, -np.inf)


def _union_seconds(segs: list[tuple[float, float]]) -> float:
    tot, end = 0.0, -np.inf
    for a, b in sorted(segs):
        if b > end:
            tot += b - max(a, end)
            end = b
    return tot


def merge(segs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(segs):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def decode(score: np.ndarray, budget_s: float, *, duration_s: float | None = None, block: int = 5, lead_s: float = 0.0, tail_s: float = 0.0,
           min_segment_s: float = 4.0, clip_s: float = CLIP_S) -> list[Segment]:
    """Highest-scoring non-overlapping blocks, padded, with total merged duration <= budget_s."""
    n = len(score)
    duration = float(duration_s if duration_s is not None else n * clip_s)
    if n == 0 or budget_s <= 0 or duration <= 0:
        return []
    block = max(1, min(block, n))
    sm = smooth(np.asarray(score, float), block)
    order = sorted((i for i in range(n) if np.isfinite(sm[i])), key=lambda i: (-sm[i], i))        # ties: earlier clip first
    taken = np.zeros(n, bool)
    chosen: list[Segment] = []
    spans: list[tuple[float, float]] = []
    for i in order:
        lo, hi = max(0, i - block // 2), min(n, i + block // 2 + 1)
        if taken[lo:hi].any():
            continue
        a, b = max(0.0, lo * clip_s - lead_s), min(duration, hi * clip_s + tail_s)
        if b <= a:
            continue
        extra = _union_seconds(spans + [(a, b)]) - _union_seconds(spans)
        left = budget_s - _union_seconds(spans)
        if extra > left + 1e-9:
            if left < min_segment_s:
                break
            b = a + left if not spans else min(b, a + left)                                           # trim the tail to fit exactly
            if _union_seconds(spans + [(a, b)]) - _union_seconds(spans) > left + 1e-9 or b - a < min_segment_s:
                continue                                                                              # overlaps earlier padding awkwardly: try the next block
        taken[lo:hi] = True
        peak = lo + int(np.argmax(np.where(np.isfinite(score[lo:hi]), score[lo:hi], -np.inf)))
        spans.append((a, b))
        chosen.append(Segment(a, b, (peak + 0.5) * clip_s, float(sm[i]), len(chosen) + 1))
        if budget_s - _union_seconds(spans) < 1e-9:
            break
    return chosen


def output_seconds(segs: list[Segment]) -> float:
    return _union_seconds([(s.start_s, s.end_s) for s in segs])


def second_mask(segs: list[Segment], duration_s: float) -> np.ndarray:
    m = np.zeros(int(np.ceil(duration_s)), bool)
    for a, b in merge([(s.start_s, s.end_s) for s in segs]):
        m[int(np.floor(a)):int(np.ceil(b))] = True
    return m


# ---------------------------------------------------------------- evaluation (labels are read only here)

def label_seconds(y: np.ndarray, clip_s: float = CLIP_S) -> np.ndarray:
    return np.repeat(y.astype(bool), int(clip_s))


def label_runs(y: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], y.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def metrics(y: np.ndarray, segs: list[Segment], budget_s: float, clip_s: float = CLIP_S) -> dict:
    lab = label_seconds(y, clip_s)
    sel = second_mask(segs, len(lab))[: len(lab)]
    inter, out_s, lab_s = int((sel & lab).sum()), float(output_seconds(segs)), int(lab.sum())
    runs = [(int(a * clip_s), int(b * clip_s)) for a, b in label_runs(y)]
    cover = [float(sel[a:b].mean()) for a, b in runs]
    return {"budget_s": budget_s, "output_s": round(out_s, 3), "within_budget": bool(out_s <= budget_s + 1e-6), "segments": len(merge([(s.start_s, s.end_s) for s in segs])),
            "precision": inter / max(1, int(sel.sum())), "recall_of_labelled_time": inter / max(1, lab_s), "temporal_iou": inter / max(1, int((sel | lab).sum())),
            "labelled_s": lab_s, "oracle_recall_upper_bound": min(budget_s, lab_s) / max(1, lab_s),
            "labelled_segments": len(runs), "segments_touched": int(sum(c > 0 for c in cover)), "segments_half_covered": int(sum(c >= 0.5 for c in cover))}


def discovery_curve(y: np.ndarray, score: np.ndarray, shares=(0.02, 0.05, 0.10, 0.15, 0.20, 0.30)) -> list[dict]:
    """Discovery mode: keep the top share of clips by score with no reel budget. Recall rises as precision falls."""
    s = np.where(np.isfinite(score), score, -np.inf)
    order = np.lexsort((np.arange(len(s)), -s))
    out = []
    for sh in shares:
        k = max(1, round(sh * len(s)))
        sel = np.zeros(len(s), bool)
        sel[order[:k]] = True
        out.append({"share_of_broadcast": sh, "seconds": k * CLIP_S, "precision": float((sel & (y == 1)).sum() / k), "recall_of_labelled_time": float((sel & (y == 1)).sum() / max(1, y.sum())),
                    "segments_touched_share": float(np.mean([sel[a:b].any() for a, b in label_runs(y)])) if y.sum() else 0.0})
    return out
