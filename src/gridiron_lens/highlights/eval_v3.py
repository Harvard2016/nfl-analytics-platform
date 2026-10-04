"""Highlights evaluation v3: the strict-budget decoder against the preserved v1 selector, on the same saved scores.

No model is retrained here. Decoder settings are chosen on the 6 validation games (which were also used for H3 early
stopping, so they are development data). The 6 test games were examined in v2: figures on them are a comparison on a
previously examined benchmark. Labels are the released editorial labels, untouched.

Registered objective for decoder settings: mean recall of labelled time at a strict 180-second output; ties go to precision.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
from sklearn.metrics import average_precision_score

from ..shared import config
from ..shared.provenance import now_utc, write_json
from . import decode as D
from . import models as M
from . import pipeline as P

OUT = config.REPORTS / "v3"
BUDGETS = {"1 minute": 60, "3 minutes": 180, "5 minutes": 300}
GRID = {"block": [3, 5, 7], "pad": [(0.0, 0.0), (2.0, 2.0), (4.0, 2.0)]}
SEED = 20261004


def _mean(rows: list[dict], k: str) -> float:
    return float(np.mean([r[k] for r in rows]))


def boot(vals: list[float], n: int = 4000) -> list[float]:
    """Percentile interval from resampling games. With 6 games this is wide and only indicative."""
    rng, a = np.random.default_rng(SEED), np.asarray(vals)
    m = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def run(log=print) -> dict:
    split, labels = P.make_split(), P.load_labels()
    z = np.load(config.REPORTS / "v2" / "highlights_scores.npz")
    score = {name: {v: z[f"{name}|{v}"] for v in P.video_ids()} for name in ("H3 temporal fusion", "H0 loudness", "H1 commentary")}
    h3, va, te = score["H3 temporal fusion"], split["validation"], split["test"]

    grid = []
    for block, (lead, tail) in itertools.product(GRID["block"], GRID["pad"]):
        rows = [D.metrics(labels[v], D.decode(h3[v], 180, block=block, lead_s=lead, tail_s=tail), 180) for v in va]
        grid.append({"block_clips": block, "lead_s": lead, "tail_s": tail, "recall_of_labelled_time": _mean(rows, "recall_of_labelled_time"),
                     "precision": _mean(rows, "precision"), "segments_half_covered_share": float(sum(r["segments_half_covered"] for r in rows) / sum(r["labelled_segments"] for r in rows)),
                     "mean_segments_in_reel": _mean(rows, "segments")})
    best = max(grid, key=lambda g: (round(g["recall_of_labelled_time"], 6), g["precision"]))
    cfg = {"block": best["block_clips"], "lead_s": best["lead_s"], "tail_s": best["tail_s"]}
    padded = {"block": best["block_clips"], "lead_s": 4.0, "tail_s": 2.0}

    def block(games: list[int], name: str, kw: dict) -> dict:
        out = {}
        for bname, b in BUDGETS.items():
            rows = {v: D.metrics(labels[v], D.decode(score[name][v], b, **kw), b) for v in games}
            r = list(rows.values())
            out[bname] = {k: _mean(r, k) for k in ("output_s", "precision", "recall_of_labelled_time", "temporal_iou", "oracle_recall_upper_bound")} | {
                "all_within_budget": all(x["within_budget"] for x in r), "max_output_s": max(x["output_s"] for x in r),
                "segments_touched_share": float(sum(x["segments_touched"] for x in r) / sum(x["labelled_segments"] for x in r)),
                "segments_half_covered_share": float(sum(x["segments_half_covered"] for x in r) / sum(x["labelled_segments"] for x in r)),
                "recall_interval_95_over_games": boot([x["recall_of_labelled_time"] for x in r]), "precision_interval_95_over_games": boot([x["precision"] for x in r]),
                "per_game": {str(v): rows[v] for v in games}}
        return out

    def v1_block(games: list[int], name: str) -> dict:
        out = {}
        for bname, k in M.BUDGETS.items():
            rows = []
            for v in games:
                sel, y = M.select_budget(score[name][v], k), labels[v]
                inter = int((sel & (y == 1)).sum())
                rows.append({"output_s": float(sel.sum() * 2), "precision": inter / max(1, int(sel.sum())), "recall_of_labelled_time": inter / max(1, int(y.sum()))})
            out[bname] = {k2: _mean(rows, k2) for k2 in ("output_s", "precision", "recall_of_labelled_time")} | {"max_output_s": max(r["output_s"] for r in rows),
                                                                                                                   "all_within_budget": all(r["output_s"] <= BUDGETS[bname] for r in rows)}
        return out

    curve = [{"budget_s": b, "recall_of_labelled_time": _mean([D.metrics(labels[v], D.decode(h3[v], b, **cfg), b) for v in te], "recall_of_labelled_time"),
              "precision": _mean([D.metrics(labels[v], D.decode(h3[v], b, **cfg), b) for v in te], "precision"),
              "oracle_recall_upper_bound": float(np.mean([min(b, labels[v].sum() * 2) / (labels[v].sum() * 2) for v in te]))} for b in (30, 60, 120, 180, 300, 450, 600, 900)]
    disc = {}
    for sh_rows in zip(*[D.discovery_curve(labels[v], h3[v]) for v in te]):
        disc[str(sh_rows[0]["share_of_broadcast"])] = {k: _mean(list(sh_rows), k) for k in ("seconds", "precision", "recall_of_labelled_time", "segments_touched_share")}
    lab_s = {str(v): int(labels[v].sum() * 2) for v in te}
    rep = {
        "module": "highlights", "version": "highlights-eval-v3", "created_at_utc": now_utc(), "scores": "saved v2 scores (no retraining)",
        "split_roles": {"decoder settings": "6 validation games (development)", "reported": "6 test games, previously examined in v2: a benchmark comparison, not a fresh test"},
        "labels": "released editorial labels, unchanged. Contiguous label runs are alignment-derived segments, not validated counts of distinct plays.",
        "objective": "mean recall of labelled time at a strict 180-second output; ties go to precision", "grid_on_validation": grid, "chosen": cfg, "padded_variant": padded,
        "decoder": {"budget": "final output seconds after padding and merging, clamped to the media duration", "ties": "earlier clip first", "export": "see local_media: cuts are re-encoded for exact durations and verified with ffprobe"},
        "mean_average_precision": {name: {"test": float(np.mean([average_precision_score(labels[v], score[name][v]) for v in te])),
                                          "test_interval_95_over_games": boot([average_precision_score(labels[v], score[name][v]) for v in te]),
                                          "validation": float(np.mean([average_precision_score(labels[v], score[name][v]) for v in va]))} for name in score},
        "oracle": {"definition": "mean over games of min(budget, labelled seconds) / labelled seconds. A generous upper bound that ignores clip coherence. Diagnostic only; labels never drive selection.",
                   "labelled_seconds_test_games": lab_s, "three_minutes": float(np.mean([min(180, s) / s for s in lab_s.values()]))},
        "test": {"v1_selector": {n: v1_block(te, n) for n in ("H3 temporal fusion", "H0 loudness")},
                 "strict_decoder": {n: block(te, n, cfg) for n in score},
                 "strict_decoder_padded": {"H3 temporal fusion": block(te, "H3 temporal fusion", padded)}},
        "validation": {"strict_decoder": {"H3 temporal fusion": block(va, "H3 temporal fusion", cfg)}},
        "budget_curve_test_h3": curve, "discovery_mode_test_h3": disc,
        "discovery_note": "Discovery mode keeps the top share of clips by score with no reel budget: recall rises as precision falls. Segment share = labelled runs with at least one selected clip.",
    }
    write_json(OUT / "highlights_eval_v3.json", rep)
    t = rep["test"]
    log(json.dumps({"chosen": cfg, "v1 3min": {k: round(v, 4) for k, v in t["v1_selector"]["H3 temporal fusion"]["3 minutes"].items() if isinstance(v, float)},
                    "strict 3min": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t["strict_decoder"]["H3 temporal fusion"]["3 minutes"].items() if k != "per_game"},
                    "oracle 3min": rep["oracle"]["three_minutes"]}, indent=1))
    return rep


if __name__ == "__main__":
    run()
