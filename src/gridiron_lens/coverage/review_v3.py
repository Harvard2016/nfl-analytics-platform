"""Development error table (out-of-fold) and the review export for the existing 100-play queue.

Error slices use the E0 control's out-of-fold predictions on weeks 7-12 (each play scored by a model that did not train on it).
Outcome-like fields (label, family, whether the throw came early) are used only to slice errors, never as inputs.
The review export gives a person what they need to judge a disagreement. Geometry flags are automated suggestions of what to
look at; they are not annotations, and no released label is changed.
"""
from __future__ import annotations

import json

import numpy as np

from ..shared import config
from ..shared.provenance import now_utc, write_json
from . import neural, relational

OUT = config.REPORTS / "v3"
WEB = config.WEB_DEMO / "real" / "coverage"
HZ = relational.HORIZONS
CTRL = "E0 control (gru+reflect)|seed42|clean"
VERDICTS = ["plausible prediction", "probable label ambiguity", "insufficient evidence", "clear model error"]


def geometry(A: dict, i: int) -> dict:
    """Plain descriptors of one play from its tensors (frames the play actually has)."""
    n = int(A["nframes"][i])
    d, r = A["defs"][i][:n][:, A["dmask"][i]], A["recs"][i][:n][:, A["rmask"][i]]
    last = n - 1
    order0, order1 = np.argsort(r[0, :, 1]), np.argsort(r[last, :, 1])
    rr0 = np.linalg.norm(r[0][:, None, :2] - r[0][None, :, :2], axis=-1) + np.eye(r.shape[1]) * 99 if r.shape[1] > 1 else np.array([[99.0]])
    sep0, sep1 = np.linalg.norm(d[0][:, None, :2] - r[0][None, :, :2], axis=-1), np.linalg.norm(d[last][:, None, :2] - r[last][None, :, :2], axis=-1)
    near0, near1 = sep0.argmin(0), sep1.argmin(0)
    return {"frames": n, "defenders": int(d.shape[1]), "route_runners": int(r.shape[1]), "thrown_before_1_5s": bool(n < 16),
            "receiver_in_motion_at_snap": bool((np.hypot(r[0, :, 2], r[0, :, 3]) > 2.0).any()), "receivers_cross": bool((order0 != order1).any()),
            "bunch_at_snap": bool(rr0.min() < 2.5), "nearest_defender_switched": bool((near0 != near1).any()),
            "mean_defender_depth_at_snap": round(float(d[0, :, 0].mean()), 2), "min_separation_last_frame": round(float(sep1.min()), 2)}


def suggest(g: dict, p: float, label: str) -> list[str]:
    out = []
    if g["thrown_before_1_5s"]:
        out.append("quick throw: little movement to judge")
    if g["defenders"] <= 4:
        out.append("few tracked defenders: the release selected a small group")
    if g["receivers_cross"] or g["nearest_defender_switched"]:
        out.append("crossing routes or a defender switch: man and zone can look alike")
    if g["bunch_at_snap"]:
        out.append("bunched receivers at the snap")
    if g["receiver_in_motion_at_snap"]:
        out.append("receiver moving at the snap")
    if abs(p - 0.5) < 0.15:
        out.append("model near 50/50")
    return out or ["no automated flag: compare the movement with the released label"]


def slice_table(rows: list[dict], key, min_n: int = 30) -> list[dict]:
    groups: dict = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out = []
    for k, g in groups.items():
        if len(g) < min_n:
            continue
        ll = float(np.mean([-(np.log(max(r["p"], 1e-6)) if r["y"] else np.log(max(1 - r["p"], 1e-6))) for r in g]))
        man = [r for r in g if r["y"] == 1]
        out.append({"slice": str(k), "plays": len(g), "error_rate": float(np.mean([r["wrong"] for r in g])), "log_loss": ll, "man_share": float(np.mean([r["y"] for r in g])),
                    "man_recall": float(np.mean([not r["wrong"] for r in man])) if man else None})
    return sorted(out, key=lambda x: -x["error_rate"])


def run(log=print) -> dict:
    A = neural.load_arrays()
    oof = np.load(OUT / "coverage_v3_oof.npz")
    ix, lg = oof["index"], oof[CTRL]
    rows = []
    for j, i in enumerate(ix):
        g = geometry(A, int(i))
        per = {h: (float(1 / (1 + np.exp(-lg[j, f]))) if g["frames"] > f else None) for h, f in HZ.items()}
        last = [h for h in HZ if per[h] is not None][-1]
        rows.append({"key": str(A["key"][i]), "week": int(A["week"][i]), "team": str(A["team"][i]), "family": str(A["family"][i]) or "unknown", "y": int(A["y"][i]), "p": per[last], "at": last,
                     "wrong": bool((per[last] >= 0.5) != bool(A["y"][i])), "per_horizon": per, **g})
    full = [r for r in rows if not r["thrown_before_1_5s"]]
    by_h = {}
    for h in HZ:
        ok = [r for r in rows if r["per_horizon"][h] is not None]
        by_h[h] = {"plays": len(ok), "error_rate": float(np.mean([(r["per_horizon"][h] >= 0.5) != bool(r["y"]) for r in ok]))}
    rep = {
        "module": "coverage", "version": "coverage-error-slices-v3", "created_at_utc": now_utc(),
        "predictions": "E0 control (v2 champion configuration, seed 42), out-of-fold on weeks 7-12, uncalibrated, lean at 0.5",
        "split_role": "development (weeks 1-12 folds). Not the benchmark weeks.", "plays": len(rows), "overall_by_cutoff": by_h,
        "note": "Slices with fewer than 30 plays are omitted. Slice fields describe the play or its outcome; none of them is a model input. Differences between slices are associations, not causes.",
        "slices_at_last_available_cutoff": {
            "released class": slice_table(rows, lambda r: "Man" if r["y"] else "Zone"), "released family": slice_table(rows, lambda r: r["family"]),
            "defensive team": slice_table(rows, lambda r: r["team"]), "week": slice_table(rows, lambda r: r["week"]),
            "thrown before +1.5 s": slice_table(rows, lambda r: r["thrown_before_1_5s"]), "tracked defenders": slice_table(rows, lambda r: r["defenders"]),
            "route runners": slice_table(rows, lambda r: r["route_runners"]), "receiver in motion at snap": slice_table(rows, lambda r: r["receiver_in_motion_at_snap"]),
            "receivers cross": slice_table(rows, lambda r: r["receivers_cross"]), "nearest defender switched": slice_table(rows, lambda r: r["nearest_defender_switched"]),
            "bunch at snap": slice_table(rows, lambda r: r["bunch_at_snap"]),
            "mean defender depth at snap": slice_table(rows, lambda r: "under 5 yd" if r["mean_defender_depth_at_snap"] < 5 else "5-8 yd" if r["mean_defender_depth_at_snap"] < 8 else "8 yd or more")},
        "full_cohort_plays_reaching_1_5s": len(full),
    }
    write_json(OUT / "coverage_error_slices.json", rep)
    write_json(WEB / "error_slices_v3.json", rep)

    # review export for the existing queue (weeks 13-14, v2 temporal seed 42 calibrated probabilities)
    q = json.loads((config.REPORTS / "v2" / "coverage_review_queue.json").read_text())
    saved = np.load(config.REPORTS / "v2" / "coverage_expB_predictions.npz")
    pos = {k: i for i, k in enumerate(A["key"])}
    plays = []
    for qname, items in q["queues"].items():
        for it in items:
            i = pos[it["key"]]
            g = geometry(A, i)
            n = g["frames"]
            ents = []
            for side, arr, mask in (("defense", A["defs"][i], A["dmask"][i]), ("offense", A["recs"][i], A["rmask"][i])):
                for s in np.where(mask)[0]:
                    ents.append({"id": f"{side[0]}{s}", "side": side, "name": None, "position": None, "jersey": None, "passer": False,
                                 "x": [round(float(v) + 50.0, 2) for v in arr[:n, s, 0]], "y": [round(float(v) + 26.65, 2) for v in arr[:n, s, 1]], "s": [None] * n, "o": [None] * n})
            per = {h: (None if np.isnan(saved["seed42"][i, j]) else round(float(saved["seed42"][i, j]), 4)) for j, h in enumerate(HZ)}
            plays.append({"key": it["key"], "queue": qname, "week": it["week"], "defense": it["defense"], "released_label": it["released_label"], "released_coverage_type": it.get("released_coverage_type"),
                          "p_man": per, "lean_at_last_cutoff": it["lean"], "geometry": g, "suggested_look": suggest(g, it["p_man"], it["released_label"]),
                          "frames": list(range(n)), "los_x": 50.0, "entities": ents})
    out = {"version": "coverage-review-v3", "created_at_utc": now_utc(), "purpose": q["purpose"], "model": q["model"], "status": "unreviewed: no person has annotated this queue",
           "verdicts": VERDICTS, "suggestion_note": "Suggested things to look at come from simple geometry. They are not annotations and never change a released label.",
           "display_note": "Positions are relative to the line of scrimmage (drawn at the 50) and the formation centre; players are anonymous.",
           "queues": {k: len(v) for k, v in q["queues"].items()}, "plays": plays}
    WEB.mkdir(parents=True, exist_ok=True)
    (WEB / "review_v3.json").write_text(json.dumps(out, separators=(",", ":")))
    log(json.dumps({"dev plays": len(rows), "by_cutoff": by_h, "class": rep["slices_at_last_available_cutoff"]["released class"],
                    "worst_family": rep["slices_at_last_available_cutoff"]["released family"][:3], "review_plays": len(plays)}, indent=1)[:1800])
    return rep


if __name__ == "__main__":
    run()
