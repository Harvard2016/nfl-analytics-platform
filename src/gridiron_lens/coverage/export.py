"""Export compact per-play demo files for the website: positions, features, predictions, explanations.

Two play sets are exported, both from the locked test split and both from saved models (no retraining):
- sample: a seeded random sample chosen without looking at any prediction, label or confidence.
- errors: for each selectable model and window, a seeded random sample of plays whose underlying
  prediction disagrees with the released label. Chosen because they are mistakes; never a performance sample.

Synthetic exports are marked synthetic in every file.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from ..shared import config
from ..shared.ids import play_key
from ..shared.provenance import now_utc, write_json
from . import schema
from .features import DESCRIPTIONS, WINDOWS, feature_names

MODEL_ORDER = ["class_prior", "context_logit", "geometry_logit", "geometry_gbm"]
SELECTABLE = ["geometry_gbm", "geometry_logit"]  # models the explorer can show, default first
DEFAULT_MODEL, DEFAULT_WINDOW = "geometry_gbm", "post_1_5s"
FRAME_RANGE = (-30, 60)
EXPORT_SCHEMA = "coverage-demo-v2"
SAMPLE_SEED, ERROR_SEED = 11, 23
TOP_TERMS = 8  # largest terms kept per model and window; descriptions live once in index.json
EXPLANATIONS = {
    "geometry_logit": {
        "method": "linear_terms",
        "description": "Each measurement's term in the logistic model's log-odds: its coefficient times its standardized value. "
                       "Computed before calibration.",
    },
    "geometry_gbm": {
        "method": "replace_with_typical",
        "description": "How far the boosted-tree model's log-odds move when this one measurement is replaced by its median over "
                       "training plays and everything else is kept. Computed before calibration. Measurements that move together "
                       "can hide each other, and the changes do not add up to the prediction.",
    },
}


def _explain(model: str, bundle: dict, row: pl.DataFrame, medians: dict[str, float]) -> list[dict]:
    """Per-play explanation that belongs to this model. Describes the fitted model, not defensive intent."""
    cols = bundle["columns"]
    x = row.select(cols).to_pandas()
    raw = row.select(cols).row(0)
    if model == "geometry_logit":
        pipe = bundle["base"]
        z = pipe[:-1].transform(x)[0]
        effect = pipe[-1].coef_[0] * z
    else:
        base = float(bundle["base"].decision_function(x)[0])
        alt = pl.concat([row.select(cols).with_columns(pl.lit(medians[f]).alias(f)) for f in cols]).to_pandas()
        effect = base - bundle["base"].decision_function(alt)
    terms = [{"feature": f, "value": None if v is None or np.isnan(v) else round(float(v), 3),
              "log_odds_toward_man": round(float(e), 4)} for f, v, e in zip(cols, raw, effect)]
    return sorted(terms, key=lambda t: -abs(t["log_odds_toward_man"]))[:TOP_TERMS]


def run(source: str, n_plays: int = 120, processed_dir: Path | None = None, features_dir: Path | None = None,
        models_dir: Path | None = None, out_dir: Path | None = None, splits_path: Path | None = None,
        report_path: Path | None = None, n_errors: int = 30) -> dict:
    processed_dir = Path(processed_dir or config.PROCESSED / source)
    features_dir = Path(features_dir or config.FEATURES / source)
    models_dir = Path(models_dir or config.MODELS / f"coverage_{source}")
    synthetic = (features_dir / "SYNTHETIC").exists()
    out_dir = Path(out_dir or config.WEB_DEMO / ("synthetic" if synthetic else "real") / "coverage")
    if (out_dir / "plays").exists():
        for old in (out_dir / "plays").glob("*.json"):
            old.unlink()
    (out_dir / "plays").mkdir(parents=True, exist_ok=True)
    splits = json.loads(Path(splits_path or config.MANIFESTS / f"{source}_splits.json").read_text())
    test, train_keys = sorted(splits["splits"]["test"]["plays"]), splits["splits"]["train"]["plays"]
    key = (pl.col("gameId").cast(pl.String) + ":" + pl.col("playId").cast(pl.String)).alias("_k")
    full = {w: pl.read_parquet(features_dir / f"features_{w}.parquet").with_columns(key) for w in WINDOWS}
    tables = {w: t.filter(pl.col("_k").is_in(test)).sort(["gameId", "playId"]) for w, t in full.items()}
    bundles = {(w, m): joblib.load(models_dir / f"{w}__{m}.joblib") for w in WINDOWS for m in MODEL_ORDER}
    medians = {w: {f: float(full[w].filter(pl.col("_k").is_in(train_keys))[f].drop_nans().median()) for f in feature_names(w)} for w in WINDOWS}

    # 1) Main sample: drawn from the sorted list of eligible test play ids only. No model output is read before this line.
    rng = np.random.default_rng(SAMPLE_SEED)
    sample = sorted(rng.choice(test, size=min(n_plays, len(test)), replace=False).tolist())

    # 2) Error sets: per model and window, plays whose underlying prediction (p_man >= 0.5) differs from the released label.
    errors: dict[str, dict[str, list[str]]] = {m: {} for m in SELECTABLE}
    err_counts: dict[str, dict[str, int]] = {m: {} for m in SELECTABLE}
    erng = np.random.default_rng(ERROR_SEED)
    for m in SELECTABLE:
        for w in WINDOWS:
            b, t = bundles[(w, m)], tables[w]
            p = b["model"].predict_proba(t.select(b["columns"]).to_pandas())[:, 1]
            wrong = np.array(t["_k"].to_list())[(p >= 0.5) != (t[schema.TARGET] == "Man").to_numpy()]
            err_counts[m][w] = len(wrong)
            errors[m][w] = sorted(erng.choice(np.sort(wrong), size=min(n_errors, len(wrong)), replace=False).tolist())
    wanted = sorted(set(sample) | {k for m in errors.values() for ks in m.values() for k in ks})

    pp = pl.read_parquet(processed_dir / "players.parquet")
    names = {i: (n, pos) for i, n, pos in pp.select(["nflId", "displayName", "position"]).iter_rows()}
    first = next(iter(WINDOWS))
    picked = tables[first].filter(pl.col("_k").is_in(wanted))
    index, pid_of = [], {}
    for week in sorted(picked["week"].unique().to_list()):
        wk = picked.filter(pl.col("week") == week)
        trk = (pl.scan_parquet(processed_dir / f"tracking_week_{week}.parquet")
               .filter(pl.col("gameId").is_in(wk["gameId"].unique().to_list()) & pl.col("rel_frame").is_between(*FRAME_RANGE)).collect())
        for meta in wk.iter_rows(named=True):
            gid, pid, k = meta["gameId"], meta["playId"], meta["_k"]
            t = trk.filter((pl.col("gameId") == gid) & (pl.col("playId") == pid)).sort(["rel_frame"])
            frames = sorted(t["rel_frame"].unique().to_list())
            pos = {f: i for i, f in enumerate(frames)}
            ents = []
            for (side, nid), g in t.partition_by(["side", "nflId"], as_dict=True).items():
                # null where a player has no observation at a frame: the UI must not draw a guess
                arr = {c: [None] * len(frames) for c in ("x", "y", "s", "o")}
                for f, x, y, s, o in g.select(["rel_frame", "x", "y", "s", "o"]).iter_rows():
                    arr["x"][pos[f]], arr["y"][pos[f]] = round(x, 2), round(y, 2)
                    arr["s"][pos[f]] = None if s is None else round(s, 2)
                    arr["o"][pos[f]] = None if o is None else round(o, 1)
                name, role = names.get(nid, (None, None))
                ents.append({"id": None if nid is None else str(nid), "side": side, "name": name, "position": role,
                             "jersey": g["jerseyNumber"][0], **arr})
            windows, summary_pred = {}, {m: {} for m in MODEL_ORDER}
            for w, (cutoff, start) in WINDOWS.items():
                row = tables[w].filter(pl.col("_k") == k)
                preds = {}
                for m in MODEL_ORDER:
                    bd = bundles[(w, m)]
                    pm = float(bd["model"].predict_proba(row.select(bd["columns"]).to_pandas())[0, 1])
                    conf = max(pm, 1 - pm)
                    preds[m] = {"version": bd["version"], "p_man": round(pm, 4), "p_zone": round(1 - pm, 4),
                                "predicted": "Man" if pm >= 0.5 else "Zone", "confidence": round(conf, 4),
                                "confidence_threshold": bd["threshold"], "accepted": bool(conf >= bd["threshold"])}
                    summary_pred[m][w] = {"p_man": preds[m]["p_man"], "predicted": preds[m]["predicted"], "accepted": preds[m]["accepted"]}
                windows[w] = {"latest_frame": cutoff, "motion_from_frame": start, "predictions": preds,
                              "explanations": {m: _explain(m, bundles[(w, m)], row, medians[w]) for m in SELECTABLE}}
            pid_of[k] = play_key(source, gid, pid)
            summary = {
                "id": pid_of[k], "file": f"plays/{gid}_{pid}.json", "week": meta["week"], "season": meta["season"],
                "offense": meta["possessionTeam"], "defense": meta["defensiveTeam"], "quarter": meta["quarter"],
                "down": meta["down"], "yardsToGo": meta["yardsToGo"], "yardsToGoal": round(meta["yards_to_goal"], 1),
                "released": {"manZone": meta[schema.TARGET], "coverage": meta["coverage_type"], "source": schema.SOURCES[source]["label_source"]},
                "predictions": {m: summary_pred[m] for m in SELECTABLE}, "split": "test",
            }
            index.append(summary)
            write_json(out_dir / "plays" / f"{gid}_{pid}.json", {
                "schema": EXPORT_SCHEMA, "synthetic": synthetic, **summary,
                "description": meta.get("playDescription"), "los_x": float(t["los_x"][0]),
                "units": {"position": "yards, offense moving toward +x", "time": "frames at 10 per second, 0 = snap"},
                "frames": frames, "entities": ents, "windows": windows})
    manifest = {
        "schema": EXPORT_SCHEMA, "synthetic": synthetic, "generated_at": now_utc(), "source": source,
        "display_rights": "not applicable (synthetic)" if synthetic else "Owner approved display on 2026-10-04; the competition rules have not been independently reviewed.",
        "eligible_test_plays": len(test),
        "sample": {"ids": [pid_of[k] for k in sample], "seed": SAMPLE_SEED,
                   "note": f"A random sample of {len(sample)} of the {len(test)} eligible held-out test plays (seed {SAMPLE_SEED}). "
                           "Drawn from the list of play ids alone, before any prediction, label or confidence was read."},
        "errors": {"ids": {m: {w: [pid_of[k] for k in ks] for w, ks in v.items()} for m, v in errors.items()},
                   "available": err_counts, "seed": ERROR_SEED,
                   "note": f"Picked because they are mistakes. For the chosen model and window, up to {n_errors} plays drawn at random (seed {ERROR_SEED}) "
                           "from every held-out test play where the model's lean (man if its man probability is at least 50%) differs from the "
                           "released coverage label. This includes plays the model abstained on. It says nothing about how often the model is wrong."},
        "decision_rule": "Lean: man if calibrated p(man) >= 0.5, else zone. Confidence = max(p(man), 1 - p(man)). Accepted if confidence >= the saved cutoff; otherwise abstained.",
        "windows": {w: {"latest_frame": c, "motion_from_frame": s, "features": feature_names(w)} for w, (c, s) in WINDOWS.items()},
        "models": MODEL_ORDER, "selectable_models": SELECTABLE, "default_model": DEFAULT_MODEL, "default_window": DEFAULT_WINDOW,
        "cutoffs": {m: {w: bundles[(w, m)]["threshold"] for w in WINDOWS} for m in SELECTABLE},
        "versions": {m: {w: bundles[(w, m)]["version"] for w in WINDOWS} for m in SELECTABLE},
        "explanations": EXPLANATIONS, "feature_descriptions": DESCRIPTIONS, "plays": index,
    }
    write_json(out_dir / "index.json", manifest)
    report_path = Path(report_path or config.REPORTS / "experiments" / ("coverage_man_zone_SYNTHETIC.json" if synthetic else "coverage_man_zone.json"))
    if report_path.exists():
        (out_dir / "evaluation.json").write_text(report_path.read_text())
    for name in ("studies", "tendencies"):  # written by `studies`; copied when present
        src = report_path.parent / f"coverage_{name}{'_SYNTHETIC' if synthetic else ''}.json"
        if src.exists():
            (out_dir / f"{name}.json").write_text(src.read_text())
    return {"out_dir": config.rel(out_dir), "sample": len(sample), "error_plays": len(wanted) - len(sample), "files": len(index), "synthetic": synthetic}
