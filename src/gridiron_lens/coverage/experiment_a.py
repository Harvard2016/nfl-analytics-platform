"""Coverage experiment A: do relational defender-receiver features add to the v1 geometry features?

Protocol (all selection on development data):
- Chronological folds inside weeks 1-12: train 1-6 / validate 7-8, train 1-8 / validate 9-10, train 1-10 / validate 11-12.
  Whole games stay together because a game belongs to one week.
- Same boosted-tree settings as v1. Man class weights 1.0, 1.5, 2.0, 2.5 (zone 1.0).
- Every model is Platt-calibrated without class weights, so probabilities refer to the natural class mix.
  Inside a fold the calibrator is fitted on out-of-fold scores from the training weeks (4 game-grouped folds).
- Weeks 13-14 are used only afterwards, for the final calibrator and operating policies.
- Weeks 15-18 were examined during v1. Scoring a frozen v2 model there is a comparison on a previously
  examined benchmark, not a fresh test.

Cohort: the v1 split manifest (plays with features at every v1 window), so v1 and v2 are compared on the same plays.
"""
from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from ..shared import config
from ..shared.provenance import write_json
from ..shared.runs import Run
from . import evalkit, features, relational, schema

SOURCE = "bdb2026"
FOLDS = [([1, 2, 3, 4, 5, 6], [7, 8]), ([1, 2, 3, 4, 5, 6, 7, 8], [9, 10]), ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12])]
WEIGHTS = [1.0, 1.5, 2.0, 2.5]
HORIZONS = ["at_snap", "post_1s", "post_1_5s"]
SEED = 20261004
GBM = {"max_depth": 3, "learning_rate": 0.06, "max_iter": 250, "l2_regularization": 1.0, "random_state": SEED}   # unchanged from v1
KEY = ["gameId", "playId"]
K = (pl.col("gameId").cast(pl.String) + ":" + pl.col("playId").cast(pl.String)).alias("_k")
OUT = config.REPORTS / "v2"


def load(horizon: str) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Joined table for one horizon plus the geometry and relational column lists."""
    fd = config.FEATURES / SOURCE
    geo = pl.read_parquet(fd / f"features_{horizon}.parquet")
    rel = pl.read_parquet(fd / f"relational_{horizon}.parquet")
    gcols = features.feature_names(horizon)
    rcols = json.loads((fd / "relational_report.json").read_text())["horizons"][horizon]["features"]
    t = geo.join(rel.select([*KEY, *rcols]), on=KEY, how="inner").with_columns(K)
    return t, gcols, rcols


def fit_platt(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Platt scaling: one-dimensional logistic regression on the model's raw score, no class weights."""
    return LogisticRegression(C=1e6, max_iter=1000).fit(scores.reshape(-1, 1), y)


def fit_calibrated(xtr, ytr, groups, man_weight: float):
    """Weighted boosted trees plus an unweighted Platt calibrator fitted on out-of-fold training scores."""
    sw = np.where(ytr == 1, man_weight, 1.0)
    oof = np.zeros(len(ytr))
    for a, b in GroupKFold(4).split(xtr, ytr, groups):
        m = HistGradientBoostingClassifier(**GBM).fit(xtr[a], ytr[a], sample_weight=sw[a])
        oof[b] = m.decision_function(xtr[b])
    model = HistGradientBoostingClassifier(**GBM).fit(xtr, ytr, sample_weight=sw)
    return model, fit_platt(oof, ytr)


def predict(model, platt, x) -> np.ndarray:
    return platt.predict_proba(model.decision_function(x).reshape(-1, 1))[:, 1]


def cross_validate() -> dict:
    splits = json.loads((config.MANIFESTS / f"{SOURCE}_splits.json").read_text())
    train_keys = splits["splits"]["train"]["plays"]
    run = Run("coverage", "expA-cv", config_={"folds": FOLDS, "weights": WEIGHTS, "horizons": HORIZONS, "gbm": GBM}, seed=SEED)
    ckpt = OUT / "coverage_expA_cv_partial.json"          # finished configurations survive an interrupted run
    results = json.loads(ckpt.read_text()) if ckpt.exists() else []
    done = {(r["horizon"], r["features"], r["man_weight"]) for r in results}
    t0 = time.time()
    for h in HORIZONS:
        t, gcols, rcols = load(h)
        t = t.filter(pl.col("_k").is_in(train_keys)).sort(KEY)
        y = (t[schema.TARGET] == "Man").to_numpy().astype(int)
        week, game = t["week"].to_numpy(), t["gameId"].to_numpy()
        sets = {"geometry": gcols, "geometry+relational": gcols + rcols, "relational": rcols}
        for sname, cols in sets.items():
            x = t.select(cols).to_numpy()
            for w in (WEIGHTS if sname != "relational" else [1.0]):
                if (h, sname, w) in done:
                    continue
                folds = []
                for fi, (tr_w, va_w) in enumerate(FOLDS):
                    a, b = np.isin(week, tr_w), np.isin(week, va_w)
                    model, platt = fit_calibrated(x[a], y[a], game[a], w)
                    p = predict(model, platt, x[b])
                    folds.append({"fold": fi, "train_weeks": tr_w, "validate_weeks": va_w, "train_plays": int(a.sum()),
                                  **evalkit.binary_metrics(y[b], p)})
                keys = ("log_loss", "brier", "accuracy", "balanced_accuracy", "macro_f1", "man_recall", "man_precision", "zone_recall", "zone_precision")
                results.append({"horizon": h, "features": sname, "n_features": len(cols), "man_weight": w, "folds": folds,
                                "mean": {k: float(np.mean([f[k] for f in folds])) for k in keys},
                                "fold_sd": {k: float(np.std([f[k] for f in folds], ddof=1)) for k in keys}})
                write_json(ckpt, results)
                print(f"{h:10s} {sname:20s} w={w} log loss {results[-1]['mean']['log_loss']:.4f} bal acc {results[-1]['mean']['balanced_accuracy']:.4f} "
                      f"man recall {results[-1]['mean']['man_recall']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    # selection rule, written before looking: lowest mean development log loss at +1.5 s, one setting used at every horizon
    cand = [r for r in results if r["horizon"] == "post_1_5s"]
    best = min(cand, key=lambda r: r["mean"]["log_loss"])
    chosen = {"features": best["features"], "man_weight": best["man_weight"],
              "rule": "lowest mean development log loss at +1.5 s across the three chronological folds; the same setting is used at every horizon"}
    report = {"experiment": "A", "question": "relational features and Man class weights, boosted trees", "cohort": "v1 split manifest, weeks 1-12",
              "results": results, "chosen": chosen}
    write_json(OUT / "coverage_expA_cv.json", report)
    run.set(target="released man/zone label (Man = 1)", population="train split of the v1 manifest", input_cutoff=relational.HORIZONS,
            split={"folds": FOLDS}, hyperparameters=GBM, calibration="Platt on out-of-fold training scores, unweighted",
            metrics={"chosen": chosen, "chosen_mean": best["mean"]})
    run.output("cv_report", OUT / "coverage_expA_cv.json")
    run.finish()
    return report
