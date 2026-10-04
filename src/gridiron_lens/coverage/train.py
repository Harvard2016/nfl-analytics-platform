"""Frozen week splits, baselines, calibration, abstention and held-out evaluation for man vs zone.

Split weeks come from schema.SOURCES. Real data (2023 season): weeks 1-12 train, weeks 13-14
development (calibration and threshold choice), weeks 15-18 locked test.
Whole games stay in one split because a game belongs to exactly one week. Nothing is fitted or tuned
on test plays; the test set is scored once per model with the configuration frozen on development.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..shared import config
from ..shared.provenance import now_utc, write_json
from . import schema
from .features import FEATURE_SCHEMA_VERSION, WINDOWS, feature_names

SEED = 20261003
CONTEXT = list(schema.CONTEXT_WHITELIST)
TARGET_ACCEPT_ACCURACY = 0.90
MIN_CLASS_TRAIN = 50


def freeze_splits(source: str, features_dir: Path, out: Path | None = None) -> dict:
    """Write the split manifest once. Only plays eligible in every window are kept, so windows compare on the same plays."""
    out = Path(out or config.MANIFESTS / f"{source}_splits.json")
    split_weeks = schema.SOURCES[source]["split_weeks"]
    if out.exists():
        import json
        return json.loads(out.read_text())
    key = ["gameId", "playId"]
    tables = [pl.read_parquet(features_dir / f"features_{w}.parquet") for w in WINDOWS]
    base = tables[0].filter(pl.col(schema.TARGET).is_in(schema.TARGET_CLASSES))
    for t in tables[1:]:
        base = base.join(t.select(key), on=key, how="inner")
    manifest = {"version": schema.SOURCES[source]["split_version"], "source": source, "frozen_at": now_utc(), "unit": "play; whole games per split",
                "weeks": split_weeks, "target": schema.TARGET, "splits": {}, "class_counts": {},
                "excluded": {"no_man_zone_label_or_other": tables[0].height - tables[0].filter(
                    pl.col(schema.TARGET).is_in(schema.TARGET_CLASSES)).height}}
    for name, weeks in split_weeks.items():
        part = base.filter(pl.col("week").is_in(weeks))
        manifest["splits"][name] = {"games": sorted(part["gameId"].unique().to_list()),
                                    "plays": [f"{g}:{p}" for g, p in part.select(key).sort(key).iter_rows()]}
        manifest["class_counts"][name] = {c: part.filter(pl.col(schema.TARGET) == c).height for c in schema.TARGET_CLASSES}
    write_json(out, manifest)
    return manifest


def _xy(table: pl.DataFrame, plays: list[str]):
    keyed = table.with_columns((pl.col("gameId").cast(pl.String) + ":" + pl.col("playId").cast(pl.String)).alias("_k"))
    part = keyed.filter(pl.col("_k").is_in(plays)).sort(["gameId", "playId"])
    return part, (part[schema.TARGET] == "Man").to_numpy().astype(int)


def _models(geo: list[str]) -> dict:
    num = lambda cols: make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    logit = lambda: LogisticRegression(C=1.0, max_iter=2000)
    ctx = ColumnTransformer([("num", num(CONTEXT), CONTEXT),
                             ("team", OneHotEncoder(handle_unknown="ignore"), ["defensiveTeam"])])
    return {
        "class_prior": (DummyClassifier(strategy="prior"), CONTEXT),
        "context_logit": (make_pipeline(ctx, logit()), CONTEXT + ["defensiveTeam"]),
        "geometry_logit": (make_pipeline(ColumnTransformer([("num", num(geo), geo)]), logit()), geo),
        "geometry_gbm": (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=250,
                                                        l2_regularization=1.0, random_state=SEED), geo),
    }


def _reliability(y, p, bins=10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return [{"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int((idx == b).sum()),
             "mean_predicted": float(p[idx == b].mean()), "observed_man_rate": float(y[idx == b].mean())}
            for b in range(bins) if (idx == b).any()]


def metrics(y: np.ndarray, p: np.ndarray, games: np.ndarray | None = None, n_boot: int = 500) -> dict:
    pred = (p >= 0.5).astype(int)
    pr, rc, _, sup = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    out = {
        "n": len(y), "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]), "support": int(sup[i])}
                      for i, c in enumerate(schema.TARGET_CLASSES)},
        "confusion": {"labels": list(schema.TARGET_CLASSES), "rows_charted_cols_predicted": cm.tolist()},
        "reliability": _reliability(y, p),
    }
    if games is not None and len(np.unique(games)) > 1:  # resample whole games, not plays or frames
        rng = np.random.default_rng(SEED)
        ug = np.unique(games)
        by_game = {g: np.where(games == g)[0] for g in ug}
        acc, f1 = [], []
        for _ in range(n_boot):
            ix = np.concatenate([by_game[g] for g in rng.choice(ug, len(ug))])
            acc.append(accuracy_score(y[ix], pred[ix]))
            f1.append(f1_score(y[ix], pred[ix], average="macro", zero_division=0))
        out["game_bootstrap_95"] = {"games": len(ug), "resamples": n_boot,
                                    "accuracy": [float(np.quantile(acc, .025)), float(np.quantile(acc, .975))],
                                    "macro_f1": [float(np.quantile(f1, .025)), float(np.quantile(f1, .975))]}
    return out


def _accept_curve(y, p) -> list[dict]:
    conf, pred = np.maximum(p, 1 - p), (p >= 0.5).astype(int)
    return [{"threshold": float(t), "accepted_fraction": float((conf >= t).mean()),
             "accepted_accuracy": float((pred[conf >= t] == y[conf >= t]).mean()) if (conf >= t).any() else None}
            for t in np.round(np.arange(0.5, 0.96, 0.05), 2)]


def _pick_threshold(curve: list[dict]) -> dict:
    """Lowest confidence threshold whose DEVELOPMENT accepted accuracy meets the target; else no abstention."""
    for row in curve:
        if row["accepted_accuracy"] is not None and row["accepted_accuracy"] >= TARGET_ACCEPT_ACCURACY and row["accepted_fraction"] >= 0.2:
            return {"threshold": row["threshold"], "rule": f"lowest threshold with development accepted accuracy >= {TARGET_ACCEPT_ACCURACY} and >= 20% accepted"}
    return {"threshold": 0.5, "rule": "no threshold met the development target; nothing is abstained"}


def run(source: str, features_dir: Path | None = None, models_dir: Path | None = None, reports_dir: Path | None = None,
        splits_path: Path | None = None) -> dict:
    features_dir = Path(features_dir or config.FEATURES / source)
    models_dir = Path(models_dir or config.MODELS / f"coverage_{source}")
    reports_dir = Path(reports_dir or config.REPORTS / "experiments")
    models_dir.mkdir(parents=True, exist_ok=True)
    splits = freeze_splits(source, features_dir, splits_path)
    counts = splits["class_counts"]
    if min(counts["train"].values()) < MIN_CLASS_TRAIN or min(counts["test"].values()) == 0 or min(counts["dev"].values()) == 0:
        raise SystemExit(f"Too few labelled plays to train honestly: {counts}")
    synthetic = (features_dir / "SYNTHETIC").exists()
    report = {"module": "coverage", "task": "man_vs_zone", "run_at": now_utc(), "synthetic": synthetic,
              "source": source, "label_source": schema.SOURCES[source]["label_source"], "split_weeks": splits["weeks"], "split_version": splits["version"], "class_counts": counts,
              "feature_schema": FEATURE_SCHEMA_VERSION, "seed": SEED, "positive_class": "Man",
              "label_meaning": "agreement with the released coverage label, not coaching intent",
              "windows": {}}
    for window in WINDOWS:
        table = pl.read_parquet(features_dir / f"features_{window}.parquet")
        geo = feature_names(window)
        (tr, ytr), (dv, ydv), (te, yte) = (_xy(table, splits["splits"][s]["plays"]) for s in ("train", "dev", "test"))
        # a feature with no variation in training carries no information and breaks binning
        flat = [c for c in geo if tr[c].drop_nans().drop_nulls().n_unique() < 2]
        geo = [c for c in geo if c not in flat]
        res = {"features": geo, "dropped_constant_features": flat,
               "context_features": CONTEXT + ["defensiveTeam"], "models": {}}
        for name, (est, cols) in _models(geo).items():
            est.fit(tr.select(cols).to_pandas(), ytr)
            cal = CalibratedClassifierCV(FrozenEstimator(est), method="sigmoid").fit(dv.select(cols).to_pandas(), ydv) \
                if name != "class_prior" else est
            p_dev = cal.predict_proba(dv.select(cols).to_pandas())[:, 1]
            p_te = cal.predict_proba(te.select(cols).to_pandas())[:, 1]
            dev_curve = _accept_curve(ydv, p_dev)
            pick = _pick_threshold(dev_curve)
            conf = np.maximum(p_te, 1 - p_te)
            acc_mask = conf >= pick["threshold"]
            version = f"cov-{window}-{name}-{FEATURE_SCHEMA_VERSION}-{splits['version']}"
            joblib.dump({"model": cal, "base": est, "columns": cols, "version": version, "threshold": pick["threshold"],
                         "window": window, "synthetic": synthetic}, models_dir / f"{window}__{name}.joblib")
            res["models"][name] = {
                "version": version, "calibration": "sigmoid, fitted on the development weeks" if name != "class_prior" else "none",
                "dev": metrics(ydv, p_dev), "test": metrics(yte, p_te, te["gameId"].to_numpy()),
                "abstention": pick | {
                    "dev_curve": dev_curve, "test_curve": _accept_curve(yte, p_te),
                    "test_accepted_fraction": float(acc_mask.mean()),
                    "test_accepted_accuracy": float(((p_te >= .5).astype(int)[acc_mask] == yte[acc_mask]).mean()) if acc_mask.any() else None},
            }
        report["windows"][window] = res
    write_json(reports_dir / ("coverage_man_zone_SYNTHETIC.json" if synthetic else "coverage_man_zone.json"), report)
    return report
