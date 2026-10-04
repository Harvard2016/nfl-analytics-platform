"""Checks and studies that sit beside the main man/zone evaluation.

- funnel: every tracked play accounted for, from the raw count down to the evaluated splits.
- short_throws: plays thrown too early for the later windows, scored with the at-snap model.
- error_analysis: where the locked-test errors are, with man recall broken down.
- team_holdout: unseen-defense stress test using weeks 1-14 only (the locked test is not touched).
- multiclass_dev: coverage-family attempt, selected and reported on development weeks only.
- tendencies: released-label rates with denominators, and predicted-label rates kept separate.

Nothing here changes a model or a split. Reports go to reports/experiments/.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from ..shared import config
from ..shared.provenance import now_utc, write_json
from . import schema, train
from .features import WINDOWS, feature_names

KEY = ["gameId", "playId"]
K = (pl.col("gameId").cast(pl.String) + ":" + pl.col("playId").cast(pl.String)).alias("_k")
LAST = list(WINDOWS)[-1]
MIN_MULTICLASS_TRAIN = 200
N_TEAM_FOLDS = 4


def _ctx(source: str) -> dict:
    fd = config.FEATURES / source
    return {
        "source": source, "features_dir": fd, "processed": config.PROCESSED / source,
        "models": config.MODELS / f"coverage_{source}",
        "splits": json.loads((config.MANIFESTS / f"{source}_splits.json").read_text()),
        "tables": {w: pl.read_parquet(fd / f"features_{w}.parquet").with_columns(K) for w in WINDOWS},
        "out": config.REPORTS / "experiments",
    }


def _split_of(c: dict) -> dict[str, str]:
    return {k: s for s, v in c["splits"]["splits"].items() for k in v["plays"]}


def _wilson(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z, p = 1.96, k / n
    d = 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(mid - half), 4), round(float(mid + half), 4)]


# ------------------------------------------------------------------ funnel

def funnel(c: dict) -> dict:
    trk = pl.concat([
        pl.scan_parquet(f).group_by(KEY).agg((pl.col("rel_frame").max() + 1).alias("frames")).collect()
        for f in sorted(c["processed"].glob("tracking_week_*.parquet"))])
    plays = pl.read_parquet(c["processed"] / "plays.parquet").join(trk, on=KEY, how="inner").with_columns(K)
    ok = {w: set(t["_k"].to_list()) for w, t in c["tables"].items()}
    w0, w1, w2 = list(WINDOWS)
    labelled = pl.col(schema.TARGET).is_in(schema.TARGET_CLASSES)
    steps, cur = [], plays
    steps.append({"step": "Pass plays with tracking in the release", "plays": cur.height})
    nolab = cur.filter(~labelled | pl.col(schema.TARGET).is_null())
    cur = cur.filter(labelled)
    steps.append({"step": "Removed: no released man/zone label (label field empty)", "removed": nolab.height, "plays": cur.height,
                  "detail": [{"gameId": g, "playId": p, "week": w, "coverage_type": t} for g, p, w, t in
                             nolab.select([*KEY, "week", "coverage_type"]).iter_rows()]})
    for w, why in ((w0, f"fewer than {4} coverage defenders or {3} route runners tracked at the snap"),
                   (w1, "thrown before 1.0 seconds (fewer than 11 frames), or too few tracked players at that frame"),
                   (w2, "thrown before 1.5 seconds (fewer than 16 frames), or too few tracked players at that frame")):
        gone = cur.filter(~pl.col("_k").is_in(ok[w]))
        cur = cur.filter(pl.col("_k").is_in(ok[w]))
        steps.append({"step": f"Removed: {why}", "removed": gone.height, "plays": cur.height,
                      "removed_labels": {k: gone.filter(pl.col(schema.TARGET) == k).height for k in schema.TARGET_CLASSES}})
    in_split = sum(len(v["plays"]) for v in c["splits"]["splits"].values())
    kept = set(cur["_k"].to_list())
    split_keys = set(_split_of(c))
    return {
        "steps": steps, "evaluated_plays": in_split, "matches_split_manifest": kept == split_keys,
        "by_split": {s: {"plays": len(v["plays"]), **c["splits"]["class_counts"][s]} for s, v in c["splits"]["splits"].items()},
        "rule": "A play is evaluated only if it has features in every window, so every model and window is scored on the same plays. "
                "Short sequences are excluded, never padded or extrapolated.",
        "same_test_plays_for_every_model_and_window": True,
    }


# ------------------------------------------------------------------ short throws

def short_throws(c: dict) -> dict:
    """Test-week plays left out of the main comparison because the ball was out early."""
    test_weeks = c["splits"]["weeks"]["test"]
    main = set(_split_of(c))
    out = {"note": "These plays are in the test weeks but outside the main comparison. They were never used for training, "
                   "calibration or threshold choice. Each is scored by the latest window it has frames for.", "groups": []}
    w0, w1, _ = list(WINDOWS)
    seen: set[str] = set()
    for w, label in ((w1, "Thrown between 1.0 and 1.5 seconds: scored with the snap + 1s model"),
                     (w0, "Thrown before 1.0 seconds: scored with the at-snap model")):
        t = c["tables"][w].filter(pl.col("week").is_in(test_weeks) & pl.col(schema.TARGET).is_in(schema.TARGET_CLASSES)
                                  & ~pl.col("_k").is_in(main) & ~pl.col("_k").is_in(seen)).sort(KEY)
        seen |= set(t["_k"].to_list())
        y = (t[schema.TARGET] == "Man").to_numpy().astype(int)
        g = {"group": label, "window": w, "plays": t.height, "man": int(y.sum()), "zone": int((1 - y).sum()), "models": {}}
        if t.height >= 30 and 0 < y.sum() < len(y):
            for m in ("class_prior", "context_logit", "geometry_logit", "geometry_gbm"):
                b = joblib.load(c["models"] / f"{w}__{m}.joblib")
                p = b["model"].predict_proba(t.select(b["columns"]).to_pandas())[:, 1]
                r = train.metrics(y, p)
                g["models"][m] = {k: r[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier", "per_class")}
        out["groups"].append(g)
    return out


# ------------------------------------------------------------------ error analysis

def _rate_table(df: pl.DataFrame, by: str, correct: str) -> list[dict]:
    rows = df.group_by(by).agg(pl.len().alias("n"), pl.col(correct).sum().alias("k")).sort("n", descending=True)
    return [{"group": str(g), "plays": n, "correct": k, "rate": round(k / n, 4), "interval_95": _wilson(k, n)} for g, n, k in rows.iter_rows()]


def error_analysis(c: dict) -> dict:
    test = c["splits"]["splits"]["test"]["plays"]
    out = {"split": "test", "note": "Descriptive breakdown of locked-test errors. Read after the models were frozen; not used to change them.",
           "windows": {}}
    for w in WINDOWS:
        t = c["tables"][w].filter(pl.col("_k").is_in(test)).sort(KEY)
        res = {}
        for m in ("geometry_logit", "geometry_gbm"):
            b = joblib.load(c["models"] / f"{w}__{m}.joblib")
            p = b["model"].predict_proba(t.select(b["columns"]).to_pandas())[:, 1]
            d = t.with_columns(
                pl.Series("p_man", p), pl.Series("pred_man", p >= 0.5),
                pl.when(pl.col("down") >= 3).then(pl.lit("3rd or 4th down")).otherwise(pl.format("{} down", pl.col("down").replace_strict({1: "1st", 2: "2nd"}, default="?"))).alias("down_group"),
                pl.when(pl.col("yardsToGo") <= 3).then(pl.lit("1-3 to go")).when(pl.col("yardsToGo") <= 7).then(pl.lit("4-7 to go")).otherwise(pl.lit("8+ to go")).alias("dist_group"),
            ).with_columns((pl.col("pred_man") == (pl.col(schema.TARGET) == "Man")).alias("ok"))
            man, zone = d.filter(pl.col(schema.TARGET) == "Man"), d.filter(pl.col(schema.TARGET) == "Zone")
            missed, caught = man.filter(~pl.col("ok")), man.filter(pl.col("ok"))
            feats = [f for f in b["columns"] if not f.startswith("snap_")]
            sd = {f: float(d[f].drop_nans().std()) or 1.0 for f in feats}
            gap = sorted(({"feature": f, "missed_man_mean": round(float(missed[f].drop_nans().mean()), 3),
                           "caught_man_mean": round(float(caught[f].drop_nans().mean()), 3),
                           "zone_mean": round(float(zone[f].drop_nans().mean()), 3),
                           "standardized_gap": round((float(missed[f].drop_nans().mean()) - float(caught[f].drop_nans().mean())) / sd[f], 3)}
                          for f in feats), key=lambda r: -abs(r["standardized_gap"]))[:6]
            res[m] = {
                "man_recall": round(caught.height / man.height, 4), "man_plays": man.height, "missed_man": missed.height,
                "zone_recall": round(zone.filter(pl.col("ok")).height / zone.height, 4), "zone_plays": zone.height,
                "man_recall_by_released_coverage_type": _rate_table(man, "coverage_type", "ok"),
                "zone_recall_by_released_coverage_type": _rate_table(zone, "coverage_type", "ok"),
                "man_recall_by_down": _rate_table(man, "down_group", "ok"),
                "man_recall_by_distance": _rate_table(man, "dist_group", "ok"),
                "missed_man_median_p_man": round(float(missed["p_man"].median()), 3),
                "missed_man_share_near_the_line": round(float((missed["p_man"] >= 0.35).mean()), 3),
                "missed_vs_caught_man_feature_gaps": gap,
            }
        out["windows"][w] = res
    return out


# ------------------------------------------------------------------ held-out teams

def team_holdout(c: dict) -> dict:
    """Unseen-defense test inside weeks 1-14. For each fold, every game involving a held-out team is removed
    from training and from calibration; the model is then scored on plays where a held-out team is on defense."""
    split = _split_of(c)
    teams = sorted(set(c["tables"][LAST]["defensiveTeam"].unique().to_list()))
    folds = [teams[i::N_TEAM_FOLDS] for i in range(N_TEAM_FOLDS)]
    out = {"design": f"{N_TEAM_FOLDS} folds of {len(folds[0])} defenses (alphabetical, every {N_TEAM_FOLDS}th team). Weeks 1-14 only: "
                     "the locked test weeks are not used. All games involving a held-out team, on either side of the ball, "
                     "are removed from training (weeks 1-12) and calibration (weeks 13-14). Scored on those teams' defensive plays. "
                     "Same model settings as the main run; nothing tuned.",
           "folds": [{"held_out": f} for f in folds], "windows": {}}
    for w in WINDOWS:
        t = c["tables"][w].filter(pl.col("_k").is_in(list(split))).with_columns(
            pl.col("_k").replace_strict(split, default=None).alias("_split")).filter(pl.col("_split") != "test").sort(KEY)
        geo = feature_names(w)
        pooled: dict[str, dict] = {}
        for fi, held in enumerate(folds):
            touch = pl.col("defensiveTeam").is_in(held) | pl.col("possessionTeam").is_in(held)
            tr, cal = t.filter(~touch & (pl.col("_split") == "train")), t.filter(~touch & (pl.col("_split") == "dev"))
            ev = t.filter(pl.col("defensiveTeam").is_in(held))
            assert not set(ev["gameId"].to_list()) & (set(tr["gameId"].to_list()) | set(cal["gameId"].to_list()))
            ytr, ycal, yev = ((x[schema.TARGET] == "Man").to_numpy().astype(int) for x in (tr, cal, ev))
            out["folds"][fi][w] = {"train_plays": tr.height, "calibration_plays": cal.height, "scored_plays": ev.height}
            for name, (est, cols) in train._models(geo).items():
                est.fit(tr.select(cols).to_pandas(), ytr)
                m = est if name == "class_prior" else CalibratedClassifierCV(FrozenEstimator(est), method="sigmoid").fit(cal.select(cols).to_pandas(), ycal)
                p = m.predict_proba(ev.select(cols).to_pandas())[:, 1]
                acc = pooled.setdefault(name, {"y": [], "p": [], "g": [], "fold_acc": []})
                acc["y"].append(yev), acc["p"].append(p), acc["g"].append(ev["gameId"].to_numpy())
                acc["fold_acc"].append(round(float(accuracy_score(yev, p >= 0.5)), 4))
        res = {}
        for name, a in pooled.items():
            y, p, g = np.concatenate(a["y"]), np.concatenate(a["p"]), np.concatenate(a["g"])
            r = train.metrics(y, p, g)
            res[name] = {k: r[k] for k in ("n", "accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier", "per_class", "game_bootstrap_95")}
            res[name]["accuracy_by_fold"] = a["fold_acc"]
        out["windows"][w] = res
    return out


# ------------------------------------------------------------------ multiclass on development weeks

def multiclass_dev(c: dict) -> dict:
    tr_k, dv_k = c["splits"]["splits"]["train"]["plays"], c["splits"]["splits"]["dev"]["plays"]
    base = c["tables"][LAST].filter(pl.col("_k").is_in(tr_k))
    counts = {k: n for k, n in base.group_by("coverage_type").len().iter_rows() if k is not None}
    classes = sorted(k for k, n in counts.items() if n >= MIN_MULTICLASS_TRAIN)
    out = {"status": "Development only. Trained on weeks 1-12 and scored on weeks 13-14. The locked test weeks have not been "
                     "scored for this task, so there is no final multiclass result yet.",
           "target": "released coverage_type", "classes": classes, "train_counts": counts,
           "excluded_classes": {k: n for k, n in counts.items() if k not in classes},
           "exclusion_rule": f"fewer than {MIN_MULTICLASS_TRAIN} training plays", "probabilities": "uncalibrated", "windows": {}}
    for w in WINDOWS:
        t = c["tables"][w].filter(pl.col("coverage_type").is_in(classes)).sort(KEY)
        tr, dv = t.filter(pl.col("_k").is_in(tr_k)), t.filter(pl.col("_k").is_in(dv_k))
        ytr, ydv = tr["coverage_type"].to_numpy(), dv["coverage_type"].to_numpy()
        res = {"train_plays": tr.height, "dev_plays": dv.height, "models": {}}
        models = train._models(feature_names(w))
        models["class_prior"] = (DummyClassifier(strategy="prior"), train.CONTEXT)
        for name, (est, cols) in models.items():
            est.fit(tr.select(cols).to_pandas(), ytr)
            proba = est.predict_proba(dv.select(cols).to_pandas())
            pred = est.classes_[proba.argmax(1)]
            pr, rc, f1, sup = precision_recall_fscore_support(ydv, pred, labels=classes, zero_division=0)
            res["models"][name] = {
                "accuracy": float(accuracy_score(ydv, pred)), "balanced_accuracy": float(balanced_accuracy_score(ydv, pred)),
                "macro_f1": float(f1_score(ydv, pred, average="macro", labels=classes, zero_division=0)),
                "log_loss": float(log_loss(ydv, proba, labels=list(est.classes_))),
                "per_class": {k: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])} for i, k in enumerate(classes)},
                "confusion": {"labels": classes, "rows_released_cols_predicted": confusion_matrix(ydv, pred, labels=classes).tolist()},
            }
        out["windows"][w] = res
    return out


# ------------------------------------------------------------------ tendencies

def tendencies(c: dict) -> dict:
    """Released-label rates over every labelled tracked play; predicted-label rates over locked-test plays only."""
    trk = pl.concat([pl.scan_parquet(f).select(KEY).unique().collect() for f in sorted(c["processed"].glob("tracking_week_*.parquet"))])
    plays = pl.read_parquet(c["processed"] / "plays.parquet").join(trk, on=KEY).filter(pl.col(schema.TARGET).is_in(schema.TARGET_CLASSES))
    dates = None
    raw = config.RAW / c["source"] / "supplementary_data.csv"
    if raw.exists():
        d = pl.read_csv(raw, null_values=schema.NULLS, infer_schema_length=20000).rename({"game_id": "gameId"}).join(
            plays.select("gameId").unique(), on="gameId").select(pl.col("game_date").str.to_date("%m/%d/%Y"))
        dates = {"first_game": str(d["game_date"].min()), "last_game": str(d["game_date"].max())}
    down = pl.when(pl.col("down") >= 3).then(pl.lit("3rd/4th")).otherwise(pl.col("down").replace_strict({1: "1st", 2: "2nd"}, default="?")).alias("down_group")
    dist = pl.when(pl.col("yardsToGo") <= 3).then(pl.lit("1-3")).when(pl.col("yardsToGo") <= 7).then(pl.lit("4-7")).otherwise(pl.lit("8+")).alias("dist_group")
    plays = plays.with_columns(down, dist, (pl.col(schema.TARGET) == "Man").alias("man"))

    def cell(df: pl.DataFrame) -> dict:
        n, k = df.height, int(df["man"].sum())
        return {"plays": n, "man": k, "man_rate": round(k / n, 4) if n else None, "interval_95": _wilson(k, n)}

    def block(df: pl.DataFrame) -> dict:
        return {"all": cell(df),
                "by_situation": [{"down": dn, "distance": ds, **cell(df.filter((pl.col("down_group") == dn) & (pl.col("dist_group") == ds)))}
                                 for dn in ("1st", "2nd", "3rd/4th") for ds in ("1-3", "4-7", "8+")],
                "coverage_types": {k: n for k, n in df.group_by("coverage_type").len().sort("len", descending=True).iter_rows()}}

    test = c["splits"]["splits"]["test"]["plays"]
    b = joblib.load(c["models"] / f"{LAST}__geometry_gbm.joblib")
    t = c["tables"][LAST].filter(pl.col("_k").is_in(test)).sort(KEY)
    p = b["model"].predict_proba(t.select(b["columns"]).to_pandas())[:, 1]
    t = t.with_columns(pl.Series("pred_man", p >= 0.5), (pl.col(schema.TARGET) == "Man").alias("man"),
                       pl.Series("abstain", np.maximum(p, 1 - p) < b["threshold"]))

    def pred_cell(df: pl.DataFrame) -> dict:
        n = df.height
        return {"plays": n, "released_man_rate": round(float(df["man"].mean()), 4) if n else None,
                "predicted_man_rate": round(float(df["pred_man"].mean()), 4) if n else None,
                "abstained": int(df["abstain"].sum()), "agreement": round(float((df["man"] == df["pred_man"]).mean()), 4) if n else None}

    teams = sorted(plays["defensiveTeam"].unique().to_list())
    return {
        "sample": {"season": sorted(plays["season"].unique().to_list()), "weeks": [int(plays["week"].min()), int(plays["week"].max())],
                   "dates": dates, "plays": plays.height,
                   "scope": "Pass plays that reached a throw and carry a released man/zone label. Not every defensive snap."},
        "released": {"league": block(plays), "teams": {tm: block(plays.filter(pl.col("defensiveTeam") == tm)) for tm in teams}},
        "predicted": {
            "scope": f"Locked test weeks {c['splits']['weeks']['test'][0]}-{c['splits']['weeks']['test'][-1]} only: the model never trained on these plays.",
            "model": b["version"], "confidence_bar": b["threshold"],
            "league": pred_cell(t), "teams": {tm: pred_cell(t.filter(pl.col("defensiveTeam") == tm)) for tm in teams}},
    }


def run(source: str) -> dict:
    c = _ctx(source)
    report = {"source": source, "run_at": now_utc(), "synthetic": (c["features_dir"] / "SYNTHETIC").exists(),
              "funnel": funnel(c), "short_throws": short_throws(c), "error_analysis": error_analysis(c),
              "team_holdout": team_holdout(c), "multiclass_dev": multiclass_dev(c)}
    tag = "_SYNTHETIC" if report["synthetic"] else ""
    write_json(c["out"] / f"coverage_studies{tag}.json", report)
    write_json(c["out"] / f"coverage_tendencies{tag}.json", {"source": source, "run_at": report["run_at"],
                                                              "synthetic": report["synthetic"]} | tendencies(c))
    return report
