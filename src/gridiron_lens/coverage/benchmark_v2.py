"""Coverage v2: freeze the development choices, then compare every model on the previously examined benchmark.

Models compared at each horizon, same plays:
  v1_gbm       boosted trees on v1 geometry features (the preserved benchmark model, loaded from disk, not refitted)
  v2_gbm_rel   boosted trees with the feature set and Man weight chosen in experiment A
  v2_temporal  the temporal interaction model chosen in experiment B (seed 42), plus the 3-seed average

Weeks 13-14: Platt calibrators and operating policies (class threshold and abstention cutoff are chosen separately).
Weeks 15-18 were examined in v1. Results there are comparisons on a previously examined benchmark, not a fresh test.

Two populations are always reported apart:
  common cohort   the 3,178 test plays that have every v1 window (so models and horizons are comparable)
  operational     every labelled test-week play that has the horizon's prefix (includes quick throws at early horizons)
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from ..shared import config
from ..shared.provenance import write_json
from ..shared.runs import Run
from . import evalkit, features, relational, schema
from .experiment_a import GBM, KEY, SOURCE, K, fit_platt

OUT = config.REPORTS / "v2"
MODELS_DIR = config.MODELS / "coverage_v2"
HZ = list(relational.HORIZONS)
V1_WINDOWS = ["at_snap", "post_1s", "post_1_5s"]
LABELS = {"v1_gbm": "Boosted trees, v1 geometry (benchmark)", "v2_gbm_rel": "Boosted trees, geometry + relational (v2)",
          "v2_temporal": "Temporal interaction model (v2)", "v2_temporal_3seed": "Temporal model, 3-seed average (v2)"}


def geometry_table(horizon: str) -> pl.DataFrame:
    """v1 geometry features. +0.5 s was not a v1 window, so it is built here with the same code and a temporary window."""
    fd = config.FEATURES / SOURCE
    path = fd / f"features_{horizon}.parquet"
    if not path.exists():
        features.WINDOWS[horizon] = (relational.HORIZONS[horizon], 0)
        try:
            t, _ = features.build_window(config.PROCESSED / SOURCE, horizon, pl.read_parquet(config.PROCESSED / SOURCE / "plays.parquet"))
            t.write_parquet(path)
        finally:
            features.WINDOWS.pop(horizon, None)
    return pl.read_parquet(path)


def geometry_names(horizon: str) -> list[str]:
    return features.STATIC if horizon == "at_snap" else features.STATIC + features.MOTION + [f"snap_{n}" for n in features.STATIC]


def table(horizon: str) -> tuple[pl.DataFrame, list[str], list[str]]:
    fd = config.FEATURES / SOURCE
    rcols = json.loads((fd / "relational_report.json").read_text())["horizons"][horizon]["features"]
    rel = pl.read_parquet(fd / f"relational_{horizon}.parquet").select([*KEY, *rcols])
    t = geometry_table(horizon).join(rel, on=KEY, how="inner").with_columns(K).filter(pl.col(schema.TARGET).is_in(schema.TARGET_CLASSES)).sort(KEY)
    return t, geometry_names(horizon), rcols


def fit_v2_tree(chosen: dict) -> dict:
    """Refit on weeks 1-12 with the experiment-A choice; Platt on weeks 13-14 without class weights. Predict every labelled play."""
    splits = json.loads((config.MANIFESTS / f"{SOURCE}_splits.json").read_text())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for h in HZ:
        t, g, r = table(h)
        cols = {"geometry": g, "geometry+relational": g + r, "relational": r}[chosen["features"]]
        # training population: the v1 train cohort where it applies; at +0.5 s the same plays (they all have 16 frames)
        tr = t.filter(pl.col("_k").is_in(splits["splits"]["train"]["plays"]))
        cal = t.filter(pl.col("_k").is_in(splits["splits"]["dev"]["plays"]))
        ytr, ycal = (tr[schema.TARGET] == "Man").to_numpy().astype(int), (cal[schema.TARGET] == "Man").to_numpy().astype(int)
        model = HistGradientBoostingClassifier(**GBM).fit(tr.select(cols).to_numpy(), ytr, sample_weight=np.where(ytr == 1, chosen["man_weight"], 1.0))
        platt = fit_platt(model.decision_function(cal.select(cols).to_numpy()), ycal)
        p = platt.predict_proba(model.decision_function(t.select(cols).to_numpy()).reshape(-1, 1))[:, 1]
        medians = {c: float(np.nanmedian(tr[c].to_numpy())) for c in cols}
        joblib.dump({"model": model, "platt": platt, "columns": cols, "horizon": h, "version": f"cov-v2-gbm-rel-{h}", "chosen": chosen, "train_medians": medians},
                    MODELS_DIR / f"gbm_rel_{h}.joblib")
        out[h] = dict(zip(t["_k"].to_list(), p.tolist()))
    return out


def v1_predictions() -> dict:
    out = {}
    for h in V1_WINDOWS:
        b = joblib.load(config.MODELS / f"coverage_{SOURCE}" / f"{h}__geometry_gbm.joblib")
        t = pl.read_parquet(config.FEATURES / SOURCE / f"features_{h}.parquet").with_columns(K).filter(pl.col(schema.TARGET).is_in(schema.TARGET_CLASSES))
        out[h] = dict(zip(t["_k"].to_list(), b["model"].predict_proba(t.select(b["columns"]).to_pandas())[:, 1].tolist()))
    return out


def temporal_predictions() -> dict[str, dict]:
    z = np.load(OUT / "coverage_expB_predictions.npz")
    keys = z["key"].tolist()
    out = {}
    for name, arr in (("v2_temporal", z["seed42"]), ("v2_temporal_3seed", z["ensemble"])):
        out[name] = {h: {k: float(p) for k, p in zip(keys, arr[:, j]) if np.isfinite(p)} for j, h in enumerate(HZ)}
    return out


def evaluate() -> dict:
    run = Run("coverage", "benchmark-v2", seed=20261004)
    splits = json.loads((config.MANIFESTS / f"{SOURCE}_splits.json").read_text())
    chosen_a = json.loads((OUT / "coverage_expA_cv.json").read_text())["chosen"]
    preds = {"v1_gbm": v1_predictions(), "v2_gbm_rel": fit_v2_tree(chosen_a), **temporal_predictions()}
    idx = pl.read_parquet(config.FEATURES / SOURCE / "tensors_v2_index.parquet").with_columns(K).filter(pl.col(schema.TARGET).is_in(schema.TARGET_CLASSES))
    info = {r["_k"]: r for r in idx.iter_rows(named=True)}
    test_weeks, _dev_weeks = splits["weeks"]["test"], splits["weeks"]["dev"]
    common = {s: splits["splits"][s]["plays"] for s in ("dev", "test")}

    def arrays(model: str, h: str, keys: list[str]):
        keys = [k for k in keys if k in preds[model][h]]
        y = np.array([info[k][schema.TARGET] == "Man" for k in keys]).astype(int)
        return keys, y, np.array([preds[model][h][k] for k in keys]), np.array([info[k]["gameId"] for k in keys])

    rep = {"labels": LABELS, "horizons": relational.HORIZONS, "chosen": {"experiment_a": chosen_a, "experiment_b": json.loads((OUT / "coverage_expB_cv.json").read_text())["chosen"]},
           "note": "Weeks 15-18 were examined during v1. Everything below on those weeks is a comparison on a previously examined benchmark.",
           "decision_rule": "Lean: man if calibrated p(man) >= class threshold (0.5 unless the balanced policy is supported). Confidence = max(p, 1 - p). Abstain when confidence < the cutoff chosen on weeks 13-14.",
           "models": {}}
    rows = []
    for model in preds:
        rep["models"][model] = {}
        for h in HZ:
            if h not in preds[model]:
                continue
            kd, yd, pd_, _ = arrays(model, h, common["dev"])
            kt, yt, pt, gt = arrays(model, h, common["test"])
            policy = evalkit.balanced_policy(yd, pd_)
            sweep_dev = evalkit.abstention_sweep(yd, pd_)
            cut = evalkit.pick_cutoff(sweep_dev)
            test = evalkit.binary_metrics(yt, pt)
            entry = {
                "calibration_weeks": {"plays": len(kd), **{k: v for k, v in evalkit.binary_metrics(yd, pd_).items() if k != "confusion"}},
                "benchmark": test, "benchmark_game_bootstrap_95": evalkit.game_bootstrap(yt, pt, gt), "reliability": evalkit.reliability(yt, pt),
                "balanced_policy": policy | ({"benchmark": evalkit.binary_metrics(yt, pt, policy["threshold"])} if policy["supported"] else {}),
                "abstention": {"cutoff": cut, "calibration_sweep": sweep_dev, "benchmark_sweep": evalkit.abstention_sweep(yt, pt),
                               "benchmark_at_cutoff": next(r for r in evalkit.abstention_sweep(yt, pt) if r["cutoff"] == cut["cutoff"])},
            }
            # operational eligibility: every labelled test-week play with this prefix, including quick throws left out of the common cohort
            op_keys = [k for k, r in info.items() if r["week"] in test_weeks]
            ko, yo, po, _ = arrays(model, h, op_keys)
            entry["operational"] = {"plays_with_prefix": len(ko), "labelled_test_week_plays": len(op_keys),
                                    "not_scored": len(op_keys) - len(ko), **{k: v for k, v in evalkit.binary_metrics(yo, po).items() if k != "confusion"}}
            if model != "v1_gbm" and h in preds["v1_gbm"]:
                _, _, p_old, _ = arrays("v1_gbm", h, kt)
                entry["paired_vs_v1_gbm"] = evalkit.paired_game_bootstrap(yt, pt, p_old, gt)
            rep["models"][model][h] = entry
            for split, ks in (("dev", kd), ("test", kt)):
                rows += [{"model": model, "horizon": h, "key": k, "split": split, "p_man": preds[model][h][k]} for k in ks]
    # selected-player sensitivity: results by how many coverage defenders the release tracked (common cohort, +1.5 s)
    sens = {}
    for model in ("v1_gbm", "v2_gbm_rel", "v2_temporal"):
        kt, yt, pt, _ = arrays(model, "post_1_5s", common["test"])
        nd = np.array([info[k]["n_def"] for k in kt])
        sens[model] = [{"tracked_defenders": lab, "plays": int(m.sum()), "man_share": float(yt[m].mean()), "accuracy": float(((pt[m] >= .5) == yt[m]).mean()),
                        "man_recall": float(((pt[m] >= .5) & (yt[m] == 1)).sum() / max(1, yt[m].sum()))}
                       for lab, m in (("5 or fewer", nd <= 5), ("6", nd == 6), ("7", nd == 7), ("8 or more", nd >= 8)) if m.sum() >= 30]
    rep["selected_player_sensitivity"] = {"note": "The number of tracked coverage defenders is set by the release after the play and is strongly tied to the label. "
                                                  "No model here uses the count directly, but every feature is computed over that set. Results by count show how much of the "
                                                  "performance holds inside each stratum.", "by_tracked_defenders": sens}
    write_json(OUT / "coverage_benchmark_v2.json", rep)
    pl.DataFrame(rows).write_parquet(OUT / "coverage_predictions_v2.parquet")
    run.set(target="released man/zone label", population="v1 common cohort and operational test-week plays", input_cutoff=relational.HORIZONS,
            split={"weeks": splits["weeks"], "manifest": "data/manifests/bdb2026_splits.json"}, calibration="Platt per model and horizon on weeks 13-14",
            metrics={m: {h: {k: e["benchmark"][k] for k in ("accuracy", "macro_f1", "log_loss", "brier", "man_recall", "zone_recall")} for h, e in v.items()} for m, v in rep["models"].items()})
    run.output("report", OUT / "coverage_benchmark_v2.json")
    run.output("predictions", OUT / "coverage_predictions_v2.parquet")
    run.finish()
    return rep


N_TEAM_FOLDS = 4


def team_holdout() -> dict:
    """Unseen-defense check for the v2 models inside weeks 1-14 (benchmark weeks untouched).

    For each fold every game involving a held-out team, on either side of the ball, is removed from training
    (weeks 1-12) and from calibration (weeks 13-14). Models are scored on the held-out teams' defensive plays in
    weeks 1-14. Settings are the frozen experiment A and B choices; nothing is tuned here.
    """
    from . import neural
    from .experiment_b import sig
    splits = json.loads((config.MANIFESTS / f"{SOURCE}_splits.json").read_text())
    ca, cb = (json.loads((OUT / f"coverage_exp{x}_cv.json").read_text())["chosen"] for x in "AB")
    cohort = set(splits["splits"]["train"]["plays"]) | set(splits["splits"]["dev"]["plays"])
    t, g, r = table("post_1_5s")
    t = t.filter(pl.col("_k").is_in(cohort))
    cols = {"geometry": g, "geometry+relational": g + r, "relational": r}[ca["features"]]
    teams = sorted(t["defensiveTeam"].unique().to_list())
    folds = [teams[i::N_TEAM_FOLDS] for i in range(N_TEAM_FOLDS)]
    A = neural.load_arrays()
    in_c = np.array([k in cohort for k in A["key"]])
    off = pl.read_parquet(config.FEATURES / SOURCE / "tensors_v2_index.parquet")["possessionTeam"].to_numpy()
    acc = {m: {"y": [], "p": [], "g": []} for m in ("v2_gbm_rel", "v2_temporal")}
    rep = {"design": team_holdout.__doc__.split("\n\n")[1].strip().replace("\n    ", " "), "horizon": "post_1_5s", "folds": []}
    for held in folds:
        touch = pl.col("defensiveTeam").is_in(held) | pl.col("possessionTeam").is_in(held)
        tr, cal, ev = t.filter(~touch & (pl.col("week") <= 12)), t.filter(~touch & pl.col("week").is_in([13, 14])), t.filter(pl.col("defensiveTeam").is_in(held))
        assert not set(ev["gameId"].to_list()) & (set(tr["gameId"].to_list()) | set(cal["gameId"].to_list()))
        ytr, ycal = (tr[schema.TARGET] == "Man").to_numpy().astype(int), (cal[schema.TARGET] == "Man").to_numpy().astype(int)
        m = HistGradientBoostingClassifier(**GBM).fit(tr.select(cols).to_numpy(), ytr, sample_weight=np.where(ytr == 1, ca["man_weight"], 1.0))
        pt = fit_platt(m.decision_function(cal.select(cols).to_numpy()), ycal).predict_proba(m.decision_function(ev.select(cols).to_numpy()).reshape(-1, 1))[:, 1]
        acc["v2_gbm_rel"]["y"].append((ev[schema.TARGET] == "Man").to_numpy().astype(int)), acc["v2_gbm_rel"]["p"].append(pt), acc["v2_gbm_rel"]["g"].append(ev["gameId"].to_numpy())
        involved = np.isin(A["team"], held) | np.isin(off, held)
        lab = in_c & ~np.isnan(A["y"])
        ntr, ncal, nev = np.where(lab & ~involved & (A["week"] <= 12))[0], np.where(lab & ~involved & np.isin(A["week"], [13, 14]))[0], np.where(lab & np.isin(A["team"], held))[0]
        res = neural.train_model(A, ntr, None, temporal=cb["temporal"], augment=cb["augment"], seed=42, fixed_epochs=cb["epochs"], log=lambda s: None)
        lg = neural.predict_logits(res["model"], A, np.concatenate([ncal, nev]))[:, 15]
        pl_ = fit_platt(lg[: len(ncal)], A["y"][ncal].astype(int))
        acc["v2_temporal"]["y"].append(A["y"][nev].astype(int)), acc["v2_temporal"]["p"].append(sig(pl_.coef_[0, 0] * lg[len(ncal):] + pl_.intercept_[0])), acc["v2_temporal"]["g"].append(A["game"][nev])
        rep["folds"].append({"held_out": held, "train_plays": len(ntr), "calibration_plays": len(ncal), "scored_plays": len(nev)})
    rep["models"] = {}
    for m, a in acc.items():
        y, p, gms = np.concatenate(a["y"]), np.concatenate(a["p"]), np.concatenate(a["g"])
        rep["models"][m] = {k: v for k, v in evalkit.binary_metrics(y, p).items() if k != "confusion"} | {
            "game_bootstrap_95": evalkit.game_bootstrap(y, p, gms), "accuracy_by_fold": [float(((pp >= .5) == yy).mean()) for yy, pp in zip(a["y"], a["p"])]}
    write_json(OUT / "coverage_team_holdout_v2.json", rep)
    return rep


def review_queue(n: int = 25, seed: int = 41) -> dict:
    """Development-week (13-14) plays worth a human look, for the temporal model at +1.5 s. Original labels are never changed here."""
    splits = json.loads((config.MANIFESTS / f"{SOURCE}_splits.json").read_text())
    p = temporal_predictions()["v2_temporal"]["post_1_5s"]
    info = {r["_k"]: r for r in pl.read_parquet(config.FEATURES / SOURCE / "tensors_v2_index.parquet").with_columns(K).iter_rows(named=True)}
    dev = [k for k in splits["splits"]["dev"]["plays"] if k in p]
    rng = np.random.default_rng(seed)
    rows = [{"key": k, "week": info[k]["week"], "defense": info[k]["defensiveTeam"], "released_label": info[k][schema.TARGET], "released_coverage_type": info[k]["coverage_type"],
             "p_man": round(p[k], 4), "lean": "Man" if p[k] >= .5 else "Zone"} for k in dev]
    wrong = [r for r in rows if r["lean"] != r["released_label"]]
    pick = lambda xs: sorted(xs, key=lambda r: r["key"])[:n] if len(xs) <= n else [xs[i] for i in sorted(rng.choice(len(xs), n, replace=False))]
    q = {
        "confident_disagreements": sorted(wrong, key=lambda r: -abs(r["p_man"] - 0.5))[:n],
        "uncertain": sorted(rows, key=lambda r: abs(r["p_man"] - 0.5))[:n],
        "minority_class_misses": pick([r for r in wrong if r["released_label"] == "Man"]),
        "random_control": pick(rows),
    }
    out = {"purpose": "Error-review queue on development weeks 13-14 (not the benchmark). Reviewer alternatives go into a separate annotation version with the reviewer's evidence basis and uncertainty; the released labels and the benchmark are never relabelled because a model disagrees.",
           "model": "v2_temporal seed 42, +1.5 s", "plays_available": len(rows), "disagreements_available": len(wrong), "queues": q}
    write_json(OUT / "coverage_review_queue.json", out)
    return out
