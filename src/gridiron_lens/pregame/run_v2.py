"""Pregame v2 driver: build features, select on development seasons, freeze, compare on the examined benchmark, write forecasts.

Commands (via bin/pregame-lens): `backtest` (features, selection, reports, site export) and `forecast` (append-only
records for games that have not kicked off). Reconstructed backtests and live forecasts are stored separately and
labelled differently; a backtest row is never written into the forecast log.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from ..shared import config
from ..shared.provenance import now_utc, write_json
from ..shared.runs import Run, sha256_path
from . import features_v2 as F
from . import model_v2 as M
from . import pipeline

OUT = config.REPORTS / "v2"
WEB = config.WEB_DEMO / "pregame"
FORECASTS = config.REPORTS / "forecasts"
FEATURES = config.FEATURES / "pregame" / "features_v2.parquet"
VERSION = "pregame-v2"
ET = ZoneInfo("America/New_York")


def build_features() -> pl.DataFrame:
    games = pipeline.load_games()
    team, qb = F.load_stats()
    feat = F.build(games, team, qb)
    FEATURES.parent.mkdir(parents=True, exist_ok=True)
    feat.write_parquet(FEATURES)
    return feat


def decided_games(feat: pl.DataFrame) -> pl.DataFrame:
    return (feat.filter(pl.col("result").is_not_null() & (pl.col("result") != 0) & (pl.col("season") >= M.FIRST))
            .with_columns((pl.col("result") > 0).cast(pl.Int8).alias("home_win")).sort(["gameday", "game_id"]))


def select(decided: pl.DataFrame, log=print) -> dict:
    """Walk-forward development selection on 2012-2022. Nothing from 2023 onward is read here."""
    dev_seasons = range(M.DEV[0], M.DEV[1] + 1)
    dev = decided.filter(pl.col("season").is_between(*M.DEV))
    y, elo = dev["home_win"].to_numpy().astype(float), dev["p_elo"].to_numpy()
    season = dev["season"].to_numpy()
    base = float(M.ll(y, elo).mean())
    rows, preds = [], {}
    for kind, pens in (("elo_offset", M.LAMBDAS), ("margin", M.ALPHAS)):
        for variant in F.VARIANTS:
            for gname, groups in M.GROUP_SETS.items():
                uses_window = any(g in ("efficiency", "rates") for g in groups)
                if not uses_window and variant != F.VARIANTS[0]:
                    continue                                   # window choice is irrelevant for these groups
                cols = F.group_columns(groups, variant)
                for pen in pens:
                    p = M.walk_forward(decided, dev_seasons, kind, cols, pen)
                    by = {int(s): float(M.ll(y[season == s], p[season == s]).mean() - M.ll(y[season == s], elo[season == s]).mean()) for s in dev_seasons}
                    key = f"{kind}|{gname}|{variant if uses_window else '-'}|{pen}"
                    preds[key] = p
                    rows.append({"key": key, "model": kind, "groups": gname, "window": variant if uses_window else None, "penalty": pen,
                                 "log_loss": float(M.ll(y, p).mean()), "brier": float(((p - y) ** 2).mean()), "vs_elo": float(M.ll(y, p).mean() - base),
                                 "seasons_better_than_elo": int(sum(v < 0 for v in by.values())), "by_season_vs_elo": by})
        log(f"{kind}: {len([r for r in rows if r['model'] == kind])} configurations scored")
    best = {k: min((r for r in rows if r["model"] == k), key=lambda r: r["log_loss"]) for k in ("elo_offset", "margin")}
    # one blend: chosen Elo-offset and chosen point-margin probabilities, weight picked on the same development predictions
    blends = [{"weight_elo_offset": w, "log_loss": float(M.ll(y, w * preds[best["elo_offset"]["key"]] + (1 - w) * preds[best["margin"]["key"]]).mean())}
              for w in (0.0, 0.25, 0.5, 0.75, 1.0)]
    blend = min(blends, key=lambda b: b["log_loss"])
    return {"development_seasons": list(M.DEV), "games": len(y), "elo_log_loss": base, "configurations": rows, "chosen": best,
            "blend_candidates": blends, "blend": blend,
            "rule": "lowest pooled walk-forward log loss on 2012-2022 for each model family; blend weight from the same development predictions",
            "note": "The winning configuration was picked from many on the same development games, so its development score is optimistic."}


def frozen_predictions(decided: pl.DataFrame, sel: dict, seasons: list[int]) -> pl.DataFrame:
    """Every model on the same games, each season fitted on all earlier seasons. Also returns pieces for per-game explanations."""
    parts = []
    eo, mg, w = sel["chosen"]["elo_offset"], sel["chosen"]["margin"], sel["blend"]["weight_elo_offset"]
    for s in seasons:
        tr, te = decided.filter(pl.col("season") < s), decided.filter(pl.col("season") == s)
        if te.height == 0:
            continue
        v1 = M.v1_models(tr, te)
        cols_e = F.group_columns(M.GROUP_SETS[eo["groups"]], eo["window"] or F.VARIANTS[0])
        cols_m = F.group_columns(M.GROUP_SETS[mg["groups"]], mg["window"] or F.VARIANTS[0])
        pe, fe = M.fit_predict(tr, te, "elo_offset", cols_e, eo["penalty"])
        pm, fm = M.fit_predict(tr, te, "margin", cols_m, mg["penalty"])
        te2, _ = M.add_qb(te, tr)
        contrib = fe["z"] * fe["w"][1:]
        parts.append(te2.with_columns(
            *[pl.Series(f"p_{k}", v) for k, v in v1.items() if k != "elo"], pl.Series("p_elo_offset", pe), pl.Series("p_margin", pm),
            pl.Series("p_blend", w * pe + (1 - w) * pm), pl.Series("pred_margin", fm["margin"]), pl.lit(float(fm["sigma"])).alias("margin_sigma"),
            pl.lit(float(fe["w"][0])).alias("offset_intercept"), pl.lit(tr.height).alias("trained_on_games"),
            *[pl.Series(f"c_{c}", contrib[:, i]) for i, c in enumerate(cols_e)], *[pl.Series(f"z_{c}", fe["z"][:, i]) for i, c in enumerate(cols_e)]))
    return pl.concat(parts, how="diagonal")


MODELS = ["home_prior", "elo", "v1_logistic", "v1_boosted_trees", "elo_offset", "margin", "blend"]
LABELS = {"home_prior": "Home-win rate only", "elo": "Elo rating", "v1_logistic": "v1 logistic regression", "v1_boosted_trees": "v1 boosted trees",
          "elo_offset": "Elo-offset logistic (v2)", "margin": "Ridge point margin (v2)", "blend": "Blend of the two v2 models"}


def block(df: pl.DataFrame) -> dict:
    y = df["home_win"].to_numpy().astype(float)
    blocks = (df["season"] * 100 + df["week"]).to_numpy()
    out = {m: M.scores(y, df[f"p_{m}"].to_numpy()) for m in MODELS}
    for m in MODELS:
        if m not in ("elo", "home_prior"):
            out[m]["vs_elo"] = M.paired(y, df[f"p_{m}"].to_numpy(), df["p_elo"].to_numpy(), blocks)
    return out


def rolling(df: pl.DataFrame, window: int = 100) -> list[dict]:
    """Rolling mean Brier and log loss over the last `window` games, in date order."""
    y = df["home_win"].to_numpy().astype(float)
    out = []
    for i in range(window, len(y) + 1, 16):
        s = slice(i - window, i)
        row = {"through": str(df["gameday"][i - 1]), "games": window}
        for m in ("elo", "elo_offset"):
            p = df[f"p_{m}"].to_numpy()[s]
            row[f"{m}_log_loss"], row[f"{m}_brier"] = float(M.ll(y[s], p).mean()), float(((p - y[s]) ** 2).mean())
        out.append(row)
    return out


def backtest(log=print) -> dict:
    t0 = time.time()
    run = Run("pregame", "v2-backtest", seed=M.SEED)
    feat = build_features()
    decided = decided_games(feat)
    violations = F.cutoff_violations(feat)
    sel = select(decided, log)
    eo = sel["chosen"]["elo_offset"]
    last = int(decided["season"].max())
    scored = frozen_predictions(decided, sel, list(range(M.DEV[0], last + 1)))
    scored = scored.join(pipeline.load_games().select(["game_id", "home_moneyline", "away_moneyline"]), on="game_id", how="left")   # reference only, never a feature
    dev, locked, later = (scored.filter(pl.col("season").is_between(*M.DEV)), scored.filter(pl.col("season").is_between(*M.LOCKED)),
                          scored.filter(pl.col("season") > M.LOCKED[1]))
    # feature-group ablations for the Elo-offset family at its chosen window and penalty, development seasons
    abl = [{"groups": r["groups"], "window": r["window"], "penalty": r["penalty"], "log_loss": r["log_loss"], "vs_elo": r["vs_elo"],
            "seasons_better_than_elo": r["seasons_better_than_elo"]}
           for r in sel["configurations"] if r["model"] == "elo_offset" and r["penalty"] == eo["penalty"] and r["window"] in (None, eo["window"])]
    mk = locked.filter(pl.col("home_moneyline").is_not_null() & pl.col("away_moneyline").is_not_null()) if "home_moneyline" in locked.columns else locked.head(0)
    perf = {
        "module": "pregame", "version": VERSION, "run_at": now_utc(), "target": "home team wins (ties excluded, playoffs included)",
        "cutoff": f"{F.CUTOFF_HOURS} hours before scheduled kickoff. A finished game counts only if it kicked off at least {F.CUTOFF_HOURS + F.GAME_HOURS} hours before the target kickoff.",
        "cutoff_violations": violations,
        "reconstructed": "Backtests are reconstructed from today's nflverse files, not from forecasts recorded before kickoff. nflverse EPA models were fitted with later data, so EPA-based features carry a small amount of hindsight.",
        "qb_assumption": "Projected quarterback = the team's primary passer in its previous game. An assumption, not confirmed starters.",
        "not_used": ["the target game's own score, box score or starter", "injuries", "weather", "betting lines", "anything from the coverage or highlights modules"],
        "selection": {k: sel[k] for k in ("development_seasons", "games", "elo_log_loss", "chosen", "blend", "blend_candidates", "rule", "note")},
        "configurations_tried": len(sel["configurations"]), "ablations": abl, "labels": LABELS,
        "features": {c: F.describe(c) for c in F.group_columns(M.GROUP_SETS[eo["groups"]], eo["window"] or F.VARIANTS[0])},
        "development": {"seasons": list(M.DEV), "models": block(dev), "by_season": {str(s): block(dev.filter(pl.col("season") == s)) for s in range(M.DEV[0], M.DEV[1] + 1)}},
        "previously_examined_benchmark": {"seasons": list(M.LOCKED), "note": "These seasons were scored during v1. The v2 choice was frozen on 2012-2022 before this comparison, but this is not a fresh test.",
                                          "models": block(locked), "by_season": {str(s): block(locked.filter(pl.col("season") == s)) for s in range(M.LOCKED[0], M.LOCKED[1] + 1)}},
        "in_progress": None if later.height == 0 else {"seasons": sorted(later["season"].unique().to_list()), "through": str(later["gameday"].max()), "models": block(later)},
        "closing_market_reference": None if mk.height == 0 else {
            "note": "De-vigged closing moneylines from the schedule file. Closing prices include information from after the 24-hour cutoff, so this is a later-information reference, not a baseline.",
            "games": mk.height, "market_log_loss": float(M.ll(mk["home_win"].to_numpy().astype(float), M.market_probability(mk)).mean()),
            "elo_log_loss_same_games": float(M.ll(mk["home_win"].to_numpy().astype(float), mk["p_elo"].to_numpy()).mean()),
            "elo_offset_log_loss_same_games": float(M.ll(mk["home_win"].to_numpy().astype(float), mk["p_elo_offset"].to_numpy()).mean())},
        "rolling_100_games": rolling(scored.sort(["gameday", "game_id"])),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "pregame_backtest_v2.json", perf)
    write_json(OUT / "pregame_selection_v2.json", sel)
    scored.write_parquet(OUT / "pregame_predictions_v2.parquet")
    export(scored, perf, sel)
    run.set(target=perf["target"], population=f"decided games {M.FIRST}-{last}", input_cutoff=perf["cutoff"], exclusions="ties",
            split={"development": list(M.DEV), "previously_examined": list(M.LOCKED), "refit": "before each season on all earlier seasons"},
            features=list(perf["features"]), hyperparameters={"chosen": sel["chosen"], "blend": sel["blend"], "elo": pipeline.ELO, "elo_clip": M.ELO_CLIP},
            metrics={"development": {m: {k: v[k] for k in ("log_loss", "brier", "accuracy")} for m, v in perf["development"]["models"].items()},
                     "previously_examined": {m: {k: v[k] for k in ("log_loss", "brier", "accuracy")} | ({"vs_elo": v["vs_elo"]} if "vs_elo" in v else {})
                                             for m, v in perf["previously_examined_benchmark"]["models"].items()}},
            ablations={"feature_groups": abl})
    run.data("games", pipeline.RAW)
    run.output("report", OUT / "pregame_backtest_v2.json")
    run.output("predictions", OUT / "pregame_predictions_v2.parquet")
    run.finish()
    log(f"backtest done in {time.time() - t0:.0f}s")
    return perf


def _game_record(g: dict, cols: list[str]) -> dict:
    terms = sorted(({"feature": c, "description": F.describe(c), "standardized": round(float(g[f"z_{c}"]), 3), "log_odds": round(float(g[f"c_{c}"]), 4)} for c in cols),
                   key=lambda t: -abs(t["log_odds"]))
    return {
        "id": g["game_id"], "season": g["season"], "week": g["week"], "type": g["game_type"], "date": str(g["gameday"]), "home": g["home_team"], "away": g["away_team"],
        "home_score": g["home_score"], "away_score": g["away_score"], "home_won": bool(g["home_win"]),
        "p": {m: round(float(g[f"p_{m}"]), 4) for m in MODELS}, "elo": [round(g["elo_home"], 1), round(g["elo_away"], 1)],
        "elo_logit": round(float(M.elo_logit(np.array([g["p_elo"]]))[0]), 4), "intercept": round(float(g["offset_intercept"]), 4), "terms": terms,
        "qb": {"home": g["home_qb_name"], "away": g["away_qb_name"], "home_dropbacks": g["home_qb_dropbacks"], "away_dropbacks": g["away_qb_dropbacks"]},
        "predicted_margin": round(float(g["pred_margin"]), 2), "trained_on_games": g["trained_on_games"],
        "missing_inputs": [n for n, v in (("home quarterback history", g["home_qb_name"]), ("away quarterback history", g["away_qb_name"])) if v is None],
    }


def export(scored: pl.DataFrame, perf: dict, sel: dict) -> None:
    eo = sel["chosen"]["elo_offset"]
    cols = F.group_columns(M.GROUP_SETS[eo["groups"]], eo["window"] or F.VARIANTS[0])
    shown = scored.filter(pl.col("season") >= M.LOCKED[0]).sort(["gameday", "game_id"])
    write_json(WEB / "v2_games.json", {"version": VERSION, "generated_at": now_utc(), "kind": "backtest (reconstructed)", "labels": LABELS,
                                       "cutoff": perf["cutoff"], "qb_assumption": perf["qb_assumption"], "reconstructed": perf["reconstructed"],
                                       "games": [_game_record(g, cols) for g in shown.iter_rows(named=True)]})
    write_json(WEB / "v2_performance.json", perf)


def forecast(log=print) -> dict:
    """Append-only records for games that have not kicked off. Uses the frozen v2 choice and data on disk at creation time."""
    sel = json.loads((OUT / "pregame_selection_v2.json").read_text())
    feat = build_features()
    decided = decided_games(feat)
    now = datetime.now(ET).replace(tzinfo=None)
    up = feat.filter(pl.col("result").is_null() & (pl.col("kickoff") > now)).sort("kickoff")
    up = up.filter(pl.col("kickoff") <= up["kickoff"].min() + timedelta(days=9)) if up.height else up      # the coming slate only
    if up.height == 0:
        return {"created": 0}
    eo, mg, w = sel["chosen"]["elo_offset"], sel["chosen"]["margin"], sel["blend"]["weight_elo_offset"]
    cols_e = F.group_columns(M.GROUP_SETS[eo["groups"]], eo["window"] or F.VARIANTS[0])
    cols_m = F.group_columns(M.GROUP_SETS[mg["groups"]], mg["window"] or F.VARIANTS[0])
    up = up.with_columns(pl.lit(0).alias("home_win"))                                                        # placeholder so the same code path runs; never read
    pe, fe = M.fit_predict(decided, up, "elo_offset", cols_e, eo["penalty"])
    pm, fm = M.fit_predict(decided, up, "margin", cols_m, mg["penalty"])
    created = now_utc()
    snapshot = {"games_csv_sha256": sha256_path(pipeline.RAW), "games_csv_modified": datetime.fromtimestamp(pipeline.RAW.stat().st_mtime, ET).isoformat(timespec="seconds"),
                "last_completed_game": str(decided["gameday"].max()), "completed_games": decided.height}
    records = []
    for i, g in enumerate(up.iter_rows(named=True)):
        cutoff = g["kickoff"] - timedelta(hours=F.CUTOFF_HOURS)
        records.append({
            "record_id": f"{g['game_id']}@{created}", "kind": "live forecast", "game_id": g["game_id"], "season": g["season"], "week": g["week"],
            "home": g["home_team"], "away": g["away_team"], "kickoff_eastern": g["kickoff"].isoformat(), "cutoff_eastern": cutoff.isoformat(),
            "created_at_utc": created, "created_before_cutoff": bool(now <= cutoff),
            "timing_note": None if now <= cutoff else "Created inside the 24 hours before kickoff, so later than the nominal cutoff. It uses only the data on disk at creation time.",
            "data_snapshot": snapshot, "model_version": VERSION, "trained_on_games": decided.height, "calibrator": "none (probabilities used as fitted)",
            "qb_assumption": {"home": g["home_qb_name"], "away": g["away_qb_name"], "basis": "primary passer in each team's previous game; not a confirmed starter"},
            "features": {c: (None if g[c] is None or np.isnan(g[c]) else round(float(g[c]), 4)) for c in ["elo_diff", *[c for c in cols_e if c in g]]},
            "probabilities": {"elo": round(float(g["p_elo"]), 4), "elo_offset": round(float(pe[i]), 4), "margin": round(float(pm[i]), 4),
                              "blend": round(float(w * pe[i] + (1 - w) * pm[i]), 4)},
            "explanation": {"unit": "log-odds of a home win", "elo_logit": round(float(M.elo_logit(np.array([g["p_elo"]]))[0]), 4), "intercept": round(float(fe["w"][0]), 4),
                            "terms": sorted(({"feature": c, "description": F.describe(c), "log_odds": round(float(fe["z"][i, j] * fe["w"][1 + j]), 4)} for j, c in enumerate(cols_e)),
                                            key=lambda t: -abs(t["log_odds"]))},
            "predicted_margin": round(float(fm["margin"][i]), 2), "outcome": None,
        })
    FORECASTS.mkdir(parents=True, exist_ok=True)
    path = FORECASTS / f"forecasts_{created.replace(':', '').replace('+0000', 'Z')}.json"
    if path.exists():
        raise SystemExit("A forecast file with this timestamp already exists; forecast records are never overwritten.")
    write_json(path, {"created_at_utc": created, "records": records})
    publish_forecasts()
    log(f"wrote {len(records)} forecast records to {config.rel(path)}")
    return {"created": len(records), "file": config.rel(path)}


def publish_forecasts() -> None:
    """Site export: every forecast record ever written, with outcomes attached beside (not inside) the original record."""
    games = {g["game_id"]: g for g in pipeline.load_games().iter_rows(named=True)}
    allrec = []
    for f in sorted(FORECASTS.glob("forecasts_*.json")):
        for r in json.loads(f.read_text())["records"]:
            g = games.get(r["game_id"])
            done = g is not None and g["result"] is not None
            allrec.append(r | {"outcome_attached": None if not done else {"home_score": g["home_score"], "away_score": g["away_score"], "home_won": g["result"] > 0,
                                                                          "tie": g["result"] == 0, "attached_at_utc": now_utc()}})
    write_json(WEB / "forecasts.json", {"generated_at": now_utc(), "note": "Live forecast records, written before kickoff and never edited. Outcomes are attached in a separate field.",
                                        "records": allrec})


def main(argv: list[str] | None = None) -> None:
    import sys
    cmd = (argv or sys.argv[1:] or ["backtest"])[0]
    if cmd == "backtest":
        p = backtest()
        for name, blk in (("development 2012-2022", p["development"]["models"]), ("previously examined 2023-2025", p["previously_examined_benchmark"]["models"])):
            print(name)
            for m, v in blk.items():
                extra = f"  vs Elo {v['vs_elo']['mean_log_loss_difference']:+.4f} {tuple(round(x, 4) for x in v['vs_elo']['interval_95'])}" if "vs_elo" in v else ""
                print(f"  {m:18s} log loss {v['log_loss']:.4f} brier {v['brier']:.4f} acc {v['accuracy']:.3f}{extra}")
    elif cmd == "forecast":
        print(forecast())
    elif cmd == "v1":
        pipeline.run()
    else:
        raise SystemExit("usage: pregame-lens backtest | forecast | v1")


if __name__ == "__main__":
    main()
