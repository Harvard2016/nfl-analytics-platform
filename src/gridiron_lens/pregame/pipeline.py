"""Pregame home-win probabilities. Independent of the coverage and highlights pipelines.

Cutoff: a game's features use only games played on an earlier calendar date. All games on one date
are built from the same state, then that date's results are applied. That is at least as strict as
a "24 hours before kickoff" cutoff for results, and it keeps same-day games from leaking into each other.

What this is not: a true point-in-time archive. The schedule file is today's nflverse file, so these
are reconstructed forecasts. Scores are final results; kickoff dates and rest days come from the
schedule, which is published before the season. Quarterback, injury and betting fields in the file
are not used because their publication time cannot be established from it.

Evaluation: features need history, so Elo warms up on 1999-2001. Train 2002-2018, development
2019-2022, locked evaluation 2023-2025 with a refit before each season on
every earlier season. Model settings were written down once, before any scoring, and have not been tuned on any split.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..shared import config
from ..shared.provenance import file_record, now_utc, write_json

SOURCE_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
RAW = config.RAW / "nflverse" / "games.csv"
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}  # one franchise key across relocations
ELO = {"k": 20.0, "home_field": 55.0, "mean": 1505.0, "offseason_keep": 2 / 3}
FORM_GAMES = 8
FEATURES = ["elo_diff", "form_diff", "win_pct_diff", "rest_diff", "home_field", "div_game"]
DESCRIPTIONS = {
    "elo_diff": "Home Elo rating minus away Elo rating before the game, plus home-field points unless the site is neutral",
    "form_diff": f"Home minus away average point margin over each team's last {FORM_GAMES} games",
    "win_pct_diff": "Home minus away win share this season before the game (0.5 each before a team's first game)",
    "rest_diff": "Home minus away days of rest, from the schedule",
    "home_field": "1 for a true home game, 0 for a neutral site",
    "div_game": "1 when the teams share a division",
}
FIRST_FEATURE_SEASON, TRAIN_END, DEV = 2002, 2018, (2019, 2022)
EVAL = (2023, 2025)
MODEL_VERSION = "pregame-v1"
SEED = 20261004


def load_games(path: Path = RAW) -> pl.DataFrame:
    g = pl.read_csv(path, null_values=["NA", ""], infer_schema_length=10000)
    return g.with_columns(pl.col("home_team").replace(FRANCHISE).alias("home"), pl.col("away_team").replace(FRANCHISE).alias("away"),
                          pl.col("gameday").str.to_date()).sort(["gameday", "game_id"])


def build_snapshots(games: pl.DataFrame) -> pl.DataFrame:
    """One row per scheduled game with past-only features. Results are joined only as the target."""
    elo: dict[str, float] = defaultdict(lambda: ELO["mean"])
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_GAMES))
    record: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # wins (ties count half), games, this season
    last_season: dict[str, int] = {}
    rows = []
    for (_day,), day_games in games.group_by(["gameday"], maintain_order=True):
        todays = list(day_games.iter_rows(named=True))
        for g in todays:  # features first, for every game on this date, from the state before the date
            for t in (g["home"], g["away"]):
                if last_season.get(t, g["season"]) != g["season"]:
                    elo[t] = ELO["mean"] + (elo[t] - ELO["mean"]) * ELO["offseason_keep"]
                    record[t] = [0, 0]
                last_season[t] = g["season"]
            h, a = g["home"], g["away"]
            hf = 0.0 if g["location"] == "Neutral" else 1.0

            def mean(q):
                return float(np.mean(q)) if q else 0.0

            def pct(r):
                return r[0] / r[1] if r[1] else 0.5

            rows.append({
                "game_id": g["game_id"], "season": g["season"], "week": g["week"], "game_type": g["game_type"],
                "gameday": g["gameday"], "home_team": g["home_team"], "away_team": g["away_team"],
                "elo_home": elo[h], "elo_away": elo[a], "elo_diff": elo[h] - elo[a] + ELO["home_field"] * hf,
                "form_diff": mean(form[h]) - mean(form[a]), "win_pct_diff": pct(record[h]) - pct(record[a]),
                "rest_diff": float(g["home_rest"] - g["away_rest"]), "home_field": hf, "div_game": float(g["div_game"]),
                "home_prior_games": len(form[h]), "away_prior_games": len(form[a]),
                "home_score": g["home_score"], "away_score": g["away_score"], "result": g["result"],
            })
        for g in todays:  # then apply the date's results
            if g["result"] is None:
                continue
            h, a, margin = g["home"], g["away"], g["result"]
            hf = 0.0 if g["location"] == "Neutral" else 1.0
            diff = elo[h] - elo[a] + ELO["home_field"] * hf
            expected = 1 / (1 + 10 ** (-diff / 400))
            actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            winner_diff = diff if margin > 0 else -diff
            mult = math.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
            delta = ELO["k"] * mult * (actual - expected)
            elo[h] += delta
            elo[a] -= delta
            form[h].append(margin)
            form[a].append(-margin)
            record[h][0] += actual
            record[h][1] += 1
            record[a][0] += 1 - actual
            record[a][1] += 1
    return pl.DataFrame(rows).with_columns((1 / (1 + 10 ** (-pl.col("elo_diff") / 400))).alias("p_elo"))


def _models() -> dict:
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
        "boosted_trees": HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=150, l2_regularization=5.0,
                                                        min_samples_leaf=40, random_state=SEED),
    }


def _scores(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    return {"games": len(y), "log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p)),
            "accuracy": float(accuracy_score(y, p >= 0.5)), "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            "home_win_rate": float(y.mean()),
            "calibration": [{"lo": float(edges[b]), "hi": float(edges[b + 1]), "games": int((idx == b).sum()),
                             "mean_predicted": float(p[idx == b].mean()), "home_won": float(y[idx == b].mean())}
                            for b in range(10) if (idx == b).any()]}


def _paired(y, p_model, p_base, blocks, n_boot=2000) -> dict:
    """Bootstrap over season-weeks of the per-game log-loss difference (model minus baseline; negative is better)."""
    ll = lambda p: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    d = ll(p_model) - ll(p_base)
    ub = np.unique(blocks)
    by = {b: d[blocks == b] for b in ub}
    rng = np.random.default_rng(SEED)
    means = [np.concatenate([by[b] for b in rng.choice(ub, len(ub))]).mean() for _ in range(n_boot)]
    return {"mean_log_loss_difference": float(d.mean()), "interval_95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            "blocks": len(ub), "block": "season-week"}


def run(raw: Path = RAW, out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir or config.WEB_DEMO / "pregame")
    out_dir.mkdir(parents=True, exist_ok=True)
    games = load_games(raw)
    snap = build_snapshots(games)
    (config.FEATURES / "pregame").mkdir(parents=True, exist_ok=True)
    snap.write_parquet(config.FEATURES / "pregame" / "snapshots.parquet")
    played = snap.filter(pl.col("result").is_not_null() & (pl.col("season") >= FIRST_FEATURE_SEASON))
    ties = played.filter(pl.col("result") == 0)
    decided = played.filter(pl.col("result") != 0).with_columns((pl.col("result") > 0).cast(pl.Int8).alias("home_win"))

    def xy(df):
        return df.select(FEATURES).to_pandas(), df["home_win"].to_numpy()

    def fit_predict(train_df, test_df) -> dict:
        xtr, ytr = xy(train_df)
        xte, _ = xy(test_df)
        prior = float(train_df.filter(pl.col("home_field") == 1)["home_win"].mean())
        preds = {"home_prior": np.where(test_df["home_field"].to_numpy() == 1, prior, 0.5), "elo": test_df["p_elo"].to_numpy()}
        fitted = {}
        for name, est in _models().items():
            est.fit(xtr, ytr)
            preds[name] = est.predict_proba(xte)[:, 1]
            fitted[name] = est
        return {"preds": preds, "fitted": fitted, "prior": prior}

    # development: fit once on 2002-2018, score 2019-2022
    dev = decided.filter(pl.col("season").is_between(*DEV))
    dev_fit = fit_predict(decided.filter(pl.col("season") <= TRAIN_END), dev)
    # locked evaluation: refit before each season on every earlier season (settings fixed above)
    parts, explain = [], {}
    seasons = list(range(EVAL[0], EVAL[1] + 1)) + [s for s in sorted(decided["season"].unique().to_list()) if s > EVAL[1]]
    for season in seasons:
        te = decided.filter(pl.col("season") == season)
        if te.height == 0:
            continue
        fp = fit_predict(decided.filter(pl.col("season") < season), te)
        parts.append(te.with_columns(*[pl.Series(f"p_{k}", v) for k, v in fp["preds"].items() if k != "elo"]))
        pipe = fp["fitted"]["logistic"]
        explain[season] = {"mean": pipe[0].mean_.tolist(), "scale": pipe[0].scale_.tolist(), "coef": pipe[1].coef_[0].tolist(),
                           "intercept": float(pipe[1].intercept_[0]), "trained_on_games": decided.filter(pl.col("season") < season).height,
                           "home_prior": fp["prior"]}
    scored = pl.concat(parts)
    names = ["home_prior", "elo", "logistic", "boosted_trees"]

    def block(df: pl.DataFrame) -> dict:
        y = df["home_win"].to_numpy()
        blocks = (df["season"] * 100 + df["week"]).to_numpy()
        out = {m: _scores(y, df[f"p_{m}"].to_numpy()) for m in names}
        for m in ("logistic", "boosted_trees"):
            out[m]["vs_elo"] = _paired(y, df[f"p_{m}"].to_numpy(), df["p_elo"].to_numpy(), blocks)
        return out

    locked = scored.filter(pl.col("season").is_between(*EVAL))
    later = scored.filter(pl.col("season") > EVAL[1])
    y_dev = dev["home_win"].to_numpy()
    performance = {
        "module": "pregame", "version": MODEL_VERSION, "run_at": now_utc(), "source": SOURCE_URL,
        "source_file": file_record(raw), "target": "home team wins (ties excluded)",
        "cutoff": "Only games played on an earlier calendar date feed a game's features.",
        "reconstructed": "Reconstructed from today's nflverse schedule file, not from forecasts recorded before kickoff.",
        "not_used": ["quarterback fields", "injuries", "betting lines (publication time not established)", "anything from the coverage or highlights modules"],
        "elo": ELO, "features": FEATURES, "feature_descriptions": DESCRIPTIONS,
        "games": {"completed_since_2002": played.height, "ties_excluded": ties.height,
                  "ties": [{"game_id": i, "season": s} for i, s in ties.select(["game_id", "season"]).iter_rows()],
                  "elo_warm_up_seasons": [1999, FIRST_FEATURE_SEASON - 1], "train_seasons": [FIRST_FEATURE_SEASON, TRAIN_END],
                  "train_games": decided.filter(pl.col("season") <= TRAIN_END).height},
        "development": {"seasons": list(DEV), "note": "Fit once on 2002-2018. Nothing was tuned on these seasons; they are a second look at the same fixed settings.",
                        "models": {m: _scores(y_dev, np.asarray(dev_fit["preds"][m])) for m in names}},
        "locked": {"seasons": list(EVAL), "refit": "Before each season, refit on every earlier season since 2002. Settings fixed.",
                   "models": block(locked), "by_season": {str(s): block(locked.filter(pl.col("season") == s)) for s in range(EVAL[0], EVAL[1] + 1)},
                   "regular_season_only": block(locked.filter(pl.col("game_type") == "REG")),
                   "playoffs_only": block(locked.filter(pl.col("game_type") != "REG"))},
        "in_progress": None if later.height == 0 else {
            "seasons": sorted(later["season"].unique().to_list()), "through": str(later["gameday"].max()),
            "note": "Season still being played. Reconstructed like the rest; too few games for a firm reading.",
            "models": block(later)},
    }
    write_json(out_dir / "performance.json", performance)
    write_json(config.REPORTS / "experiments" / "pregame_backtest.json", performance)

    index = []
    for g in scored.sort(["gameday", "game_id"]).iter_rows(named=True):
        e = explain[g["season"]]
        z = [(g[f] - e["mean"][i]) / e["scale"][i] for i, f in enumerate(FEATURES)]
        index.append({
            "id": g["game_id"], "season": g["season"], "week": g["week"], "type": g["game_type"], "date": str(g["gameday"]),
            "home": g["home_team"], "away": g["away_team"], "home_score": g["home_score"], "away_score": g["away_score"],
            "home_won": bool(g["home_win"]), "locked": EVAL[0] <= g["season"] <= EVAL[1],
            "p": {m: round(float(g[f"p_{m}"]), 4) for m in names},
            "elo": [round(g["elo_home"], 1), round(g["elo_away"], 1)], "prior_games": [g["home_prior_games"], g["away_prior_games"]],
            "features": {f: round(float(g[f]), 3) for f in FEATURES},
            "contributions": {f: round(float(e["coef"][i] * z[i]), 4) for i, f in enumerate(FEATURES)},
            "intercept": round(e["intercept"], 4), "trained_on_games": e["trained_on_games"],
        })
    write_json(out_dir / "games.json", {"version": MODEL_VERSION, "generated_at": now_utc(), "feature_descriptions": DESCRIPTIONS,
                                         "cutoff": performance["cutoff"], "reconstructed": performance["reconstructed"], "games": index})
    return performance
