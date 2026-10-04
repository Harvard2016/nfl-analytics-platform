"""Game prediction v3 experiments P1-P4. Registered in docs/experiments/pregame_v3.md before the first run.

Opponent-adjusted team ratings are estimated for each target kickoff from games that kicked off at least 28 hours earlier:
a weighted ridge regression of each offence's per-play value in a game on (offence, defence, home) indicators. Weights decay
with the number of games each team has played since, and games from the previous season carry a further discount. Nothing
from the target game or later is read.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from ..shared import config
from ..shared.provenance import now_utc, write_json
from ..shared.runs import Run
from . import features_v2 as F
from . import model_v2 as M
from . import pipeline

OUT = config.REPORTS / "v3"
RIDGE_GAMES, PRIOR_SEASON_WEIGHT, TO_PRIOR_PLAYS = 8.0, 0.35, 400.0
HALF_LIVES = (6, 12)
NEST_FROM = 2015


def adjusted_ratings(games: pl.DataFrame, team: pl.DataFrame, value: str, half_life: float) -> pl.DataFrame:
    """Per target game: home/away opponent-adjusted offence and defence ratings from prior games only."""
    g = F.kickoff(games).sort(["kickoff", "game_id"])
    stat = {(r["game_id"], r["posteam"]): r for r in team.iter_rows(named=True)}
    obs = []                                                    # (kickoff, season, offence, defence, home flag, value, plays)
    for r in g.filter(pl.col("result").is_not_null()).iter_rows(named=True):
        for off, de, home in ((r["home"], r["away"], 1.0), (r["away"], r["home"], 0.0)):
            if value == "points":
                v, n = float(r["home_score"] if home else r["away_score"]), 1.0
            else:
                s = stat.get((r["game_id"], off))
                if s is None or s[value] is None:
                    continue
                v, n = float(s[value]), float(s["plays"])
            obs.append((r["kickoff"], r["season"], off, de, home if r["location"] != "Neutral" else 0.5, v, n))
    by_season = defaultdict(list)
    for o in obs:
        by_season[o[1]].append(o)
    rows, need = [], timedelta(hours=F.CUTOFF_HOURS + F.GAME_HOURS)
    for kick, grp in g.group_by("kickoff", maintain_order=True):
        k = kick[0]
        season = grp["season"][0]
        cand = [o for s in (season - 1, season) for o in by_season.get(s, []) if o[0] <= k - need]
        teams = sorted({t for o in cand for t in (o[2], o[3])})
        rating = {}
        if len(cand) >= 8:
            idx = {t: i for i, t in enumerate(teams)}
            nt = len(teams)
            played = defaultdict(int)
            age = []
            for o in reversed(cand):                           # games each offence has played since this one
                age.append(played[o[2]])
                played[o[2]] += 1
            age = np.array(age[::-1], float)
            w = 0.5 ** (age / half_life) * np.array([PRIOR_SEASON_WEIGHT if o[1] < season else 1.0 for o in cand])
            if value != "points":
                w = w * np.array([o[6] for o in cand]) / 60.0
            X = np.zeros((len(cand), 2 * nt + 2))
            y = np.array([o[5] for o in cand])
            for i, o in enumerate(cand):
                X[i, idx[o[2]]], X[i, nt + idx[o[3]]], X[i, 2 * nt], X[i, 2 * nt + 1] = 1.0, 1.0, o[4], 1.0
            pen = np.r_[np.full(2 * nt, RIDGE_GAMES), 0.0, 0.0]                      # team effects shrunk toward the league mean; home and intercept free
            beta = np.linalg.solve((X * w[:, None]).T @ X + np.diag(pen), (X * w[:, None]).T @ y)
            rating = {t: (beta[idx[t]], beta[nt + idx[t]]) for t in teams}
        for r in grp.iter_rows(named=True):
            h, a = rating.get(r["home"], (0.0, 0.0)), rating.get(r["away"], (0.0, 0.0))
            rows.append({"game_id": r["game_id"], "d_adj_off": h[0] - a[0], "d_adj_def": h[1] - a[1], "d_adj_net": (h[0] - a[1]) - (a[0] - h[1])})
    return pl.DataFrame(rows)


def shrunk_turnovers(games: pl.DataFrame, team: pl.DataFrame) -> pl.DataFrame:
    """Season-to-date giveaway and takeaway rates shrunk toward the league rate of earlier games (previous season at a discount)."""
    g = F.kickoff(games).sort(["kickoff", "game_id"])
    stat = {(r["game_id"], r["posteam"]): r for r in team.iter_rows(named=True)}
    give, take, plays_o, plays_d = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    lg_to, lg_pl, season_now, rows = 0.0, 0.0, None, []
    for (_d,), day in g.group_by(["gameday"], maintain_order=True):
        todays = list(day.iter_rows(named=True))
        if todays[0]["season"] != season_now:                  # offseason: keep a fraction of last season's evidence
            season_now = todays[0]["season"]
            for d in (give, take, plays_o, plays_d):
                for k in d:
                    d[k] *= PRIOR_SEASON_WEIGHT
        league = lg_to / lg_pl if lg_pl else 0.025
        rate = lambda num, den, t, league=league: (num[t] + TO_PRIOR_PLAYS * league) / (den[t] + TO_PRIOR_PLAYS)
        for r in todays:
            rows.append({"game_id": r["game_id"], "d_giveaway_shrunk": rate(give, plays_o, r["home"]) - rate(give, plays_o, r["away"]),
                         "d_takeaway_shrunk": rate(take, plays_d, r["home"]) - rate(take, plays_d, r["away"])})
        for r in todays:
            if r["result"] is None:
                continue
            for t, o in ((r["home"], r["away"]), (r["away"], r["home"])):
                s = stat.get((r["game_id"], t))
                if s is None:
                    continue
                to = s["turnover_rate"] * s["plays"]
                give[t] += to
                plays_o[t] += s["plays"]
                take[o] += to
                plays_d[o] += s["plays"]
                lg_to += to
                lg_pl += s["plays"]
    return pl.DataFrame(rows)


def nested_pick(scores: dict[str, dict[int, np.ndarray]], y: dict[int, np.ndarray], seasons: list[int]) -> tuple[dict[int, str], np.ndarray]:
    """For each season, the configuration with the lowest pooled log loss on earlier development seasons."""
    pick, out = {}, []
    for s in seasons:
        prior = [q for q in y if q < s]
        best = min(scores, key=lambda k: np.concatenate([M.ll(y[q], scores[k][q]) for q in prior]).mean())
        pick[s] = best
        out.append(scores[best][s])
    return pick, np.concatenate(out)


def run(log=print) -> dict:
    from . import forecast_v3
    snap = forecast_v3.latest_snapshot()
    sdir = forecast_v3.SNAPSHOTS / snap["snapshot_id"] if snap else None
    games = pipeline.load_games(sdir / "games.csv" if sdir else pipeline.RAW)
    team, qb = F.load_stats(sdir / "pbp" if sdir else F.PBP_DIR)
    feat = F.build(games, team, qb)
    extra = None
    for hl in HALF_LIVES:
        for value, tag in (("off_epa", "epa"), ("points", "pts")):
            r = adjusted_ratings(games, team, value, hl).rename({c: f"{c}_{tag}_h{hl}" for c in ("d_adj_off", "d_adj_def", "d_adj_net")})
            extra = r if extra is None else extra.join(r, on="game_id")
        log(f"ratings built for half-life {hl}")
    feat = feat.join(extra, on="game_id", how="left").join(shrunk_turnovers(games, team), on="game_id", how="left")
    decided = (feat.filter(pl.col("result").is_not_null() & (pl.col("result") != 0) & (pl.col("season") >= M.FIRST))
               .with_columns((pl.col("result") > 0).cast(pl.Int8).alias("home_win")).sort(["gameday", "game_id"]))
    sel = json.loads((config.REPORTS / "v2" / "pregame_selection_v2.json").read_text())
    eo, mg = sel["chosen"]["elo_offset"], sel["chosen"]["margin"]
    v2e, v2m = F.group_columns(M.GROUP_SETS[eo["groups"]], eo["window"] or F.VARIANTS[0]), F.group_columns(M.GROUP_SETS[mg["groups"]], mg["window"] or F.VARIANTS[0])
    all_seasons = range(M.DEV[0], M.LOCKED[1] + 1)
    scope = decided.filter(pl.col("season").is_between(M.DEV[0], M.LOCKED[1]))
    season, y_all = scope["season"].to_numpy(), scope["home_win"].to_numpy().astype(float)
    blocks = np.array([f"{s}-{w}" for s, w in zip(scope["season"].to_list(), scope["week"].to_list())])
    P = {"elo": scope["p_elo"].to_numpy(), "v2 elo-offset (frozen)": M.walk_forward(decided, all_seasons, "elo_offset", v2e, eo["penalty"]),
         "v2 margin (frozen)": M.walk_forward(decided, all_seasons, "margin", v2m, mg["penalty"])}
    sets = {}
    for hl in HALF_LIVES:
        sets[f"P1 adjusted EPA h{hl}"] = [f"d_adj_off_epa_h{hl}", f"d_adj_def_epa_h{hl}"]
        sets[f"P2 adjusted points h{hl}"] = [f"d_adj_off_pts_h{hl}", f"d_adj_def_pts_h{hl}"]
    sets["P3 adjusted EPA h12 + shrunk turnovers"] = ["d_adj_off_epa_h12", "d_adj_def_epa_h12", "d_giveaway_shrunk", "d_takeaway_shrunk"]
    cand = {}
    for name, cols in sets.items():
        for pen in M.LAMBDAS:
            cand[f"{name} | penalty {pen}"] = M.walk_forward(decided, all_seasons, "elo_offset", cols + ["d_qb_epa_shrunk", "d_qb_log_dropbacks"], pen)
    for pen in (10.0, 100.0):
        cand[f"P4 margin on adjusted EPA h12 | alpha {pen}"] = M.walk_forward(decided, all_seasons, "margin", sets["P1 adjusted EPA h12"] + ["d_qb_epa_shrunk", "d_qb_log_dropbacks"], pen)
    log(f"{len(cand)} v3 configurations scored walk-forward")

    dev = (season >= M.DEV[0]) & (season <= M.DEV[1])
    by = lambda p: {int(s): p[season == s] for s in np.unique(season)}
    ys = by(y_all)
    nest_seasons = list(range(NEST_FROM, M.DEV[1] + 1))
    pick, p_nested = nested_pick({k: {s: v for s, v in by(p).items() if s <= M.DEV[1]} for k, p in cand.items()}, {s: v for s, v in ys.items() if s <= M.DEV[1]}, nest_seasons)
    nm = (season >= NEST_FROM) & (season <= M.DEV[1])

    def platt_prior(p: np.ndarray) -> np.ndarray:
        """Calibrate each season with a Platt map fitted on that model's forecasts for earlier development seasons only."""
        out = p.copy()
        for s in range(NEST_FROM, M.LOCKED[1] + 1):
            tr = (season >= M.DEV[0]) & (season < s)
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(M.logit(np.clip(p[tr], 1e-6, 1 - 1e-6)).reshape(-1, 1), y_all[tr])
            out[season == s] = lr.predict_proba(M.logit(np.clip(p[season == s], 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]
        return out

    def block(mask: np.ndarray, p: np.ndarray, base: dict[str, np.ndarray]) -> dict:
        sc = M.scores(y_all[mask], p[mask])
        return {k: sc[k] for k in ("games", "log_loss", "brier", "accuracy", "calibration_slope", "calibration_intercept")} | {
            f"vs_{b}": M.paired(y_all[mask], p[mask], q[mask], blocks[mask]) for b, q in base.items()}

    base = {"elo": P["elo"], "v2_elo_offset": P["v2 elo-offset (frozen)"]}
    final_key = min(cand, key=lambda k: M.ll(y_all[dev], cand[k][dev]).mean())               # single frozen choice from all development seasons
    locked = season >= M.LOCKED[0]
    rep = {
        "module": "pregame", "version": "pregame-v3-experiments", "created_at_utc": now_utc(), "registered_in": "docs/experiments/pregame_v3.md",
        "source_snapshot": snap and {k: snap[k] for k in ("snapshot_id", "content_sha256")}, "configurations_scored": len(cand),
        "reconstructed": "Backtests from today's nflverse files. EPA values come from models fitted with later data (hindsight); P2 uses points only as a check.",
        "development_all_configurations": {k: {"log_loss": float(M.ll(y_all[dev], p[dev]).mean()), "vs_elo": float(M.ll(y_all[dev], p[dev]).mean() - M.ll(y_all[dev], P["elo"][dev]).mean())}
                                           for k, p in sorted(cand.items(), key=lambda kv: M.ll(y_all[dev], kv[1][dev]).mean())},
        "development_controls": {k: {"log_loss": float(M.ll(y_all[dev], p[dev]).mean())} for k, p in P.items()},
        "nested_development": {"seasons": [NEST_FROM, M.DEV[1]], "picked_per_season": {str(s): k for s, k in pick.items()},
                               "v3 nested": block(nm, np.where(nm, np.r_[np.zeros((season < NEST_FROM).sum()), p_nested, np.zeros((season > M.DEV[1]).sum())], 0.0), base),
                               "controls": {k: block(nm, p, {"elo": P["elo"]} if k != "elo" else {}) for k, p in P.items()},
                               "note": "Each season's configuration was chosen using earlier development seasons only. Still optimistic as a guide to future seasons: the experiment list itself was designed after seeing 2012-2022."},
        "calibration_from_prior_out_of_fold": {k: {"nested_seasons_raw": float(M.ll(y_all[nm], p[nm]).mean()), "nested_seasons_platt": float(M.ll(y_all[nm], platt_prior(p)[nm]).mean())}
                                               for k, p in {**P, "v3 final choice": cand[final_key]}.items()},
        "frozen_v3_choice": {"key": final_key, "rule": "lowest pooled development log loss (2012-2022)"},
        "previously_examined_benchmark": {"seasons": list(M.LOCKED), "note": "2023-2025 were examined in v1 and v2. One frozen v3 configuration is scored here once; this is not a fresh test.",
                                          "v3 final choice": block(locked, cand[final_key], base), "controls": {k: block(locked, p, {"elo": P["elo"]} if k != "elo" else {}) for k, p in P.items()}},
        "per_season_vs_elo": {k: {str(s): float(M.ll(ys[s], by(p)[s]).mean() - M.ll(ys[s], by(P["elo"])[s]).mean()) for s in sorted(ys)} for k, p in {"v2 elo-offset (frozen)": P["v2 elo-offset (frozen)"], "v3 final choice": cand[final_key]}.items()},
        "not_run": {"quarterback scenarios from timestamped depth charts": "nflverse documents timestamped depth-chart snapshots from 2025 on, so they cannot be tested on development seasons 2012-2022.",
                    "market comparison": "No price series with timestamps before the 24-hour cutoff is on disk; closing prices are later information."},
    }
    n = rep["nested_development"]["v3 nested"]
    promoted = bool(n["vs_elo"]["interval_95"][1] < 0 and n["vs_v2_elo_offset"]["interval_95"][1] < 0)
    b = rep["previously_examined_benchmark"]["v3 final choice"]["vs_elo"]["interval_95"]
    parts = [f"On nested development seasons the v3 choice is {'better than Elo with an interval excluding zero' if n['vs_elo']['interval_95'][1] < 0 else 'not distinguishable from Elo'}",
             f"and {'better than' if n['vs_v2_elo_offset']['interval_95'][1] < 0 else 'not distinguishable from'} the frozen v2 model.",
             f"On the previously examined 2023-2025 seasons its interval against Elo {'excludes' if b[1] < 0 else 'includes'} zero."]
    rep["decision"] = {"promote_v3_to_forecasts": promoted, "statement": " ".join(parts) + (" It replaces v2 in forecasts." if promoted else " Not promoted: forecasting stays on Elo and the frozen v2 model.")}
    write_json(OUT / "pregame_experiments_v3.json", rep)
    run_ = Run("pregame", "v3-experiments", "v3", seed=M.SEED)
    run_.set(target="home team wins", population="decided games 2012-2025", input_cutoff="24 hours before kickoff; prior games at least 28 hours earlier",
             split={"development": list(M.DEV), "nested_from": NEST_FROM, "previously_examined": list(M.LOCKED)}, metrics={"nested": n, "decision": rep["decision"]})
    run_.output("report", OUT / "pregame_experiments_v3.json")
    run_.finish()
    log(json.dumps({"nested v3": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in n.items() if k in ("games", "log_loss", "brier", "accuracy")},
                    "vs elo": n["vs_elo"], "vs v2": n["vs_v2_elo_offset"], "controls": {k: round(v["log_loss"], 4) for k, v in rep["nested_development"]["controls"].items()},
                    "final": final_key, "benchmark": {k: round(v, 4) for k, v in rep["previously_examined_benchmark"]["v3 final choice"].items() if isinstance(v, float)},
                    "benchmark_vs_elo": rep["previously_examined_benchmark"]["v3 final choice"]["vs_elo"], "decision": rep["decision"]}, indent=1))
    return rep


if __name__ == "__main__":
    run()
