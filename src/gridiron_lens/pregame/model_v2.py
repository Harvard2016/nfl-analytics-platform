"""Pregame models, version 2: does anything beat plain Elo once selection is done honestly?

Models on the same games: home-win rate, Elo, v1 logistic and boosted trees, a ridge-penalized Elo-offset logistic
model, an interpretable ridge point-margin model, and at most one two-model blend.

  Elo offset:   logit P(home win) = logit(P_elo) + b0 + b . z        (z = standardized features, ridge penalty on b)
  Point margin: margin = ridge(z); P(home win) = Phi(margin / sigma), sigma = sd of training residuals

Selection uses walk-forward backtests on development seasons 2012-2022 only (each season predicted by a model fitted
on every earlier season since 2002). The 2023-2025 seasons were examined in v1; scoring the frozen v2 choice there is a
comparison on a previously examined benchmark. Ties are excluded; playoffs are included, as in v1.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features_v2 as F
from . import pipeline

FIRST, DEV, LOCKED = 2002, (2012, 2022), (2023, 2025)
ELO_CLIP = (0.02, 0.98)            # numerical stability of logit(P_elo) only
LAMBDAS = [0.01, 0.1, 1.0]
ALPHAS = [10.0, 100.0, 1000.0]
GROUP_SETS = {
    "rest": ["rest"], "form": ["form"], "efficiency": ["efficiency"], "rates": ["rates"], "qb": ["qb"],
    "efficiency+qb": ["efficiency", "qb"], "rest+efficiency+rates+qb": ["rest", "efficiency", "rates", "qb"],
    "all": ["rest", "form", "efficiency", "rates", "qb"],
}
SEED = 20261004


def logit(p):
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def add_qb(df: pl.DataFrame, train: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """Shrunk quarterback form. The league prior is the dropback-weighted mean over training rows only."""
    n = float(train["home_qb_dropbacks"].sum() + train["away_qb_dropbacks"].sum())
    mu = float(train["home_qb_epa_sum"].sum() + train["away_qb_epa_sum"].sum()) / max(n, 1.0)
    k = F.QB_PRIOR_DROPBACKS
    sh = lambda s: (pl.col(f"{s}_qb_epa_sum") + k * mu) / (pl.col(f"{s}_qb_dropbacks") + k)
    return df.with_columns((sh("home") - sh("away")).alias("d_qb_epa_shrunk"),
                           (pl.col("home_qb_dropbacks").log1p() - pl.col("away_qb_dropbacks").log1p()).alias("d_qb_log_dropbacks")), mu


class Scaler:
    """Training-only mean and sd. Missing values become 0 after scaling, which is the training mean."""

    def __init__(self, x: np.ndarray):
        self.mean, self.sd = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
        self.sd[self.sd == 0] = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.nan_to_num((x - self.mean) / self.sd, nan=0.0)


def fit_offset(z: np.ndarray, y: np.ndarray, base_logit: np.ndarray, lam: float) -> np.ndarray:
    """Ridge logistic regression with a fixed Elo offset. Returns [b0, b...]. The intercept is not penalized."""
    def f(w):
        p = sigmoid(base_logit + w[0] + z @ w[1:])
        g = p - y
        return ll(y, p).mean() + lam * (w[1:] @ w[1:]), np.concatenate([[g.mean()], z.T @ g / len(y) + 2 * lam * w[1:]])
    return minimize(f, np.zeros(z.shape[1] + 1), jac=True, method="L-BFGS-B").x


def elo_logit(p_elo: np.ndarray) -> np.ndarray:
    return logit(np.clip(p_elo, *ELO_CLIP))


def fit_predict(train: pl.DataFrame, test: pl.DataFrame, kind: str, cols: list[str], pen: float) -> tuple[np.ndarray, dict]:
    """One model fitted on `train` and applied to `test`. Returns probabilities and the fitted pieces needed to explain them."""
    train, mu = add_qb(train, train)
    test, _ = add_qb(test, train)
    y = train["home_win"].to_numpy().astype(float)
    if kind == "elo_offset":
        sc = Scaler(train.select(cols).to_numpy())
        w = fit_offset(sc(train.select(cols).to_numpy()), y, elo_logit(train["p_elo"].to_numpy()), pen)
        z = sc(test.select(cols).to_numpy())
        return sigmoid(elo_logit(test["p_elo"].to_numpy()) + w[0] + z @ w[1:]), {"scaler": sc, "w": w, "qb_prior": mu, "z": z}
    if kind == "margin":
        mc = ["elo_diff", *cols]
        sc = Scaler(train.select(mc).to_numpy())
        ztr = sc(train.select(mc).to_numpy())
        m = Ridge(alpha=pen).fit(ztr, train["result"].to_numpy().astype(float))
        sigma = float(np.std(train["result"].to_numpy() - m.predict(ztr)))
        z = sc(test.select(mc).to_numpy())
        pred = m.predict(z)
        return norm.cdf(pred / sigma), {"scaler": sc, "coef": m.coef_, "intercept": float(m.intercept_), "sigma": sigma, "margin": pred, "z": z, "cols": mc}
    raise ValueError(kind)


def v1_models(train: pl.DataFrame, test: pl.DataFrame) -> dict[str, np.ndarray]:
    xtr, ytr, xte = train.select(pipeline.FEATURES).to_pandas(), train["home_win"].to_numpy(), test.select(pipeline.FEATURES).to_pandas()
    prior = float(train.filter(pl.col("home_field") == 1)["home_win"].mean())
    out = {"home_prior": np.where(test["home_field"].to_numpy() == 1, prior, 0.5), "elo": test["p_elo"].to_numpy()}
    out["v1_logistic"] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(xtr, ytr).predict_proba(xte)[:, 1]
    out["v1_boosted_trees"] = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=150, l2_regularization=5.0, min_samples_leaf=40,
                                                             random_state=pipeline.SEED).fit(xtr, ytr).predict_proba(xte)[:, 1]
    return out


def walk_forward(decided: pl.DataFrame, seasons: range, kind: str, cols: list[str], pen: float) -> np.ndarray:
    """Out-of-sample probabilities for every game in `seasons`, each season fitted on all earlier seasons. Order follows `decided` filtered to seasons."""
    out = []
    for s in seasons:
        te = decided.filter(pl.col("season") == s)
        if te.height:
            out.append(fit_predict(decided.filter(pl.col("season") < s), te, kind, cols, pen)[0])
    return np.concatenate(out)


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    lg = logit(p).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(lg, y)
    return {"games": len(y), "log_loss": float(ll(y, p).mean()), "brier": float(((p - y) ** 2).mean()), "accuracy": float(((p >= 0.5) == y).mean()),
            "home_win_rate": float(y.mean()), "calibration_slope": float(cal.coef_[0, 0]), "calibration_intercept": float(cal.intercept_[0]),
            "calibration": [{"lo": float(edges[b]), "hi": float(edges[b + 1]), "games": int((idx == b).sum()), "mean_predicted": float(p[idx == b].mean()),
                             "home_won": float(y[idx == b].mean())} for b in range(10) if (idx == b).any()]}


def paired(y: np.ndarray, p_model: np.ndarray, p_base: np.ndarray, blocks: np.ndarray, n_boot: int = 2000) -> dict:
    """Per-game log-loss difference (model minus baseline; negative favours the model), bootstrap over season-weeks."""
    d = ll(y, p_model) - ll(y, p_base)
    ub = np.unique(blocks)
    by = {b: d[blocks == b] for b in ub}
    rng = np.random.default_rng(SEED)
    means = [np.concatenate([by[b] for b in rng.choice(ub, len(ub))]).mean() for _ in range(n_boot)]
    return {"mean_log_loss_difference": float(d.mean()), "interval_95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            "share_of_resamples_better": float(np.mean(np.array(means) < 0)), "blocks": len(ub), "block": "season-week"}


def market_probability(df: pl.DataFrame) -> np.ndarray:
    """De-vigged home probability from the moneylines in the schedule file. These are closing prices: later information than a 24-hour cutoff."""
    def dec(ml):
        ml = ml.astype(float)
        return np.where(ml > 0, 1 + ml / 100, 1 + 100 / np.abs(ml))
    h, a = 1 / dec(df["home_moneyline"].to_numpy()), 1 / dec(df["away_moneyline"].to_numpy())
    return h / (h + a)
