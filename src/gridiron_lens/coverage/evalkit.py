"""Shared evaluation helpers for coverage experiments: metrics, policies, abstention sweeps, paired game bootstrap.

Classification threshold and abstention cutoff are separate things here and stay separate everywhere:
- the class policy turns a calibrated man probability into a lean (man or zone);
- the uncertainty policy decides whether that lean is confident enough to count.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (p >= threshold).astype(int)
    pr, rc, _, _sup = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
    return {
        "n": len(y), "man": int(y.sum()), "zone": int((1 - y).sum()), "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "man_precision": float(pr[0]), "man_recall": float(rc[0]), "zone_precision": float(pr[1]), "zone_recall": float(rc[1]),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])), "brier": float(brier_score_loss(y, p)),
        "confusion": {"order": "man->man, man->zone, zone->man, zone->zone", "values": confusion_matrix(y, pred, labels=[1, 0]).ravel().tolist()},
    }


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return [{"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int((idx == b).sum()),
             "mean_predicted": float(p[idx == b].mean()), "observed_man_rate": float(y[idx == b].mean())}
            for b in range(bins) if (idx == b).any()]


def balanced_policy(y: np.ndarray, p: np.ndarray, min_man_precision: float = 0.80) -> dict:
    """Man threshold that maximizes balanced accuracy on calibration data subject to man precision >= the target.

    If no threshold reaches the precision target the standard 0.5 rule is kept and that is reported.
    """
    best = None
    for t in np.round(np.arange(0.20, 0.71, 0.01), 2):
        m = binary_metrics(y, p, t)
        if m["man_precision"] >= min_man_precision and (best is None or m["balanced_accuracy"] > best["balanced_accuracy"]):
            best = m
    if best is None:
        return {"supported": False, "threshold": 0.5, "note": f"No threshold between 0.20 and 0.70 reached man precision {min_man_precision} on the calibration weeks; the standard 0.5 rule is kept."}
    return {"supported": True, "threshold": best["threshold"], "calibration_balanced_accuracy": best["balanced_accuracy"],
            "calibration_man_precision": best["man_precision"], "calibration_man_recall": best["man_recall"],
            "rule": f"maximize balanced accuracy on weeks 13-14 subject to man precision >= {min_man_precision}"}


def abstention_sweep(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> list[dict]:
    """Confidence = max(p, 1 - p) of the calibrated probability. Denominators are stated in the field names."""
    pred, conf = (p >= threshold).astype(int), np.maximum(p, 1 - p)
    out = []
    for c in np.round(np.arange(0.50, 0.96, 0.05), 2):
        a = conf >= c
        out.append({
            "cutoff": float(c), "accepted": int(a.sum()), "accepted_fraction_of_all": float(a.mean()),
            "accepted_accuracy": float((pred[a] == y[a]).mean()) if a.any() else None,
            "man_accepted_fraction_of_man": float(a[y == 1].mean()), "zone_accepted_fraction_of_zone": float(a[y == 0].mean()),
            "man_correct_accepted_over_all_man": float(((pred == 1) & (y == 1) & a).sum() / max(1, (y == 1).sum())),
            "zone_correct_accepted_over_all_zone": float(((pred == 0) & (y == 0) & a).sum() / max(1, (y == 0).sum())),
        })
    return out


def pick_cutoff(sweep: list[dict], target: float = 0.90, min_fraction: float = 0.20) -> dict:
    """Same rule as v1: lowest cutoff whose accepted accuracy on calibration weeks reaches the target."""
    for row in sweep:
        if row["accepted_accuracy"] is not None and row["accepted_accuracy"] >= target and row["accepted_fraction_of_all"] >= min_fraction:
            return {"cutoff": row["cutoff"], "rule": f"lowest cutoff with calibration-week accepted accuracy >= {target} and >= {min_fraction:.0%} accepted"}
    return {"cutoff": 0.5, "rule": "no cutoff met the target on calibration weeks; nothing is abstained"}


def game_bootstrap(y: np.ndarray, p: np.ndarray, games: np.ndarray, n_boot: int = 1000, seed: int = 20261004) -> dict:
    """95% intervals from resampling whole games."""
    rng = np.random.default_rng(seed)
    ug = np.unique(games)
    by = {g: np.where(games == g)[0] for g in ug}
    stats = {k: [] for k in ("accuracy", "macro_f1", "log_loss", "man_recall")}
    for _ in range(n_boot):
        ix = np.concatenate([by[g] for g in rng.choice(ug, len(ug))])
        m = binary_metrics(y[ix], p[ix])
        for k in stats:
            stats[k].append(m[k])
    return {"games": len(ug), "resamples": n_boot, **{k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in stats.items()}}


def paired_game_bootstrap(y: np.ndarray, p_new: np.ndarray, p_old: np.ndarray, games: np.ndarray, n_boot: int = 2000, seed: int = 20261004) -> dict:
    """Differences (new minus old) on the same plays, resampling whole games. Lower is better for log loss and Brier."""
    rng = np.random.default_rng(seed)
    ug = np.unique(games)
    by = {g: np.where(games == g)[0] for g in ug}
    d = {k: [] for k in ("accuracy", "macro_f1", "log_loss", "brier", "man_recall", "balanced_accuracy")}
    for _ in range(n_boot):
        ix = np.concatenate([by[g] for g in rng.choice(ug, len(ug))])
        a, b = binary_metrics(y[ix], p_new[ix]), binary_metrics(y[ix], p_old[ix])
        for k in d:
            d[k].append(a[k] - b[k])
    a, b = binary_metrics(y, p_new), binary_metrics(y, p_old)
    return {"games": len(ug), "resamples": n_boot,
            "difference": {k: {"point": float(a[k] - b[k]), "interval_95": [float(np.quantile(v, .025)), float(np.quantile(v, .975))]} for k, v in d.items()}}
