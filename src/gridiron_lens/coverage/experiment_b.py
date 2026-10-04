"""Coverage experiment B driver: select the temporal model on development folds, then refit with fixed choices.

Selection (weeks 1-12 only): GRU vs causal temporal convolution vs GRU with left-right reflection, seed 42, on the same
three chronological folds as experiment A. Criterion: mean validation log loss averaged over the four prefixes.
The stopping rule carried forward is a fixed epoch count: the median best epoch across folds.

Refit: weeks 1-12 with that architecture and epoch count, seeds 42, 7 and 2026. Seed 42 is the primary model (named
before training); the three-seed probability average is reported as a secondary entry. Platt calibration per horizon on
weeks 13-14. Weeks 15-18 were examined in v1, so scores there are comparisons on a previously examined benchmark.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from ..shared import config
from ..shared.provenance import write_json
from ..shared.runs import Run
from . import evalkit, neural, relational
from .experiment_a import FOLDS, fit_platt

OUT = config.REPORTS / "v2"
MODELS = config.MODELS / "coverage_v2"
SEEDS = [42, 7, 2026]
HZ = relational.HORIZONS
CONFIGS = {"gru": {"temporal": "gru", "augment": False}, "tcn": {"temporal": "tcn", "augment": False}, "gru+reflect": {"temporal": "gru", "augment": True}}
BASE = {"optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "batch": 64, "grad_clip": 1.0, "max_epochs": 50, "patience": 8,
            "pair_mlp": [64, 64], "pooled": "mean+max over receivers, then defenders", "hidden": 96, "layers": 2, "dropout": 0.1}


def sig(x):
    return 1 / (1 + np.exp(-x))


def horizon_metrics(A: dict, ix: np.ndarray, logits: np.ndarray) -> dict:
    """Uncalibrated metrics per horizon on plays that have that prefix."""
    out = {}
    for h, f in HZ.items():
        ok = A["nframes"][ix] > f
        out[h] = evalkit.binary_metrics(A["y"][ix][ok].astype(int), sig(logits[ok, f]))
    return out


def cross_validate(log=print) -> dict:
    A = neural.load_arrays()
    lab = ~np.isnan(A["y"])
    run = Run("coverage", "expB-cv", config_={"folds": FOLDS, "configs": CONFIGS, "base": BASE}, seed=42)
    ckpt = OUT / "coverage_expB_cv_partial.json"          # finished folds survive an interrupted run
    saved = json.loads(ckpt.read_text()) if ckpt.exists() else {}
    results = []
    for name, cfg in CONFIGS.items():
        folds = []
        for fi, (tr_w, va_w) in enumerate(FOLDS):
            if f"{name}|{fi}" in saved:
                folds.append(saved[f"{name}|{fi}"])
                continue
            tr, va = np.where(lab & np.isin(A["week"], tr_w))[0], np.where(lab & np.isin(A["week"], va_w))[0]
            r = neural.train_model(A, tr, va, seed=42, log=lambda s: None, **cfg)
            m = horizon_metrics(A, va, neural.predict_logits(r["model"], A, va))
            folds.append({"fold": fi, "train_plays": len(tr), "validate_plays": len(va), "best_epoch": r["best_epoch"], "epochs_run": len(r["history"]),
                          "val_loss": r["best_val_loss"], "seconds": r["seconds"], "seconds_per_epoch": r["seconds_per_epoch"],
                          "plays_per_second": r["plays_per_second"], "parameters": r["parameters"], "by_horizon": m})
            saved[f"{name}|{fi}"] = folds[-1]
            write_json(ckpt, saved)
            log(f"{name} fold {fi}: best epoch {r['best_epoch']} val loss {r['best_val_loss']:.4f} +1.5s log loss {m['post_1_5s']['log_loss']:.4f} "
                f"bal acc {m['post_1_5s']['balanced_accuracy']:.4f} ({r['seconds']}s)")
        results.append({"config": name, **cfg, "folds": folds, "mean_val_loss": float(np.mean([f["val_loss"] for f in folds])),
                        "val_loss_sd": float(np.std([f["val_loss"] for f in folds], ddof=1)),
                        "median_best_epoch": int(np.median([f["best_epoch"] for f in folds])),
                        "mean_by_horizon": {h: {k: float(np.mean([f["by_horizon"][h][k] for f in folds])) for k in ("log_loss", "accuracy", "balanced_accuracy", "man_recall", "macro_f1")} for h in HZ}})
    best = min(results, key=lambda r: r["mean_val_loss"])
    chosen = {"config": best["config"], "temporal": best["temporal"], "augment": best["augment"], "epochs": best["median_best_epoch"],
              "rule": "lowest mean validation log loss over the three chronological folds (averaged over the four prefixes); epochs = median best epoch"}
    rep = {"experiment": "B", "device": "cpu", "device_note": "Apple MPS was benchmarked at about half the CPU speed for this small model, so CPU is used.",
           "base_settings": BASE, "pair_features": neural.PAIR_FEATS, "results": results, "chosen": chosen, "structural_checks": neural.structural_checks()}
    write_json(OUT / "coverage_expB_cv.json", rep)
    run.set(target="released man/zone label", population="labelled plays in weeks 1-12 with the prefix available", input_cutoff=HZ,
            split={"folds": FOLDS}, hyperparameters=BASE, metrics={"chosen": chosen, "mean_val_loss": best["mean_val_loss"]})
    run.output("cv_report", OUT / "coverage_expB_cv.json")
    run.finish()
    return rep


def final(log=print) -> dict:
    """Refit the chosen architecture for the fixed epoch count; calibrate per horizon on weeks 13-14; save predictions for every play."""
    chosen = json.loads((OUT / "coverage_expB_cv.json").read_text())["chosen"]
    A = neural.load_arrays()
    lab = ~np.isnan(A["y"])
    tr, cal = np.where(lab & (A["week"] <= 12))[0], np.where(lab & np.isin(A["week"], [13, 14]))[0]
    everyone = np.arange(len(A["y"]))
    MODELS.mkdir(parents=True, exist_ok=True)
    run = Run("coverage", "expB-final", config_={"chosen": chosen, "seeds": SEEDS, "base": BASE}, seed=SEEDS[0])
    probs, seeds_out = {}, {}
    for seed in SEEDS:
        r = neural.train_model(A, tr, None, temporal=chosen["temporal"], augment=chosen["augment"], seed=seed, fixed_epochs=chosen["epochs"], log=lambda s: None)
        logits = neural.predict_logits(r["model"], A, everyone)
        platt = {}
        p = np.full((len(everyone), len(HZ)), np.nan, np.float32)
        for j, (h, f) in enumerate(HZ.items()):
            okc = A["nframes"][cal] > f
            pl_ = fit_platt(logits[cal][okc, f], A["y"][cal][okc].astype(int))
            platt[h] = {"a": float(pl_.coef_[0, 0]), "b": float(pl_.intercept_[0])}
            ok = A["nframes"] > f                                        # no prediction where the prefix does not exist
            p[ok, j] = sig(platt[h]["a"] * logits[ok, f] + platt[h]["b"])
        probs[seed] = p
        torch.save({"state_dict": r["model"].state_dict(), "config": chosen, "seed": seed, "platt": platt, "pair_features": neural.PAIR_FEATS,
                    "epochs": chosen["epochs"]}, MODELS / f"temporal_{chosen['temporal']}_seed{seed}.pt")
        seeds_out[seed] = {"platt": platt, "seconds": r["seconds"], "final_train_loss": r["history"][-1]["train_loss"], "parameters": r["parameters"]}
        run.output(f"model_seed{seed}", MODELS / f"temporal_{chosen['temporal']}_seed{seed}.pt")
        log(f"seed {seed}: trained {chosen['epochs']} epochs in {r['seconds']}s")
    np.savez_compressed(OUT / "coverage_expB_predictions.npz", key=A["key"], horizons=np.array(list(HZ)),
                        **{f"seed{s}": probs[s] for s in SEEDS}, ensemble=np.nanmean(np.stack([probs[s] for s in SEEDS]), axis=0))
    rep = {"chosen": chosen, "seeds": seeds_out, "primary": f"seed {SEEDS[0]}", "secondary": "mean probability of the three seeds",
           "train_plays": len(tr), "calibration_plays": len(cal), "calibration": "Platt per horizon on weeks 13-14"}
    write_json(OUT / "coverage_expB_final.json", rep)
    run.set(target="released man/zone label", population="labelled plays, weeks 1-12", input_cutoff=HZ, calibration=rep["calibration"], hyperparameters=BASE,
            metrics={"seeds": seeds_out})
    run.output("predictions", OUT / "coverage_expB_predictions.npz")
    run.finish()
    return rep
