"""Coverage experiment C: coverage family (released coverage_type), reported separately from the binary task.

Classes: Cover 0, Cover 1 and Cover 2 man (man group); Cover 2, 3, 4 and 6 zone (zone group). PREVENT is excluded
(too few plays) and the exclusion is reported. The man/zone group of every family matches the released man/zone
label exactly, so the hierarchy P(family) = P(group) x P(family | group) is consistent with the labels.

Compared: flat boosted-tree multiclass (experiment A feature set) and the temporal model with hierarchical heads
(experiment B architecture and epoch count). Nothing is tuned here. Trained on weeks 1-12 and scored on weeks 13-14;
the locked test weeks are not scored for this task. Probabilities are not calibrated.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from torch import nn

from ..shared import config
from ..shared.provenance import write_json
from ..shared.runs import Run
from . import neural, relational
from .benchmark_v2 import table
from .experiment_a import GBM, SOURCE

OUT = config.REPORTS / "v2"
MAN = ["COVER_0_MAN", "COVER_1_MAN", "COVER_2_MAN"]
ZONE = ["COVER_2_ZONE", "COVER_3_ZONE", "COVER_4_ZONE", "COVER_6_ZONE"]
CLASSES = MAN + ZONE
EXCLUDED = ["PREVENT"]
HZ = relational.HORIZONS


def metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(1)
    labels = list(range(len(CLASSES)))
    pr, rc, _f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {"plays": len(y), "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
            "log_loss": float(log_loss(y, np.clip(proba, 1e-6, 1), labels=labels)),
            "per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]), "support": int(sup[i])} for i, c in enumerate(CLASSES)},
            "confusion": {"labels": CLASSES, "rows_released_cols_predicted": confusion_matrix(y, pred, labels=labels).tolist()}}


def run(log=print) -> dict:
    run_rec = Run("coverage", "expC-family", seed=42)
    splits = json.loads((config.MANIFESTS / f"{SOURCE}_splits.json").read_text())
    chosen_a = json.loads((OUT / "coverage_expA_cv.json").read_text())["chosen"]
    chosen_b = json.loads((OUT / "coverage_expB_cv.json").read_text())["chosen"]
    rep = {"task": "coverage family (released coverage_type)", "classes": CLASSES, "excluded": EXCLUDED,
           "status": "Development only: trained on weeks 1-12, scored on weeks 13-14. The locked test weeks have not been scored for this task.",
           "hierarchy": "P(family) = P(man or zone) x P(family | group); the group is predicted, never given as an input",
           "probabilities": "uncalibrated", "horizons": {}}
    # flat boosted trees
    for h in ("post_1s", "post_1_5s"):
        t, g, r = table(h)
        t = t.filter(pl.col("coverage_type").is_in(CLASSES))
        cols = {"geometry": g, "geometry+relational": g + r, "relational": r}[chosen_a["features"]]
        tr, dv = t.filter(pl.col("_k").is_in(splits["splits"]["train"]["plays"])), t.filter(pl.col("_k").is_in(splits["splits"]["dev"]["plays"]))
        ytr, ydv = np.array([CLASSES.index(c) for c in tr["coverage_type"]]), np.array([CLASSES.index(c) for c in dv["coverage_type"]])
        m = HistGradientBoostingClassifier(**GBM).fit(tr.select(cols).to_numpy(), ytr)
        proba = np.zeros((dv.height, len(CLASSES)))
        proba[:, m.classes_] = m.predict_proba(dv.select(cols).to_numpy())
        rep["horizons"][h] = {"train_plays": tr.height, "train_counts": {c: int((ytr == i).sum()) for i, c in enumerate(CLASSES)},
                              "models": {"boosted_trees_flat": metrics(ydv, proba), "class_prior": metrics(ydv, np.tile(np.bincount(ytr, minlength=len(CLASSES)) / len(ytr), (dv.height, 1)))}}
        dv["_k"].to_list()
        log(f"{h} boosted trees: accuracy {rep['horizons'][h]['models']['boosted_trees_flat']['accuracy']:.3f} macro F1 {rep['horizons'][h]['models']['boosted_trees_flat']['macro_f1']:.3f}")
    # hierarchical temporal model
    A = neural.load_arrays()
    fam = np.array([CLASSES.index(c) if c in CLASSES else -1 for c in A["family"]])
    cohort = set(splits["splits"]["train"]["plays"]) | set(splits["splits"]["dev"]["plays"])
    in_c = np.array([k in cohort for k in A["key"]])
    tr = np.where((fam >= 0) & in_c & (A["week"] <= 12))[0]
    dv = np.where((fam >= 0) & in_c & np.isin(A["week"], [13, 14]))[0]
    torch.manual_seed(42)
    rng, gen = np.random.default_rng(42), torch.Generator().manual_seed(42)
    model = neural.CoverageNet(chosen_b["temporal"], n_family=(len(MAN), len(ZONE)))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    hz = torch.tensor(list(HZ.values()))
    y_group = torch.as_tensor((fam < len(MAN)).astype(np.float32))
    y_fam, nf = torch.as_tensor(fam), torch.as_tensor(A["nframes"])
    for epoch in range(chosen_b["epochs"]):
        model.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), 64):
            ix = perm[i:i + 64]
            refl = rng.random(len(ix)) < 0.5 if chosen_b["augment"] else None
            g_logit, man_l, zone_l = model.forward_family(*neural.to_batch(A, ix, "cpu", refl))
            pick = hz[torch.multinomial((nf[ix][:, None] > hz[None, :]).float() + 1e-9, 1, generator=gen).squeeze(1)]
            rows = torch.arange(len(ix))
            gl, ml, zl = g_logit[rows, pick], man_l[rows, pick], zone_l[rows, pick]
            is_man = y_group[ix] == 1
            loss = nn.functional.binary_cross_entropy_with_logits(gl, y_group[ix])
            if is_man.any():
                loss = loss + nn.functional.cross_entropy(ml[is_man], y_fam[ix][is_man]) * is_man.float().mean()
            if (~is_man).any():
                loss = loss + nn.functional.cross_entropy(zl[~is_man], y_fam[ix][~is_man] - len(MAN)) * (~is_man).float().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        g_logit, man_l, zone_l = model.forward_family(*neural.to_batch(A, dv, "cpu"))
    for h in ("post_1s", "post_1_5s"):
        f = HZ[h]
        pg = torch.sigmoid(g_logit[:, f])[:, None]
        proba = torch.cat([pg * torch.softmax(man_l[:, f], 1), (1 - pg) * torch.softmax(zone_l[:, f], 1)], 1).numpy()
        rep["horizons"][h]["models"]["temporal_hierarchical"] = metrics(fam[dv], proba)
        rep["horizons"][h]["dev_plays"] = len(dv)
        log(f"{h} hierarchical temporal: accuracy {rep['horizons'][h]['models']['temporal_hierarchical']['accuracy']:.3f} macro F1 {rep['horizons'][h]['models']['temporal_hierarchical']['macro_f1']:.3f}")
    write_json(OUT / "coverage_expC_family.json", rep)
    run_rec.set(target=rep["task"], population="v1 train and development cohorts with a family in the class list", exclusions=EXCLUDED, input_cutoff={h: HZ[h] for h in rep["horizons"]},
                metrics={h: {m: {k: v[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss")} for m, v in e["models"].items()} for h, e in rep["horizons"].items()})
    run_rec.output("report", OUT / "coverage_expC_family.json")
    run_rec.finish()
    return rep
