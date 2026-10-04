"""Coverage site export, schema coverage-demo-v3: every selectable model and horizon, evidence, prefix outputs, similar plays.

Play sets (all from test weeks 15-18, all from saved models; nothing is retrained here):
- sample: the same seeded, prediction-blind random sample of the common cohort used since the audit fix.
- quick throws: a seeded random sample of test-week plays thrown before 1.5 s. Horizons past the throw are null and
  the site disables them; no positions are invented to reach a later horizon.
- errors: per model and horizon, a seeded random sample of plays whose lean disagrees with the released label.

Every prediction shown for a model and horizon comes from that model's saved artifact and that horizon's calibrator.
Explanations are computed for the same model and horizon they are shown with.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import polars as pl
import torch

from ..shared import config
from ..shared.ids import play_key
from ..shared.provenance import now_utc, write_json
from . import benchmark_v2 as B
from . import neural, relational, schema
from .experiment_a import SOURCE, K
from .features import DESCRIPTIONS

OUT = config.WEB_DEMO / "real" / "coverage"
SCHEMA = "coverage-demo-v3"
HZ = relational.HORIZONS
MODELS = {
    "v1_gbm": {"label": "Boosted trees, v1 geometry", "short": "Boosted trees (v1 benchmark)", "horizons": B.V1_WINDOWS, "explain": "replace_with_typical"},
    "v1_logit": {"label": "Logistic regression, v1 geometry", "short": "Logistic regression (v1)", "horizons": B.V1_WINDOWS, "explain": "linear_terms"},
    "v2_gbm_rel": {"label": "Boosted trees, geometry + relational (v2)", "short": "Boosted trees + relational (v2)", "horizons": list(HZ), "explain": "replace_with_typical"},
    "v2_temporal": {"label": "Temporal interaction model (v2)", "short": "Temporal model (v2)", "horizons": list(HZ), "explain": "input_ablation"},
}
DEFAULT_MODEL, DEFAULT_HORIZON = "v1_gbm", "post_1_5s"
SAMPLE_SEED, ERROR_SEED, QUICK_SEED = 11, 23, 31
N_SAMPLE, N_ERRORS, N_QUICK, TOP_TERMS, N_SIMILAR = 120, 15, 12, 8, 3
EXPLAIN = {
    "linear_terms": "Each measurement's term in the logistic model's log-odds (coefficient times standardized value), before calibration.",
    "replace_with_typical": "How far the model's log-odds move when one measurement is replaced by its training median and the rest are kept, before calibration. "
                            "Related measurements can hide each other, and the changes do not add up to the prediction.",
    "input_ablation": "How far the temporal model's log-odds move when one group of pair inputs is replaced by its training average, or one player is removed, "
                      "before calibration. This shows what the model is sensitive to. Removing inputs creates situations the model never trained on, so treat it as a rough guide, not a cause.",
}
GROUPS = {"separation and relative position": [0, 1, 2], "relative movement": [3, 4, 5, 6], "defender facing": [7],
          "defender position and speed": [8, 9, 10, 11, 12], "receiver position and speed": [13, 14, 15, 16, 17]}
REL_DESCRIPTIONS = {
    "d_near_sep": "Distance from each defender to his nearest route runner", "d_near_relx": "Depth of each defender relative to his nearest route runner",
    "d_near_rely_abs": "Lateral offset between each defender and his nearest route runner", "d_near_closing": "How fast each defender is closing on his nearest route runner",
    "d_near_facing": "How squarely each defender faces his nearest route runner", "r_near_sep": "Distance from each route runner to his nearest defender",
    "d_near_margin": "Gap between a defender's nearest and second-nearest route runner (small = ambiguous)", "match_sep": "Separation within the closest one-to-one pairing of defenders and route runners",
    "match_facing": "How squarely paired defenders face their route runner", "match_within_3": "Share of paired route runners with their defender within 3 yards",
    "match_within_5": "Share of paired route runners with their defender within 5 yards", "def_width": "Width of the tracked defense", "def_depth_range": "Depth range of the tracked defense",
    "def_depth_std": "Spread of defender depths", "def_speed_mean": "Mean defender speed", "def_move_coherence": "How much defenders move in one shared direction",
    "match_sep_mean_t": "Average separation of each pair since the snap", "match_sep_min_t": "Smallest separation of each pair since the snap",
    "match_sep_std_t": "How much each pair's separation varied since the snap", "match_sep_change": "Change in each pair's separation since the snap",
    "match_closing_mean_t": "Average closing speed of each pair since the snap", "match_dircos_mean_t": "How closely paired defenders moved in the same direction as their route runner",
    "d_follow_dircos": "How closely each defender moved in the same direction as whichever route runner was nearest", "d_near_persist": "How steadily each defender stayed nearest to the same route runner",
    "d_near_switch_mean": "Average number of times a defender's nearest route runner changed", "d_near_switch_any": "Share of defenders whose nearest route runner changed",
    "r_near_def_changes": "Average number of times a route runner's nearest defender changed", "switch_dircos_mean": "Movement agreement with the new nearest route runner at a change",
    "d_disp_depth": "Defender movement in depth since the snap", "d_disp_lateral_abs": "Defender lateral movement since the snap",
    "def_width_change": "Change in the defense's width since the snap", "def_depth_mean_change": "Change in mean defender depth since the snap",
}
STAT = {"mean": "average", "min": "smallest", "max": "largest", "std": "spread"}


def describe(f: str) -> tuple[str, str]:
    """(plain description, evidence kind). Evidence kind tells the site which players to highlight."""
    base = f.removeprefix("snap_")
    pre = "At the snap: " if f.startswith("snap_") else ""
    if base in DESCRIPTIONS:
        kind = "deepest" if base in ("depth_max", "depth_2nd", "n_deep_10", "n_deep_7", "deep2_lateral_sep") else "nearest" if base.startswith(("cov_", "press", "cover_", "follow", "motion")) else "all"
        return pre + DESCRIPTIONS[base], kind
    for stat, word in STAT.items():
        if base.endswith("_" + stat) and base[: -len(stat) - 1] in REL_DESCRIPTIONS:
            b = base[: -len(stat) - 1]
            return f"{pre}{REL_DESCRIPTIONS[b]} ({word} over players)", "pairs" if b.startswith("match_") else "nearest" if b.startswith(("d_near", "d_follow", "r_near")) else "all"
    if base in REL_DESCRIPTIONS:
        return pre + REL_DESCRIPTIONS[base], "pairs" if base.startswith("match_") else "nearest" if base.startswith(("d_near", "r_near", "switch")) else "all"
    return f, "all"


def _compact(path, obj) -> None:
    """Play files hold long number arrays; write them without indentation to keep the export small."""
    path.write_text(json.dumps(obj, separators=(",", ":"), default=str))


def tree_terms(base_model, cols: list[str], row: np.ndarray, medians: dict[str, float]) -> list[dict]:
    x = row.reshape(1, -1)
    ref = float(base_model.decision_function(x)[0])
    alt = np.repeat(x, len(cols), 0)
    alt[np.arange(len(cols)), np.arange(len(cols))] = [medians[c] for c in cols]
    eff = ref - base_model.decision_function(alt)
    order = np.argsort(-np.abs(eff))[:TOP_TERMS]
    return [{"feature": cols[i], "value": None if np.isnan(row[i]) else round(float(row[i]), 3), "log_odds_toward_man": round(float(eff[i]), 4)} for i in order]


def run() -> dict:
    splits = json.loads((config.MANIFESTS / f"{SOURCE}_splits.json").read_text())
    bench = json.loads((B.OUT / "coverage_benchmark_v2.json").read_text())
    v1_eval = json.loads((config.REPORTS / "v1" / "coverage_man_zone.json").read_text())
    test = sorted(splits["splits"]["test"]["plays"])
    A = neural.load_arrays()
    pos = {k: i for i, k in enumerate(A["key"])}
    meta = pl.read_parquet(config.FEATURES / SOURCE / "tensors_v2_index.parquet").with_columns(K)
    plays = pl.read_parquet(config.PROCESSED / SOURCE / "plays.parquet").with_columns(K)
    info = {r["_k"]: r for r in plays.iter_rows(named=True)}
    midx = {r["_k"]: r for r in meta.iter_rows(named=True)}
    label = {k: r[schema.TARGET] for k, r in info.items()}

    # ---- saved models and their probabilities for every play that has the horizon
    v1 = {(m, h): joblib.load(config.MODELS / f"coverage_{SOURCE}" / f"{h}__geometry_{'gbm' if m == 'v1_gbm' else 'logit'}.joblib") for m in ("v1_gbm", "v1_logit") for h in B.V1_WINDOWS}
    v2 = {h: joblib.load(B.MODELS_DIR / f"gbm_rel_{h}.joblib") for h in HZ}
    chosen_b = bench["chosen"]["experiment_b"]
    ck = torch.load(B.MODELS_DIR / f"temporal_{chosen_b['temporal']}_seed42.pt", weights_only=False)
    net = neural.CoverageNet(chosen_b["temporal"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    tables = {h: B.table(h)[0] for h in HZ}
    v1_tables = {h: pl.read_parquet(config.FEATURES / SOURCE / f"features_{h}.parquet").with_columns(K) for h in B.V1_WINDOWS}
    rows = {h: {k: i for i, k in enumerate(t["_k"].to_list())} for h, t in tables.items()}
    v1_rows = {h: {k: i for i, k in enumerate(t["_k"].to_list())} for h, t in v1_tables.items()}
    prob: dict[str, dict[str, dict[str, float]]] = {m: {} for m in MODELS}
    for (m, h), b in v1.items():
        prob[m][h] = dict(zip(v1_tables[h]["_k"].to_list(), b["model"].predict_proba(v1_tables[h].select(b["columns"]).to_pandas())[:, 1].tolist()))
    for h, b in v2.items():
        prob["v2_gbm_rel"][h] = dict(zip(tables[h]["_k"].to_list(), b["platt"].predict_proba(b["model"].decision_function(tables[h].select(b["columns"]).to_numpy()).reshape(-1, 1))[:, 1].tolist()))
    logits = neural.predict_logits(net, A, np.arange(len(A["key"])))
    for h, f in HZ.items():
        a, b_ = ck["platt"][h]["a"], ck["platt"][h]["b"]
        prob["v2_temporal"][h] = {k: float(1 / (1 + np.exp(-(a * logits[i, f] + b_)))) for i, k in enumerate(A["key"]) if A["nframes"][i] > f and np.isfinite(A["y"][i])}
    cutoff = {"v1_gbm": {h: v1[("v1_gbm", h)]["threshold"] for h in B.V1_WINDOWS}, "v1_logit": {h: v1[("v1_logit", h)]["threshold"] for h in B.V1_WINDOWS},
              "v2_gbm_rel": {h: bench["models"]["v2_gbm_rel"][h]["abstention"]["cutoff"]["cutoff"] for h in HZ},
              "v2_temporal": {h: bench["models"]["v2_temporal"][h]["abstention"]["cutoff"]["cutoff"] for h in HZ}}
    version = {"v1_gbm": {h: v1[("v1_gbm", h)]["version"] for h in B.V1_WINDOWS}, "v1_logit": {h: v1[("v1_logit", h)]["version"] for h in B.V1_WINDOWS},
               "v2_gbm_rel": {h: v2[h]["version"] for h in HZ}, "v2_temporal": {h: f"cov-v2-temporal-{chosen_b['temporal']}-seed42-{h}" for h in HZ}}

    # ---- play sets
    sample = sorted(np.random.default_rng(SAMPLE_SEED).choice(test, size=N_SAMPLE, replace=False).tolist())
    short = sorted(k for k, r in midx.items() if r["week"] in splits["weeks"]["test"] and r["n_frames"] < 16 and r["n_def"] >= 4 and r["n_rec"] >= 3 and label.get(k) in schema.TARGET_CLASSES)
    quick = sorted(np.random.default_rng(QUICK_SEED).choice(short, size=min(N_QUICK, len(short)), replace=False).tolist())
    erng, errors, available = np.random.default_rng(ERROR_SEED), {m: {} for m in MODELS}, {m: {} for m in MODELS}
    for m, spec in MODELS.items():
        for h in spec["horizons"]:
            wrong = np.array(sorted(k for k in test if k in prob[m][h] and (prob[m][h][k] >= 0.5) != (label[k] == "Man")))
            available[m][h] = len(wrong)
            errors[m][h] = sorted(erng.choice(wrong, size=min(N_ERRORS, len(wrong)), replace=False).tolist())
    wanted = sorted(set(sample) | set(quick) | {k for m in errors.values() for ks in m.values() for k in ks})

    # ---- similar-play reference pool: training weeks only, standardized v2 features at +1.5 s
    ref_t = tables["post_1_5s"].filter(pl.col("_k").is_in(splits["splits"]["train"]["plays"]))
    cols15 = v2["post_1_5s"]["columns"]
    R = ref_t.select(cols15).to_numpy()
    mu, sd = np.nanmean(R, 0), np.nanstd(R, 0) + 1e-9
    Rz = np.nan_to_num((R - mu) / sd)
    ref_keys = ref_t["_k"].to_list()

    # ---- neural ablation references: training-average pair inputs
    tr_ix = np.array([pos[k] for k in splits["splits"]["train"]["plays"][:1500]])
    with torch.no_grad():
        d, r, dm, rm = neural.to_batch(A, tr_ix, "cpu")
        pf = neural.pair_features(d, r)
        pm = (dm[:, None, :, None] & rm[:, None, None, :]).expand(pf.shape[:-1])
        pair_mean = pf[pm].mean(0)

    def temporal_explain(k: str, f: int) -> dict:
        i = np.array([pos[k]])
        d, r, dm, rm = neural.to_batch(A, i, "cpu")
        with torch.no_grad():
            base = float(net(d, r, dm, rm)[0, f])
            groups = []
            for name, idxs in GROUPS.items():
                def edit(x, idxs=idxs):
                    x = x.clone()
                    x[..., idxs] = pair_mean[idxs]
                    return x
                net.pair_edit = edit
                groups.append({"group": name, "log_odds_toward_man": round(base - float(net(d, r, dm, rm)[0, f]), 4)})
            net.pair_edit = None
            players = []
            for side, mask, n in (("defense", dm, int(dm.sum())), ("offense", rm, int(rm.sum()))):
                if n <= 2:
                    continue
                for j in range(n):
                    m2 = mask.clone()
                    m2[0, j] = False
                    out = net(d, r, m2 if side == "defense" else dm, m2 if side == "offense" else rm)
                    players.append({"side": side, "slot": j, "log_odds_toward_man": round(base - float(out[0, f]), 4)})
        return {"groups": sorted(groups, key=lambda g: -abs(g["log_odds_toward_man"])), "players": players}

    names = {i: (n, p) for i, n, p in pl.read_parquet(config.PROCESSED / SOURCE / "players.parquet").select(["nflId", "displayName", "position"]).iter_rows()}
    for old in (OUT / "plays").glob("*.json"):
        old.unlink()
    (OUT / "plays").mkdir(parents=True, exist_ok=True)
    index = []
    by_week: dict[int, list[str]] = {}
    for k in wanted:
        by_week.setdefault(info[k]["week"], []).append(k)
    for week, keys in sorted(by_week.items()):
        gids = sorted({info[k]["gameId"] for k in keys})
        trk = pl.scan_parquet(config.PROCESSED / SOURCE / f"tracking_week_{week}.parquet").filter(pl.col("gameId").is_in(gids) & pl.col("rel_frame").is_between(0, 60)).collect()
        for k in keys:
            r0 = info[k]
            gid, pid = r0["gameId"], r0["playId"]
            t = trk.filter((pl.col("gameId") == gid) & (pl.col("playId") == pid)).sort("rel_frame")
            frames = sorted(t["rel_frame"].unique().to_list())
            fpos = {f: i for i, f in enumerate(frames)}
            ents, slot_ids = [], {"defense": [], "offense": []}
            for (side, nid), g in sorted(t.partition_by(["side", "nflId"], as_dict=True).items(), key=lambda kv: (kv[0][0], kv[0][1])):
                arr = {c: [None] * len(frames) for c in ("x", "y", "s", "o")}
                for f, x, y, s, o in g.select(["rel_frame", "x", "y", "s", "o"]).iter_rows():
                    arr["x"][fpos[f]], arr["y"][fpos[f]] = round(x, 2), round(y, 2)
                    arr["s"][fpos[f]] = None if s is None else round(s, 2)
                    arr["o"][fpos[f]] = None if o is None else round(o, 1)
                nm, role = names.get(nid, (None, None))
                passer = g["role"][0] == "Passer"
                ents.append({"id": str(nid), "side": side, "name": nm, "position": role, "jersey": None, "passer": bool(passer), **arr})
                if not passer:
                    slot_ids[side].append(str(nid))        # same order as the tensors: sorted by player id within side
            n_frames = midx[k]["n_frames"]
            ai = pos[k]
            dvalid, rvalid = A["dmask"][ai], A["rmask"][ai]
            horizons, summary_pred = {}, {m: {} for m in MODELS}
            for h, f in HZ.items():
                if n_frames <= f:
                    horizons[h] = None                       # prefix does not exist: no prediction, nothing fabricated
                    for m in MODELS:
                        if h in MODELS[m]["horizons"]:
                            summary_pred[m][h] = None
                    continue
                preds, expl = {}, {}
                for m, spec in MODELS.items():
                    if h not in spec["horizons"] or k not in prob[m][h]:
                        if h in spec["horizons"]:
                            summary_pred[m][h] = None
                        continue
                    p = prob[m][h][k]
                    conf = max(p, 1 - p)
                    preds[m] = {"version": version[m][h], "p_man": round(p, 4), "p_zone": round(1 - p, 4), "predicted": "Man" if p >= 0.5 else "Zone",
                                "confidence": round(conf, 4), "confidence_threshold": cutoff[m][h], "accepted": bool(conf >= cutoff[m][h])}
                    summary_pred[m][h] = {"p_man": preds[m]["p_man"], "predicted": preds[m]["predicted"], "accepted": preds[m]["accepted"]}
                    if m in ("v1_gbm", "v1_logit"):
                        b = v1[(m, h)]
                        row = v1_tables[h][v1_rows[h][k]]
                        if m == "v1_logit":
                            z = b["base"][:-1].transform(row.select(b["columns"]).to_pandas())[0]
                            eff = b["base"][-1].coef_[0] * z
                            vals = row.select(b["columns"]).row(0)
                            order = np.argsort(-np.abs(eff))[:TOP_TERMS]
                            expl[m] = [{"feature": b["columns"][i], "value": None if vals[i] is None or np.isnan(vals[i]) else round(float(vals[i]), 3), "log_odds_toward_man": round(float(eff[i]), 4)} for i in order]
                        else:
                            base_est = b["base"]
                            med = {c: float(np.nanmedian(v1_tables[h].filter(pl.col("_k").is_in(splits["splits"]["train"]["plays"]))[c].to_numpy())) for c in b["columns"]} if (m, h, "med") not in v1 else v1[(m, h, "med")]
                            v1[(m, h, "med")] = med
                            expl[m] = tree_terms(base_est, b["columns"], row.select(b["columns"]).to_numpy()[0].astype(float), med)
                    elif m == "v2_gbm_rel":
                        b = v2[h]
                        expl[m] = tree_terms(b["model"], b["columns"], tables[h][rows[h][k]].select(b["columns"]).to_numpy()[0].astype(float), b["train_medians"])
                    else:
                        e = temporal_explain(k, f)
                        for pl_ in e["players"]:
                            pl_["id"] = slot_ids[pl_.pop("side")][pl_.pop("slot")] if True else None
                        expl[m] = e
                # evidence: observed geometry at this horizon that the relational measurements are built from
                dd = A["defs"][ai][: f + 1][:, dvalid].astype(float)
                rr = A["recs"][ai][: f + 1][:, rvalid].astype(float)
                P = relational.pair_arrays(dd, rr)
                from scipy.optimize import linear_sum_assignment
                ri, ci = linear_sum_assignment(P["sep"][-1])
                near = P["sep"][-1].argmin(1)
                deep = np.argsort(-dd[-1, :, 0])[:2]
                evidence = {
                    "pairs": [{"defender": slot_ids["defense"][a], "receiver": slot_ids["offense"][b], "separation": round(float(P["sep"][-1, a, b]), 2),
                               "closing_speed": round(float(P["closing"][-1, a, b]), 2), "separation_at_snap": round(float(P["sep"][0, a, b]), 2)} for a, b in zip(ri, ci)],
                    "nearest": [{"defender": slot_ids["defense"][a], "receiver": slot_ids["offense"][int(b)], "separation": round(float(P["sep"][-1, a, b]), 2)} for a, b in enumerate(near)],
                    "deepest": [slot_ids["defense"][int(a)] for a in deep],
                }
                horizons[h] = {"latest_frame": f, "predictions": preds, "explanations": expl, "evidence": evidence}
            similar = []
            if n_frames > 15 and k in rows["post_1_5s"]:
                q = np.nan_to_num((tables["post_1_5s"][rows["post_1_5s"][k]].select(cols15).to_numpy()[0].astype(float) - mu) / sd)
                dist = np.sqrt(((Rz - q) ** 2).mean(1))
                for j in np.argsort(dist)[:N_SIMILAR]:
                    rk = ref_keys[int(j)]
                    ri_, rj = info[rk], pos[rk]
                    dv, rv = A["dmask"][rj], A["rmask"][rj]
                    similar.append({"id": play_key(SOURCE, ri_["gameId"], ri_["playId"]), "week": ri_["week"], "offense": ri_["possessionTeam"], "defense": ri_["defensiveTeam"],
                                    "down": ri_["down"], "yardsToGo": ri_["yardsToGo"], "released": {"manZone": ri_[schema.TARGET], "coverage": ri_["coverage_type"]},
                                    "distance": round(float(dist[j]), 3),
                                    "tracks": {"defense": np.round(A["defs"][rj][:16][:, dv][:, :, :2], 1).transpose(1, 0, 2).tolist(),
                                               "offense": np.round(A["recs"][rj][:16][:, rv][:, :, :2], 1).transpose(1, 0, 2).tolist()}})
            pid_key = play_key(SOURCE, gid, pid)
            los = float(t["los_x"][0])
            summary = {"id": pid_key, "file": f"plays/{gid}_{pid}.json", "week": r0["week"], "season": r0["season"], "offense": r0["possessionTeam"], "defense": r0["defensiveTeam"],
                       "quarter": r0["quarter"], "down": r0["down"], "yardsToGo": r0["yardsToGo"], "yardsToGoal": round(110 - los, 1), "frames_observed": int(n_frames),
                       "released": {"manZone": label[k], "coverage": r0["coverage_type"], "source": schema.SOURCES[SOURCE]["label_source"]},
                       "predictions": summary_pred, "split": "test"}
            index.append(summary)
            _compact(OUT / "plays" / f"{gid}_{pid}.json", {"schema": SCHEMA, "synthetic": False, **summary, "description": r0.get("playDescription"), "los_x": los,
                                                             "ref_y": float(t["ref_y"][0]), "frames": frames, "entities": ents, "horizons": horizons,
                                                             "similar": {"pool": "training weeks 1-12 only; the play itself is never in the pool", "basis": "average squared difference of standardized v2 relational and geometry measurements at snap + 1.5 s",
                                                                         "note": "Geometrically comparable, by this model's measurements. It does not mean the same call or the same tactic.", "plays": similar}})
    key_to_id = {":".join(s["id"].split(":")[1:]): s["id"] for s in index}
    feats = sorted({t["feature"] for p in (OUT / "plays").glob("*.json") for hz in json.loads(p.read_text())["horizons"].values() if hz for m, e in hz["explanations"].items() if isinstance(e, list) for t in e})
    manifest = {
        "schema": SCHEMA, "synthetic": False, "generated_at": now_utc(), "source": SOURCE, "eligible_test_plays": len(test),
        "display_rights": "Owner approved display on 2026-10-04; the competition rules have not been independently reviewed.",
        "models": {m: {k: v for k, v in spec.items()} | {"explanation": EXPLAIN[spec["explain"]]} for m, spec in MODELS.items()},
        "horizons": {h: {"latest_frame": f, "seconds": f / 10} for h, f in HZ.items()}, "default_model": DEFAULT_MODEL, "default_horizon": DEFAULT_HORIZON,
        "cutoffs": cutoff, "versions": version,
        "decision_rule": "Lean: man if calibrated p(man) >= 0.5, else zone. Confidence = max(p(man), 1 - p(man)). Accepted if confidence >= the cutoff chosen on weeks 13-14; otherwise abstained.",
        "sample": {"ids": [key_to_id[k] for k in sample], "seed": SAMPLE_SEED,
                   "note": f"A random sample of {len(sample)} of the {len(test)} eligible held-out test plays (seed {SAMPLE_SEED}). Drawn from the list of play ids alone, before any prediction, label or confidence was read."},
        "quick_throws": {"ids": [key_to_id[k] for k in quick], "seed": QUICK_SEED, "available": len(short),
                         "note": f"A random sample of {len(quick)} of the {len(short)} test-week plays thrown before 1.5 seconds (seed {QUICK_SEED}). These are outside the common cohort. Horizons after the throw are unavailable and nothing is extrapolated."},
        "errors": {"ids": {m: {h: [key_to_id[k] for k in ks] for h, ks in v.items()} for m, v in errors.items()}, "available": available, "seed": ERROR_SEED,
                   "note": f"Picked because they are mistakes. For the chosen model and horizon, up to {N_ERRORS} plays drawn at random (seed {ERROR_SEED}) from every eligible test play where the model's lean differs from the released label, abstained plays included. It says nothing about how often the model is wrong."},
        "feature_descriptions": {f: {"text": describe(f)[0], "evidence": describe(f)[1]} for f in feats},
        "benchmark_note": bench["note"], "plays": index,
    }
    write_json(OUT / "index.json", manifest)
    (OUT / "evaluation.json").write_text(json.dumps(v1_eval))
    (OUT / "benchmark_v2.json").write_text((B.OUT / "coverage_benchmark_v2.json").read_text())
    for name in ("coverage_expA_cv", "coverage_expB_cv", "coverage_expC_family", "coverage_neural_checks", "coverage_team_holdout_v2", "coverage_review_queue"):
        src = B.OUT / f"{name}.json"
        if src.exists():
            (OUT / f"{name.removeprefix('coverage_')}.json").write_text(src.read_text())
    return {"plays": len(index), "sample": len(sample), "quick": len(quick), "out": config.rel(OUT)}
