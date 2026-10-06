"""Audit of the BDB 2026 Prediction archive as a candidate for fresh coverage validation, read straight from the ZIP.

Question: does it contain coverage labels for plays the models have not seen? Every CSV header is checked for a coverage
label, every file is hashed, seasons and game ids are listed and compared with the BDB 2026 Analytics data already used.

If there are no labels, no accuracy can be measured on it. Its unlabelled 2024 plays can still be pushed through the upload
contract to check that the inputs are compatible and to produce experimental, unscored predictions. Fields that describe
what happened after the throw (`num_frames_output`, `ball_land_x/y`, `player_to_predict`, the Targeted Receiver role and the
`output_*.csv` future coordinates) are never read as model inputs.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from ..shared import config
from ..shared.provenance import now_utc, sha256_file, write_json
from . import relational
from . import upload as U

ZIP_DEFAULT = Path.home() / "Downloads" / "nfl-big-data-bowl-2026-prediction.zip"
RAW = config.RAW / "bdb2026_prediction"
OUT = config.REPORTS / "v3"
LABEL_HINTS = ("coverage", "man_zone", "manzone", "pff_", "team_coverage")
POST_THROW = ("num_frames_output", "ball_land_x", "ball_land_y", "player_to_predict")


def _hash(z: zipfile.ZipFile, name: str) -> str:
    h = hashlib.sha256()
    with z.open(name) as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def audit(zip_path: Path = ZIP_DEFAULT, log=print) -> dict:
    z = zipfile.ZipFile(zip_path)
    files, label_columns, games, roles = [], {}, defaultdict(set), Counter()
    for i in z.infolist():
        if i.is_dir():
            continue
        rec = {"name": i.filename, "bytes": i.file_size, "sha256": _hash(z, i.filename)}
        if i.filename.endswith(".csv"):
            with z.open(i) as f:
                rd = csv.reader(io.TextIOWrapper(f))
                header = next(rd)
                gi = header.index("game_id") if "game_id" in header else None
                ri = header.index("player_role") if "player_role" in header else None
                n = 0
                for row in rd:
                    n += 1
                    if gi is not None:
                        games[i.filename].add(row[gi])
                    if ri is not None:
                        roles[row[ri]] += 1
            rec |= {"columns": header, "rows": n}
            hits = [c for c in header if any(h in c.lower() for h in LABEL_HINTS)]
            if hits:
                label_columns[i.filename] = hits
        files.append(rec)
    ours = {p.name: sha256_file(p) for p in sorted((config.RAW / "bdb2026").glob("*.csv"))}
    ours_by_hash = {v: k for k, v in ours.items()}
    same = {f["name"]: ours_by_hash[f["sha256"]] for f in files if f["sha256"] in ours_by_hash}
    known_games = set(pl.read_parquet(config.PROCESSED / "bdb2026" / "plays.parquet")["gameId"].cast(pl.String).to_list())
    train_games = set().union(*[g for n, g in games.items() if n.startswith("train/")])
    test_games = games.get("test_input.csv", set())
    season = lambda g: int(g[:4]) - (1 if int(g[4:6]) <= 2 else 0)
    with z.open("test_input.csv") as f:
        t = pl.read_csv(f.read(), null_values=["NA", ""], infer_schema_length=20000)
    rep = {
        "created_at_utc": now_utc(), "archive": {"name": zip_path.name, "bytes": zip_path.stat().st_size, "sha256": sha256_file(zip_path), "files": len(files),
                                                 "uncompressed_bytes": sum(f["bytes"] for f in files)},
        "files": files,
        "label_check": {"columns_matching_coverage_label_names": label_columns, "separate_label_table": False,
                        "conclusion": "No CSV has a man/zone or coverage-family column and there is no label table. 'Defensive Coverage' is a value of player_role (which players were tracked), not a coverage label. output_*.csv holds future x,y for players after the throw."},
        "player_role_values": dict(roles),
        "train": {"files": sum(1 for f in files if f["name"].startswith("train/")), "seasons": sorted({season(g) for g in train_games}), "games": len(train_games),
                  "games_already_in_bdb2026_analytics": len(train_games & known_games), "input_files_byte_identical_to_bdb2026_analytics": len([k for k in same if "input_" in k]),
                  "identical_files": same},
        "test": {"plays": t.select(["game_id", "play_id"]).unique().height, "games": sorted(test_games), "seasons": sorted({season(g) for g in test_games}),
                 "games_already_in_bdb2026_analytics": len(test_games & known_games), "rows": t.height,
                 "columns_describing_events_after_the_throw": [c for c in POST_THROW if c in t.columns]},
        "fresh_coverage_validation": "The archive alone has no coverage labels. Its 143 test-input plays do have released labels in the BDB 2026 Analytics label table (supplementary_data.csv), which lists 2024-season plays that had no tracking there.",
    }
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "test_input.csv").write_bytes(z.read("test_input.csv"))
    write_json(RAW / "CHECKSUMS.json", {f["name"]: f["sha256"] for f in files} | {"_archive": rep["archive"]["sha256"]})
    log(json.dumps({k: rep[k] for k in ("archive", "label_check", "player_role_values", "test")}, indent=1)[:2500])
    log(json.dumps({k: v for k, v in rep["train"].items() if k != "identical_files"}))
    return rep


def experimental_predictions(log=print) -> dict:
    """Unlabelled 2024 plays through the upload contract. Compatibility and model outputs only: there is nothing to score."""
    t = pl.read_csv(RAW / "test_input.csv", null_values=["NA", ""], infer_schema_length=20000)
    rows_out, problems, shape = [], Counter(), defaultdict(list)
    for (gid, pid), p in t.partition_by(["game_id", "play_id"], as_dict=True).items():
        rows = [{"player_id": str(r["nfl_id"]), "frame": r["frame_id"], "side": r["player_side"].lower(), "role": "passer" if r["player_role"] == "Passer" else "route_runner",
                 "x": r["x"], "y": r["y"], "s": r["s"], "dir": r["dir"], "vx": None, "vy": None, "o": r["o"], "play_direction": r["play_direction"], "los_x": float(r["absolute_yardline_number"])}
                for r in p.iter_rows(named=True)]                                    # only fields available at the cutoff; post-throw columns are not copied
        try:
            tens = U.build(U.Parsed(f"{gid}:{pid}", rows, None))
        except U.UploadError as e:
            problems[e.problems[0][:70]] += 1
            continue
        res = U.infer(tens, None, f"{gid}:{pid}")
        shape["defenders"].append(tens["defenders"]), shape["route_runners"].append(tens["route_runners"]), shape["frames"].append(tens["frames_supplied"])
        shape["snap_speed"].append(float(np.median(np.hypot(tens["defs"][0][tens["dmask"]][:, 2], tens["defs"][0][tens["dmask"]][:, 3]))))
        rows_out.append({"play": f"{gid}:{pid}", "defenders": tens["defenders"], "route_runners": tens["route_runners"], "frames_usable": tens["frames_usable"],
                         "p_man": {h: (round(v["p_man"], 4) if v["available"] else None) for h, v in res["horizons"].items()}})
    A = __import__("gridiron_lens.coverage.neural", fromlist=["x"]).load_arrays()
    ref = {"defenders": A["dmask"].sum(1), "route_runners": A["rmask"].sum(1), "frames": A["nframes"]}
    saved = np.load(config.REPORTS / "v2" / "coverage_expB_predictions.npz")["seed42"]
    p15 = np.array([r["p_man"]["post_1_5s"] for r in rows_out if r["p_man"]["post_1_5s"] is not None])
    ref15 = saved[:, 3][~np.isnan(saved[:, 3])]
    dist = lambda a: {"mean": round(float(np.mean(a)), 3), "median": float(np.median(a)), "p10": float(np.quantile(a, .1)), "p90": float(np.quantile(a, .9))}
    rep = {
        "created_at_utc": now_utc(), "status": "EXPERIMENTAL AND UNSCORED. These plays have no coverage label. Nothing here is an accuracy measurement.",
        "model": U.bundle()["version"], "source": "test_input.csv of the BDB 2026 Prediction archive: 2024 season, 3 games",
        "inputs_not_read": list(POST_THROW) + ["Targeted Receiver role (all non-passer offence is treated alike)", "output_*.csv"],
        "plays": len(rows_out) + sum(problems.values()), "accepted_by_the_upload_contract": len(rows_out), "rejected": dict(problems),
        "input_compatibility": {k: {"2024 unlabelled": dist(shape[k]), "2023 release (all plays)": dist(ref[k])} for k in ("defenders", "route_runners", "frames")} | {
            "median_defender_speed_at_first_frame_yards_per_s": dist(shape["snap_speed"]),
            "note": "Same file layout, same tracked-player selection (passer, route runners, defensive coverage players), same 10 Hz sampling. The first frame is treated as the snap, as for 2023."},
        "prediction_distribution_post_1_5s": {"2024 unlabelled": {"plays": len(p15), "mean_p_man": round(float(p15.mean()), 3), "share_leaning_man": round(float((p15 >= 0.5).mean()), 3),
                                                                 "share_confident_above_0.9_or_below_0.1": round(float(((p15 > 0.9) | (p15 < 0.1)).mean()), 3)},
                                              "2023 all plays (same model, includes its training weeks)": {"plays": len(ref15), "mean_p_man": round(float(ref15.mean()), 3), "share_leaning_man": round(float((ref15 >= 0.5).mean()), 3),
                                                                                                           "share_confident_above_0.9_or_below_0.1": round(float(((ref15 > 0.9) | (ref15 < 0.1)).mean()), 3)},
                                              "reading": "A similar share of man leans and a similar confidence profile say the inputs look familiar to the model. They do not say the predictions are right."},
        "predictions": rows_out,
    }
    write_json(OUT / "coverage_2024_unlabelled_experimental.json", rep)
    log(json.dumps({k: v for k, v in rep.items() if k != "predictions"}, indent=1))
    return rep


if __name__ == "__main__":
    a = audit()
    write_json(OUT / "coverage_prediction_archive_audit.json", a)
    experimental_predictions()
    # The labelled evaluation is a one-time run: `fresh_evaluate()` was executed once on 2026-10-06 and is not part of the default entry point.


def fresh_evaluate(log=print) -> dict:
    """The registered one-time evaluation (docs/experiments/coverage_fresh_2024.md). Labels are joined only after predictions exist."""
    import torch
    from sklearn.metrics import average_precision_score

    from . import evalkit, neural
    t = pl.read_csv(RAW / "test_input.csv", null_values=["NA", ""], infer_schema_length=20000)
    preds, meta = {}, {}
    seeds = (42, 7, 2026)
    bundles = {}
    for s in seeds:
        f = config.MODELS / "coverage_v2" / f"temporal_gru_seed{s}.pt"
        b = torch.load(f, weights_only=False)
        m = neural.CoverageNet(b["config"]["temporal"])
        m.load_state_dict(b["state_dict"])
        bundles[s] = (m.eval(), b["platt"], sha256_file(f))
    for (gid, pid), p in t.partition_by(["game_id", "play_id"], as_dict=True).items():
        rows = [{"player_id": str(r["nfl_id"]), "frame": r["frame_id"], "side": r["player_side"].lower(), "role": "passer" if r["player_role"] == "Passer" else "route_runner",
                 "x": r["x"], "y": r["y"], "s": r["s"], "dir": r["dir"], "vx": None, "vy": None, "o": r["o"], "play_direction": r["play_direction"], "los_x": float(r["absolute_yardline_number"])}
                for r in p.iter_rows(named=True)]
        tens = U.build(U.Parsed(f"{gid}:{pid}", rows, None))
        out = {}
        with torch.no_grad():
            for s, (m, platt, _) in bundles.items():
                lg = m(*(torch.as_tensor(tens[k])[None] for k in ("defs", "recs", "dmask", "rmask")))[0].numpy()
                out[s] = {h: (float(1 / (1 + np.exp(-(platt[h]["a"] * lg[f] + platt[h]["b"])))) if tens["frames_usable"] > f else None) for h, f in relational.HORIZONS.items()}
        preds[(gid, pid)], meta[(gid, pid)] = out, gid
    # labels are read only now
    sup = pl.read_csv(config.RAW / "bdb2026" / "supplementary_data.csv", null_values=["NA", ""], infer_schema_length=20000)
    lab = {(r["game_id"], r["play_id"]): r for r in sup.select(["game_id", "play_id", "season", "week", "team_coverage_man_zone", "team_coverage_type"]).iter_rows(named=True)}
    keys = [k for k in preds if k in lab and lab[k]["team_coverage_man_zone"] in ("MAN_COVERAGE", "ZONE_COVERAGE")]
    y_all = np.array([1 if lab[k]["team_coverage_man_zone"] == "MAN_COVERAGE" else 0 for k in keys])

    def wilson(k: int, n: int) -> list[float]:
        z, ph = 1.96, k / n
        c, h = (ph + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return [float(c - h), float(c + h)]

    def block(prob: np.ndarray, y: np.ndarray, games: np.ndarray) -> dict:
        m = evalkit.binary_metrics(y, prob)
        ll = -(y * np.log(np.clip(prob, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - prob, 1e-6, 1)))
        rng = np.random.default_rng(20261006)
        bs = [ll[rng.integers(0, len(ll), len(ll))].mean() for _ in range(4000)]
        prior = y.mean()
        return {k: m[k] for k in ("n", "man", "zone", "accuracy", "balanced_accuracy", "macro_f1", "man_precision", "man_recall", "zone_precision", "zone_recall", "log_loss", "brier", "confusion")} | {
            "accuracy_interval_95_wilson": wilson(round(m["accuracy"] * len(y)), len(y)), "log_loss_interval_95_play_bootstrap": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "man_pr_auc": float(average_precision_score(y, prob)), "always_zone_accuracy": float(1 - prior), "class_prior_log_loss": float(-(prior * np.log(prior) + (1 - prior) * np.log(1 - prior))),
            "reliability": evalkit.reliability(y, prob, bins=5),
            "by_game": {str(g): {"plays": int((games == g).sum()), "accuracy": float(((prob[games == g] >= 0.5) == (y[games == g] == 1)).mean()), "man_plays": int(y[games == g].sum())} for g in np.unique(games)}}

    res = {}
    for name, get in (("primary: seed 42", lambda k, h: preds[k][42][h]), ("secondary: three-seed average", lambda k, h: None if preds[k][42][h] is None else float(np.mean([preds[k][s][h] for s in seeds])))):
        res[name] = {}
        for h in relational.HORIZONS:
            ok = [i for i, k in enumerate(keys) if get(k, h) is not None]
            res[name][h] = block(np.array([get(keys[i], h) for i in ok]), y_all[ok], np.array([meta[keys[i]] for i in ok]))
    bench = json.loads((config.REPORTS / "v2" / "coverage_benchmark_v2.json").read_text())["models"]
    rep = {
        "module": "coverage", "version": "coverage-fresh-2024-v1", "created_at_utc": now_utc(), "registered_in": "docs/experiments/coverage_fresh_2024.md (committed before scoring)",
        "what_this_is": "A one-time evaluation on 143 labelled plays from three 2024-season games. Tracking from the BDB 2026 Prediction test input; released labels from the BDB 2026 Analytics label table. No model here trained on, tuned on or previously scored any 2024 play.",
        "limits": ["143 plays from 3 games in weeks 14, 15 and 18: intervals are wide and three games cannot represent a season", "same release format and the same post-play selection of tracked players as 2023: this tests a new season, not new tracking or a new player selection",
                   "labels are the released labels; who charted them is not stated", "now examined: this sample cannot serve as a fresh test again"],
        "bundles": {f"seed{s}": bundles[s][2] for s in seeds}, "plays": len(keys), "seasons": sorted({lab[k]["season"] for k in keys}), "weeks": sorted({lab[k]["week"] for k in keys}),
        "label_counts": {"man": int(y_all.sum()), "zone": int((1 - y_all).sum())}, "results": res,
        "for_reference_2023_previously_examined_benchmark_post_1_5s": {m: {k: bench[m]["post_1_5s"]["benchmark"][k] for k in ("n", "accuracy", "man_recall", "log_loss", "brier")} for m in ("v2_temporal", "v2_temporal_3seed")},
    }
    write_json(OUT / "coverage_fresh_2024.json", rep)
    for name, r in res.items():
        for h, b in r.items():
            log(f"{name:32s} {h:10s} n {b['n']:3d} acc {b['accuracy']:.3f} {[round(x, 3) for x in b['accuracy_interval_95_wilson']]} bal {b['balanced_accuracy']:.3f} manR {b['man_recall']:.3f} manP {b['man_precision']:.3f} "
                f"ll {b['log_loss']:.3f} {[round(x, 3) for x in b['log_loss_interval_95_play_bootstrap']]} brier {b['brier']:.3f} prauc {b['man_pr_auc']:.3f} | always-zone {b['always_zone_accuracy']:.3f} prior ll {b['class_prior_log_loss']:.3f}")
    log(json.dumps(res["primary: seed 42"]["post_1_5s"]["by_game"]))
    log(json.dumps(res["primary: seed 42"]["post_1_5s"]["reliability"]))
    return rep
