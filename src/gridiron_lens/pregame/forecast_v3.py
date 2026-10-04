"""Prospective forecasts with auditable provenance (pregame v3 forecasting protocol).

What changed from `run_v2.forecast` (which is left untouched, with its one legacy file):

- A forecast is built from a **source snapshot**: the schedule file and every play-by-play file, with the bytes preserved
  locally, a SHA-256 per file, the fetch time, the source URL and the server's Last-Modified/ETag when it sends them.
- The creation time is read **after** fitting, immediately before the record is persisted, and timing eligibility is derived
  from that persisted time, the snapshot time and the kickoff converted from US-Eastern to UTC.
- The fitted model is saved as a bundle and its hash, the training cohort hash and the pipeline hash go into every record.
- Records are idempotent per (game, model version, cutoff, source snapshot) and are never overwritten.
- A game with no listed kickoff time is not eligible for an official forecast.

The 28-hour prior-kickoff rule in `features_v2` remains a reconstruction proxy for "this game had finished and been
published"; a snapshot proves only that the bytes were on this machine at the recorded time.
The model is the frozen v2 choice (hyperparameters selected on 2012-2022). No gain over Elo has been established.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import polars as pl

from ..shared import config
from ..shared.provenance import sha256_file, write_json
from . import features_v2 as F
from . import model_v2 as M
from . import pipeline

ET = ZoneInfo("America/New_York")
SNAPSHOTS = config.ROOT / "data" / "snapshots" / "nflverse"
FORECASTS = config.REPORTS / "forecasts" / "v3"
LEGACY = config.REPORTS / "forecasts"
BUNDLES = config.MODELS / "pregame" / "v3"
WEB = config.WEB_DEMO / "pregame"
MODEL_VERSION = "pregame-v2-frozen"            # hyperparameters frozen by the v2 development selection
PROTOCOL = "forecast-protocol-v3"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
PRIMARY, CONTROL = "elo_offset", "elo"
POLICY = {
    "update": "Before each run the frozen v2 configuration is refitted on every decided game in the source snapshot. Hyperparameters, feature groups and the blend weight do not change.",
    "official_record": "For each game and model version: the latest record created at or before the cutoff, built from a snapshot fetched at or before the cutoff, whose kickoff equals the kickoff actually played. Earlier eligible records are early previews. Records created after the cutoff are late and are never scored in the official cohort.",
    "ties": "A tie is recorded as an outcome and excluded from log loss, Brier score and accuracy.",
    "postponed_or_cancelled": "If the scheduled kickoff changes after a record is written, the record is kept and marked as made for a superseded kickoff; a new record is needed for the new cutoff. Cancelled games are never scored.",
    "missing_kickoff": "A game with no kickoff time in the schedule gets no official forecast.",
    "primary_model": PRIMARY, "control": CONTROL,
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso(t: datetime) -> str:
    return t.astimezone(UTC).isoformat(timespec="seconds")


def kickoff_utc(gameday, gametime: str | None) -> datetime | None:
    """Published kickoff (US-Eastern wall time) as an aware UTC time. None when the schedule lists no time."""
    if gametime is None or gameday is None:
        return None
    h, m = (int(x) for x in gametime.split(":")[:2])
    return datetime(gameday.year, gameday.month, gameday.day, h, m, tzinfo=ET).astimezone(UTC)


# ---------------------------------------------------------------- snapshots

def _fetch(url: str, dest: Path) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "gridiron-lens-snapshot"})
    with urllib.request.urlopen(req, timeout=120) as r:
        dest.write_bytes(r.read())
        head = {"last_modified": r.headers.get("Last-Modified"), "etag": r.headers.get("ETag")}
    return {"source_url": url, "fetched_at_utc": iso(utc_now()), "publication": head, "obtained": "downloaded for this snapshot"}


def snapshot(season: int | None = None, fetch: bool = True, clock=utc_now, log=print) -> dict:
    """Preserve the bytes of every pregame input and describe them. Raises if a refresh fails: stale data is never used silently."""
    t = clock()
    sid = t.strftime("%Y%m%dT%H%M%SZ")
    d = SNAPSHOTS / sid
    (d / "pbp").mkdir(parents=True, exist_ok=False)
    season = season or (t.year if t.month >= 8 else t.year - 1)
    files = []
    if fetch:
        files.append({"name": "games.csv", **_fetch(pipeline.SOURCE_URL, d / "games.csv")})
        files.append({"name": f"pbp/play_by_play_{season}.parquet", **_fetch(PBP_URL.format(season=season), d / "pbp" / f"play_by_play_{season}.parquet")})
    else:
        os.link(pipeline.RAW, d / "games.csv")
        files.append({"name": "games.csv", "source_url": pipeline.SOURCE_URL, "fetched_at_utc": None, "publication": None, "obtained": "local file; original fetch time not recorded"})
    for f in sorted(F.PBP_DIR.glob("play_by_play_*.parquet")):
        if not (d / "pbp" / f.name).exists():
            os.link(f, d / "pbp" / f.name)                                 # hard link: bytes are preserved even if the raw file is later replaced
            files.append({"name": f"pbp/{f.name}", "source_url": PBP_URL.format(season=f.stem[-4:]), "fetched_at_utc": None, "publication": None,
                          "obtained": "local file from an earlier download; original fetch time not recorded"})
    for rec in files:
        p = d / rec["name"]
        rec |= {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    games = pipeline.load_games(d / "games.csv")
    done = games.filter(pl.col("result").is_not_null())
    cur = pl.read_parquet(d / "pbp" / f"play_by_play_{season}.parquet", columns=["game_id", "week", "game_date"])
    meta = {
        "snapshot_id": sid, "created_at_utc": iso(t), "protocol": PROTOCOL, "season": season, "files": files,
        "content_sha256": hashlib.sha256("".join(f"{r['name']}:{r['sha256']}" for r in sorted(files, key=lambda r: r["name"])).encode()).hexdigest(),
        "coverage": {"schedule_last_completed_game": str(done["gameday"].max()), "schedule_completed_games_this_season": done.filter(pl.col("season") == season).height,
                     "pbp_latest_game_date": str(cur["game_date"].max()), "pbp_latest_week": int(cur["week"].max()), "pbp_games_this_season": cur["game_id"].n_unique()},
        "schema": {"games_columns_sha256": hashlib.sha256(",".join(games.columns).encode()).hexdigest()},
        "limits": "Proves these bytes were on this machine at created_at_utc. It does not prove when each row was first published upstream, and today's historical rows may have been revised since the games were played.",
    }
    write_json(d / "snapshot.json", meta)
    log(f"snapshot {sid}: schedule through {meta['coverage']['schedule_last_completed_game']}, play-by-play through {meta['coverage']['pbp_latest_game_date']}")
    return meta


def latest_snapshot() -> dict | None:
    s = sorted(SNAPSHOTS.glob("*/snapshot.json"))
    return json.loads(s[-1].read_text()) if s else None


# ---------------------------------------------------------------- forecasts

def pipeline_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()


def timing(created: datetime, snapshot_created: datetime, kickoff: datetime | None) -> dict:
    """Eligibility from persisted times only."""
    if kickoff is None:
        return {"timing": "ineligible", "eligible_for_official_cohort": False, "reason": "no kickoff time in the schedule", "cutoff_utc": None}
    cutoff = kickoff - timedelta(hours=F.CUTOFF_HOURS)
    if created > kickoff:
        t, why = "after_kickoff", "created after the scheduled kickoff: a reconstruction, not a forecast"
    elif created > cutoff:
        t, why = "late", "created after the 24-hour cutoff"
    elif snapshot_created > cutoff:
        t, why = "late", "source snapshot fetched after the cutoff"
    else:
        t, why = "before_cutoff", None
    return {"timing": t, "eligible_for_official_cohort": t == "before_cutoff", "reason": why, "cutoff_utc": iso(cutoff)}


def existing_keys() -> set[str]:
    return {r["idempotency_key"] for f in FORECASTS.glob("forecasts_*.json") for r in json.loads(f.read_text())["records"]}


def forecast(snapshot_id: str | None = None, clock=utc_now, days_ahead: int = 9, log=print) -> dict:
    snap = json.loads((SNAPSHOTS / snapshot_id / "snapshot.json").read_text()) if snapshot_id else latest_snapshot()
    if snap is None:
        raise SystemExit("No source snapshot. Run `bin/pregame-lens snapshot` first.")
    sdir = SNAPSHOTS / snap["snapshot_id"]
    sel = json.loads((config.REPORTS / "v2" / "pregame_selection_v2.json").read_text())
    games = pipeline.load_games(sdir / "games.csv")
    feat = F.build(games, *F.load_stats(sdir / "pbp"))
    decided = (feat.filter(pl.col("result").is_not_null() & (pl.col("result") != 0) & (pl.col("season") >= M.FIRST))
               .with_columns((pl.col("result") > 0).cast(pl.Int8).alias("home_win")).sort(["gameday", "game_id"]))
    started = clock()
    listed = {g["game_id"]: g["gametime"] for g in games.iter_rows(named=True)}
    up = feat.filter(pl.col("result").is_null()).sort("kickoff")
    up = up.filter(pl.Series([(k := kickoff_utc(g["gameday"], listed[g["game_id"]])) is None or started < k <= started + timedelta(days=days_ahead)
                              for g in up.iter_rows(named=True)], dtype=pl.Boolean))
    if up.height == 0:
        return {"created": 0, "skipped": 0, "snapshot": snap["snapshot_id"]}
    eo, mg, w = sel["chosen"]["elo_offset"], sel["chosen"]["margin"], sel["blend"]["weight_elo_offset"]
    cols_e = F.group_columns(M.GROUP_SETS[eo["groups"]], eo["window"] or F.VARIANTS[0])
    cols_m = F.group_columns(M.GROUP_SETS[mg["groups"]], mg["window"] or F.VARIANTS[0])
    up = up.with_columns(pl.lit(0).alias("home_win"))                       # placeholder so the shared code path runs; never read
    pe, fe = M.fit_predict(decided, up, "elo_offset", cols_e, eo["penalty"])
    pm, fm = M.fit_predict(decided, up, "margin", cols_m, mg["penalty"])
    cohort = hashlib.sha256(",".join(decided["game_id"].to_list()).encode()).hexdigest()
    bundle = {"model_version": MODEL_VERSION, "selection": {"elo_offset": eo, "margin": mg, "blend_weight_elo_offset": w}, "elo": pipeline.ELO,
              "elo_offset": {"columns": cols_e, "mean": fe["scaler"].mean, "sd": fe["scaler"].sd, "weights": fe["w"], "qb_prior": fe["qb_prior"]},
              "margin": {"columns": fm["cols"], "mean": fm["scaler"].mean, "sd": fm["scaler"].sd, "coef": fm["coef"], "intercept": fm["intercept"], "sigma": fm["sigma"]},
              "training_cohort_sha256": cohort, "trained_on_games": decided.height, "trained_through": str(decided["gameday"].max()), "snapshot_id": snap["snapshot_id"]}
    BUNDLES.mkdir(parents=True, exist_ok=True)
    bpath = BUNDLES / f"bundle_{snap['snapshot_id']}.joblib"
    joblib.dump(bundle, bpath)
    bhash, phash = sha256_file(bpath), pipeline_hash()
    seen, snap_t = existing_keys(), datetime.fromisoformat(snap["created_at_utc"])
    created = clock()                                                       # read after fitting, immediately before persisting
    records, skipped = [], 0
    for i, g in enumerate(up.iter_rows(named=True)):
        k = kickoff_utc(g["gameday"], listed[g["game_id"]])
        tm = timing(created, snap_t, k)
        key = hashlib.sha256(f"{g['game_id']}|{MODEL_VERSION}|{tm['cutoff_utc']}|{snap['content_sha256']}".encode()).hexdigest()[:24]
        if key in seen:
            skipped += 1
            continue
        raw = [c for c in ["elo_diff", *cols_e, "home_qb_epa_sum", "home_qb_dropbacks", "away_qb_epa_sum", "away_qb_dropbacks"] if c in g]
        feats = {c: (None if g[c] is None or np.isnan(g[c]) else float(g[c])) for c in raw}
        records.append({
            "record_id": f"{g['game_id']}@{iso(created)}", "idempotency_key": key, "protocol": PROTOCOL, "kind": "live forecast",
            "game_id": g["game_id"], "season": g["season"], "week": g["week"], "home": g["home_team"], "away": g["away_team"],
            "kickoff_utc": None if k is None else iso(k), "kickoff_eastern": None if k is None else g["kickoff"].isoformat(),
            "cutoff_utc": tm["cutoff_utc"], "cutoff_eastern": None if k is None else (g["kickoff"] - timedelta(hours=F.CUTOFF_HOURS)).isoformat(),
            "created_at_utc": iso(created), "timing": tm["timing"], "eligible_for_official_cohort": tm["eligible_for_official_cohort"], "timing_reason": tm["reason"],
            "created_before_cutoff": tm["eligible_for_official_cohort"], "timing_note": tm["reason"],
            "model_version": MODEL_VERSION, "pipeline_sha256": phash, "bundle_sha256": bhash, "bundle_file": config.rel(bpath),
            "training": {"through": bundle["trained_through"], "games": decided.height, "cohort_sha256": cohort}, "trained_on_games": decided.height,
            "calibration_version": "none (probabilities used as fitted)", "calibrator": "none (probabilities used as fitted)",
            "source_snapshot": {"snapshot_id": snap["snapshot_id"], "created_at_utc": snap["created_at_utc"], "content_sha256": snap["content_sha256"],
                                "coverage": snap["coverage"], "files": [{k2: f[k2] for k2 in ("name", "sha256", "fetched_at_utc")} for f in snap["files"] if f["fetched_at_utc"]],
                                "historical_files": sum(1 for f in snap["files"] if not f["fetched_at_utc"])},
            "data_snapshot": {"last_completed_game": snap["coverage"]["schedule_last_completed_game"], "completed_games": decided.height},
            "prior_game_rule": f"A finished game counts only if it kicked off at least {F.CUTOFF_HOURS + F.GAME_HOURS} hours before this kickoff (a proxy for publication time).",
            "qb_assumption": {"home": g["home_qb_name"], "away": g["away_qb_name"], "basis": "primary passer in each team's previous game; not a confirmed starter"},
            "features": feats, "standardized_model_inputs": {c: float(fe["z"][i, j]) for j, c in enumerate(cols_e)},
            "quarterback_prior_epa_per_dropback": float(fe["qb_prior"]),
            "probabilities": {"elo": float(g["p_elo"]), "elo_offset": float(pe[i]), "margin": float(pm[i]), "blend": float(w * pe[i] + (1 - w) * pm[i])},
            "explanation": {"unit": "log-odds of a home win", "elo_logit": float(M.elo_logit(np.array([g["p_elo"]]))[0]), "intercept": float(fe["w"][0]),
                            "terms": sorted(({"feature": c, "description": F.describe(c), "log_odds": float(fe["z"][i, j] * fe["w"][1 + j])} for j, c in enumerate(cols_e)),
                                            key=lambda t: -abs(t["log_odds"]))},
            "predicted_margin": float(fm["margin"][i]), "outcome": None,
        })
    if not records:
        return {"created": 0, "skipped": skipped, "snapshot": snap["snapshot_id"]}
    FORECASTS.mkdir(parents=True, exist_ok=True)
    path = FORECASTS / f"forecasts_{created.strftime('%Y%m%dT%H%M%SZ')}.json"
    if path.exists():
        raise SystemExit("A forecast file with this timestamp already exists; forecast records are never overwritten.")
    write_json(path, {"created_at_utc": iso(created), "protocol": PROTOCOL, "policy": POLICY, "fit_started_at_utc": iso(started), "records": records})
    counts = {t: sum(r["timing"] == t for r in records) for t in ("before_cutoff", "late", "after_kickoff", "ineligible")}
    log(f"wrote {len(records)} records to {config.rel(path)}: {counts}; skipped {skipped} already recorded")
    return {"created": len(records), "skipped": skipped, "file": config.rel(path), "timing": counts, "snapshot": snap["snapshot_id"]}


# ---------------------------------------------------------------- outcomes and site export

def _ll(p: float, y: int) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)))


def publish(games_path: Path | None = None, clock=utc_now) -> dict:
    """Site export. Outcomes are attached beside each record from the newest schedule snapshot; records themselves are never edited."""
    snap = latest_snapshot()
    gp = games_path or (SNAPSHOTS / snap["snapshot_id"] / "games.csv" if snap else pipeline.RAW)
    games = {g["game_id"]: g for g in pipeline.load_games(gp).iter_rows(named=True)}
    recs = []
    for f in sorted(LEGACY.glob("forecasts_*.json")):
        for r in json.loads(f.read_text())["records"]:
            recs.append(r | {"protocol": "legacy", "timing": "before_cutoff" if r["created_before_cutoff"] else "late", "eligible_for_official_cohort": False,
                             "provenance": "limited: only the schedule file was hashed and the creation time was read before fitting. Kept for audit; never in the official cohort."})
    for f in sorted(FORECASTS.glob("forecasts_*.json")):
        recs += [r | {"provenance": "full: source snapshot, bundle, cohort and pipeline hashes recorded"} for r in json.loads(f.read_text())["records"]]
    official: dict[tuple, dict] = {}
    for r in recs:
        g = games.get(r["game_id"])
        k = None if g is None else kickoff_utc(g["gameday"], g["gametime"])
        same_kick = r.get("kickoff_utc") is not None and k is not None and iso(k) == r["kickoff_utc"]
        r["schedule_status"] = "legacy" if r["protocol"] == "legacy" else "current kickoff" if same_kick else "made for a superseded kickoff"
        done = g is not None and g["result"] is not None
        r["outcome_attached"] = None if not done else {"home_score": g["home_score"], "away_score": g["away_score"], "home_won": g["result"] > 0, "tie": g["result"] == 0,
                                                       "source": config.rel(gp) if snap is None else f"snapshot {snap['snapshot_id']}"}
        r["role"] = "legacy (not scored)" if r["protocol"] == "legacy" else "late" if r["timing"] != "before_cutoff" else "early preview"
        if r["eligible_for_official_cohort"] and same_kick:
            key = (r["game_id"], r["model_version"])
            if key not in official or official[key]["created_at_utc"] < r["created_at_utc"]:
                official[key] = r
    for r in official.values():
        r["role"] = "official"
    scored = [r for r in official.values() if r["outcome_attached"] and not r["outcome_attached"]["tie"]]
    board = None
    if scored:
        y = [int(r["outcome_attached"]["home_won"]) for r in scored]
        board = {m: {"log_loss": float(np.mean([_ll(r["probabilities"][m], yy) for r, yy in zip(scored, y)])),
                     "brier": float(np.mean([(r["probabilities"][m] - yy) ** 2 for r, yy in zip(scored, y)])),
                     "accuracy": float(np.mean([(r["probabilities"][m] >= 0.5) == bool(yy) for r, yy in zip(scored, y)]))} for m in ("elo", "elo_offset", "margin", "blend")}
    out = {"generated_at": iso(clock()), "protocol": PROTOCOL, "policy": POLICY,
           "note": "Forecast records written before kickoff and never edited. Outcomes are attached in a separate field. Only records with role 'official' are scored.",
           "counts": {"records": len(recs), "official": len(official), "early_preview": sum(r["role"] == "early preview" for r in recs), "late": sum(r["role"] == "late" for r in recs),
                      "legacy_limited_provenance": sum(r["protocol"] == "legacy" for r in recs), "official_with_outcome": len(scored)},
           "leaderboard": {"season": 2026, "games_scored": len(scored), "models": board,
                           "caution": "A prospective record over this few games cannot separate these models. No gain over Elo has been established."},
           "source_through": None if snap is None else snap["coverage"], "records": recs}
    write_json(WEB / "forecasts.json", out)
    return out["counts"]
