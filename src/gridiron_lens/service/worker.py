"""The single background worker. Each job moves through real stages; nothing here reports progress it cannot measure."""
from __future__ import annotations

import hashlib
import json
import resource
import threading
import time
from pathlib import Path

import numpy as np

from ..coverage import relational
from ..coverage import upload as cov_upload
from ..highlights import decode as D
from ..highlights import models as HM
from ..shared.provenance import sha256_file, write_json
from . import media
from .store import Store

LIMITS = {
    "coverage": {"max_bytes": 2 * 2**20, "formats": [".csv", ".json"]},
    "highlights": {"max_bytes": 600 * 2**20, "max_duration_s": 1200.0, "min_duration_s": 10.0, "formats": "any container ffprobe can read with an audio stream",
                   "full_games": "use the command line: PYTHONPATH=src .venv/bin/python -m gridiron_lens.service.cli highlights <file>"},
}


class Cancelled(Exception):
    pass


def check(store: Store, jid: str) -> None:
    if store.cancelled(jid):
        raise Cancelled


# ---------------------------------------------------------------- coverage (tracking classification)

def run_coverage(store: Store, row, path: Path) -> dict:
    store.update(row["id"], state="validating", stage_note="checking the schema, coordinates and player counts")
    opts = json.loads(row["options"])
    parsed = cov_upload.parse(path.read_bytes(), "json" if path.suffix == ".json" else "csv")
    tens = cov_upload.build(parsed, "broader" if opts.get("mode") == "broader" else "release-compatible")
    check(store, row["id"])
    store.update(row["id"], state="inferring", stage_note="running the saved temporal model at each available cutoff", warnings=tens["warnings"])
    res = cov_upload.infer(tens, parsed.label, parsed.play_id)
    T = tens["frames_usable"]
    ent = []
    for side, arr, mask in (("defense", tens["defs"], tens["dmask"]), ("offense", tens["recs"], tens["rmask"])):
        for i, pid in enumerate(tens["ids"][side]):
            ent.append({"id": pid, "side": side, "name": pid, "position": None, "jersey": None, "passer": False,
                        "x": [round(float(v) + tens["los_x"], 2) for v in arr[:T, i, 0]], "y": [round(float(v) + tens["ref_y"], 2) for v in arr[:T, i, 1]],
                        "s": [round(float(np.hypot(*arr[f, i, 2:4])), 2) for f in range(T)],
                        "o": [round(float(np.degrees(np.arctan2(arr[f, i, 4], arr[f, i, 5])) % 360), 1) if mask[i] and tens["has_orientation"] else None for f in range(T)]})
    res["playback"] = {"frames": list(range(T)), "los_x": tens["los_x"], "entities": ent, "note": "Canonical view: offense moves toward larger x. Only the uploaded players are drawn.",
                       "cutoff_frames": relational.HORIZONS}
    return res


# ---------------------------------------------------------------- highlights (clip ranking)

def run_highlights(store: Store, row, path: Path) -> dict:
    jid, d = row["id"], store.dir(row["id"])
    opts = json.loads(row["options"])
    store.update(jid, state="validating", stage_note="reading the container with ffprobe")
    if not media.available():
        raise media.MediaError("ffmpeg and ffprobe are not installed on the machine running the service.")
    info = media.probe(path)
    lim = LIMITS["highlights"]
    if not info["has_audio"]:
        raise media.MediaError("The file has no audio stream. The loudness baseline needs audio.")
    if not lim["min_duration_s"] <= info["duration_s"] <= (float("inf") if opts.get("local_cli") else lim["max_duration_s"]):
        raise media.MediaError(f"Duration {info['duration_s']:.0f} s is outside the {lim['min_duration_s']:.0f}-{lim['max_duration_s']:.0f} s range for uploads. Longer files run from the command line.")
    check(store, jid)
    store.update(jid, state="extracting", stage_note="extracting mono 16 kHz audio and measuring loudness per 2-second clip")
    t0 = time.time()
    wav = d / "audio.wav"
    media.extract_audio(path, wav, lambda: store.cancelled(jid))
    vol = media.loudness_db(wav)
    wav.unlink(missing_ok=True)
    if len(vol) < 3:
        raise media.MediaError("Too little audio to analyse.")
    check(store, jid)
    store.update(jid, state="inferring", stage_note="loudness baseline (H0): level above the local background, then the strict-budget decoder")
    feats = HM.loudness_features(vol)
    score = feats[:, 0]
    budget = float(min(max(4.0, float(opts.get("reel_seconds", 60))), info["duration_s"]))
    lead, tail = float(opts.get("lead_s", 2.0)), float(opts.get("tail_s", 2.0))
    segs = D.decode(score, budget, duration_s=info["duration_s"], block=3, lead_s=lead, tail_s=tail)
    order = np.argsort(-np.where(np.isfinite(score), score, -np.inf))
    rank = np.empty(len(score), int)
    rank[order] = np.arange(len(score), 0, -1) * 100 // len(score)
    check(store, jid)
    store.update(jid, state="rendering", stage_note=f"cutting {len(segs)} clips and the reel (re-encoded for exact durations)")
    assets, cuts, parts = {}, [], []
    ext = ".mp4" if info["has_video"] else ".m4a"
    for s in sorted(segs, key=lambda s: s.start_s):
        name = f"cut_{len(cuts) + 1:02d}{ext}"
        media.cut(path, d / "media" / name, s.start_s, s.end_s, info["has_video"], lambda: store.cancelled(jid))
        got = media.probe(d / "media" / name)["duration_s"]
        assets[name] = name
        parts.append(d / "media" / name)
        cuts.append({"asset": name, "rank_in_reel": s.rank, "candidate_moment_s": round(s.moment_s, 2), "clip_start_s": round(s.start_s, 3), "clip_end_s": round(s.end_s, 3),
                     "requested_s": round(s.end_s - s.start_s, 3), "ffprobe_duration_s": round(got, 3), "loudness_above_background_db": round(float(s.score), 2),
                     "sha256": sha256_file(d / "media" / name)})
    reel = None
    if parts:
        media.concat(parts, d / "media" / f"reel{ext}", d / "media" / "concat.txt", lambda: store.cancelled(jid))
        (d / "media" / "concat.txt").unlink(missing_ok=True)
        assets[f"reel{ext}"] = f"reel{ext}"
        reel = {"asset": f"reel{ext}", "ffprobe_duration_s": round(media.probe(d / "media" / f"reel{ext}")["duration_s"], 3), "budget_s": budget,
                "decoder_output_s": round(D.output_seconds(segs), 3),
                "render_tolerance": "The selected seconds never exceed the budget. The rendered container can be longer by up to one audio frame per file (about 64 ms at 16 kHz) because AAC is written in whole frames.", "sha256": sha256_file(d / "media" / f"reel{ext}")}
    assets["source"] = path.name
    store.update(jid, assets=assets)
    short = len(vol) < HM.BACKGROUND_CLIPS
    return {
        "schema": "highlights-upload-v1", "module": "highlights", "kind": "highlight ranking", "mode": "loudness baseline (H0)",
        "mode_note": "Loudness above the local background only. This is not the trained multimodal model and carries none of its benchmark figures.",
        "media": {k: info[k] for k in ("duration_s", "has_video", "width", "height", "format")}, "clip_seconds": media.CLIP_S,
        "time_origin": "seconds from the start of the uploaded file",
        "timeline": {"loudness_db": [round(float(v), 2) for v in vol], "loudness_above_background_db": [round(float(v), 2) for v in score], "rank_within_file": rank.tolist()},
        "score_meaning": "A ranking score within this file (100 = loudest relative to its surroundings). Not a probability and not a confidence.",
        "candidates": sorted(cuts, key=lambda c: c["rank_in_reel"]), "reel": reel,
        "evidence_streams_used": ["audio loudness"], "event_types": "none: no event is detected or claimed",
        "domain_shift": ("This file is shorter than the 5-minute background window, so loudness is compared with the whole file. Rankings in short clips are less meaningful than in a full broadcast."
                         if short else "The background is the surrounding 5 minutes, as in the benchmark."),
        "measured": {"seconds": round(time.time() - t0, 2), "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
                     "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.stat().st_size < 64 * 2**20 else sha256_file(path)},
    }


RUNNERS = {"coverage": run_coverage, "highlights": run_highlights}


def process(store: Store, row) -> None:
    jid = row["id"]
    path = store.dir(jid) / row["input_file"]
    try:
        res = RUNNERS[row["module"]](store, row, path)
        check(store, jid)
        write_json(store.dir(jid) / "result.json", res)
        store.update(jid, state="complete", stage_note=None)
    except Cancelled:
        store.update(jid, state="cancelled", stage_note=None)
    except (cov_upload.UploadError, media.MediaError) as e:
        if str(e) == "cancelled":
            store.update(jid, state="cancelled", stage_note=None)
        else:
            store.update(jid, state="failed", error=str(e), stage_note=None)
    except Exception as e:                                                   # noqa: BLE001 - a job must never take the worker down
        store.update(jid, state="failed", error=f"Unexpected error ({type(e).__name__}). The job was not completed.", stage_note=None)


class Worker(threading.Thread):
    def __init__(self, store: Store, poll_s: float = 0.3):
        super().__init__(daemon=True)
        self.store, self.poll_s, self.stop = store, poll_s, threading.Event()

    def run(self) -> None:
        last_sweep = 0.0
        while not self.stop.is_set():
            if time.time() - last_sweep > 600:
                self.store.sweep()
                last_sweep = time.time()
            row = self.store.next_queued()
            if row is None:
                self.stop.wait(self.poll_s)
                continue
            process(self.store, row)
