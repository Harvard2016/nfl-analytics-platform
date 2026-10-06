"""The single background worker. Each job moves through real stages; nothing here reports progress it cannot measure."""
from __future__ import annotations

import hashlib
import json
import resource
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..coverage import relational
from ..coverage import upload as cov_upload
from ..highlights import decode as D
from ..highlights import models as HM
from ..shared.provenance import sha256_file, write_json
from . import asr, media
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
    bounds = media.audio_bounds(wav, len(vol), info["audio_start_s"] or 0.0)
    if len(vol) < 3:
        wav.unlink(missing_ok=True)
        raise media.MediaError("Too little audio to analyse.")
    commentary = None
    if row["mode"] == "commentary_experimental":
        if not asr.available():
            raise media.MediaError("Speech-to-text is not installed on the machine running the service.")
        store.update(jid, stage_note="transcribing speech locally in 10-minute chunks (whisper.cpp)")
        ta = time.time()
        try:
            tr = asr.transcribe(wav, d, lambda: store.cancelled(jid))
        except RuntimeError as e:
            wav.unlink(missing_ok=True)
            raise media.MediaError(str(e)) from e
        commentary = commentary_scores(tr, len(vol)) | {"transcription_seconds": round(time.time() - ta, 2)}
        for group in ("segments", "sounds"):
            commentary[group] = [g | {"start_s": g["start_s"] + bounds["audio_start_s"], "end_s": g["end_s"] + bounds["audio_start_s"]} for g in commentary[group]]
    wav.unlink(missing_ok=True)
    check(store, jid)
    feats = HM.loudness_features(vol)
    loud = feats[:, 0]
    score = commentary["score"] if commentary else loud
    store.update(jid, state="inferring", stage_note="ranking clips by the commentary word model (experimental)" if commentary else "loudness baseline (H0): level above the local background, then the strict-budget decoder")
    budget = float(min(max(4.0, float(opts.get("reel_seconds", 60))), info["duration_s"]))
    lead, tail = float(opts.get("lead_s", 2.0)), float(opts.get("tail_s", 2.0))
    audio_start = bounds["audio_start_s"]
    analysed_s = min(bounds["audio_end_s"], info["duration_s"]) - audio_start
    segs = D.decode(score, budget, duration_s=analysed_s, block=3, lead_s=lead, tail_s=tail)
    shifted = []
    for s in segs:
        i = min(len(vol) - 1, int(s.moment_s // media.CLIP_S))
        shifted.append(replace(s, start_s=s.start_s + audio_start, end_s=s.end_s + audio_start,
                               moment_s=(bounds["bin_start_s"][i] + bounds["bin_end_s"][i]) / 2))
    segs = shifted
    order = np.argsort(-np.where(np.isfinite(score), score, -np.inf))
    rank = np.empty(len(score), int)
    rank[order] = np.arange(len(score), 0, -1) * 100 // len(score)
    check(store, jid)
    store.update(jid, state="rendering", stage_note="cutting clips and the reel (re-encoded for exact durations)")
    assets, cuts, parts = {}, [], []
    ext = ".mp4" if info["has_video"] else ".m4a"
    # Cut merged spans, not individual segments: two segments whose padding overlaps become one clip, so no second plays twice.
    for a, b in D.merge([(s.start_s, s.end_s) for s in segs]):
        inside = sorted((s for s in segs if s.start_s >= a - 1e-9 and s.end_s <= b + 1e-9), key=lambda s: s.rank)
        name = f"cut_{len(cuts) + 1:02d}{ext}"
        media.cut(path, d / "media" / name, a, b, info["has_video"], lambda: store.cancelled(jid))
        got = media.probe(d / "media" / name)["duration_s"]
        assets[name] = name
        parts.append(d / "media" / name)
        cuts.append({"asset": name, "rank_in_reel": inside[0].rank, "candidate_moment_s": round(inside[0].moment_s, 2), "moments_s": [round(s.moment_s, 2) for s in inside],
                     "clip_start_s": round(a, 3), "clip_end_s": round(b, 3), "requested_s": round(b - a, 3), "ffprobe_duration_s": round(got, 3),
                     "loudness_above_background_db": round(float(loud[min(len(loud) - 1, int((inside[0].moment_s - audio_start) // media.CLIP_S))]), 2), "ranking_score": round(float(inside[0].score), 3), "sha256": sha256_file(d / "media" / name)})
    reel = None
    if cuts:
        spans = [(c["clip_start_s"], c["clip_end_s"]) for c in cuts]
        media.reel(path, d / "media" / f"reel{ext}", spans, budget, info["has_video"], info.get("fps"), lambda: store.cancelled(jid))
        assets[f"reel{ext}"] = f"reel{ext}"
        pr = media.probe(d / "media" / f"reel{ext}")
        longest = max(x for x in (pr["duration_s"], pr["video_duration_s"], pr["audio_duration_s"]) if x is not None)
        reel = {"asset": f"reel{ext}", "ffprobe_duration_s": round(pr["duration_s"], 3), "ffprobe_video_s": pr["video_duration_s"] and round(pr["video_duration_s"], 3),
                "ffprobe_audio_s": pr["audio_duration_s"] and round(pr["audio_duration_s"], 3), "budget_s": budget, "decoder_output_s": round(D.output_seconds(segs), 3),
                "within_budget": bool(longest <= budget + 1e-3),
                "duration_rule": "The exported reel's container, video and audio durations are each at or under the budget. It is encoded in one pass and ends up to about a quarter of a second before the selected seconds do, never after. Individual cut files can each run one frame over their own span.",
                "sha256": sha256_file(d / "media" / f"reel{ext}")}
        if not reel["within_budget"]:
            raise media.MediaError(f"The rendered reel is {longest:.3f} s for a {budget:.0f} s budget. It was not released.")
    assets["source"] = path.name
    store.update(jid, assets=assets)
    short = len(vol) < HM.BACKGROUND_CLIPS
    lrank = np.empty(len(loud), int)
    lrank[np.argsort(-np.where(np.isfinite(loud), loud, -np.inf))] = np.arange(len(loud), 0, -1) * 100 // len(loud)
    if commentary:
        head = {"mode": "commentary (experimental)", "mode_note": "Clips are ranked by a word model trained on NFL broadcast commentary, applied to a local machine transcript of this file. Speech here may be a stadium announcer, coaches or the crowd, not a commentator: treat the ranking as an experiment. It is not the trained multimodal model and carries no benchmark figure.",
                "evidence_streams_used": ["machine transcript (words and timing)", "audio loudness (shown, not used for ranking)"],
                "commentary": {"model": commentary["asr_model"], "word_model": "v2 commentary word model (TF-IDF + logistic regression), trained on 28 NFL broadcasts", "words": commentary["words"],
                               "segments": commentary["segments"], "sounds": commentary["sounds"], "transcription_seconds": commentary["transcription_seconds"],
                               "note": "Machine transcript: names and football terms are often misheard. It stays on this machine."}}
    else:
        head = {"mode": "loudness baseline (H0)", "mode_note": "Loudness above the local background only. This is not the trained multimodal model and carries none of its benchmark figures.",
                "evidence_streams_used": ["audio loudness"], "commentary": None}
    return {
        "schema": "highlights-upload-v2", "module": "highlights", "kind": "highlight ranking", **head,
        "media": {k: info[k] for k in ("duration_s", "audio_start_s", "audio_duration_s", "has_video", "width", "height", "format")}, "clip_seconds": media.CLIP_S,
        "time_origin": "seconds from the start of the uploaded file",
        "timeline": {**bounds, "loudness_db": [round(float(v), 2) for v in vol], "loudness_above_background_db": [round(float(v), 2) for v in loud], "rank_within_file": rank.tolist(),
                     "loudness_rank": lrank.tolist(), "ranked_by": "commentary word model" if commentary else "loudness above background"},
        "score_meaning": "A rank within this file (100 = highest). Not a probability and not a confidence.",
        "candidates": sorted(cuts, key=lambda c: c["rank_in_reel"]), "reel": reel,
        "event_types": "none: no event is detected or claimed",
        "domain_shift": ("This file is shorter than the 5-minute background window, so loudness is compared with the whole file. Rankings in short clips are less meaningful than in a full broadcast."
                         if short else "The background is the surrounding 5 minutes, as in the benchmark."),
        "measured": {"seconds": round(time.time() - t0, 2), "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
                     "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.stat().st_size < 64 * 2**20 else sha256_file(path)},
    }


_H1: dict = {}


def commentary_scores(tr: dict, n: int) -> dict:
    """v2 commentary word model on a local transcript. Same clip windows as the benchmark: words from 2 s before to 6 s after, and 6-20 s after."""
    import joblib

    from ..shared import config
    if not _H1:
        _H1.update(joblib.load(config.MODELS / "highlights" / "commentary_tfidf_logistic.joblib"))
    now, after = [[] for _ in range(n)], [[] for _ in range(n)]
    for w in tr["words"]:
        tok = "".join(ch for ch in w["word"].lower() if ch.isalnum() or ch == "'")
        if not tok:
            continue
        t = w["start_s"]
        for i in range(max(0, int((t - 6) // 2)), min(n, int((t + 2) // 2) + 1)):
            if 2 * i - 2 <= t < 2 * i + 6:
                now[i].append("n_" + tok)
        for i in range(max(0, int((t - 20) // 2)), min(n, int((t - 6) // 2) + 1)):
            if 2 * i + 6 <= t < 2 * i + 20:
                after[i].append("a_" + tok)
    docs = [" ".join(a + b) for a, b in zip(now, after)]
    score = _H1["model"].decision_function(_H1["vectorizer"].transform(docs)).astype(np.float32)
    return {"score": score, "words": len(tr["words"]), "segments": asr.segments(tr["words"]), "sounds": tr["sounds"], "asr_model": tr["model"]}


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
