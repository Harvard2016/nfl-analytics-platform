"""Local processing for a video file the user is authorized to use. Nothing is uploaded; outputs stay beside the file.

Status: written but NOT yet run end to end on this machine. ffmpeg is not installed here and no authorized footage has
been supplied, so this path has no measured results. It produces loudness-based candidates only (experiment H0);
the embedding models need the same feature extractors the benchmark used, which are not bundled.

    PYTHONPATH=src .venv/bin/python -m gridiron_lens.highlights.local_media /path/to/game.mp4 --minutes 3

Steps: ffmpeg extracts mono 16 kHz audio -> loudness per 2-second clip -> level above a 5-minute local background ->
top non-overlapping 10-second blocks up to the budget -> an editable cut list (JSON) with separate fields for the
candidate moment and the clip start/end, padded by default so a cut includes the lead-up. Transcription with a local
Whisper model is an optional later step and is not wired in.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from . import models as M

CLIP = 2.0
RATE = 16000


def extract_audio(video: Path, wav: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install it (for example `brew install ffmpeg`) and run again.")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(RATE), str(wav)], check=True, stdin=subprocess.DEVNULL)


def loudness_db(wav: Path) -> np.ndarray:
    with wave.open(str(wav), "rb") as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2
        n = int(CLIP * RATE)
        out = []
        while True:
            buf = w.readframes(n)
            if len(buf) < 2 * n:
                break
            x = np.frombuffer(buf, dtype=np.int16).astype(np.float64) / 32768.0
            out.append(20 * np.log10(max(np.sqrt(np.mean(x * x)), 1e-6)))
    return np.array(out, np.float32)


def cut_list(video: Path, minutes: float, lead_s: float = 6.0, tail_s: float = 4.0) -> dict:
    wav = video.with_suffix(".gridiron_audio.wav")
    extract_audio(video, wav)
    vol = loudness_db(wav)
    score = M.loudness_features(vol)[:, 0]
    sel = M.select_budget(score, int(minutes * 60 / CLIP))
    cuts = [{"candidate_moment_s": float((a + int(np.argmax(score[a:b]))) * CLIP), "clip_start_s": max(0.0, a * CLIP - lead_s), "clip_end_s": b * CLIP + tail_s,
             "loudness_above_background_db": round(float(score[a:b].max()), 1), "source": "loudness only (H0)", "reviewed": False, "event_labels": [], "replay": "unknown"}
            for a, b in M.runs_of_ones(sel.astype(int))]
    return {"video": video.name, "clips": len(vol), "clip_seconds": CLIP, "note": "Candidates from loudness alone. Edit clip_start_s and clip_end_s before cutting; the candidate moment is kept separately.", "cuts": cuts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--minutes", type=float, default=3.0)
    a = ap.parse_args()
    out = a.video.with_suffix(".gridiron_cuts.json")
    out.write_text(json.dumps(cut_list(a.video, a.minutes), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
