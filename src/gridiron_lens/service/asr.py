"""Local speech-to-text with word timing, using whisper.cpp (`whisper-cli`) and a model file kept in models/asr/.

Nothing leaves the machine. Long recordings are transcribed in fixed chunks so memory stays flat and a cancelled job stops
between chunks. The model writes sound tags such as "(crowd cheering)"; those are kept apart from speech.

Profile on this machine (Apple M2, base.en, 148 MB model): 5 minutes of audio in about 14 s, about 400 MB peak.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import wave
from pathlib import Path

from ..shared import config

MODEL = config.MODELS / "asr" / "ggml-base.en.bin"
MODEL_NAME = "whisper.cpp base.en (ggml)"
CHUNK_S = 600
TAG = re.compile(r"^[\(\[].*[\)\]]$")


def available() -> bool:
    return shutil.which("whisper-cli") is not None and MODEL.exists()


def parse(obj: dict, offset_s: float = 0.0) -> tuple[list[dict], list[dict]]:
    """whisper.cpp JSON (one token group per entry) -> (spoken words, sound tags), times in seconds from the file start."""
    words, sounds, tag = [], [], None
    for t in obj.get("transcription", []):
        text = t["text"].strip()
        if not text or text == "-":
            continue
        a, b = t["offsets"]["from"] / 1000 + offset_s, t["offsets"]["to"] / 1000 + offset_s
        if tag is not None or text[0] in "([":                           # sound tags can span several entries: "(crowd" "cheering)"
            tag = (tag[0], f"{tag[2]} {text}", f"{tag[2]} {text}") if tag else (a, text, text)
            if text[-1] in ")]":
                sounds.append({"start_s": round(tag[0], 2), "end_s": round(b, 2), "label": tag[2].strip("()[] ")})
                tag = None
            continue
        words.append({"start_s": round(a, 2), "end_s": round(b, 2), "word": text})
    return words, sounds


def transcribe(wav: Path, workdir: Path, cancelled=None) -> dict:
    """Transcribe a 16 kHz mono WAV in chunks. Returns words, sound tags and what was run."""
    with wave.open(str(wav), "rb") as w:
        rate, total = w.getframerate(), w.getnframes()
        words, sounds, chunks = [], [], 0
        for start in range(0, total, CHUNK_S * rate):
            if cancelled is not None and cancelled():
                raise RuntimeError("cancelled")
            part = workdir / f"asr_{chunks:03d}.wav"
            w.setpos(start)
            with wave.open(str(part), "wb") as out:
                out.setnchannels(1), out.setsampwidth(2), out.setframerate(rate)
                out.writeframes(w.readframes(CHUNK_S * rate))
            r = subprocess.run(["whisper-cli", "-m", str(MODEL), "-f", str(part), "-ml", "1", "-sow", "-oj", "-of", str(part.with_suffix("")), "-np"],
                               capture_output=True, text=True, check=False, timeout=3600)
            js = part.with_suffix(".json")
            if r.returncode != 0 or not js.exists():
                raise RuntimeError("speech-to-text failed on a chunk: " + (r.stderr or "")[-200:])
            ws, ss = parse(json.loads(js.read_text(errors="replace")), start / rate)
            words += ws
            sounds += ss
            part.unlink(missing_ok=True), js.unlink(missing_ok=True)
            chunks += 1
    return {"words": words, "sounds": sounds, "model": MODEL_NAME, "chunks": chunks, "chunk_seconds": CHUNK_S, "audio_seconds": round(total / rate, 2)}


def segments(words: list[dict], gap_s: float = 1.2, max_s: float = 12.0) -> list[dict]:
    """Group words into short spoken segments for display: a new one after a pause or once it gets long."""
    out: list[dict] = []
    for w in words:
        if out and w["start_s"] - out[-1]["end_s"] <= gap_s and w["end_s"] - out[-1]["start_s"] <= max_s:
            out[-1]["text"] += " " + w["word"]
            out[-1]["end_s"] = w["end_s"]
        else:
            out.append({"start_s": w["start_s"], "end_s": w["end_s"], "text": w["word"]})
    return out
