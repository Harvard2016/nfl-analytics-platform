"""Speech-to-text plumbing. Parsing and grouping are checked on a hand-written transcript; the real model runs only where it is installed."""
import shutil
import subprocess

import pytest

from gridiron_lens.service import asr


def _t(a, b, text):
    return {"offsets": {"from": int(a * 1000), "to": int(b * 1000)}, "text": text}


def test_words_and_sound_tags_are_separated_and_offset():
    obj = {"transcription": [_t(0.0, 0.4, " (crowd"), _t(0.4, 1.0, " cheering)"), _t(1.2, 1.5, " Second"), _t(1.5, 1.8, " down"), _t(1.8, 2.0, " -"), _t(2.0, 2.3, " [BLANK_AUDIO]"), _t(9.0, 9.4, " incomplete.")]}
    words, sounds = asr.parse(obj, offset_s=600.0)
    assert [w["word"] for w in words] == ["Second", "down", "incomplete."]
    assert words[0]["start_s"] == 601.2 and words[-1]["end_s"] == 609.4                    # chunk offset applied
    assert [s["label"] for s in sounds] == ["crowd cheering", "BLANK_AUDIO"] and sounds[0]["start_s"] == 600.0


def test_segments_break_on_pauses_and_length():
    words = [{"start_s": i * 0.4, "end_s": i * 0.4 + 0.3, "word": f"w{i}"} for i in range(10)] + [{"start_s": 20.0, "end_s": 20.3, "word": "later"}]
    seg = asr.segments(words)
    assert len(seg) == 2 and seg[0]["text"].startswith("w0 w1") and seg[1]["text"] == "later"
    long = asr.segments([{"start_s": i * 1.0, "end_s": i * 1.0 + 0.9, "word": "x"} for i in range(30)], max_s=12.0)
    assert all(s["end_s"] - s["start_s"] <= 12.0 for s in long) and len(long) >= 3


@pytest.mark.skipif(not asr.available() or shutil.which("ffmpeg") is None, reason="whisper.cpp and its model are not installed")
def test_real_model_runs_in_chunks_on_generated_audio(tmp_path, monkeypatch):
    wav = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=25", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
    monkeypatch.setattr(asr, "CHUNK_S", 10)
    out = asr.transcribe(wav, tmp_path)
    assert out["chunks"] == 3 and out["audio_seconds"] == 25.0 and isinstance(out["words"], list)   # a tone has no speech: no invented words is acceptable
    assert not list(tmp_path.glob("asr_*"))                                                       # chunk files are cleaned up
