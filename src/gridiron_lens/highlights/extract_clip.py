"""Experimental CLIP stream for new video, and what can and cannot be verified about it.

The benchmark's `vid_clip` features came from the HERO Video Feature Extractor. What the release and the features show:
512 dimensions, one vector per 2-second clip, not L2-normalized (norm about 10), which matches OpenAI CLIP ViT-B/32
`encode_image` output. What is NOT documented and cannot be read from the features: which frame of each clip was encoded,
the exact resize and crop, and the decoder's frame-rounding.

This module re-creates that stream as closely as the documentation allows (one frame per 2 s, short side 224, centre crop,
CLIP normalization, ViT-B/32 OpenAI weights, no L2 normalization). It then compares the result with the released
features by a fingerprint that does not need the original video: CLIP image features share a large constant component,
so the per-dimension mean vector of any video is close to that of any other video **from the same model and
preprocessing**, and far from it otherwise.

A matching fingerprint supports "same model family and scale". It cannot establish frame-level equivalence: no benchmark
source video is on disk and downloading one is out of bounds. So anything scored from these features is EXPERIMENTAL and
is never reported with benchmark figures.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import torch

CLIP_S, SIZE, BATCH = 2.0, 224, 64
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
MODEL_NAME = "open_clip ViT-B-32-quickgelu / openai (the original OpenAI CLIP ViT-B/32 weights)"
_M: dict = {}


def model():
    if not _M:
        import open_clip
        m, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        _M["m"] = m.eval()
    return _M["m"]


def extract(video: Path, cancelled=None, log=None) -> np.ndarray:
    """One 512-d vector per 2-second clip. Frames are decoded by ffmpeg in a stream and encoded in batches: memory stays flat."""
    vf = f"fps=1/{CLIP_S:g},scale={SIZE}:{SIZE}:force_original_aspect_ratio=increase,crop={SIZE}:{SIZE}"
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
    m, out, n = model(), [], SIZE * SIZE * 3
    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    t0 = time.time()
    while True:
        if cancelled is not None and cancelled():
            p.kill()
            raise RuntimeError("cancelled")
        buf = p.stdout.read(n * BATCH)
        if not buf:
            break
        x = torch.frombuffer(bytearray(buf[: len(buf) // n * n]), dtype=torch.uint8).view(-1, SIZE, SIZE, 3).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            out.append(m.encode_image((x - mean) / std).float().numpy())
        if log:
            log(f"  CLIP: {sum(len(o) for o in out)} clips ({time.time() - t0:.0f}s)")
    p.wait()
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def fingerprint(new: np.ndarray, reference: list[np.ndarray]) -> dict:
    """Compare per-dimension mean vectors. `reference` is a list of benchmark games' released vid_clip arrays."""
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    ref_means = [r.mean(0) for r in reference]
    between = [cos(ref_means[i], ref_means[j]) for i in range(len(ref_means)) for j in range(i + 1, len(ref_means))]
    mine = [cos(new.mean(0), r) for r in ref_means]
    rng = np.random.default_rng(0)
    return {"new_clips": len(new), "new_norm_mean": float(np.linalg.norm(new, axis=1).mean()), "reference_norm_mean": float(np.mean([np.linalg.norm(r, axis=1).mean() for r in reference])),
            "mean_vector_cosine_new_vs_benchmark_games": {"mean": float(np.mean(mine)), "min": float(np.min(mine))},
            "mean_vector_cosine_between_benchmark_games": {"mean": float(np.mean(between)), "min": float(np.min(between))},
            "mean_vector_cosine_shuffled_dimensions_control": float(np.mean([cos(rng.permutation(new.mean(0)), r) for r in ref_means])),
            "adjacent_clip_cosine": {"new": float(np.mean(np.sum(new[1:] * new[:-1], 1) / (np.linalg.norm(new[1:], axis=1) * np.linalg.norm(new[:-1], axis=1)))),
                                     "reference": float(np.mean([np.mean(np.sum(r[1:] * r[:-1], 1) / (np.linalg.norm(r[1:], axis=1) * np.linalg.norm(r[:-1], axis=1))) for r in reference]))}}
