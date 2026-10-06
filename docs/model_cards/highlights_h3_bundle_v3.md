# Model card: highlight ranker H3, inference bundle v3

**Version:** `highlights-h3-bundle-v3` (`models/highlights/v3/`: `weights.pt`, `transforms.joblib`, `manifest.json` with checksums). Same weights as v2; the bundle adds the transforms v2 did not save.

| | |
|---|---|
| Task | Rank 2-second clips of a full broadcast by agreement with the editors' highlight selection. Not event detection. Scores are not probabilities |
| Inputs | Per clip: CLIP (512), SlowFast (2304) and PANN (2048) embeddings as released with SVHighlights, plus loudness in dB |
| Transforms | Per stream: IncrementalPCA to 64 components, then standardize; fitted on the 28 training games. Loudness: level above a 5-minute centred median, change from the previous clip, within-video z-score |
| Architecture | 64-d projection per stream with a stream mask, three 1-D convolution blocks (kernel 5). Sees 6 clips (12 s) on each side: offline analysis, not live |
| Missing streams | Passed as zeros with mask bit 0. Trained with 15% stream dropout. Benchmark figures apply only with all four streams |

## Parity (measured 2026-10-04)

- Reconstructed transforms reproduce the cached model inputs: max absolute difference 2e-13 (CLIP), 4e-9 (SlowFast), 2e-12 (PANN).
- Raw embeddings through the bundle reproduce the saved v2 scores on all 40 games: max difference 6e-8.
- Chunked inference (2,048 clips, 16-clip overlap) equals a single pass to 4e-6.

Details: `reports/v3/highlights_bundle_parity.json`. Rebuild: `PYTHONPATH=src .venv/bin/python -m gridiron_lens.highlights.bundle`.

## Measured ranking quality

6 test games, previously examined in v2: mean average precision 0.619 (random 0.088). Strict 180-second reel: precision 0.780, recall of labelled time 0.193; no reel can exceed 0.272 because the editors' reels are far longer than 3 minutes. See `reports/v3/highlights_eval_v3.json`.

## What it cannot do yet

The bundle does **not** include the CLIP, SlowFast and PANN extractors. They must match the upstream ones exactly (checkpoint, frame sampling, resolution, normalization, clip boundaries). They are not installed or parity-checked here, so the trained model cannot score a new video. Uploads use the loudness baseline, labelled as such.

## Limits

40 NFL broadcasts, one channel, 2016–2024. Labels come from frame alignment with official highlight videos and include replays. CC BY-NC 4.0: non-commercial use, attribution required. No broadcast rights are granted by the dataset.
