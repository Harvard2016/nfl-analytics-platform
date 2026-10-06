# Highlight feature extractors: what is known, what was measured, what cannot be verified

Written 2026-10-06. Purpose: decide whether the trained H3 ranker can honestly score a newly uploaded video.

**Conclusion.** Not yet. The trained ranker needs three feature streams produced exactly as the benchmark's were. One of the three (CLIP) was re-created and checked as far as a check is possible without a benchmark source video; it matches in model family and scale and differs measurably in distribution. The other two were not built. No pipeline here may be called equivalent to benchmark H3. Uploads keep two labelled modes: the loudness baseline and experimental commentary.

## What the benchmark release says

SVHighlights states that video features came from the HERO Video Feature Extractor and audio features from PANN (`audioset_tagging_cnn`). It does not state checkpoints, frame sampling, resolution or clip timing.

## What the released features show (measured on this machine)

| Stream | Dimensions | Per 2-second clip | Observed properties | Consistent with |
|---|---:|---|---|---|
| `vid_clip` | 512 | yes (clip length 2.000 s from trim bounds) | not L2-normalized, norm about 10, signed values | OpenAI CLIP ViT-B/32 `encode_image` output |
| `vid_slowfast` | 2304 | yes | non-negative, norm about 23, 2% zeros | SlowFast R50 pooled features (2048 slow + 256 fast) after ReLU |
| `aud_pann` | 2048 | yes | non-negative, 83% zeros, norm about 12 | PANN CNN14 embedding after ReLU |

"Consistent with" is an inference from dimensions and value ranges. It is not confirmation of the checkpoint.

## What is not documented and cannot be read from the features

- Which frame (or frames) of each 2-second clip CLIP encoded, and the decoder's frame rounding.
- Resize, crop and colour handling before each video model.
- SlowFast's frame rate, frames per clip and spatial crop.
- The audio sample rate and windowing fed to PANN, and how a clip-level vector was pooled.
- Whether videos were re-encoded before extraction.

## The reference problem

Verifying an extractor needs one benchmark video together with its released features. The release contains no video, the source videos are on YouTube, and downloading them is outside what this project does. Without a reference, frame-level equivalence cannot be established for any stream.

## What was built and measured: the CLIP stream only

`src/gridiron_lens/highlights/extract_clip.py`: one frame per 2 s, short side 224, centre crop, CLIP normalization, OpenAI ViT-B/32 weights through `open_clip`, no L2 normalization.

| | Measured |
|---|---|
| Dependencies added | `open_clip_torch`, `torchvision`, `timm`, `pillow` and 4 small packages (optional extra `extract`); 577 MB of model cache including the weights |
| Speed | 112-minute 1080p game: 3,373 clips in 188 s. 5-minute excerpt: 33 s including model load |
| Peak memory | 1.1 GB |
| Scale | mean vector norm 9.92 against 10.05 for benchmark games |
| Model fingerprint | cosine between the per-dimension mean vector of the new features and of each benchmark game: 0.70. Between benchmark games: 0.93. Shuffled-dimension control: about 0 |
| After the bundle's PCA and scaling | 3.8% of inputs beyond 3 standard deviations, against 0.8% for a benchmark game |

Reading: the same model family and scale, not the same distribution. A night high-school game from one press-box camera with no broadcast graphics is far from NFL broadcasts, and undocumented preprocessing differences cannot be separated from that. This is evidence against treating the features as interchangeable.

## What the missing streams would cost (not installed; sizes are approximate and unverified)

| Stream | Likely upstream | Needs | Why it was not built |
|---|---|---|---|
| SlowFast | `SLOWFAST_8X8_R50.pkl` (Kinetics) inside HERO's Docker image, built for CUDA | a few hundred MB of weights, a PySlowFast or PyTorchVideo stack, dense frame decoding (tens of frames per clip): by far the slowest stream on a CPU | no reference to verify against, and the largest dependency and runtime cost |
| PANN | CNN14 AudioSet checkpoint from `audioset_tagging_cnn` | about 300 MB of weights, `torchlibrosa`, 32 kHz audio | cheapest to add; still unverifiable without a reference |

## What a partial model can do, measured on the benchmark itself

H3 run on the NFL games with released features, masking streams (mean average precision):

| Streams present | Validation | Examined test |
|---|---:|---:|
| all four | 0.688 | 0.619 |
| CLIP + loudness | 0.563 | 0.466 |
| loudness only, through H3 | 0.418 | 0.341 |
| CLIP only | 0.384 | 0.349 |

So even with perfect features, CLIP plus loudness is a clearly weaker ranker than full H3. On new footage the re-extracted CLIP stream is also out of distribution. The partial model was run once on the owner's game as an experiment (`reports/v3/highlights_real_video_review.json`); it is not offered as an upload mode.

## To make trained scoring of uploads real

1. Obtain one benchmark game's source video with permission, so each stream can be compared clip by clip with the released features.
2. Or retrain a ranker on features this project extracts itself, from footage it may use, and evaluate that as a new version with its own numbers.
