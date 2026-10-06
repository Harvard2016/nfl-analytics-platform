# V3 execution status

Branch `feat/inference-and-model-v3`, started from `main` at `af14fa0` on 2026-10-04. Everything below is one of three things, and says which: **measured** (run on this machine, file named), **proposed** (not run), or **missing** (blocked on an input).

## 0. Inventory (measured: `reports/v3/inventory.json`)

| | |
|---|---|
| Machine | Apple M2, 8 cores, 8 GB RAM, 13.7 GB free disk at the start. Training on CPU (MPS was slower in v2) |
| Software | Python 3.13.9, Node 26.10 (arm64), ffmpeg/ffprobe 9.0.2 installed during this work with Homebrew |
| Data on disk | `bdb2026` 824 MB, `nflverse` 473 MB (schedule + play-by-play 1999–2026), `svhighlights` 2.9 GB (features and annotations, no video) |
| Not on disk | Kaggle CLI or credentials; BDB 2026 Prediction release; Helmet Assignment data; any authorized video (`data/local_media/` is empty) |
| Models | 21 real model files with hashes in the inventory; v3 adds `models/highlights/v3/` and `models/pregame/v3/` |
| Tests at the start | 28 pytest tests passing. Now 74 pass (`.venv/bin/python -m pytest -q`), ruff clean, web lint and build pass, 18 browser tests pass on the public build (1 skipped: the live YouTube probe) |

### Saved-model reproduction (measured: `reports/v3/reproduction.json`)

- Coverage temporal model, seeds 42, 7, 2026: recomputed calibrated probabilities equal the saved ones on 500 random plays (max difference 0.0). No saved prediction exists where the play ended before the cutoff.
- Highlights H3: recomputed scores equal the saved ones on three games (max difference 0.0).
- Game prediction: v2 saved no model file (it refits each run). v3 now saves a bundle with every forecast run.

Command: `PYTHONPATH=src .venv/bin/python -m gridiron_lens.shared.v3_audit`

### The six code findings

| # | Finding | Status |
|---|---|---|
| 1 | `local_media.py` is loudness-only (H0) | Confirmed. The upload path is labelled "loudness baseline (H0)" everywhere |
| 2 | H3's PCA and scaling were not saved | Confirmed. Rebuilt deterministically and saved in a bundle with parity (section 2C) |
| 3 | `select_budget` can exceed the budget | Confirmed: game 40 selects 94 clips = 188 s for a 180 s reel. v1 kept as is; a strict decoder added (section 2A) |
| 4 | Forecast snapshot hashed only `games.csv` | Confirmed. Protocol v3 preserves and hashes every source file, the pipeline, the training cohort and the fitted bundle |
| 5 | `created_before_cutoff` used a time read before fitting | Confirmed. v3 reads the time after fitting, just before the record is written, and derives eligibility from it. Test: `test_run_that_crosses_the_cutoff_while_fitting_is_late` |
| 6 | The 28-hour rule is a proxy | Confirmed and stated in every v3 record and on the model card. It is not replaced: no publication-time archive exists |

Old results and logs were not rewritten. The one legacy forecast file is published as "legacy, limited provenance" and is never scored.

## 1. Coverage

**Fresh-season validation: missing.** No Kaggle client or credentials here, so the BDB 2026 Prediction files could not be listed or audited. Whether its 2024 data has compatible coverage labels is unknown. Exact owner action and the audit protocol are in `reports/v3/coverage_fresh_data.json`. Weeks 15–18 were not reused as development data and were not scored in v3.

**Development error slices (measured: `reports/v3/coverage_error_slices.json`).** Control model, out-of-fold on weeks 7–12, 4,510 plays. Error rate falls from 11.0% at the snap to 6.1% at +1.5 s. Man plays are missed far more often than zone (11.9% vs 3.7% error); Cover 2 man is the weakest family (19.7%, 61 plays). By defence the error rate ranges from about 1% to 15%. Plays with 5 or 6 tracked defenders are harder than those with 7 (8% vs 5%). Crossing routes, bunches, motion and early throws show no clear difference. These are associations on slices, not causes.

**Review queue: tool built, queue unreviewed.** `/coverage/review` shows each of the existing 100 development plays with playback, the model's probability at each cutoff, the released label, automated things to look at, and four judgements (plausible prediction, probable label ambiguity, insufficient evidence, clear model error) with notes, saved in the browser and exportable. No person has annotated it; no label was changed.

**Experiments (measured: `reports/v3/coverage_v3_experiments.json`; registered in `docs/experiments/coverage_v3.md`).** Out-of-fold on weeks 7–12, +1.5 s cutoff, seed 42 unless noted. 7 trained configurations plus 3 post-hoc combinations.

| Model | Mean validation loss (4 cutoffs) | Log loss | Accuracy | Man recall | Log loss on degraded input | Difference from control (95%, games) |
|---|---:|---:|---:|---:|---:|---|
| E0 control (v2 configuration), seed 42 | 0.2002 | 0.1456 | 0.939 | 0.880 | 0.379 | — |
| E0 control, seed 7 | 0.1971 | 0.1440 | 0.941 | 0.886 | 0.396 | −0.011 to +0.007 |
| E0 control, seed 2026 | 0.1849 | 0.1279 | 0.945 | 0.911 | 0.359 | −0.027 to −0.008 |
| E1 attention pooling | 0.2016 | 0.1440 | 0.940 | 0.884 | 0.375 | −0.012 to +0.009 |
| E2 robustness training | 0.2152 | 0.1501 | 0.938 | 0.865 | **0.200** | −0.004 to +0.014 |

- **E1: no gain.** Attention pooling is indistinguishable from the control.
- **E2: a trade.** Slightly worse on clean input (not significant), much better on noisy, gappy input (log loss 0.20 vs 0.38, accuracy 0.916 vs 0.877). Not promoted under the registered rule, which is about clean input. It is the natural starting point for inputs derived from video. It says nothing about selected-player bias.
- **Seed noise is larger than either effect.** Three seeds of the same control span 0.018 log loss.
- **E3.** The three-seed average beats a single seed (0.1238 vs 0.1456; difference −0.028 to −0.016). A blend with the relational trees also helps (0.1359; −0.015 to −0.005). Refitting Platt on other folds made things slightly worse than the raw logits (+0.002 to +0.006); temperature scaling changed nothing.
- **E4 coverage family** (7 classes, PREVENT excluded): flat head 85.5% accuracy, macro F1 0.836; hierarchical 85.1%, 0.832. The hierarchy does not help. Cover 2 man recall is 0.72 / 0.68 on 61 plays. Development only, uncalibrated.

**Decision.** No candidate met the rule. The v2 temporal model stays the champion; nothing was scored on weeks 15–18. The three-seed average already exists as a saved v2 entry and is the better-calibrated option, but the primary model was named before training and is not switched after the fact.

**Policies and interface.** The explorer now labels each model's role (champion, preserved benchmark, comparison) and says which is shown; the page default is unchanged and the note says a default is a display choice. Uploads report the standard lean and the reliable-predictions policy separately. *Proposed, not built:* a risk-versus-accepted-fraction plot per class on the evaluation page (the abstention sweep data already exists in `benchmark_v2.json`).

## 2. Highlights

**2A decoder and evaluation (measured: `reports/v3/highlights_eval_v3.json`).** `highlights/decode.py` measures the budget on final output seconds after padding and merging, clamps to the media length, breaks ties by time and ignores non-finite scores. Settings chosen on the 6 validation games: 3-clip blocks, 2 s lead and tail. On the 6 test games (examined in v2):

| Reel | Selector | Longest output | Precision | Recall of labelled time | Upper bound |
|---|---|---:|---:|---:|---:|
| 3 min | v1 (preserved) | 188 s | 0.776 | 0.194 | |
| 3 min | strict v3 | 180 s | 0.780 | 0.193 (0.160–0.233 over games) | 0.272 |
| 1 min | strict v3 | 60 s | 0.839 | 0.071 | 0.091 |
| 5 min | strict v3 | 300 s | 0.727 | 0.290 (v1: 0.304 at 308 s) | 0.454 |

The strict decoder removes the overshoot and does not find more highlights. In discovery mode (no reel budget) the top 10% of clips reach 64% of labelled time at 55% precision; the top 20% reach 84% at 37%. mAP on the original labels is unchanged: 0.619 (0.520–0.707 over 6 games).

**2B experiments (measured: `reports/v3/highlights_experiments_v3.json`, `highlights_candidates_v3.json`).** 28 training games, 4 grouped folds, every transform fitted inside the fold. Mean out-of-fold average precision:

| Ranker | mAP | Difference from control (95%, games) |
|---|---:|---|
| H-C control (H3) | 0.6507 | — |
| H-R ranking-aware loss | 0.6604 | +0.006 to +0.014 |
| H-W positive weight, game-balanced | 0.6534 | −0.003 to +0.009 |
| H-F learned fusion with commentary | 0.6696 | +0.013 to +0.025 |
| H-F learned fusion with negation-aware commentary | 0.6693 | +0.013 to +0.025 |
| v2 equal-weight sum with commentary | 0.6019 | −0.066 to −0.031 |
| Commentary alone / negation-aware | 0.4241 / 0.4225 | negation features: −0.003 to 0.000 |

- Negation, reversal, hypothetical and replay word counts add nothing over the word model.
- **Segment-aware decoding is worse.** Shot- or sentence-based segments reach 13–16% recall at a strict 180 s against 19.5% for the block decoder.
- Per the registered rule, the two passing ideas were rebuilt on all 28 training games and scored once on used data. **Ranking-aware H3 did not hold up** (validation 0.679 vs 0.688 shipped; test 0.617 vs 0.619). **Learned fusion** is level on validation (0.690 vs 0.688) and better on the examined test games (0.639 vs 0.619; +0.008 to +0.035; all 6 games).
- **Decision.** H3 stays shipped. The fusion candidate is saved under `models/highlights/v3/candidates/`; it needs a transcript, which uploads do not have yet, and both held-out splits had been used before.

**2C inference bundle (measured: `reports/v3/highlights_bundle_parity.json`).** Transforms refitted on the 28 training games reproduce the cached inputs (max difference 4e-9) and the bundle reproduces saved scores on all 40 games (6e-8); chunked inference equals a single pass (4e-6). Rebuild: `python -m gridiron_lens.highlights.bundle`. Test: `tests/test_highlights_bundle.py`.

**Trained multimodal inference on new media: not complete.** The bundle has no CLIP, SlowFast or PANN extractor; the upstream extractors were not installed or parity-checked. **Commentary mode: not built** (no local speech-to-text installed or profiled). Only the loudness baseline scores a new file.

**2D event detection: unchanged.** No reviewed event labels or source-to-play anchors exist, so there are no event metrics. The existing review page records annotations; nothing new was built for anchors.

## 3. Upload lab

**Measured.** `src/gridiron_lens/service/` (FastAPI, one worker thread, SQLite, private per-job folders), `bin/gridiron-api`, pages `/coverage/analyze` and `/highlights/analyze`.

| | Tracking classification | Clip ranking |
|---|---|---|
| Working mode | v2 temporal model, release-compatible and broader (experimental) | loudness baseline (H0) only |
| Input | one play, CSV or JSON, 10 Hz, schema `coverage-upload-v1` (template served by the API); ≤ 2 MB | any container ffprobe reads with audio; 10 s–20 min and ≤ 600 MB by upload, longer from the CLI |
| Parity | a real release play through the upload path gives the offline tensors and probabilities within 1e-3 (20+ plays, `tests/test_coverage_upload.py`) | cut durations verified with ffprobe; selected seconds never exceed the budget; the rendered file can be up to one audio frame (about 64 ms) longer |
| Measured cost | about 4 s from the command line including model load; 0.1–0.4 s per job inside the running service (3 runs) | 64 s test clip: about 1.0 s per job end to end (3 runs), 346 MB peak |
| Refuses | mixed plays, duplicate rows, out-of-range coordinates, too many players, missing line of scrimmage, missing orientation (unless experimental); disables cutoffs the play does not reach or where a frame is missing | unreadable or truncated files, no audio, over-long or oversized uploads |

- Jobs: `queued → validating → extracting → inferring → rendering → complete | failed | cancelled`. Stages only, no invented percentage. A restart marks running jobs failed. Tokens are per job; another job's token gets 404. Delete removes the files. Retention 24 hours.
- The API binds to 127.0.0.1 and allows only the local frontends by CORS. The public build has no service address, makes no request to the visitor's machine, shows the setup steps and has no file input.
- **Real-media smoke test: done on 2026-10-06** with a game video the owner supplied (high-school game, 1080p, 112 min 25 s, 3.48 GB; input SHA-256 `3da9c23c…5438c`). It stays in `data/local_media/`; no frame, cut or screenshot of it is published.
  - Full game from the command line, loudness baseline, 180 s reel: 35.7 s of processing, 352 MB peak in the worker (514 MB for the whole process), 12 clips, selected 180.0 s, rendered reel 180.2 s by ffprobe.
  - **It found a real bug.** Two picks whose padding overlapped were cut separately, so their shared seconds played twice and the first rendered reel ran 188.2 s. Fixed by cutting merged spans; regression test `test_overlapping_padding_does_not_play_twice_in_the_reel`.
  - The file's audio track ends 145 s before its video; the timeline covers the audio.
  - What the baseline picked: the third-ranked moment is the halftime marching band. Loudness finds loud things, not plays. That is the baseline's known limit, now seen on real footage.
  - A 5-minute excerpt went through the upload page in Chrome at 1440 and 390 px: the job completed, a candidate seeks to its start and stops at its end, the playhead follows, clicking the graph seeks, export and delete work.
  - The clip-upload screenshots in `docs/screenshots/upload-highlights-*.jpg` still show the generated test pattern, because the real footage is not published.
- **Video to coverage: partial.** `src/gridiron_lens/videocov/geometry.py` fits and validates a field homography on held-out landmarks, maps reviewed tracks with visibility masks, derives causal velocity and applies gates that return `insufficient evidence` with reasons. **Not built:** player detection, tracking, the review interface. No clip has been processed end to end, and no labelled video evaluation of coverage exists. The API reports this capability as not ready.
- **Calibration measured on real footage (2026-10-06, `reports/v3/videocov_helmet_eval.json`).** The owner supplied the NFL Helmet Assignment release: 60 plays, sideline and end-zone views, labelled helmet boxes and 10 Hz tracking for all 22 players. Only its tables and one sample play were extracted. For 3,600 frames in the 3 s after the snap, a homography was fitted from helmet centres to tracked positions on three quarters of the players and scored on the rest:

  | View | Median error | 90th percentile | Within 1 yard | Frames passing the 1-yard gate |
  |---|---:|---:|---:|---:|
  | Sideline | 0.48 yd | 1.19 yd | 85.7% | 75% |
  | End zone | 0.60 yd | 1.77 yd | 74.4% | 50% |

  - **One calibration goes stale fast.** Fitted at the snap and reused, the median error is 0.6 yd at the snap, 1.0 yd after 1 s, 1.3 yd after 1.5 s and 3.1 yd after 3 s (sideline). A play needs re-calibration at least every half second, or camera-motion tracking.
  - **Timing.** The release's stated alignment is close but not the best fit: shifting the video by about 6 frames (0.1 s) lowers the error from 0.85 to 0.81 yd. Nothing was tuned to it.
  - **What this does not show.** Helmet identities came from the released labels, not a detector. Points are helmet centres, not feet. These are fixed coaching cameras, not a broadcast. The release has no coverage labels and 2 of its 60 plays are passes, so it says nothing about coverage accuracy.
  - The owner's game video is a single panning, zooming press-box camera. With no detector or tracker built, it could not be run through this path.
- *Proposed:* container and cost estimate for hosting; private object storage with direct uploads. Nothing was provisioned.

## 4. Game prediction

**Protocol v3 (measured).** `bin/pregame-lens snapshot` fetched the schedule and 2026 play-by-play at 2026-10-04 22:34 UTC (schedule and play-by-play through 2026-10-04, week 4, 58 games) and preserved all 29 source files with hashes in `data/snapshots/nflverse/20261004T223450Z/`. `bin/pregame-lens forecast-v3` wrote **17 records** at 22:37 UTC: **16 before their cutoff** (Monday's ATL at NO and all 15 week-5 games) and **1 late** (DET at CAR, already inside 24 hours). A second run on the same snapshot wrote nothing (idempotent). With the 30 legacy records the public log holds 47: 16 official, 1 late, 30 legacy and unscored. No outcome is attached yet, so the 2026 leaderboard is empty.

The week-5 records are early: a later run before each cutoff, on a fresher snapshot, supersedes them as the official record. Forecasting uses Elo and the frozen v2 model.

**Experiments (measured: `reports/v3/pregame_experiments_v3.json`; registered in `docs/experiments/pregame_v3.md`).** 17 configurations. Nested choice by season, 2015–2022, 2,164 games:

| | Log loss |
|---|---:|
| Elo | 0.6357 |
| Frozen v2 Elo-offset | 0.6322 |
| Frozen v2 margin | 0.6290 |
| v3 nested choice (opponent-adjusted ratings) | 0.6322 |

v3 against Elo: −0.0062 to −0.0008. Against frozen v2: −0.0030 to +0.0030. On the previously examined 2023–2025 seasons the frozen v3 choice scores 0.6327 against Elo's 0.6362 (−0.0095 to +0.0027). Points-only ratings were not better than EPA ratings. Platt calibration from earlier seasons' forecasts changed log loss by less than 0.001. **Not promoted. No gain over Elo is established beyond what v2 already showed.**

**Not run:** quarterback scenarios from depth charts (timestamped snapshots exist only from 2025, so they cannot be backtested on 2012–2022); market comparison (no prices timestamped before the cutoff).

**Scheduler (written, not installed):** `ops/com.gridironlens.forecast.plist` and `ops/forecast_job.sh`. Installing it is one command in the plist's header; a sleeping laptop misses runs and a late record stays late.

## 5. Checks

| Check | Result |
|---|---|
| pytest | 74 passed |
| ruff | clean |
| Web lint, type check, build | pass (20 routes) |
| Web unit tests | 6 passed |
| Browser suite (`npm run test:ui`) on the public build | 18 passed, 1 skipped (live YouTube probe). With a local service configured, the two no-service checks skip instead |
| Bundle parity on real data | pass (highlights 40 games; coverage 500 plays, three seeds) |
| Upload parity on real plays | pass |
| API: rejection, isolation, cancel, restart, deletion, range and path safety | pass (`tests/test_service_api.py`) |
| Strict reel duration, incl. the 188 s case | pass (`tests/test_highlights_decode.py`) |
| Forecast timing, DST, missing kickoff | pass (`tests/test_pregame_forecast_v3.py`) |
| Browser: tracking upload, clip upload, playhead and seek sync, clip stop at end, export, errors, keyboard cutoffs, 390 px | pass, against the local service |
| Real authorized media smoke test | pass after one fix (reel overshoot from duplicated overlap) |
| Calibration on real paired video and tracking | measured (table in section 3) |
| Fresh-season coverage validation | **blocked: dataset not downloaded** |
| Screen-reader pass | not done |

## What the owner needs to do

1. Done on 2026-10-06: a game video and the Helmet Assignment release were supplied.
2. Still needed for fresh coverage validation: download the BDB 2026 Prediction archive into `data/raw/bdb2026_prediction/` (see `reports/v3/coverage_fresh_data.json`).
3. Decide whether to install the forecast schedule.

The Helmet Assignment release is a different dataset from the BDB 2026 Prediction release; it cannot validate the coverage model.

## Next smallest experiment

Score the saved three-seed coverage ensemble and the E2 robust model once on a labelled season the models have never seen. That single run answers the two open coverage questions (does the benchmark figure hold up, and is robustness training worth its clean-input cost) and needs only the download in item 2.
