# Coverage: one-time evaluation on a fresh 2024 sample. Registered before scoring

Written 2026-10-06, before any prediction was compared with a label.

## What was found

- The BDB 2026 Prediction archive has no coverage labels (audit: `reports/v3/coverage_prediction_archive_audit.json`).
- Its `test_input.csv` holds tracking for 143 plays from three 2024-season games (`2024120805`, `2024121502`, `2025010515`; weeks 14, 15 and 18).
- The BDB 2026 Analytics label table already on disk (`supplementary_data.csv`) lists 3,901 plays from the 2024 season with released coverage labels. Those plays had no tracking in the Analytics release, so they were never trained on, tuned on or scored.
- All 143 test-input plays are in that table with a released man/zone label: 108 zone, 35 man.

So tracking from one release and labels from the other give 143 labelled plays from a season no model here has seen.

## What has and has not been looked at

Before this file was written: the model's predictions on the 143 plays were generated without labels and their overall distribution was printed (mean man probability 0.229; 21.8% lean man), and the label counts above were printed. No prediction was paired with a label. No setting was changed in response.

## Frozen before scoring

- Primary model: `cov-v2-temporal-gru-seed42` with its saved Platt calibrators; lean is man when the calibrated probability is at least 0.5. Named as primary before training in v2.
- Secondary: the mean probability of seeds 42, 7 and 2026 (saved in v2).
- Bundle hashes are recorded in the result file. Nothing is refitted, recalibrated or re-thresholded on these plays.
- Inputs: the upload contract (`coverage/upload.py`), which reproduces the offline pipeline on 2023 plays. Not read: `num_frames_output`, `ball_land_x`, `ball_land_y`, `player_to_predict`, the Targeted Receiver role, any `output_*.csv`.
- Cohort: every one of the 143 plays the contract accepts, at each cutoff the play reaches. Labels come from `supplementary_data.csv` unchanged.

## Reported once

Accuracy, balanced accuracy, macro F1, man and zone precision and recall, PR-AUC for man, log loss, Brier score, reliability bins and results per game, at each cutoff; the class-prior baseline on the same plays. Intervals: Wilson for accuracy and a play-level bootstrap for log loss. With three games a game-level interval is not meaningful and is not claimed.

## How it must be described

A small fresh-season sample: 143 plays, 3 games, late-season weeks, one look. Same release format and the same post-play player selection as 2023, so it tests a new season, not new tracking or new player selection. After this run the sample is examined and cannot be used again as a fresh test. It does not replace the 2023 benchmark figure and is reported beside it.
