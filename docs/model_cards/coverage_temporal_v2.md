# Model card: coverage temporal model (v2), champion

**Version:** `cov-v2-temporal-gru-seed42` (files `models/coverage_v2/temporal_gru_seed{42,7,2026}.pt`; seed 42 is primary). Unchanged in v3.

| | |
|---|---|
| Task | Agreement with the released man/zone coverage label for a pass play |
| Input | Tracking of route runners and the coverage defenders the release selected, snap to +1.5 s, 10 Hz: position relative to the line of scrimmage and formation centre, velocity, observed orientation |
| Architecture | Shared MLP on every defender–receiver pair (18 features), masked mean+max pooling over receivers then defenders, 2-layer GRU (96), one logit per frame. About 0.14 M parameters |
| Causality | The output at frame t uses frames 0..t only. Cutoffs: snap, +0.5 s, +1.0 s, +1.5 s. No prediction where the play ended before the cutoff |
| Training | Weeks 1–12 of 2023, 28 epochs, left-right reflection augmentation. Selection on three chronological folds inside weeks 1–12 |
| Calibration | Platt per cutoff on weeks 13–14 |
| Never inputs | Coverage labels, coverage family, target receiver, pass result, ball landing point, frames after the cutoff, tracked-defender count |

## Measured

Weeks 15–18 (3,178 plays at +1.5 s). **Previously examined benchmark, not a fresh test.**

| | Accuracy | Man recall | Log loss | Brier |
|---|---:|---:|---:|---:|
| seed 42 | 0.952 | 0.900 | 0.128 | 0.038 |
| three-seed average | 0.954 | 0.903 | 0.117 | 0.034 |

Full tables, intervals, reliability bins and policies: `reports/v2/coverage_benchmark_v2.json`. v3 development experiments and error slices: `reports/v3/coverage_v3_experiments.json`, `reports/v3/coverage_error_slices.json`.

## Limits

- The tracked defenders were chosen by the release with knowledge of the play. Masks and geometry carry that selection. Results do not transfer to all-22 tracking or to video-derived coordinates.
- 2023 season only. No external season has been scored (`reports/v3/coverage_fresh_data.json`).
- Labels are the released labels; who charted them is not stated. Disagreement is not proof of a label error.
- Ablation "explanations" are sensitivity checks, not causes, and not evidence of a defender's assignment.

## Reproduce

`.venv/bin/python -m gridiron_lens.shared.v3_audit` recomputes saved probabilities from the bundles (max difference 0.0 on 500 plays).
Upload path parity: `.venv/bin/python -m pytest tests/test_coverage_upload.py -q`.
