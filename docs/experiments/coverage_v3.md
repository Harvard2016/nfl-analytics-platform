# Coverage v3 experiments: registered before any run

Written 2026-10-04, before the first v3 training run. Results are appended below the line at the end; this section is not edited afterwards.

## Fixed setting

- Data: BDB 2026 Analytics release, 2023 season, selected-player tracking. Same limits as v2 (players chosen by the release after the play).
- Development data: weeks 1–12, the three chronological folds from v2 (`train ≤ 6 → validate 7–8`, `≤ 8 → 9–10`, `≤ 10 → 11–12`).
- Weeks 13–14 keep their v2 role (calibration and policies). **Weeks 15–18 are not scored in these experiments.** They are a previously examined benchmark and are touched only if a model is promoted, once, with that label.
- Cohort: every labelled play that has the requested prefix. A play thrown before a cutoff gets no prediction at that cutoff.
- Control: the v2 champion configuration (`gru+reflect`, pair MLP 64-64, mean+max pooling, GRU 96×2), retrained on the same folds with early stopping exactly as in v2.
- Never inputs: released man/zone label, coverage family, target receiver, pass result, frames after the cutoff, ball landing point, tracked-defender count as an explicit feature.

## Experiments (run in this order, seed 42)

| Id | Change from the control | Question |
|---|---|---|
| E0 | none (control) | reference out-of-fold predictions |
| E1 | masked attention pooling over receivers and over defenders in place of the mean (max kept); same GRU. Parameter count reported next to the control's | does learned weighting of defender–receiver relations help? |
| E2 | control architecture trained with position noise σ 0.15 yd, velocity noise σ 0.3 yd/s, 10% of frames replaced by the previous frame, 10% of defenders and 5% of receivers dropped (never below 3 defenders / 1 receiver). Orientation is left untouched | does it cost clean accuracy, and does it hold up on inputs degraded the same way? |
| E3 | no new network: (a) average of three control seeds (42, 7, 2026); (b) blend of control and relational boosted trees, weight fitted on out-of-fold predictions from the other folds; (c) Platt vs temperature calibration fitted the same nested way | are ensembles or a different calibrator better calibrated? |
| E4 | coverage family, 7 classes (PREVENT excluded): flat 7-way head vs hierarchical head (group × family within group), same trunk as the control | does the hierarchy help; which families are unsupported? |

Configurations counted: E0 1, E1 1, E2 1, E3 3 post-hoc combinations (+2 control seeds), E4 2. Finalists (any E1/E2 model that passes the rule) are rerun with seeds 7 and 2026. Nothing else is tried; no hyperparameter search.

## Selection rule

Primary: mean out-of-fold log loss over the four cutoffs (snap, +0.5 s, +1.0 s, +1.5 s), uncalibrated logits, as in v2.

A candidate replaces the control only if all of these hold:
1. its primary score is lower in each of the three folds;
2. the game-clustered bootstrap interval (2,000 resamples) of the per-play log-loss difference at +1.5 s excludes zero;
3. at +1.5 s, macro F1, man precision and man recall are each no more than 0.01 below the control.

Otherwise the v2 temporal model stays the champion and the result is reported as not established. There is no accuracy target.
Rollback: the v2 bundles in `models/coverage_v2/` are never modified; a promoted model is saved under `models/coverage/v3/`.

---

## Results

Run on 2026-10-04. Report: `reports/v3/coverage_v3_experiments.json`. Summary and decision: `docs/V3_EXECUTION_STATUS.md`.
