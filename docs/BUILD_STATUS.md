# Build status

Last updated 2026-10-04 (experiments v2). Everything below was run on this machine. Local git only, branch `experiments-v2`: nothing pushed or deployed.

**Location changed.** The repository now lives at `~/Developer/nfl-analytics-platform`. `~/Documents` is synced by iCloud, which evicted 13,000+ project files and stalled every read. The old copy in `~/Documents/nfl-analytics-platform` is stale and can be deleted.

## Local URLs (site: `cd apps/web && npx next start -p 3111`)

| Module | Pages |
|---|---|
| Coverage | `/coverage` (random sample), `/coverage/quick` (quick throws), `/coverage/errors` (error gallery), `/coverage/evaluation`, `/coverage/tendencies` |
| Game predictor | `/predictions` (backtests), `/predictions/performance`, `/predictions/forecasts` (live log) |
| Highlights | `/highlights` (timeline, benchmark, recap), `/highlights/review` (candidate review tool) |
| Shared | `/research`, `/research/experiments` (run records), `/engineering` |

A recruiter path: `/` → `/coverage?model=v2_temporal` (play a snap, click a pairing, read "Prediction change") → `/coverage/evaluation?model=v2_temporal` → `/coverage/errors?model=v2_temporal` → `/predictions/performance` (an honest null result) → `/highlights`.

## Preserved v1 benchmark

`reports/v1/baseline_manifest.json` (commit, data checksums, split ids, feature lists, model hashes, export hashes, metrics with game-level intervals, interface defaults) and `docs/audits/coverage_explorer_audit_v1.md`. Counts re-confirmed locally: 14,108 tracked pass plays, 2023 weeks 1–18; 14,105 labelled (4,013 man, 10,092 zone); 3 unlabelled; roles tracked are passer, route runners and coverage defenders only; no ball rows and no pre-snap frames. v1 boosted trees at +1.5 s: accuracy 0.8883, macro F1 0.8535, log loss 0.2569, Brier 0.0791, man recall 0.7297, 94.1% accepted. The explorer fixes from the audit were already in place and were re-verified.

## Coverage (independent module)

**Standing limit.** The release tracks only the passer, route runners and the defenders it marks as coverage players, chosen with knowledge of the play. Every model input is computed over that set, and masks or aggregates can still reveal it. These are predictions from the released selected-player tracking. Not a pre-snap, full-field or leakage-free system.

**Protocol.** Train weeks 1–12, calibrate and choose policies on weeks 13–14, compare on weeks 15–18 (3,178 plays, 64 games). All v2 selection used three chronological folds inside weeks 1–12 (train 1–6 / validate 7–8; 1–8 / 9–10; 1–10 / 11–12). **Weeks 15–18 were examined during v1, so every v2 number there is a comparison on a previously examined benchmark, not a fresh test.**

### Experiment A: relational features and Man class weights (development folds)

New order-invariant measurements (`coverage/relational.py`): defender-to-receiver separation, relative position and velocity, closing speed, movement agreement, a minimum-distance one-to-one pairing and its history since the snap, nearest-receiver persistence and switches, defender displacement, group width/depth/coherence. Geometry, not confirmed matchups. Mean over folds (fold spread):

| Cutoff | Features | Man weight | Log loss | Balanced acc. | Man recall |
|---|---|---|---|---|---|
| Snap | geometry | 1.0 | 0.3884 (0.0184) | 0.759 | 0.583 |
| Snap | geometry | 1.5 | 0.3885 (0.0192) | 0.760 | 0.590 |
| Snap | geometry+relational | 1.0 | 0.3443 (0.0263) | 0.783 | 0.634 |
| Snap | geometry+relational | 1.5 | 0.3451 (0.0259) | 0.787 | 0.643 |
| Snap | relational | 1.0 | 0.3796 (0.0340) | 0.762 | 0.589 |
| +1.0 s | geometry | 1.0 | 0.3179 (0.0213) | 0.807 | 0.679 |
| +1.0 s | geometry | 1.5 | 0.3168 (0.0202) | 0.807 | 0.678 |
| +1.0 s | geometry+relational | 1.0 | 0.2542 (0.0245) | 0.851 | 0.748 |
| +1.0 s | geometry+relational | 1.5 | 0.2543 (0.0249) | 0.851 | 0.751 |
| +1.0 s | relational | 1.0 | 0.2696 (0.0301) | 0.839 | 0.728 |
| +1.5 s | geometry | 1.0 | 0.2798 (0.0171) | 0.835 | 0.723 |
| +1.5 s | geometry | 1.5 | 0.2798 (0.0161) | 0.834 | 0.722 |
| +1.5 s | geometry+relational | 1.0 | 0.2302 (0.0141) | 0.868 | 0.787 |
| +1.5 s | geometry+relational | 1.5 | 0.2292 (0.0161) | 0.871 | 0.793 |
| +1.5 s | relational | 1.0 | 0.2459 (0.0219) | 0.861 | 0.773 |

Chosen: **geometry+relational, man weight 1.5**. Weights 2.0 and 2.5 did not improve log loss; weighting is recalibrated on the natural class mix, so it moves the operating point slightly, not the representation.

### Experiment B: temporal interaction model (development folds)

`coverage/neural.py`: frames × defenders × receivers × 18 pair features, shared two-layer pair MLP (64), masked mean+max pooling over receivers then defenders, two-layer unidirectional GRU (96, dropout 0.1), binary head. Trained on random permitted prefixes (snap, +0.5, +1.0, +1.5 s). AdamW 1e-3, weight decay 1e-4, batch 64, clip 1.0, early stopping 8. CPU: Apple MPS measured at about half the CPU speed for this model.

| Model | Mean validation loss (spread) | Median best epoch | +1.5 s log loss | +1.5 s man recall | Parameters |
|---|---|---|---|---|---|
| gru | 0.2161 (0.0177) | 23 | 0.1582 | 0.886 | 134,689 |
| tcn | 0.2103 (0.0259) | 31 | 0.1536 | 0.878 | 106,177 |
| gru+reflect | 0.2002 (0.0182) | 28 | 0.1460 | 0.880 | 134,689 |

Chosen: **gru+reflect, 28 epochs**, refit on weeks 1–12 with seeds 42 (primary), 7 and 2026. Mechanical checks (random inputs): player reordering changes outputs by under 1e-7; rewriting later frames changes earlier outputs by exactly 0; garbage in masked slots changes outputs by exactly 0; tensor reflection equals mirroring raw coordinates.

### Comparison on the previously examined benchmark (weeks 15–18, common cohort)

Standard 0.5 decision on calibrated probabilities. Accuracy range resamples whole games. Last four columns: abstention cutoff chosen on weeks 13–14, share accepted, share of man plays accepted, man plays correct and accepted over all man plays.

| Model | Cutoff | Accuracy (95%) | Balanced acc. | Macro F1 | Man recall / precision | Zone recall | Log loss | Brier | Conf. cutoff | Accepted | Man accepted | Man correct+accepted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Boosted trees, v1 geometry (benchmark) | Snap | 0.831 (0.816–0.845) | 0.749 | 0.768 | 0.566 / 0.758 | 0.931 | 0.377 | 0.119 | 0.75 | 0.750 | 0.556 | 0.346 |
| Boosted trees, v1 geometry (benchmark) | +1.0 s | 0.870 (0.856–0.884) | 0.808 | 0.826 | 0.670 / 0.823 | 0.945 | 0.306 | 0.095 | 0.65 | 0.890 | 0.805 | 0.564 |
| Boosted trees, v1 geometry (benchmark) | +1.5 s | 0.888 (0.874–0.900) | 0.839 | 0.853 | 0.730 / 0.843 | 0.948 | 0.257 | 0.079 | 0.60 | 0.941 | 0.880 | 0.668 |
| Boosted trees, geometry + relational (v2) | Snap | 0.853 (0.838–0.868) | 0.789 | 0.805 | 0.647 / 0.780 | 0.931 | 0.333 | 0.104 | 0.70 | 0.834 | 0.701 | 0.496 |
| Boosted trees, geometry + relational (v2) | +0.5 s | 0.874 (0.858–0.889) | 0.824 | 0.836 | 0.714 / 0.806 | 0.935 | 0.292 | 0.090 | 0.60 | 0.936 | 0.884 | 0.653 |
| Boosted trees, geometry + relational (v2) | +1.0 s | 0.894 (0.883–0.905) | 0.852 | 0.863 | 0.759 / 0.839 | 0.945 | 0.245 | 0.075 | 0.60 | 0.941 | 0.892 | 0.708 |
| Boosted trees, geometry + relational (v2) | +1.5 s | 0.906 (0.894–0.917) | 0.866 | 0.878 | 0.779 / 0.864 | 0.954 | 0.218 | 0.066 | 0.50 | 1.000 | 1.000 | 0.779 |
| Temporal interaction model (v2) | Snap | 0.896 (0.883–0.909) | 0.855 | 0.866 | 0.765 / 0.842 | 0.946 | 0.249 | 0.076 | 0.55 | 0.972 | 0.952 | 0.746 |
| Temporal interaction model (v2) | +0.5 s | 0.922 (0.911–0.933) | 0.894 | 0.901 | 0.833 / 0.878 | 0.956 | 0.195 | 0.058 | 0.50 | 1.000 | 1.000 | 0.833 |
| Temporal interaction model (v2) | +1.0 s | 0.941 (0.932–0.951) | 0.922 | 0.926 | 0.877 / 0.907 | 0.966 | 0.147 | 0.043 | 0.50 | 1.000 | 1.000 | 0.877 |
| Temporal interaction model (v2) | +1.5 s | 0.952 (0.944–0.960) | 0.936 | 0.939 | 0.900 / 0.923 | 0.971 | 0.128 | 0.038 | 0.50 | 1.000 | 1.000 | 0.900 |
| Temporal model, 3-seed average (v2) | Snap | 0.901 (0.889–0.912) | 0.857 | 0.870 | 0.762 / 0.860 | 0.953 | 0.236 | 0.071 | 0.55 | 0.972 | 0.948 | 0.743 |
| Temporal model, 3-seed average (v2) | +0.5 s | 0.928 (0.918–0.938) | 0.899 | 0.908 | 0.835 / 0.897 | 0.964 | 0.181 | 0.054 | 0.50 | 1.000 | 1.000 | 0.835 |
| Temporal model, 3-seed average (v2) | +1.0 s | 0.943 (0.934–0.952) | 0.924 | 0.928 | 0.880 / 0.911 | 0.967 | 0.138 | 0.041 | 0.50 | 1.000 | 1.000 | 0.880 |
| Temporal model, 3-seed average (v2) | +1.5 s | 0.954 (0.946–0.963) | 0.938 | 0.942 | 0.903 / 0.929 | 0.974 | 0.117 | 0.034 | 0.50 | 1.000 | 1.000 | 0.903 |

Paired difference from the v1 model on the same plays (95% range, whole games resampled):

| Model | Cutoff | Accuracy | Man recall | Log loss | Brier |
|---|---|---|---|---|---|
| Boosted trees, geometry + relational (v2) | Snap | +0.022 (+0.012 to +0.032) | +0.081 (+0.058 to +0.105) | -0.043 (-0.055 to -0.031) | -0.015 (-0.019 to -0.011) |
| Boosted trees, geometry + relational (v2) | +1.0 s | +0.024 (+0.014 to +0.035) | +0.089 (+0.061 to +0.118) | -0.061 (-0.072 to -0.050) | -0.020 (-0.024 to -0.016) |
| Boosted trees, geometry + relational (v2) | +1.5 s | +0.017 (+0.009 to +0.026) | +0.049 (+0.024 to +0.075) | -0.039 (-0.049 to -0.028) | -0.013 (-0.016 to -0.009) |
| Temporal interaction model (v2) | Snap | +0.065 (+0.055 to +0.076) | +0.199 (+0.167 to +0.234) | -0.127 (-0.145 to -0.111) | -0.043 (-0.050 to -0.037) |
| Temporal interaction model (v2) | +1.0 s | +0.072 (+0.061 to +0.083) | +0.207 (+0.179 to +0.237) | -0.159 (-0.177 to -0.140) | -0.052 (-0.058 to -0.046) |
| Temporal interaction model (v2) | +1.5 s | +0.064 (+0.053 to +0.074) | +0.171 (+0.142 to +0.201) | -0.129 (-0.145 to -0.113) | -0.042 (-0.047 to -0.036) |
| Temporal model, 3-seed average (v2) | Snap | +0.070 (+0.060 to +0.080) | +0.196 (+0.161 to +0.230) | -0.141 (-0.157 to -0.127) | -0.048 (-0.054 to -0.043) |
| Temporal model, 3-seed average (v2) | +1.0 s | +0.074 (+0.062 to +0.086) | +0.210 (+0.177 to +0.240) | -0.168 (-0.184 to -0.151) | -0.054 (-0.060 to -0.049) |
| Temporal model, 3-seed average (v2) | +1.5 s | +0.066 (+0.056 to +0.076) | +0.173 (+0.143 to +0.201) | -0.140 (-0.155 to -0.126) | -0.045 (-0.050 to -0.039) |

Reading: both v2 models improve on v1 with intervals that exclude zero at every shared cutoff; the temporal model's gain is large (man recall 0.73 → 0.90 at +1.5 s). Because this benchmark was already examined, treat the size as provisional until a new season or release is scored once.

**Balanced class policy** (man threshold maximizing balanced accuracy on weeks 13–14 with man precision ≥ 0.80), +1.5 s: v1 threshold 0.44; relational trees 0.39 (benchmark man recall 0.833, precision 0.822); temporal 0.25 (man recall 0.951, precision 0.847). A different operating point for the same probabilities, not a better model. Reliability tables, full abstention sweeps with class-specific acceptance and operational-eligibility numbers are in `reports/v2/coverage_benchmark_v2.json` and on `/coverage/evaluation`.

**Selected-player sensitivity** (+1.5 s; accuracy / man recall by number of tracked coverage defenders: 5 or fewer, 6, 7, 8+):

| Model | ≤5 | 6 | 7 | 8+ |
|---|---|---|---|---|
| Boosted trees, v1 geometry (benchmark) | 0.831 / 0.805 (n 231) | 0.810 / 0.683 (n 658) | 0.916 / 0.731 (n 2176) | 0.920 / 0.737 (n 113) |
| Boosted trees, geometry + relational (v2) | 0.874 / 0.864 (n 231) | 0.856 / 0.757 (n 658) | 0.923 / 0.756 (n 2176) | 0.929 / 0.842 (n 113) |
| Temporal interaction model (v2) | 0.935 / 0.941 (n 231) | 0.950 / 0.944 (n 658) | 0.954 / 0.855 (n 2176) | 0.965 / 0.842 (n 113) |

Gains hold inside each stratum, so they are not only the defender count. The limit itself remains.

**Unseen defenses** (weeks 1–14 only; every game involving a held-out team removed from training and calibration; +1.5 s):

| Model | Plays | Accuracy (95%) | Macro F1 | Log loss | Man recall |
|---|---|---|---|---|---|
| Boosted trees, geometry + relational (v2) | 10544 | 0.895 (0.887–0.902) | 0.868 | 0.246 | 0.785 |
| Temporal interaction model (v2) | 10544 | 0.928 (0.922–0.935) | 0.910 | 0.168 | 0.846 |

### Experiment C: coverage family (development weeks 13–14 only; benchmark not scored)

Classes: Cover 0, 1, 2 man; Cover 2, 3, 4, 6 zone. Excluded: prevent (too few plays). Hierarchy P(family) = P(group) × P(family | group); group never given as input. Uncalibrated.

| Cutoff | Model | Accuracy | Balanced acc. | Macro F1 | Log loss | Recall by class |
|---|---|---|---|---|---|---|
| +1.0 s | boosted trees flat | 0.696 | 0.612 | 0.632 | 0.826 | C0m 0.71, C1m 0.73, C2m 0.28, C2z 0.67, C3z 0.82, C4z 0.70, C6z 0.38 |
| +1.0 s | class prior | 0.314 | 0.143 | 0.068 | 1.704 | C0m 0.00, C1m 0.00, C2m 0.00, C2z 0.00, C3z 1.00, C4z 0.00, C6z 0.00 |
| +1.0 s | temporal hierarchical | 0.837 | 0.824 | 0.834 | 0.430 | C0m 0.79, C1m 0.82, C2m 0.84, C2z 0.73, C3z 0.86, C4z 0.92, C6z 0.79 |
| +1.5 s | boosted trees flat | 0.742 | 0.669 | 0.695 | 0.706 | C0m 0.77, C1m 0.78, C2m 0.41, C2z 0.70, C3z 0.87, C4z 0.72, C6z 0.43 |
| +1.5 s | class prior | 0.314 | 0.143 | 0.068 | 1.704 | C0m 0.00, C1m 0.00, C2m 0.00, C2z 0.00, C3z 1.00, C4z 0.00, C6z 0.00 |
| +1.5 s | temporal hierarchical | 0.837 | 0.817 | 0.829 | 0.415 | C0m 0.81, C1m 0.83, C2m 0.78, C2z 0.71, C3z 0.87, C4z 0.92, C6z 0.79 |

### Product
Explorer defaults to the v1 boosted trees at +1.5 s (the validated benchmark model); selectable: relational trees, temporal model, v1 logistic; cutoffs snap/+0.5/+1.0/+1.5. Random sample (120, prediction-blind), quick throws (12 plays thrown before 1.5 s; later cutoffs disabled, nothing extrapolated), error gallery per model and cutoff. Evidence mode highlights the players and observed path segments behind a measurement; observed pairings table; prediction-change table from saved prefix outputs (no interpolation); three similar plays from training weeks; explanations per model (logistic terms, replace-with-median for trees, grouped-input and player-removal ablations for the temporal model). 304 play files, 15 MB. Error-review queue for development weeks in `reports/v2/coverage_review_queue.json`; no reviewer annotations exist yet.

## Game predictor (independent module)

Cutoff: 24 hours before scheduled kickoff. A finished game counts only if it kicked off at least 28 hours before the target kickoff. Violations: 0. Backtests are reconstructed from today's nflverse files, not from forecasts recorded before kickoff. nflverse EPA models were fitted with later data, so EPA-based features carry a small amount of hindsight. Projected quarterback = the team's primary passer in its previous game. An assumption, not confirmed starters.

Selection: 108 configurations (feature groups × 4-game, 8-game, exponential windows × penalties) by walk-forward log loss on 2012–2022. Chosen Elo-offset: all, w4, λ 0.1. Chosen point margin: rest+efficiency+rates+qb, w4, α 10.0. Blend weight 0.25 on Elo-offset. Elo input clipped to [0.02, 0.98] for numerical stability only.

Development 2012–2022 (optimistic for the chosen models, which were picked on these games):

| Model | Games | Log loss | Brier | Accuracy | vs Elo (95%) |
|---|---|---|---|---|---|
| Home-win rate only | 2962 | 0.6860 | 0.2464 | 0.558 |  |
| Elo rating | 2962 | 0.6301 | 0.2201 | 0.649 |  |
| v1 logistic regression | 2962 | 0.6295 | 0.2199 | 0.644 | -0.0006 (-0.0032 to +0.0020) |
| v1 boosted trees | 2962 | 0.6324 | 0.2212 | 0.639 | +0.0024 (-0.0010 to +0.0057) |
| Elo-offset logistic (v2) | 2962 | 0.6278 | 0.2192 | 0.647 | -0.0022 (-0.0048 to +0.0003) |
| Ridge point margin (v2) | 2962 | 0.6263 | 0.2187 | 0.640 | -0.0037 (-0.0083 to +0.0009) |
| Blend of the two v2 models | 2962 | 0.6259 | 0.2185 | 0.641 | -0.0041 (-0.0080 to -0.0002) |

Previously examined benchmark 2023–2025:

| Model | Games | Log loss | Brier | Accuracy | vs Elo (95%) |
|---|---|---|---|---|---|
| Home-win rate only | 854 | 0.6898 | 0.2483 | 0.549 |  |
| Elo rating | 854 | 0.6362 | 0.2226 | 0.641 |  |
| v1 logistic regression | 854 | 0.6354 | 0.2227 | 0.637 | -0.0007 (-0.0064 to +0.0052) |
| v1 boosted trees | 854 | 0.6359 | 0.2230 | 0.630 | -0.0002 (-0.0068 to +0.0061) |
| Elo-offset logistic (v2) | 854 | 0.6349 | 0.2221 | 0.638 | -0.0012 (-0.0064 to +0.0037) |
| Ridge point margin (v2) | 854 | 0.6319 | 0.2210 | 0.644 | -0.0042 (-0.0115 to +0.0030) |
| Blend of the two v2 models | 854 | 0.6321 | 0.2211 | 0.644 | -0.0041 (-0.0105 to +0.0024) |

**Verdict: no established gain over Elo.** Every range on 2023–2025 includes zero and the winner flips by season. Ablations: rest/bye and recent-form add nothing; efficiency, rate and quarterback features each help by about 0.001 on development seasons. Closing-market reference (later information, not a baseline): market 0.6077 vs Elo 0.6362 on 854 games.

Live forecasts: `bin/pregame-lens forecast` writes append-only records (`reports/forecasts/`). First file has 30 records created 2026-10-04 07:56 UTC: 14 for same-day games (flagged as created inside 24 hours) and 16 created before their cutoff. Outcomes attach in a separate field on the next export. Too few to score.

Tests: rewriting a target game's score, box score and starter leaves its features unchanged; changing an earlier game changes later ones; appending future games changes nothing earlier; unplayed-game path equals training path; projected QB is the previous starter.

## Highlights (independent module)

Data: SVHighlights American-football subset, revision `fdedf750`, CC BY-NC 4.0. **40 NFL games, seasons 2016-2024, all from the NFL channel** — verified from source titles, no duplicates, and all 40 mapped to exactly one nflverse game id (`data/manifests/svhighlights_game_map.json`). 167,006 two-second clips, 9.5% labelled. Label = clip appears in the official highlight video (editorial selection, replays included). Nine games had one extra trailing loudness/feature value; cut to label length. Split by game, frozen before scoring: 28 train / 6 validation / 6 test.

| Experiment | Validation mAP | Test mAP (range) | 3-min precision | 3-min recall | Segments missed |
|---|---|---|---|---|---|
| H0 loudness | 0.314 | 0.246 (0.05–0.37) | 0.407 | 0.089 | 0.886 |
| H1 commentary | 0.402 | 0.404 (0.26–0.53) | 0.633 | 0.163 | 0.813 |
| H2 visual CLIP, logistic | 0.309 | 0.291 (0.17–0.42) | 0.424 | 0.106 | 0.874 |
| H2 visual SlowFast, logistic | 0.344 | 0.347 (0.17–0.54) | 0.526 | 0.131 | 0.848 |
| H2 audio PANN, logistic | 0.385 | 0.353 (0.18–0.48) | 0.656 | 0.160 | 0.811 |
| H2 all embeddings + loudness, logistic | 0.507 | 0.467 (0.23–0.62) | 0.719 | 0.180 | 0.791 |
| H2 all embeddings + loudness, boosted trees | 0.505 | 0.477 (0.23–0.65) | 0.762 | 0.191 | 0.787 |
| H3 temporal fusion | 0.688 | 0.619 (0.42–0.75) | 0.776 | 0.194 | 0.785 |
| H3 without aud_pann | 0.627 | 0.536 (0.34–0.70) | 0.699 | 0.173 | 0.805 |
| H3 without vid_clip | 0.675 | 0.616 (0.40–0.75) | 0.772 | 0.192 | 0.789 |
| H3 without vid_slowfast | 0.653 | 0.583 (0.39–0.71) | 0.785 | 0.198 | 0.785 |
| H3 without loudness | 0.664 | 0.600 (0.42–0.74) | 0.778 | 0.193 | 0.789 |
| H3 + commentary (equal-weight z-score sum) | 0.595 | 0.573 (0.37–0.71) | 0.737 | 0.180 | 0.795 |

Chance average precision ≈ 0.088. Shipped (by validation mAP): **H3 temporal fusion**. Scores are ranking scores, not probabilities: no calibrator was fitted with only 6 validation games. Model-generated captions were not used. Negative results: adding the commentary score to H3 with equal weights lowered mAP; the commentary model scores "no touchdown" as high as "touchdown" (no negation handling), though it does score historical, hypothetical and replay phrasing low.

Event extension: taxonomy defined (touchdown, interception, sack, fumble lost, derived turnover, big play = 20+ yard pass or 10+ yard rush); metadata-led recaps from play-by-play are shown as a separate product, not aligned to video time; a review tool records event labels, evidence basis, uncertainty, event moment, clip boundaries and replay status in the browser. **No event labels or event metrics exist.** `highlights/local_media.py` (ffmpeg audio → loudness candidates → editable cut list) is written but has never been run: ffmpeg is not installed and no authorized footage was supplied.

## Engineering

Run records: `reports/v2/runs/*.json` (module, run id, commit, seed, packages, device, data hashes, split, target, population, cutoff, settings, metrics, outputs, seconds, peak memory); published to the site as frozen JSON. Rights manifest: `data/manifests/rights.json`. MLflow not used: the JSON contract covers the same fields with no server.

| Run | Seconds | Peak memory (MB) |
|---|---|---|
| `coverage-benchmark-v2-20261004T091405` | 225.3 | 509.1 |
| `coverage-expA-cv-20261004T081739` | 1506.6 | 278.2 |
| `coverage-expB-cv-20261004T081741` | 2974.4 | 914.4 |
| `coverage-expB-final-20261004T090716` | 402.9 | 914.4 |
| `coverage-expC-family-20261004T091815` | 138.1 | 1300.8 |
| `highlights-ranking-v1-20261004T084256` | 359.4 | 1269.5 |
| `pregame-v2-backtest-20261004T075113` | 92.4 | 332.6 |
| `pregame-v2-backtest-20261004T075359` | 89.1 | 335.7 |

Checks: 28 pytest tests pass; `ruff` clean; `npm run lint` and `npm run build` pass. Export parity: 4,111 exported coverage predictions match saved records (max difference 0.00005, zero decision mismatches). Browser: 42 view × model × cutoff combinations match saved records; 14 pages checked at 390 px: one overflow found on `/predictions` (a wide select) and fixed, none on the rest; all controls named; one `h1` per page. Not done: a real VoiceOver pass.

## Repository and deployment (2026-10-04)

- GitHub: `Harvard2016/nfl-analytics-platform` (public), branch `main`, published with a fresh history. CI: `.github/workflows/ci.yml`.
- Vercel: project `nfl-analytics-platform` (team `harvard2016s-projects`), framework Next.js, root directory `apps/web`, production URL https://nfl-analytics-platform-theta.vercel.app (public; per-deployment URLs stay behind Vercel login).
- Deployed from `main` with the Vercel CLI. **Not yet done:** the GitHub integration could not be connected from the CLI because the Vercel GitHub App has no access to the repository; until the owner grants it, pushes do not deploy automatically (see `docs/CODEX_HANDOFF.md`).
- History audit before the push: no secrets, no raw datasets, archives, media or model binaries in any commit; largest blob 5 MB (a site export). No environment variables are needed.
- Public display of the site exports (including per-play tracking for the sampled test plays) was confirmed by the owner on 2026-10-04. The Kaggle and NFL terms were not independently reviewed; a web search surfaced NFL Big Data Bowl terms that restrict redistribution without written consent, so this rests on the owner's confirmation.
- Deployed site checked at 1440 and 390 px: 14 pages return 200 with no failed requests or console errors, 33 internal links resolve, fonts load, coverage / predictions / highlights selectors work, mobile menu works, highlight source links open YouTube (no embeds exist).

## Visual redesign (branch `ui-redesign`, 2026-10-04)

Presentation and interaction only: no model, calibrator, threshold, label, export or report was changed (`git status` shows changes only under `apps/web` source and `docs`). Design system: `docs/DESIGN_SYSTEM.md`.

- Shared: charcoal / ivory / chartreuse / sage / powder blue / amber tokens, Barlow Condensed + Inter + IBM Plex Mono, new nav (active state, compact on scroll, mobile menu, skip link), footer, page transitions.
- Home: "Read the field." hero over an original SVG stadium illustration (not footage), three chapters each with a real result and its limit.
- Coverage: field-first film room ("Read the defense."): library with search and filters, field with playback and expand dialog, Evidence / Similar plays / Errors tabs. Default model unchanged (v1 boosted trees). Man/Zone only.
- Predictions: matchup layout ("Before kickoff.") with Elo baseline, 24 h cutoff, backtest label, observed result, log-odds breakdown, "No established gain over Elo".
- Highlights: amber timeline editor ("Find the moment.") with Evidence / Missed moments / Review tabs, audio and transcript evidence, play-by-play recap, benchmark table. Ranks, not probabilities; no hosted video.
- Research: "The playbook." with three case studies drawn from the exported records and an experiment notebook listing the run records.
- Engineering: "Built to be inspected." with an architecture figure, data-to-screen steps, commands, stack, checks, rights table.

Checks after the redesign: `npm run lint` and `npm run build` pass; 28 pytest tests pass; `ruff` clean. Browser (Chrome via Playwright): 14 pages at 1440, 1024, 768 and 390 px with no horizontal overflow (two found on `/predictions` and fixed) and no console errors; 42 play × model × cutoff combinations match the exported records; expand dialog, Esc and focus return, tab arrow keys, playback, search, mobile menu, library drawer, game strip, highlight tabs and reel length all exercised; motion on and `prefers-reduced-motion` both verified. Not done: VoiceOver pass; contrast was not machine-audited.

Not available, so not shown: GitHub link (no remote configured), generated or photographic hero image (an original SVG illustration is used), an animation library (CSS only).

Screenshots: `docs/screenshots/{home,coverage,predictions,highlights,research,engineering}-{desktop,mobile}.jpg`.

## Failed or abandoned
- iCloud eviction of the project folder (fixed by moving the repo).
- Experiment runs starved when launched as low-priority background jobs on a 3% battery; rerun with checkpointing.
- MPS training slower than CPU for the temporal model.
- Commentary + H3 fusion hurt; pregame v2 did not beat Elo; larger Man weights did not help.

## Blocked: exact inputs needed
1. **Event detection metrics and visual review:** authorized game video (any of the 40 mapped games) plus either scoreboard-clock reading or 5–10 manual time anchors per game to verify video-time to play alignment.
2. **Local media path:** `brew install ffmpeg` and one authorized video file.
3. **A fresh coverage test:** a tracking release with labels not yet examined (another season or weeks), to score the frozen v2 models once.
4. **Timestamped market or injury data** for a fair pregame comparison (optional).

## Next
Confirm the temporal model on unseen data when available; calibrate and benchmark-score the family model once; add reviewer annotations for the development error queue; VoiceOver pass; grant the Vercel GitHub App access so pushes deploy automatically.

## Cinematic website redesign — 2026-10-04

Review branch: `design/cinematic-film-room`. Presentation only: original stadium/helmet artwork, richer mastheads, restrained decorative routes, GitHub links, and interaction-loaded YouTube interval playback synchronized to saved model/audio evidence. Existing datasets, models, metrics and frozen outputs are unchanged. Source mapping uses released trim offsets; intervals are ranking windows, not verified play boundaries. No broadcast media is hosted.

Local lint, production build and four timestamp/export unit tests pass. GitHub Actions run 37201021592 passed both Python and web jobs, including the four browser tests (six pages at 1440/390 px, matchup identity, mocked playback). The 12 screenshots were reviewed: layout and typography are readable with no tested page overflow. Screenshot capture now also waits for image decoding. Vercel GitHub integration is active and the branch preview deployed successfully. Preview Authentication blocks direct browser review in the Codex session; actual YouTube playback and a screen-reader pass remain manual checks. Draft PR #1 contains the changes; production is unchanged.

## Follow-up: module tours and YouTube playback

Added five-step first-visit tours to Coverage, Predictions and Highlights with highlighted controls, module-specific copy, replay and local completion memory. Added keyboard/mobile regression tests. Player setup now includes iframe referrer/permission attributes and page origin before loading, bounded loadVideoById calls, specific YouTube error messages/codes and retry. Six timestamp/embed/error unit tests pass locally. GitHub Actions run 37202389644 passed Python/web checks and all 12 deterministic browser tests (tour steps, focus, completion/replay, mobile bounds, independent module memory, playback retry and existing page/data checks). The six tour screenshots were reviewed. A separate one-source check returned YouTube error 150 for the default Super Bowl 52 game: the owner has disabled embedding. The player connection works, but this video can only be opened on YouTube. This diagnosis does not establish the status of the other 11 source games; embed availability can change. See docs/audits/youtube-player-check.json. The public production URL still showed the older link-only Highlights page when checked. Changes remain on draft PR #1 until merged.

## V3: inference, uploads, forecasts and a third round of experiments (2026-10-04)

Branch `feat/inference-and-model-v3`. Full detail, commands and blocked items: `docs/V3_EXECUTION_STATUS.md`. Model cards: `docs/model_cards/`. Dataset notes: `docs/DATASETS.md`. Experiment plans registered before running: `docs/experiments/`.

**What now works**
- Saved coverage and highlight models reproduce their saved predictions exactly from the files on disk.
- Highlight ranker H3 has an inference bundle (weights, PCA transforms, scaling, manifest) that reproduces cached inputs and saved scores on all 40 games.
- Reels obey an exact output budget; the first selector, which could run 188 s for a 180 s reel, is preserved as v1.
- 2026 forecasts follow protocol v3: preserved and hashed sources, creation time read after fitting, saved bundle, idempotent append-only records. 16 official forecasts and 1 late one were written on 2026-10-04.
- A local inference service and two pages: tracking classification (same probabilities as the offline pipeline for real plays) and clip ranking with the loudness baseline, synchronized playback and exact cuts.
- A review tool for the 100-play development queue and out-of-fold error slices.

**What the experiments showed (development data unless stated)**
- Coverage: attention pooling gave no gain; robustness training costs a little on clean input and helps a lot on degraded input; a three-seed average and a blend with trees are better calibrated than one seed; a hierarchical family head does not beat a flat one. Champion unchanged. Weeks 15–18 not scored.
- Highlights: a learned fusion with commentary and a ranking-aware loss beat the control on development folds; rebuilt and scored once on used games, the ranking-aware model did not hold up and the fusion was level on validation and better on the examined test games. Negation features and segment-aware decoding did not help. H3 stays shipped.
- Game prediction: opponent-adjusted ratings beat Elo on nested development seasons and match the frozen v2 model. No established gain over Elo beyond v2. Not promoted.

**Not done, and why**
- Fresh-season coverage validation: the candidate dataset needs a manual download.
- Trained multimodal scoring of new video: the CLIP, SlowFast and PANN extractors are not installed or parity-checked.
- Commentary mode for uploads: no local speech-to-text.
- Video to coverage: calibration and gating library only; no detection, tracking or review interface; never run on footage.
- Real-media smoke test: no authorized footage supplied.
- Event detection: no reviewed labels.
- The public site has no inference service; the upload pages explain local operation there.
