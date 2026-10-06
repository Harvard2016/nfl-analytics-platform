# Highlights v3 experiments: registered before any run

Written 2026-10-04 before the first v3 training run. Results: `reports/v3/highlights_experiments_v3.json`.

- Data: the 28 training games, in the four grouped development folds frozen in `svh-football-split-v1`. Every learned transform (PCA, scaling, text vocabulary) is fitted inside each fold on its 21 training games.
- The 6 validation and 6 test games are **not scored** in these experiments. The test games were examined in v2 and stay a previously examined benchmark.
- Labels: released editorial labels, unchanged. No alignment artefact, caption or label is a model input.
- Control: the v2 H3 architecture and loss. Training length is fixed at 5 epochs for every variant (the v2 best epoch), so no held-out game steers stopping.

| Id | Change | Settings |
|---|---|---|
| H-C | control: H3 with binary cross-entropy | none |
| H-R | ranking-aware: cross-entropy plus a within-game pairwise ranking term (weight 0.5) | one |
| H-W | positive weight 3 with every game contributing the same number of windows per epoch | one |
| H-N | negation-aware commentary: the v2 word model plus counts of negation, reversal, hypothetical, replay and past-reference words and speech rate | one, compared with the v2 word model alone |
| H-F | learned late fusion of H3 with commentary (and with H-N), stacked on out-of-fold scores from the other folds; compared with H3 alone and the v2 equal-weight sum | one each |
| H-S | segment-aware decoding on H-C out-of-fold scores: segments from shot boundaries or transcript sentences, scored by peak, mean or 75th-percentile saliency, chosen without overlap under the exact output budget | 2 sources × 3 scores |

Seed 42. Ten configurations in total; nothing else is tried.

**Objectives.** Rankers: mean out-of-fold average precision over the 28 games (higher is better), with the game-level bootstrap interval of the paired difference from the control. Decoders: mean recall of labelled time at a strict 180-second output, ties to precision.

**Promotion.** A ranker replaces H3 only if its paired difference in average precision has a game-bootstrap interval above zero, and then it must be retrained on all 28 training games as a new version before any comparison on the validation or examined test games. Otherwise H3 stays.

## Results

Run on 2026-10-04. Report: `reports/v3/highlights_experiments_v3.json and highlights_candidates_v3.json`. Summary and decision: `docs/V3_EXECUTION_STATUS.md`.
