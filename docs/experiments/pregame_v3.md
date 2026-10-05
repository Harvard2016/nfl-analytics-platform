# Game prediction v3 experiments: registered before any run

Written 2026-10-04 before the first v3 run. Results are in `reports/v3/pregame_experiments_v3.json`.

- Controls: Elo, and the frozen v2 Elo-offset and ridge-margin models (v2 hyperparameters, refit before each season).
- Development seasons: 2012–2022 (walk-forward, each season predicted by a model fitted on all earlier seasons since 2002).
  2023–2025 were examined in v1 and v2: the single frozen v3 choice is scored there once, as a comparison on a previously examined benchmark.
- Cutoff: 24 hours before kickoff; a prior game counts only if it kicked off at least 28 hours before the target kickoff (a proxy, unchanged).

| Id | Addition | Settings tried |
|---|---|---|
| P1 | Opponent-adjusted offence and defence ratings (EPA per play) from a ridge fit over prior games only, shrunk toward the league mean, recency-weighted, previous season carried at a discount | recency half-life 6 or 12 team-games; ridge fixed at 8 pseudo-games |
| P2 | Same ratings from points only (no EPA): a raw-stat check on whether EPA hindsight matters | same two half-lives |
| P3 | Shrunk giveaway and takeaway rates (toward the league rate, 400 pseudo-plays) added to P1 | one setting |
| P4 | Ridge point-margin model on the P1 ratings, and Platt calibration of every model fitted on earlier seasons' out-of-fold forecasts only | margin penalty 10, 100 |

Each feature set enters the Elo-offset logistic model with penalties 0.01, 0.1, 1.0 (the v2 grid). Total: 2 + 2 + 1 feature sets × 3 penalties, plus 2 margin fits, plus calibration.

**Not run:** quarterback scenarios from timestamped depth charts. nflverse documents those snapshots from 2025 on, so there is no way to test them on development seasons 2012–2022; they are a future prospective experiment, not a backtest.

**Selection:** nested by season. For each development season from 2015, the configuration with the lowest pooled log loss on the earlier development seasons is applied to that season. The nested score is what is reported as the v3 development result. A v3 model replaces the frozen v2 model in forecasts only if its nested log loss is lower than both Elo and v2 with a season-week bootstrap interval excluding zero. Otherwise forecasting stays on Elo and frozen v2.

## Results

Run on 2026-10-04. Report: `reports/v3/pregame_experiments_v3.json`. Summary and decision: `docs/V3_EXECUTION_STATUS.md`.
