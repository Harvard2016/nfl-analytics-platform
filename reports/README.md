# Results map

These files are saved evidence, not a database to edit by hand. The site reads exported copies of selected reports.

| Folder | Meaning | Rule |
|---|---|---|
| `v1/` | Preserved first benchmark and its manifest | Never overwrite |
| `v2/` | Relational and temporal coverage, pregame comparisons, H0–H3 highlights | Keep model names and splits explicit |
| `v3/` | Third-round experiments, reproduction audits, small 2024 sample, calibration research | A candidate is not automatically a new champion |
| `v2/runs/`, `v3/runs/` | Commit, seed, inputs, settings, metrics and run times | Preserve the original record |
| `forecasts/` | Legacy prospective forecast records | Append-only; legacy protocol is not scored |
| `forecasts/v3/` | Timestamped v3 forecasts | Append-only; late forecasts are excluded from official scores |
| `experiments/` | Earlier summaries, including the clearly named synthetic fixture | Use versioned reports for current comparisons |

## Start with these

| Question | Evidence |
|---|---|
| How did coverage improve? | [v2/coverage_benchmark_v2.json](v2/coverage_benchmark_v2.json) |
| Did it work on any new season data? | [v3/coverage_fresh_2024.json](v3/coverage_fresh_2024.json), [registered protocol](../docs/experiments/coverage_fresh_2024.md) |
| Did v3 produce a better coverage model? | [v3/coverage_v3_experiments.json](v3/coverage_v3_experiments.json) |
| How good are the highlight ranks and strict reels? | [v2/highlights_ranking.json](v2/highlights_ranking.json), [v3/highlights_eval_v3.json](v3/highlights_eval_v3.json) |
| Does game prediction beat Elo? | [v2/pregame_backtest_v2.json](v2/pregame_backtest_v2.json), [v3/pregame_experiments_v3.json](v3/pregame_experiments_v3.json) |
| What reproduced exactly? | [v3/reproduction.json](v3/reproduction.json), [v3/highlights_bundle_parity.json](v3/highlights_bundle_parity.json) |

Whole-game bootstrap intervals in the coverage benchmark and play-level Wilson intervals in the small 2024 sample answer different questions. Keep that distinction when making a chart or writing a headline. Benchmarks already examined remain examined.

Do not reorganize these by renaming frozen files: run records, tests and site publishers refer to their current paths. This index supplies a clear entry point while preserving those links.
