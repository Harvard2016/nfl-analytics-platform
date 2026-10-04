# Gridiron Lens

One NFL analytics website with three independent machine-learning systems. Each has its own data, target, split,
models and limits. They share identifiers, run records, storage helpers and interface components, and nothing else.

| Module | Question | Data | State |
|---|---|---|---|
| Defensive coverage | Does the released label say man or zone, and what in the tracking supports it? | Big Data Bowl 2026 tracking, 2023 season | Three model families compared, film-room explorer |
| Game predictor | Who wins, using only what was known 24 hours before kickoff? | nflverse schedules and play-by-play | Backtests vs Elo, live forecast log |
| Highlights | Which moments of a broadcast did the editors put in the highlight reel? | SVHighlights, 40 NFL games | Ranking benchmark and evidence timeline |

Live site: https://nfl-analytics-platform-theta.vercel.app (static; reads frozen exports only).

Current results, caveats and next steps: [docs/BUILD_STATUS.md](docs/BUILD_STATUS.md). Handoff notes for the next agent: [docs/CODEX_HANDOFF.md](docs/CODEX_HANDOFF.md). Design system: [docs/DESIGN_SYSTEM.md](docs/DESIGN_SYSTEM.md).

## Run it

```sh
uv sync                                              # Python 3.13 environment
.venv/bin/python -m pytest -q                        # mechanics, leakage and export checks
cd apps/web && npm install && npm run dev            # http://localhost:3000
```

The site reads frozen exports under `apps/web/public/demo/`, so it runs without any dataset on disk.

## Reproduce the experiments

Raw data is not in the repository. Put it where the commands expect it:

| Data | Where | How to get it |
|---|---|---|
| Coverage tracking | `data/raw/bdb2026/` | Kaggle, NFL Big Data Bowl 2026 Analytics, after accepting its rules |
| Schedules, results | `data/raw/nflverse/games.csv` | `https://github.com/nflverse/nfldata` |
| Play-by-play | `data/raw/nflverse/pbp/` | nflverse-data releases, `play_by_play_<season>.parquet` |
| Highlight features | `data/raw/svhighlights/` | Hugging Face `idong1004/SVHighlights`, football archive and annotations |

```sh
bin/coverage-lens audit && bin/coverage-lens normalize && bin/coverage-lens features && bin/coverage-lens train   # v1 benchmark
bin/coverage-lens relational && bin/coverage-lens exp-a && bin/coverage-lens exp-b && bin/coverage-lens exp-c     # v2 experiments
bin/coverage-lens benchmark-v2 && bin/coverage-lens export-v3
bin/pregame-lens backtest && bin/pregame-lens forecast
bin/highlights-lens audit && bin/highlights-lens run && bin/highlights-lens export
.venv/bin/python -m gridiron_lens.shared.publish
```

Each run writes a record to `reports/v2/runs/` with the commit, seed, data hashes, split, cutoff, settings, metrics,
time and memory. The preserved first benchmark lives in `reports/v1/` and is never overwritten.

## Ground rules

- No invented results. Synthetic fixtures test code mechanics and are labelled synthetic.
- Labels are called what they are: a released coverage label, an editorial highlight selection, a final score.
- Model selection happens on development data. Benchmarks that have been looked at are labelled as previously examined.
- Explanations describe a fitted model. They are not causes.
- Raw data, media and model binaries stay out of Git. See `data/manifests/rights.json` for sources and display limits.
