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

## Analyze your own play or clip (local)

The public site has no inference service behind it. To run uploads on your own machine:

```sh
uv sync --extra service && brew install ffmpeg
bin/gridiron-api                                                     # loopback only, http://127.0.0.1:8765
cd apps/web && NEXT_PUBLIC_GRIDIRON_API=http://127.0.0.1:8765 npm run dev
```

- **Tracking classification** (`/coverage/analyze`): one play as CSV or JSON; the saved temporal coverage model returns man/zone probabilities at each cutoff the play reaches.
- **Clip ranking** (`/highlights/analyze`): the loudness baseline finds loud moments and cuts clips within an exact reel length. The trained multimodal ranker cannot score new files yet (its feature extractors are not installed).
- Files stay on your machine, are private to each job and are deleted on request or after 24 hours. Use only footage you may process.

Model weights are not in the repository, so a fresh clone must run the pipelines below first.

## 2026 forecasts

```sh
bin/pregame-lens snapshot        # download and preserve the schedule and play-by-play, with hashes
bin/pregame-lens forecast-v3     # append-only records for upcoming games; refreshes the site export
```

Schedule it with `ops/com.gridironlens.forecast.plist`. A record written after the 24-hour cutoff is kept and labelled late; it is never scored.

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

Round-three experiments are registered in `docs/experiments/` and run with `python -m gridiron_lens.<module>.experiment_v3` (see `docs/V3_EXECUTION_STATUS.md`).

Each run writes a record to `reports/v2/runs/` (or `reports/v3/runs/`) with the commit, seed, data hashes, split, cutoff, settings, metrics,
time and memory. The preserved first benchmark lives in `reports/v1/` and is never overwritten.

## Ground rules

- No invented results. Synthetic fixtures test code mechanics and are labelled synthetic.
- Labels are called what they are: a released coverage label, an editorial highlight selection, a final score.
- Model selection happens on development data. Benchmarks that have been looked at are labelled as previously examined.
- Explanations describe a fitted model. They are not causes.
- Raw data, media and model binaries stay out of Git. See `data/manifests/rights.json` for sources and display limits.
