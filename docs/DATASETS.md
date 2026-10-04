# Dataset notes (v3)

Rights and display terms: `data/manifests/rights.json`. Nothing raw is in Git.

| Source | Local path | Used for | State on 2026-10-04 |
|---|---|---|---|
| NFL Big Data Bowl 2026 Analytics (2023 tracking, coverage labels) | `data/raw/bdb2026/` | coverage training and evaluation | present; 14,108 pass plays, 14,105 labelled (4,013 man, 10,092 zone) |
| NFL Big Data Bowl 2026 Prediction (candidate 2024 data) | `data/raw/bdb2026_prediction/` | possible fresh coverage validation | **not present**; needs a manual Kaggle download; label compatibility unknown (`reports/v3/coverage_fresh_data.json`) |
| nflverse schedules and play-by-play | `data/raw/nflverse/`, snapshots in `data/snapshots/nflverse/<id>/` | game prediction | present through 2026 week 4 (snapshot `20261004T223450Z`) |
| SVHighlights, American-football subset | `data/raw/svhighlights/`, `data/processed/svhighlights/` | highlight ranking | present; 40 games, features and annotations only, no video |
| NFL Helmet Assignment (video + tracking) | none | possible video research | not downloaded; no coverage labels; terms not reviewed |
| Authorized local media | `data/local_media/` | upload smoke tests | **none supplied** |

## Source snapshots (game prediction)

`bin/pregame-lens snapshot` downloads the schedule and the current season's play-by-play, hard-links the earlier seasons' files, and writes `snapshot.json` with a SHA-256 per file, fetch times, the server's Last-Modified/ETag where sent, and what the data covers. A snapshot proves the bytes were on this machine at that time. It does not prove when each row was first published upstream, and historical rows may have been revised since the games were played.

## Splits (unchanged, frozen)

- Coverage: weeks 1–12 train (three chronological development folds inside), 13–14 calibration and policies, 15–18 previously examined benchmark.
- Highlights: 28 / 6 / 6 games (`svh-football-split-v1`), four grouped development folds inside the 28.
- Game prediction: 2012–2022 development (walk-forward), 2023–2025 previously examined, 2026 prospective forecasts only.
