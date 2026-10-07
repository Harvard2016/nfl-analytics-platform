# Dataset notes (v3)

Rights and display terms: `data/manifests/rights.json`. Nothing raw is in Git. Start with the [data map](../data/README.md) and [source catalog](../data/catalog.json). A fresh clone does not include the files listed as present on the owner's machine.

| Source | Local path | Used for | Owner-local state reported through 2026-10-06 |
|---|---|---|---|
| NFL Big Data Bowl 2026 Analytics (2023 tracking, coverage labels) | `data/raw/bdb2026/` | coverage training and evaluation | present; 14,108 pass plays, 14,105 labelled (4,013 man, 10,092 zone) |
| NFL Big Data Bowl 2026 Prediction | `data/raw/bdb2026_prediction/` (only `test_input.csv` and checksums extracted) | one-time fresh evaluation | supplied 2026-10-06. No coverage labels in the archive. Its 143 test plays (three 2024 games) are labelled in the Analytics `supplementary_data.csv`, which also lists 3,758 more labelled 2024 plays that have no public tracking |
| nflverse schedules and play-by-play | `data/raw/nflverse/`, snapshots in `data/snapshots/nflverse/<id>/` | game prediction | present through 2026 week 4 (snapshot `20261004T223450Z`) |
| SVHighlights, American-football subset | `data/raw/svhighlights/`, `data/processed/svhighlights/` | highlight ranking | present; 40 games, features and annotations only, no video |
| NFL Helmet Assignment (video + tracking) | `data/raw/helmet_assignment/` (tables and one sample play extracted from the owner's archive) | measuring field calibration on real footage | supplied 2026-10-06; 60 plays x 2 views; no coverage labels; 2 pass plays; terms not reviewed; nothing from it is published except aggregate error figures |
| Owner-supplied game video | `data/local_media/` | upload smoke test | supplied 2026-10-06: one high-school game, 112 minutes. Private; never published |

## Source snapshots (game prediction)

`bin/pregame-lens snapshot` downloads the schedule and the current season's play-by-play, copies earlier seasons' files as independent bytes, and writes `snapshot.json` with a SHA-256 per file, fetch times, the server's Last-Modified/ETag where sent, and what the data covers. File sizes and hashes are verified before forecasting and publication, including older snapshots referenced by forecast records. A snapshot proves the bytes were on the research machine at that time. It does not prove when each row was first published upstream, and historical rows may have been revised since the games were played.

## Splits (unchanged, frozen)

- Coverage: weeks 1–12 train (three chronological development folds inside), 13–14 calibration and policies, 15–18 previously examined benchmark.
- Highlights: 28 / 6 / 6 games (`svh-football-split-v1`), four grouped development folds inside the 28.
- Game prediction: 2012–2022 development (walk-forward), 2023–2025 previously examined, 2026 prospective forecasts only.
