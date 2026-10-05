# Gridiron Lens

One NFL analytics website, three independent ML systems: Coverage (centerpiece), Game Prediction, Highlights.
They share IDs, run records, storage helpers, contracts and design. They never share features, models or evaluation,
and the game predictor never reads coverage or highlight outputs.

Read `docs/BUILD_STATUS.md` first for the current state.

**Location:** `~/Developer/nfl-analytics-platform`. Do not work in `~/Documents`: iCloud syncs it and evicts files.

## Rules
- No invented results. Synthetic fixtures test code mechanics only and are labelled synthetic.
- Coverage label = "released coverage label", never ground truth. Highlight label = editorial selection, not "a big play happened".
- Coverage results are predictions from the released selected-player tracking (defenders chosen by the release after the play).
  Never describe them as snap-time-only, full-field, leakage-free or deployment-ready.
- All model, feature and threshold selection happens on development data. Coverage weeks 15-18 and pregame 2023-2025 have been
  examined: anything scored there is a "comparison on a previously examined benchmark", never a fresh test.
- Keep observed inputs, released labels, model predictions, model-generated interpretations and statistical associations visibly separate.
- Features read only what the stated cutoff allows. See `FORBIDDEN_SUBSTRINGS` in `coverage/schema.py`, the excluded fields in
  `coverage/relational.py`, and the 28-hour rule in `pregame/features_v2.py`.
- `reports/v1/` is the preserved first benchmark: never overwrite it. New outputs go to `reports/v2/` (or a later version).
- Split manifests in `data/manifests/` are frozen. Forecast records in `reports/forecasts/` are append-only.
- Raw data, processed data, media and model binaries stay out of Git. Site exports under `apps/web/public/demo/` are committed.
  Sources and display limits: `data/manifests/rights.json`.
- Training stays local. The repo is public on GitHub (`Harvard2016/nfl-analytics-platform`) and `main` is deployed on Vercel
  (https://nfl-analytics-platform-theta.vercel.app). Everything under `apps/web/public/` is public: add nothing there that
  `data/manifests/rights.json` does not allow. No paid services without an explicit instruction from the owner.

- v3 (2026-10-04): new reports go to `reports/v3/`, model bundles to `models/<module>/v3/`. Experiment lists and selection rules are
  written in `docs/experiments/` **before** the first run and are not edited afterwards. v1 and v2 outputs stay untouched.
- Forecasts: only `bin/pregame-lens forecast-v3` records count. Never backfill a played game as a forecast, never edit a record, and
  never mark a late record as official. Source snapshots live in `data/snapshots/` (private, ignored by Git).
- Uploads: user files, job results and `data/local_media/` are private and never committed or copied under `apps/web/public/`.
  The public site must never accept a file or imitate inference when no inference service is configured.
- Highlight scores are ranks. The loudness baseline and the trained model must always be named separately; the trained model cannot
  score new media until the CLIP/SlowFast/PANN extractors are installed and parity-checked.

## Layout
- `src/gridiron_lens/{shared,coverage,pregame,highlights}`: one package per module; `shared/runs.py` is the run-record contract.
- `apps/web`: Next.js 16 site reading frozen exports. Read `apps/web/AGENTS.md` before changing Next.js code.
- `data/raw/{bdb2026,nflverse,svhighlights}`; `models/`; `reports/{v1,v2,v3,forecasts}`.
- `src/gridiron_lens/service`: local inference API (FastAPI, SQLite jobs). `src/gridiron_lens/videocov`: calibration library (partial prototype).

## Commands
- Coverage: `bin/coverage-lens audit|normalize|features|train` (v1), `relational|exp-a|exp-b|exp-c|benchmark-v2|export-v3` (v2).
- Game predictor: `bin/pregame-lens backtest|snapshot|forecast-v3|publish-forecasts`. Highlights: `bin/highlights-lens audit|run|export`.
- v3 modules run with `PYTHONPATH=src .venv/bin/python -m gridiron_lens.<module>`: `shared.v3_audit`, `highlights.bundle`, `highlights.eval_v3`,
  `highlights.experiment_v3`, `highlights.candidate_v3`, `coverage.experiment_v3`, `coverage.review_v3`, `pregame.experiment_v3`.
- Upload lab: `uv sync --extra service`, `bin/gridiron-api` (http://127.0.0.1:8765), then
  `cd apps/web && NEXT_PUBLIC_GRIDIRON_API=http://127.0.0.1:8765 npm run dev`. Command line: `python -m gridiron_lens.service.cli`.
- Site research export: `.venv/bin/python -m gridiron_lens.shared.publish`.
- Checks: `.venv/bin/python -m pytest -q`, `.venv/bin/ruff check src tests`, `cd apps/web && npm run lint && npm run build`.
- Site: `cd apps/web && npx next start -p 3111` after a build (restart it after adding new files under `public/`). Needs arm64 Node at `/opt/homebrew/bin`.
- Long jobs: launch with `nohup ... &` and thread limits (`OMP_NUM_THREADS=3`); harness background jobs run at low priority and starve.
