# Getting started

## Website only: no datasets or Python needed

Use Node.js 24, matching CI. From a fresh clone:

```sh
git clone https://github.com/Harvard2016/nfl-analytics-platform.git
cd nfl-analytics-platform/apps/web
npm ci
npm run dev
```

Open `http://localhost:3000`. The app uses frozen JSON exports already in the repository. You can browse coverage, predictions, highlights and research without fitting a model or downloading a game.

For a local production build:

```sh
npm run build
npm run start
```

Vercel uses `apps/web` as its root directory. The public deployment has no inference-service address and no working upload input. Website images are original illustrations or screenshots; there is no hosted NFL footage.

## Python checks and research

Use Python 3.13 and [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
uv sync --locked --extra service
uv run --no-sync ruff check src tests
uv run --no-sync python -m pytest -q
python scripts/check_repository.py
```

Tests use synthetic inputs for mechanics. Tests requiring private datasets or trained models skip if those files are absent; a green test suite is not a new model evaluation. On Linux the locked PyTorch build can download large CUDA packages even on a CPU machine; allow disk space before installing.

Web checks run in `apps/web`:

```sh
npm run lint
npm test
npm run build
npx playwright install chromium
npm run test:ui
npm run test:inference
```

The inference browser suite mocks API responses and the media clock. It checks upload-state behavior without private football footage.

## Local uploads: two terminals

Install `ffmpeg` and `ffprobe` (macOS: `brew install ffmpeg`; Ubuntu: install the `ffmpeg` package). `uv sync --locked --extra service` is required.

Terminal 1, repository root:

```sh
bin/gridiron-api
```

Terminal 2, repository root:

```sh
cd apps/web
NEXT_PUBLIC_GRIDIRON_API=http://127.0.0.1:8765 npm run dev
```

Open `/coverage/analyze` or `/highlights/analyze` on your local website. The API accepts loopback hosts and configured local browser origins. Do not expose it through a public tunnel or change the bind address to a public interface. It is not a multi-user upload server.

| Mode | What a fresh clone needs | What it does |
|---|---|---|
| Loudness baseline | ffmpeg; no trained ranker | Finds loud moments; these may be bands or timeouts |
| Commentary experiment | whisper.cpp binary plus its model under `models/asr/` | Adds locally transcribed words; not reliable event detection |
| Tracking classification | Local trained temporal coverage bundle | Man/zone outputs for valid 10 Hz tracking, not video |
| Trained H3 on new video | Not yet available | Released feature extractors still need verification |
| Video to coverage | Not yet available | Calibration research exists; detection/tracking are unfinished |

Jobs live in ignored `data/jobs/`. Tokens protect each job; files are deleted on request and expired completed jobs are swept while the service runs. A stopped service does not run a deletion timer. Do not share a job's token or media URL.

## Data and reproduction

Get the sources listed in [data/README.md](../data/README.md); keep credentials outside the checkout. Model binaries are not shipped. Use a separate research checkout and a separate `GRIDIRON_ROOT` working directory for reruns so original reports, splits, site exports and append-only forecasts stay preserved.

The root README links each model card and its existing reproduction commands. `bin/coverage-lens --help`, `bin/pregame-lens --help` and `bin/highlights-lens --help` list the module-specific commands. Some commands rebuild saved report files; do not run a full experiment sequence casually in the preserved portfolio checkout. Copy the frozen manifests into the isolated working directory first. Rerunning previously examined benchmarks does not make them fresh tests.

## Prospective forecasts

From the owner's research checkout with the historical nflverse inputs present:

```sh
bin/pregame-lens snapshot
bin/pregame-lens forecast-v3
```

Snapshots copy source bytes and verify hashes before forecasting and publication. Records written inside the 24-hour pregame cutoff are retained as late and excluded from official scoring. The launchd schedule in `ops/` is an optional macOS template, not an installed scheduler.
