# Codex handoff

State on 2026-10-04. Read `AGENTS.md` (rules), `docs/BUILD_STATUS.md` (results and caveats) and
`docs/DESIGN_SYSTEM.md` (tokens, components, motion) alongside this file. `apps/web/AGENTS.md` applies to all Next.js work:
this Next.js version differs from older ones, so check `node_modules/next/dist/docs/` before using an API from memory.

## Where things are

| | |
|---|---|
| GitHub | `Harvard2016/nfl-analytics-platform` (public) |
| Live site | https://nfl-analytics-platform-theta.vercel.app |
| Vercel project | `nfl-analytics-platform`, team `harvard2016s-projects`, root directory `apps/web`, framework Next.js |
| Branches | `main` (deployed). The repository was published with a fresh history; commit hashes quoted in reports and manifests (for example the frozen v1 baseline) refer to the earlier local history and do not exist here. |

## Repository structure

```
apps/web/                 Next.js 16 site (App Router, React 19, Tailwind v4, TypeScript). The only thing Vercel builds.
  app/                    routes: / , /coverage (+ /evaluation /errors /quick /tendencies), /predictions (+ /performance /forecasts),
                          /highlights (+ /review), /research (+ /experiments), /engineering
  app/layout.tsx          fonts, skip link, nav, footer        app/template.tsx   page transition
  app/globals.css         design tokens, .paper ivory surface, motion classes
  components/             one client component per page section (CoverageExplorer, Predictions, Highlights, ResearchCases, ...)
  lib/demo.ts             read contract for coverage exports: types, model and cutoff keys, status definition, loaders
  public/demo/            frozen JSON exports the browser fetches (public once deployed)
src/gridiron_lens/        Python: coverage/, pregame/, highlights/ (independent) and shared/ (ids, run records, publish)
bin/                      coverage-lens, pregame-lens, highlights-lens wrappers
tests/                    pytest: mechanics, leakage, cutoff and export checks
reports/v1, reports/v2    saved metrics and run records (reports/v1 is frozen); reports/forecasts is append-only
data/manifests/           frozen split manifests, audits, rights.json
docs/                     BUILD_STATUS, DESIGN_SYSTEM, this file, audits, screenshots
```

Not in Git and not needed by the site: `data/raw`, `data/processed`, `data/features`, `models/`, `*.parquet`, `*.npz`, `*.pt`.

## Install, run, check

```sh
# site (Node 22 or later)
cd apps/web && npm ci
npm run dev                      # http://localhost:3000
npm run lint && npm run build    # must both pass before pushing
npx next start -p 3111           # serve the production build

# python (only for pipeline or test work; uv, Python 3.13)
uv sync
.venv/bin/python -m pytest -q    # 28 tests; tests that need the datasets skip themselves
.venv/bin/ruff check src tests
```

No environment variables are required (`.env.example` documents that). The site has no backend, database or API routes.
The pipelines cannot be rerun without the datasets listed in `README.md`; the site does not need them.

## What is public and what may be shown

Everything under `apps/web/public/` is downloadable by anyone. Sources and display terms: `data/manifests/rights.json`
(mirrored to `public/demo/research/rights.json`, shown on `/engineering`).

| Path | Content | Basis |
|---|---|---|
| `public/demo/real/coverage/plays/*.json`, `index.json` | tracking, labels and model outputs for 304 sampled 2023 test plays | owner confirmed public display on 2026-10-04; terms not independently reviewed |
| `public/demo/real/coverage/*.json` (others) | aggregate evaluation, cross-validation, tendencies, studies | derived results |
| `public/demo/pregame/` | per-game features, probabilities, scores | nflverse, CC BY 4.0 as stated |
| `public/demo/highlights/` | ranking scores, editorial labels, loudness, transcript excerpts of 100 characters or fewer, links to source videos | SVHighlights, CC BY-NC 4.0 as stated; non-commercial only |
| `public/demo/research/` | run records, rights manifest, baseline manifest | project output |

Rules to keep: never add raw tracking files, full transcripts, audio, video or broadcast frames; never embed or host game
footage (the site links out to YouTube only); keep the attributions; keep the site non-commercial. The hero image is an
original SVG (`components/StadiumScene.tsx`), not footage. There are no team logos or NFL marks.

## Model outputs and experiment records

- Coverage explorer data: `public/demo/real/coverage/index.json` + `plays/`; written by `coverage/export_v3.py` (schema `coverage-demo-v3`).
- Coverage metrics: `benchmark_v2.json`, `evaluation.json`, `expA_cv.json`, `expB_cv.json`, `expC_family.json`, `team_holdout_v2.json`; sources in `reports/v2/`.
- Game predictor: `public/demo/pregame/v2_games.json`, `v2_performance.json`; forecast log in `reports/forecasts/`.
- Highlights: `public/demo/highlights/index.json`, `games/<id>.json`, `recaps/<id>.json`.
- Run records: `reports/v2/runs/*.json`, published as `public/demo/research/runs.json`.

The site only displays saved outputs. No number is computed in the browser beyond formatting and sums of saved terms.

## Behaviour that design changes must preserve

Coverage
- Default model stays `v1_gbm`, default cutoff `post_1_5s` (`lib/demo.ts`). Links from the home page deliberately pass `?model=v2_temporal`.
- Model cutoffs are `at_snap`, `post_0_5s`, `post_1s`, `post_1_5s`; v1 models have no `post_0_5s`. Each cutoff is a separately saved output. Playback only moves the picture: never interpolate or recompute predictions between cutoffs.
- Status has one definition (`statusOf`): abstained = confidence below the saved cutoff; correct / incorrect = accepted and agrees / disagrees with the released label. Counts, filters, badges and colours all use it.
- Classes are Man and Zone only. Say "released coverage label", never ground truth.
- Random sample, quick throws and the error gallery are separate views; the error gallery must keep its "picked because they are mistakes" note.
- The selected-player limit (defenders chosen by the release after the play) must stay visible on the explorer and evaluation pages. Weeks 15–18 results are a "previously examined benchmark".
- `data-testid` hooks (`lean`, `status`, `probs`, `label`, `settings`, `counts`, `shown`, `prefix`, `explanation`, `version`) are used by parity checks; keep them.

Game predictor
- Inputs stop 24 hours before kickoff. Browsable games are reconstructed backtests and must be labelled so; only `/predictions/forecasts` shows forecasts recorded before games.
- Contributions are in log-odds, with the running probability beside them. Do not present them as percentage points.
- "No established gain over Elo" stays visible wherever the model is shown.

Highlights
- Scores are ranks within a game (0–100), not probabilities. The label is editorial selection, not event detection. No event metrics exist.
- The editorial label is evaluation only and is shown separately from the reasons for a rank.

General
- Keep observed inputs, released labels, model predictions and derived explanations visibly separate (`Badge` in `components/ui.tsx`).
- Every headline number keeps its limitation next to it. Do not invent, round up or restate results; read them from the JSON.
- The three systems stay independent: no shared features, models or evaluation.
- Motion respects `prefers-reduced-motion`; no scroll hijacking or custom cursors.

## Deployments

- Current production was deployed from `main` with the Vercel CLI (`npx vercel@latest deploy --prod` from the repo root; the project's root directory setting points the build at `apps/web`).
- **GitHub-triggered deployments are not active yet.** `vercel git connect` failed because the Vercel GitHub App has not been given access to the repository. Owner action: open https://github.com/apps/vercel → Configure → the `Harvard2016` account → Repository access → add `nfl-analytics-platform`; then in the Vercel dashboard, project `nfl-analytics-platform` → Settings → Git → connect `Harvard2016/nfl-analytics-platform` (or run `npx vercel@latest git connect`). After that, a push to `main` deploys to production and a push to any other branch creates a preview deployment.
- Preview and per-deployment URLs are behind Vercel Authentication (project Settings → Deployment Protection). The production domain is public.
- GitHub Actions (`.github/workflows/ci.yml`) runs ruff, pytest, lint and build on every push.

## Unfinished

- Vercel GitHub integration (above).
- Screen-reader (VoiceOver) pass and a machine contrast audit.
- Subpages (evaluation, tendencies, performance, forecasts, experiments, review) inherit the new tokens and pass width and error checks but were not individually art-directed.
- Minor: the "Line of scrimmage" label can overlap a player marker; player rows in the coverage explanation table are taller than the group rows.
- Independent review of dataset terms (see BUILD_STATUS, "Repository and deployment").
- Science backlog (fresh coverage test data, event detection, ffmpeg path): BUILD_STATUS, "Blocked" and "Next".
