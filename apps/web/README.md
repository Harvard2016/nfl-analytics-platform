# Gridiron Lens website

[Open the live product](https://nfl-analytics-platform-theta.vercel.app/) · [Visual overview](../../README.md) · [Full setup guide](../../docs/GETTING_STARTED.md)

This Next.js 16 / React 19 / TypeScript app presents **three independent systems**. It reads frozen JSON exports under `public/demo/`; no Python server or downloaded dataset is needed to browse it.

```sh
# From apps/web, with Node.js 24
npm ci
npm run dev
```

| Folder | Purpose |
|---|---|
| `app/` | Routes: home, coverage, predictions, highlights, research and engineering |
| `components/` | Field playback, timelines, explanations, page tours and shared layout |
| `lib/` | Typed export contracts, loaders, time mapping and upload client helpers |
| `public/demo/` | Intentionally public saved results and approved display samples |
| `public/art/` | Original decorative stadium and helmet illustrations |
| `tests/` | Unit tests, public-browser checks and mocked configured-upload regressions |

## Checks

```sh
npm run lint
npm test
npm run build
npm run test:ui
npm run test:inference
```

Playwright needs Chromium installed first (`npx playwright install chromium`). The inference suite starts a separate development build at port 3112 and mocks the local API. It does not need private model weights or real media.

## Optional local inference

Start `bin/gridiron-api` from the repository root, then:

```sh
NEXT_PUBLIC_GRIDIRON_API=http://127.0.0.1:8765 npm run dev
```

`NEXT_PUBLIC_GRIDIRON_API` is a public browser configuration value, never a secret. Keep it unset for the Vercel product. The public build explains local operation and has no file input. Model and ffmpeg/Whisper requirements are in the full setup guide.

Everything in `public/` is downloadable. Review new exports against `data/manifests/rights.json` and run the repository publication check before pushing them. Read [AGENTS.md](AGENTS.md) and the bundled Next.js docs before changing framework behavior.
