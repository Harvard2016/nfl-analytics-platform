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

## TypeScript compiler and tooling API

TypeScript 7 provides the native `tsc` command but no JavaScript compiler API. Following [Microsoft's migration guide](https://devblogs.microsoft.com/typescript/announcing-typescript-7-0/#running-side-by-side-with-typescript-6.0), `@typescript/native` aliases TypeScript 7 while `typescript` aliases the official TypeScript 6 compatibility package. Next's default CLI type checker uses `tsc` 7; typescript-eslint and other API consumers import TypeScript 6. Run `npx tsc --version` to check the CLI. The lint tests verify that TypeScript syntax and the existing rules still work.

## Checks

```sh
npm run lint
npm test
npm run build
npm run test:ui
npm run test:inference
```

Playwright needs Chromium installed first (`npx playwright install chromium`). The inference suite starts a separate development build at port 3112 and mocks the local API. It does not need private model weights or real media.

### ESLint compatibility

ESLint 10 removed methods still used by Next's bundled React, import and accessibility plugins. The official `@eslint/compat` adapter wraps those plugins in `eslint.config.mjs`; no rules are disabled or lowered. The lint tests check that React, accessibility and TypeScript findings still appear at their existing severities. Some plugins still advertise ESLint 9 peer ranges, so npm prints peer warnings despite this deliberate adapter. Remove the adapter only after the plugins support ESLint 10 and the lint tests pass.

## Optional local inference

Start `bin/gridiron-api` from the repository root, then:

```sh
NEXT_PUBLIC_GRIDIRON_API=http://127.0.0.1:8765 npm run dev
```

`NEXT_PUBLIC_GRIDIRON_API` is a public browser configuration value, never a secret. Keep it unset for the Vercel product. The public build explains local operation and has no file input. Model and ffmpeg/Whisper requirements are in the full setup guide.

Everything in `public/` is downloadable. Review new exports against `data/manifests/rights.json` and run the repository publication check before pushing them. Read [AGENTS.md](AGENTS.md) and the bundled Next.js docs before changing framework behavior.
