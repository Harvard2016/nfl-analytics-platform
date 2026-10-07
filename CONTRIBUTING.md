# Working on Gridiron Lens

Start with the [visual README](README.md), [setup guide](docs/GETTING_STARTED.md), [data map](data/README.md) and [AGENTS.md](AGENTS.md). Use a branch and a PR for changes.

## Keep the three systems independent

Coverage, pregame prediction and highlight ranking each have their own inputs, fitted models, targets and evaluation. Shared IDs and UI are useful; joining their ML pipelines is not the project design.

## A model change needs evidence

1. Write the experiment plan before running it: target, development split, comparison, metric and promotion rule.
2. Select features, settings, thresholds and calibrators using development data only.
3. Store a run record with commit, seed, source hashes, split, cutoff and results.
4. Include error slices and limitations, including negative results.
5. Keep the shipped model unless the planned comparison supports a replacement.

2023 coverage weeks 15–18, the small 2024 sample, highlight test games and pregame 2023–2025 have been examined. Do not rename them fresh tests or use them to keep choosing a model. Released coverage labels are not known defensive intent; model explanations are not causes.

## Protect data and the record

Keep raw inputs, features, model binaries, credentials, uploads and media ignored. Review public exports against `data/manifests/rights.json`. Never overwrite `reports/v1/`, frozen split manifests or append-only forecasts. Use separate working directories for research reruns. Do not upload dataset archives to a PR.

## Before a PR

```sh
python scripts/check_repository.py
uv run --no-sync ruff check src tests
uv run --no-sync python -m pytest -q
```

For website changes, run lint, unit tests and build in `apps/web`; run the relevant browser tests for interaction changes. CI also scans history for secrets. Never paste raw scanner findings or private data into the PR body.

Explain the final behavior, why it helps, what was tested and any remaining limits. New screenshots must contain only approved public displays or clearly labelled synthetic inputs.

## Updating the README

Keep the live website link and real product screenshots near the top. Numerical graphics come from saved reports, not invented demonstration values. Run `python scripts/build_readme_assets.py` to rebuild `results.svg`; the optional playback command is documented in `docs/assets/readme/README.md`. Preserve simple wording and alt text.
