# README visual assets

| Asset | Origin |
|---|---|
| `home.jpg` | Deployed homepage screenshot, 2026-10-06 |
| `coverage.jpg` | Deployed `/coverage?model=v2_temporal`, moved to its +1.5 s cutoff |
| `predictions.jpg` | Deployed `/predictions`, historical DAL at PHI matchup |
| `highlights.jpg` | Deployed `/highlights`, source-viewer placeholder and ranking evidence; no video frame |
| `highlight-signals.jpg` | Same deployed page, scrolled to saved audio/rank signals and full-game timeline |
| `coverage-playback.gif` | Rendered from public play `2023121400_1879.json`, observed 10 Hz frames 0..15 |
| `results.svg` | Chart generated directly from committed coverage, highlight and pregame reports |

Screenshots show real product output, not design mockups. They were resized to 1,080 px wide and saved as JPEGs to keep the README fast. No private upload, access token or NFL broadcast frame is included. The decorative stadium and helmets have separate provenance in [ASSETS.md](../../ASSETS.md).

The GIF follows the explorer's field orientation: offense moving up, normalized +x forward and +y to the offense's left. Each frame uses an observed coordinate; trails show prior observations. It is an illustrative rendering of the public export, not a screen recording. It does not generate new model predictions or infer responsibilities.

Regenerate the chart with Python's standard library:

```sh
python scripts/build_readme_assets.py
```

Optionally regenerate the playback with Pillow, without installing the full research environment:

```sh
uv run --no-project --with pillow python scripts/build_readme_assets.py --playback
```

Take screenshots again when the product design changes; do not edit values in the images to make results look better.
