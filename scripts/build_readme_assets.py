"""Draw README charts from saved evidence; optionally animate approved tracking.

    python scripts/build_readme_assets.py
    uv run --no-project --with pillow python scripts/build_readme_assets.py --playback

The SVG needs only the standard library. Pillow is needed only for the optional
GIF. No training, test scoring or source data download is performed.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/readme"
INK, MUTED, LIME, BLUE, AMBER = "#f2efe5", "#a5b4a8", "#d8f16e", "#8ebaf1", "#e8ad66"


def load(name):
    return json.loads((ROOT / name).read_text())


def text(x, y, value, size=16, fill=INK, weight=400):
    return (f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}">{html.escape(str(value))}</text>')


def results_svg():
    cov = load("reports/v2/coverage_benchmark_v2.json")
    high = load("reports/v2/highlights_ranking.json")
    game = load("reports/v2/pregame_backtest_v2.json")["previously_examined_benchmark"]["models"]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1260 440" role="img" aria-labelledby="title desc">',
             '<title id="title">Three independent systems, three measured comparisons</title>',
             '<desc id="desc">Coverage accuracy at +1.5 seconds from selected-player tracking; highlight mean average precision; pregame log loss. All comparisons are on previously examined benchmarks. Bars start at zero; results are not comparable across tasks.</desc>',
             '<rect width="1260" height="440" rx="8" fill="#0d1912"/>',
             text(30, 40, "MEASURED. COMPARED. QUALIFIED.", 25, INK, 700),
             text(30, 64, "Different targets and datasets. Every chart comes from a saved report.", 15, MUTED)]
    panels = [
        (30, "01  COVERAGE", f"{100 * cov['models']['v2_temporal']['post_1_5s']['benchmark']['accuracy']:.1f}%", "accuracy / higher is better", LIME,
         [("v1 trees", cov["models"]["v1_gbm"]["post_1_5s"]["benchmark"]["accuracy"]),
          ("relational trees", cov["models"]["v2_gbm_rel"]["post_1_5s"]["benchmark"]["accuracy"]),
          ("temporal model", cov["models"]["v2_temporal"]["post_1_5s"]["benchmark"]["accuracy"])],
         "3,178 plays / +1.5 s / 2023 W15–18", "Selected players; examined benchmark."),
        (450, "02  HIGHLIGHTS", f"{high['results']['H3 temporal fusion']['test']['summary']['mean_average_precision']:.3f}", "mAP / higher is better", AMBER,
         [("loudness", high["results"]["H0 loudness"]["test"]["summary"]["mean_average_precision"]),
          ("commentary", high["results"]["H1 commentary"]["test"]["summary"]["mean_average_precision"]),
          ("temporal fusion", high["results"]["H3 temporal fusion"]["test"]["summary"]["mean_average_precision"])],
         "6 test games / editorial labels", "Rank quality, not event detection."),
        (870, "03  PREGAME", "~ ELO", "log loss / lower is better", BLUE,
         [("home prior", game["home_prior"]["log_loss"]), ("Elo", game["elo"]["log_loss"]),
          ("Elo-offset", game["elo_offset"]["log_loss"])],
         "854 games / 2023–2025", "Paired improvement interval includes zero."),
    ]
    for x, heading, headline, metric, accent, rows, scope, limit in panels:
        parts += [text(x, 105, heading, 15, accent, 700), text(x, 161, headline, 47, INK, 700),
                  text(x, 186, metric, 17, MUTED)]
        for j, (label, value) in enumerate(rows):
            y = 222 + j * 42
            fill = accent if j == len(rows) - 1 else "#526356"
            parts += [text(x, y, label, 14), f'<rect x="{x}" y="{y + 7}" width="310" height="9" rx="2" fill="#26362b"/>',
                      f'<rect x="{x}" y="{y + 7}" width="{310 * value:.3f}" height="9" rx="2" fill="{fill}"/>',
                      text(x + 323, y + 15, f"{100 * value:.1f}%" if x == 30 else f"{value:.4f}" if x == 870 else f"{value:.3f}", 14, MUTED if j != len(rows) - 1 else accent)]
        parts += [text(x, 356, "0", 11, MUTED), text(x + 281, 356, "100%" if x == 30 else "1.0", 11, MUTED),
                  text(x, 383, scope, 14), text(x, 407, limit, 13, MUTED)]
    parts += ['<path d="M420 92V415M840 92V415" stroke="#344237"/>', '</svg>']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.svg").write_text("\n".join(parts) + "\n")


def tracking_gif():
    from PIL import Image, ImageDraw, ImageFont

    # Fixed play from the already-public, owner-approved 2023 tracking sample.
    play = load("apps/web/public/demo/real/coverage/plays/2023121400_1879.json")
    width, height = 820, 600
    left, top, field_w, field_h = 24, 100, 620, 450
    x0, x1 = max(0, play["los_x"] - 12), min(120, play["los_x"] + 30)
    field_w = 53.3 * field_h / (x1 - x0)
    font = ImageFont.load_default(size=19)
    small = ImageFont.load_default(size=15)
    title = ImageFont.load_default(size=28)
    frames = []

    def point(e, i):
        x, y = e["x"][i], e["y"][i]
        if x is None or y is None:
            return None
        return (left + (53.3 - y) / 53.3 * field_w,
                top + (x1 - x) / (x1 - x0) * field_h)

    # Exactly observed frames 0..15; no invented motion or probability interpolation.
    for i in range(16):
        image = Image.new("RGB", (width, height), "#0d1912")
        draw = ImageDraw.Draw(image)
        draw.text((24, 18), f"{play['offense']} vs {play['defense']} / COVERAGE FILM ROOM", font=title, fill=INK)
        draw.text((24, 58), f"OBSERVED TRACKING     +{play['frames'][i] / 10:.1f}s after snap", font=font, fill=LIME)
        draw.rectangle((left, top, left + field_w, top + field_h), fill="#1c3527")
        for x in range(int(x0) + 1, int(x1) + 1):
            y = top + (x1 - x) / (x1 - x0) * field_h
            if x % 5 == 0:
                draw.line((left, y, left + field_w, y), fill="#58705e", width=1)
            for side_y in (23.58, 29.72):
                sx = left + (53.3 - side_y) / 53.3 * field_w
                draw.line((sx - 3, y, sx + 3, y), fill="#637664", width=1)
        los_y = top + (x1 - play["los_x"]) / (x1 - x0) * field_h
        draw.line((left, los_y, left + field_w, los_y), fill=LIME, width=2)
        for e in play["entities"]:
            p = point(e, i)
            if p is None or not top <= p[1] <= top + field_h:
                continue
            color = BLUE if e["side"] == "defense" else INK
            trail = [point(e, k) for k in range(i + 1)]
            trail = [p for p in trail if p is not None]
            if len(trail) > 1:
                draw.line(trail, fill="#6486ad" if e["side"] == "defense" else "#7d897d", width=2)
            px, py = p
            if e["side"] == "defense":
                draw.line((px - 6, py - 6, px + 6, py + 6), fill=color, width=3)
                draw.line((px - 6, py + 6, px + 6, py - 6), fill=color, width=3)
            else:
                draw.ellipse((px - 7, py - 7, px + 7, py + 7), outline=color, width=2,
                             fill=INK if e["passer"] else "#1c3527")
        draw.text((635, 147), "X  DEFENDER", font=small, fill=BLUE)
        draw.text((635, 177), "O  ROUTE RUNNER", font=small, fill=INK)
        draw.ellipse((636, 211, 646, 221), fill=INK)
        draw.text((655, 207), "PASSER", font=small, fill=INK)
        draw.text((635, 273), "SELECTED PLAYERS", font=small, fill=LIME)
        draw.text((635, 304), "No ball or linemen", font=small, fill=MUTED)
        draw.text((635, 329), "No pre-snap frames", font=small, fill=MUTED)
        draw.line((635, 366, 653, 366), fill=LIME, width=2)
        draw.text((660, 355), "SNAP LOS", font=small, fill=LIME)
        draw.text((635, 402), "Playback is observed", font=small, fill=MUTED)
        draw.text((635, 427), "movement, not a", font=small, fill=MUTED)
        draw.text((635, 452), "changing prediction.", font=small, fill=MUTED)
        draw.text((24, 570), f"Public play export {play['id']}  /  BDB 2026 Analytics, 2023 season", font=small, fill=MUTED)
        frames.append(image)
    durations = [180] * len(frames)
    durations[-1] = 1100
    frames[0].save(OUT / "coverage-playback.gif", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--playback", action="store_true", help="Also build the observed tracking GIF (requires Pillow)")
    args = parser.parse_args()
    results_svg()
    if args.playback:
        tracking_gif()
    print("README assets generated from committed public evidence; no models rerun.")


if __name__ == "__main__":
    main()
