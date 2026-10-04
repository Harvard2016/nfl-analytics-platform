import type { Metadata } from "next";
import Architecture from "@/components/Architecture";
import Engineering from "@/components/Engineering";
import Reveal from "@/components/Reveal";

export const metadata: Metadata = { title: "Engineering" };

const STEPS = [
  ["Raw files", "Stay unchanged on disk with checksums. Nothing raw is committed or published."],
  ["Tables", "Each pipeline writes Parquet tables for features and predictions, keyed by source ids."],
  ["Run record", "Every run writes JSON: commit, seed, packages, data hashes, split, target, cutoff, settings, metrics, time and memory."],
  ["Export", "A step writes compact JSON for the site: per-play tracking and predictions, per-game forecasts, per-game highlight timelines."],
  ["Screen", "The site loads those files in the browser. Tests check that exported predictions match the saved models."],
];
const CHECKS = [
  ["Coverage", "Direction rotation and angle convention against actual displacement; features unchanged when later frames are scrambled; whole-game splits; exported predictions equal saved-model predictions."],
  ["Temporal model", "Output unchanged under player reordering; earlier outputs unchanged when later frames are rewritten; masked player slots have no effect; left-right reflection equals mirroring the raw coordinates."],
  ["Game predictor", "Rewriting a target game's score, box score or starter does not change its features; changing an earlier game does; appending future games changes nothing earlier; the unplayed-game path matches the training path."],
  ["Highlights", "Split by game before any scoring; reduction transforms fitted on training games only; labels and alignment files never used as inputs."],
  ["Fixtures", "Synthetic fixtures test code mechanics only. They are never reported as model performance."],
];
const H2 = "display text-4xl sm:text-5xl";

export default function Page() {
  return (
    <>
      <header className="grain border-b border-line">
        <div className="mx-auto max-w-[1500px] px-4 pb-10 pt-12 lg:px-8 lg:pt-20">
          <p className="kicker">Engineering</p>
          <h1 className="display mt-2 text-6xl sm:text-8xl lg:text-9xl">Built to be <span className="outline-text">inspected</span><span className="text-teal">.</span></h1>
          <p className="narrow mt-4 text-3xl">One product. Three independent systems.</p>
          <p className="mt-3 max-w-[66ch] text-muted">Training, evaluation and feature building run locally in Python, one package per system. The website is a Next.js app that only reads frozen exports.</p>
        </div>
      </header>

      <section aria-labelledby="e-arch" className="mx-auto max-w-[1500px] px-4 py-12 lg:px-8">
        <h2 id="e-arch" className="sr-only">Architecture</h2>
        <Reveal><Architecture /></Reveal>
      </section>

      <div className="paper">
        <div className="mx-auto max-w-[1500px] px-4 py-14 lg:px-8">
          <section aria-labelledby="e-flow">
            <h2 id="e-flow" className={H2}>From data to the screen<span className="text-teal">.</span></h2>
            <ol className="mt-6 grid gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-5">
              {STEPS.map(([t, d], i) => (
                <Reveal as="li" key={t} className="border-t-2 border-ink pt-3">
                  <span className="display block text-5xl text-line" aria-hidden="true">{String(i + 1).padStart(2, "0")}</span>
                  <span className="narrow block text-2xl font-semibold">{t}</span>
                  <span className="mt-1 block text-sm text-muted">{d}</span>
                </Reveal>
              ))}
            </ol>
          </section>

          <section aria-labelledby="e-cmd" className="mt-14 grid gap-8 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
            <div className="min-w-0">
              <h2 id="e-cmd" className={H2}>Commands</h2>
              <pre tabIndex={0} aria-label="Project commands" className="mono mt-4 overflow-x-auto bg-[#101713] p-4 text-xs leading-6 text-[#f2efe5]">{`bin/coverage-lens audit | normalize | features | train      # first benchmark (preserved)
bin/coverage-lens relational | exp-a | exp-b | exp-c          # experiments, selection on weeks 1-12
bin/coverage-lens benchmark-v2 | export-v3                    # comparison and site export
bin/pregame-lens backtest | forecast                          # backtest; append-only live forecasts
bin/highlights-lens audit | run | export                      # ranking benchmark and timeline export
.venv/bin/python -m gridiron_lens.shared.publish              # run records and rights for the site
.venv/bin/python -m pytest -q                                 # data, leakage and export checks
cd apps/web && npm run lint && npm run build                  # site checks
cd apps/web && npx next start -p 3111                         # serve the built site locally`}</pre>
            </div>
            <div>
              <h2 className={H2}>Stack</h2>
              <dl className="mt-4 text-sm">
                {[["Pipelines", "Python 3.13, Polars, Parquet, DuckDB"], ["Models", "scikit-learn, PyTorch on CPU (Apple M2, 8 GB)"], ["Site", "Next.js, React, TypeScript, Tailwind"], ["Figures", "Hand-written SVG and HTML: field, timelines, charts"], ["Hosting", "Static site on Vercel; no backend, queues or databases"]].map(([k, v]) => (
                  <div key={k} className="grid grid-cols-[6.5rem_1fr] gap-3 border-t border-line py-2"><dt className="kicker pt-0.5">{k}</dt><dd className="text-muted">{v}</dd></div>
                ))}
              </dl>
            </div>
          </section>

          <section aria-labelledby="e-checks" className="mt-14">
            <h2 id="e-checks" className={H2}>Checks that run</h2>
            <dl className="mt-4 grid gap-x-10 text-sm md:grid-cols-2">
              {CHECKS.map(([k, v]) => <div key={k} className="border-t border-line py-3"><dt className="narrow text-xl font-semibold">{k}</dt><dd className="mt-1 max-w-[62ch] text-muted">{v}</dd></div>)}
            </dl>
          </section>

          <Engineering />
        </div>
      </div>
    </>
  );
}
