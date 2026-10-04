"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import CoverageStudies from "@/components/CoverageStudies";
import { SelectionNote } from "@/components/ui";
import { type HorizonKey, HORIZONS, TYPE_LABELS, horizonFrom, loadDataset, loadJson, modelFrom, pct, signed } from "@/lib/demo";

type M = { n: number; man: number; zone: number; accuracy: number; balanced_accuracy: number; macro_f1: number; man_precision: number; man_recall: number; zone_precision: number;
  zone_recall: number; log_loss: number; brier: number; confusion?: { values: number[] } };
type Sweep = { cutoff: number; accepted: number; accepted_fraction_of_all: number; accepted_accuracy: number | null; man_accepted_fraction_of_man: number; zone_accepted_fraction_of_zone: number;
  man_correct_accepted_over_all_man: number; zone_correct_accepted_over_all_zone: number };
type Diff = { point: number; interval_95: [number, number] };
type Entry = {
  calibration_weeks: M & { plays: number }; benchmark: M; benchmark_game_bootstrap_95: { games: number; accuracy: [number, number]; macro_f1: [number, number]; log_loss: [number, number]; man_recall: [number, number] };
  reliability: { lo: number; n: number; mean_predicted: number; observed_man_rate: number }[];
  balanced_policy: { supported: boolean; threshold: number; note?: string; rule?: string; calibration_man_precision?: number; benchmark?: M };
  abstention: { cutoff: { cutoff: number; rule: string }; benchmark_sweep: Sweep[]; benchmark_at_cutoff: Sweep };
  operational: M & { plays_with_prefix: number; labelled_test_week_plays: number; not_scored: number };
  paired_vs_v1_gbm?: { games: number; resamples: number; difference: Record<"accuracy" | "macro_f1" | "log_loss" | "brier" | "man_recall" | "balanced_accuracy", Diff> };
};
type Bench = { labels: Record<string, string>; note: string; decision_rule: string; models: Record<string, Partial<Record<HorizonKey, Entry>>>;
  chosen: { experiment_a: { features: string; man_weight: number; rule: string }; experiment_b: { config: string; epochs: number; rule: string } };
  selected_player_sensitivity: { note: string; by_tracked_defenders: Record<string, { tracked_defenders: string; plays: number; man_share: number; accuracy: number; man_recall: number }[]> } };
type CvA = { results: { horizon: string; features: string; n_features: number; man_weight: number; mean: Record<string, number>; fold_sd: Record<string, number> }[] };
type CvB = { device_note: string; results: { config: string; mean_val_loss: number; val_loss_sd: number; median_best_epoch: number; folds: { seconds_per_epoch: number; plays_per_second: number; parameters: number }[];
  mean_by_horizon: Record<string, Record<string, number>> }[];
  structural_checks: Record<string, Record<string, number | boolean> | number> };
type Holdout = { design: string; folds: { held_out: string[]; scored_plays: number }[]; models: Record<string, { n: number; accuracy: number; macro_f1: number; log_loss: number; man_recall: number; man_precision: number;
  game_bootstrap_95: { accuracy: [number, number] }; accuracy_by_fold: number[] }> };
type Queue = { purpose: string; plays_available: number; disagreements_available: number; queues: Record<string, { key: string; week: number; defense: string; released_label: string; p_man: number; lean: string }[]> };
type Family = { status: string; classes: string[]; excluded: string[]; hierarchy: string; horizons: Record<string, { dev_plays: number; models: Record<string, { accuracy: number; balanced_accuracy: number; macro_f1: number; log_loss: number;
  per_class: Record<string, { precision: number; recall: number; support: number }> }> }> };

const TH = "py-1.5 pr-3 text-left font-normal text-muted";
const H2 = "narrow text-lg font-semibold";
const BENCH_MODELS = ["v1_gbm", "v2_gbm_rel", "v2_temporal", "v2_temporal_3seed"];
const diff = (d?: Diff, digits = 3) => (d ? `${signed(d.point, digits)} (${signed(d.interval_95[0], digits)} to ${signed(d.interval_95[1], digits)})` : "");

function Reliability({ rows }: { rows: Entry["reliability"] }) {
  const S = 220, P = 26, max = Math.max(...rows.map((r) => r.n));
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="h-56 w-56 shrink-0" role="img" aria-label="Reliability diagram: predicted man probability against the share of plays labelled man. The table gives the same numbers.">
      <rect x={P} y={P} width={S - 2 * P} height={S - 2 * P} fill="none" stroke="var(--color-line)" />
      <line x1={P} y1={S - P} x2={S - P} y2={P} stroke="var(--color-muted)" strokeDasharray="3 3" />
      {rows.map((r) => <circle key={r.lo} cx={P + r.mean_predicted * (S - 2 * P)} cy={S - P - r.observed_man_rate * (S - 2 * P)} r={2.5 + 5 * Math.sqrt(r.n / max)} fill="var(--color-teal)" fillOpacity={0.75} />)}
      <text x={S / 2} y={S - 6} textAnchor="middle" fontSize={10} fill="var(--color-muted)">Predicted man</text>
      <text x={10} y={S / 2} textAnchor="middle" fontSize={10} fill="var(--color-muted)" transform={`rotate(-90 10 ${S / 2})`}>Released label man</text>
    </svg>
  );
}

export default function CoverageEvaluation() {
  const params = useSearchParams();
  const [b, setB] = useState<Bench | null | undefined>(undefined);
  const [cvA, setCvA] = useState<CvA | null>(null);
  const [cvB, setCvB] = useState<CvB | null>(null);
  const [fam, setFam] = useState<Family | null>(null);
  const [hold, setHold] = useState<Holdout | null>(null);
  const [queue, setQueue] = useState<Queue | null>(null);
  const horizon = horizonFrom(params.get("window"));
  const asked = modelFrom(params.get("model"));
  const model = params.get("model") === "v2_temporal_3seed" ? "v2_temporal_3seed" : asked === "v1_logit" ? "v1_gbm" : asked;

  useEffect(() => {
    loadDataset().then(async (ds) => {
      if (!ds) return setB(null);
      setB(await loadJson<Bench>(ds, "benchmark_v2.json"));
      setCvA(await loadJson<CvA>(ds, "expA_cv.json")); setCvB(await loadJson<CvB>(ds, "expB_cv.json")); setFam(await loadJson<Family>(ds, "expC_family.json"));
      setHold(await loadJson<Holdout>(ds, "team_holdout_v2.json")); setQueue(await loadJson<Queue>(ds, "review_queue.json"));
    });
  }, []);
  if (b === undefined) return <p className="p-6 text-muted">Loading evaluation…</p>;
  if (b === null) return <p className="p-6 text-muted">No evaluation export found. Run bin/coverage-lens benchmark-v2 and export-v3, then reload.</p>;

  const h: HorizonKey = b.models[model]?.[horizon] ? horizon : "post_1_5s";
  const e = b.models[model][h]!;
  const v1 = b.models.v1_gbm[h];
  const cm = e.benchmark.confusion!.values;
  const at = e.abstention.benchmark_at_cutoff;
  const explorerModel = model === "v2_temporal_3seed" ? "v2_temporal" : model;
  const best = cvA?.results.filter((r) => r.horizon === h) ?? [];

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-8 p-4 lg:p-6">
      <SelectionNote />
      <header>
        <h1 className="display text-4xl">Man or zone: every model on the same plays</h1>
        <p className="mt-3 max-w-[72ch] text-muted">
          Train weeks 1–12, calibrate and choose operating policies on weeks 13–14, compare on weeks 15–18 ({e.benchmark.n.toLocaleString()} plays, {e.benchmark_game_bootstrap_95.games} games).
          Scores measure agreement with the released coverage label. {b.note}
        </p>
      </header>

      <nav aria-label="Model input cutoff" className="flex w-fit flex-wrap border border-line text-sm">
        {HORIZONS.map((w) => b.models[model][w.key]
          ? <Link key={w.key} href={`?window=${w.key}&model=${model}`} aria-current={w.key === h ? "true" : undefined} className={`narrow px-4 py-2 ${w.key === h ? "bg-teal font-semibold text-bg" : "text-muted hover:text-ink"}`}>{w.label}</Link>
          : <span key={w.key} className="narrow px-4 py-2 text-line" title="This model was not trained for this cutoff">{w.label}</span>)}
      </nav>

      <section aria-labelledby="e-all">
        <h2 id="e-all" className={H2}>Standard 0.5 decision, common cohort, model input up to {HORIZONS.find((w) => w.key === h)!.label}</h2>
        <div className="mt-2 overflow-x-auto">
          <table className="num w-full min-w-[820px] text-sm">
            <thead><tr><th className={TH}>Model</th><th className={TH}>Accuracy</th><th className={TH}>Balanced acc.</th><th className={TH}>Macro F1</th><th className={TH}>Man recall / precision</th><th className={TH}>Zone recall / precision</th><th className={TH}>Log loss</th><th className={TH}>Brier</th></tr></thead>
            <tbody>
              {BENCH_MODELS.filter((m) => b.models[m]?.[h]).map((m) => { const x = b.models[m][h]!.benchmark; return (
                <tr key={m} className={`border-t border-line ${m === model ? "bg-surface" : ""}`}>
                  <td className="py-1.5 pr-3"><Link className="underline decoration-line underline-offset-4 hover:decoration-teal" href={`?window=${h}&model=${m}`}>{b.labels[m]}</Link></td>
                  <td>{pct(x.accuracy, 1)}</td><td>{pct(x.balanced_accuracy, 1)}</td><td>{x.macro_f1.toFixed(3)}</td><td>{x.man_recall.toFixed(3)} / {x.man_precision.toFixed(3)}</td><td>{x.zone_recall.toFixed(3)} / {x.zone_precision.toFixed(3)}</td><td>{x.log_loss.toFixed(3)}</td><td>{x.brier.toFixed(3)}</td>
                </tr>); })}
            </tbody>
          </table>
        </div>
        {v1 && (
          <div className="mt-4 overflow-x-auto">
            <table className="num w-full min-w-[760px] text-sm">
              <caption className="pb-1 text-left text-muted">Difference from the v1 benchmark model on the same plays (95% range from resampling whole games). Lower is better for log loss and Brier.</caption>
              <thead><tr><th className={TH}>Model</th><th className={TH}>Accuracy</th><th className={TH}>Macro F1</th><th className={TH}>Man recall</th><th className={TH}>Log loss</th><th className={TH}>Brier</th></tr></thead>
              <tbody>
                {BENCH_MODELS.filter((m) => b.models[m]?.[h]?.paired_vs_v1_gbm).map((m) => { const d = b.models[m][h]!.paired_vs_v1_gbm!.difference; return (
                  <tr key={m} className="border-t border-line"><td className="py-1.5 pr-3">{b.labels[m]}</td><td>{diff(d.accuracy)}</td><td>{diff(d.macro_f1)}</td><td>{diff(d.man_recall)}</td><td>{diff(d.log_loss)}</td><td>{diff(d.brier)}</td></tr>); })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <div className="grid gap-8 md:grid-cols-2">
        <section aria-labelledby="e-conf">
          <h2 id="e-conf" className={H2}>{b.labels[model]}: confusion</h2>
          <table className="num mt-3 text-sm">
            <thead><tr><th /><th colSpan={2} className="pb-1 text-left font-normal text-muted">Predicted</th></tr><tr><th className="pr-3 text-left font-normal text-muted">Released label</th><th className="w-24 pb-1 font-normal">Man</th><th className="w-24 pb-1 font-normal">Zone</th></tr></thead>
            <tbody>
              <tr><th className="pr-3 text-left font-normal">Man</th><td className="border border-line p-4 text-center text-xl">{cm[0]}</td><td className="border border-line p-4 text-center text-xl text-coral">{cm[1]}</td></tr>
              <tr><th className="pr-3 text-left font-normal">Zone</th><td className="border border-line p-4 text-center text-xl text-coral">{cm[2]}</td><td className="border border-line p-4 text-center text-xl">{cm[3]}</td></tr>
            </tbody>
          </table>
          <p className="num mt-3 text-sm text-muted">Accuracy 95% range {pct(e.benchmark_game_bootstrap_95.accuracy[0], 1)} to {pct(e.benchmark_game_bootstrap_95.accuracy[1], 1)}; man recall {e.benchmark_game_bootstrap_95.man_recall[0].toFixed(3)} to {e.benchmark_game_bootstrap_95.man_recall[1].toFixed(3)} (whole games resampled).</p>
          <p className="mt-3 text-sm"><Link className="text-teal underline" data-testid="errors-link" href={`/coverage/errors?model=${explorerModel}&window=${h}`}>Inspect this model&apos;s errors at this cutoff</Link></p>
          <h3 className="narrow mt-6 font-semibold">Balanced class policy</h3>
          <p className="mt-1 text-sm text-muted">Goal: the man threshold that maximizes balanced accuracy on weeks 13–14 while keeping man precision at 0.80 or better.{" "}
            {e.balanced_policy.supported && e.balanced_policy.benchmark
              ? `Chosen threshold ${e.balanced_policy.threshold.toFixed(2)}. On the benchmark: balanced accuracy ${pct(e.balanced_policy.benchmark.balanced_accuracy, 1)}, man recall ${e.balanced_policy.benchmark.man_recall.toFixed(3)}, man precision ${e.balanced_policy.benchmark.man_precision.toFixed(3)}, accuracy ${pct(e.balanced_policy.benchmark.accuracy, 1)}. This is a different operating point for the same probabilities, not a better model.`
              : e.balanced_policy.note}</p>
        </section>
        <section aria-labelledby="e-cal">
          <h2 id="e-cal" className={H2}>Is the probability honest?</h2>
          <div className="mt-2 flex flex-wrap gap-4">
            <Reliability rows={e.reliability} />
            <table className="num h-fit text-sm">
              <thead><tr><th className={TH}>Predicted man</th><th className={TH}>Label man</th><th className={TH}>Plays</th></tr></thead>
              <tbody>{e.reliability.map((r) => <tr key={r.lo} className="border-t border-line"><td className="py-1 pr-3">{pct(r.mean_predicted)}</td><td className="pr-3">{pct(r.observed_man_rate)}</td><td>{r.n}</td></tr>)}</tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-muted">The confidence shown in the explorer is the larger of the calibrated man and zone probabilities. Platt scaling, fitted per model and cutoff on weeks 13–14.</p>
        </section>
      </div>

      <section aria-labelledby="e-abs">
        <h2 id="e-abs" className={H2}>Abstention: who gets left out?</h2>
        <p className="mt-1 max-w-[72ch] text-sm text-muted">{e.abstention.cutoff.rule}. At the chosen cutoff of {pct(e.abstention.cutoff.cutoff)}: {pct(at.accepted_fraction_of_all, 1)} of plays accepted, {pct(at.man_accepted_fraction_of_man, 1)} of man plays and {pct(at.zone_accepted_fraction_of_zone, 1)} of zone plays.
          High accepted accuracy that comes from declining man plays is not a full success, so the last two columns count correct accepted calls against all plays of each class.</p>
        <div className="overflow-x-auto">
          <table className="num mt-2 w-full min-w-[760px] text-sm">
            <thead><tr><th className={TH}>Confidence cutoff</th><th className={TH}>Accepted (of all)</th><th className={TH}>Accuracy on accepted</th><th className={TH}>Man accepted (of man)</th><th className={TH}>Zone accepted (of zone)</th><th className={TH}>Man correct and accepted (of all man)</th><th className={TH}>Zone correct and accepted (of all zone)</th></tr></thead>
            <tbody>
              {e.abstention.benchmark_sweep.map((r) => (
                <tr key={r.cutoff} className={`border-t border-line ${r.cutoff === e.abstention.cutoff.cutoff ? "bg-surface" : ""}`}>
                  <td className="py-1 pr-3">{pct(r.cutoff)}</td><td>{pct(r.accepted_fraction_of_all, 1)}</td><td>{r.accepted_accuracy == null ? "none" : pct(r.accepted_accuracy, 1)}</td>
                  <td>{pct(r.man_accepted_fraction_of_man, 1)}</td><td>{pct(r.zone_accepted_fraction_of_zone, 1)}</td><td>{pct(r.man_correct_accepted_over_all_man, 1)}</td><td>{pct(r.zone_correct_accepted_over_all_zone, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section aria-labelledby="e-time">
        <h2 id="e-time" className={H2}>How much does more input time help, and who is eligible?</h2>
        <div className="overflow-x-auto">
          <table className="num mt-2 w-full min-w-[760px] text-sm">
            <caption className="pb-1 text-left text-muted">{b.labels[model]}. Common cohort = plays with every cutoff available. Operational = every labelled test-week play that reached the cutoff, quick throws included.</caption>
            <thead><tr><th className={TH}>Cutoff</th><th className={TH}>Common: accuracy</th><th className={TH}>Common: man recall</th><th className={TH}>Common: log loss</th><th className={TH}>Operational plays</th><th className={TH}>Not scored (thrown earlier)</th><th className={TH}>Operational: accuracy</th><th className={TH}>Operational: man recall</th></tr></thead>
            <tbody>
              {HORIZONS.filter((w) => b.models[model][w.key]).map((w) => { const x = b.models[model][w.key]!; return (
                <tr key={w.key} className={`border-t border-line ${w.key === h ? "bg-surface" : ""}`}><td className="py-1 pr-3">{w.label}</td><td>{pct(x.benchmark.accuracy, 1)}</td><td>{x.benchmark.man_recall.toFixed(3)}</td><td>{x.benchmark.log_loss.toFixed(3)}</td>
                  <td>{x.operational.plays_with_prefix}</td><td>{x.operational.not_scored}</td><td>{pct(x.operational.accuracy, 1)}</td><td>{x.operational.man_recall.toFixed(3)}</td></tr>); })}
            </tbody>
          </table>
        </div>
      </section>

      <section aria-labelledby="e-dev">
        <h2 id="e-dev" className={H2}>How the v2 models were chosen (weeks 1–12 only)</h2>
        <p className="mt-1 max-w-[72ch] text-sm text-muted">Three chronological folds: train weeks 1–6, 1–8, 1–10 and validate on the next two weeks. Chosen for the boosted trees: {b.chosen.experiment_a.features}, man weight {b.chosen.experiment_a.man_weight} ({b.chosen.experiment_a.rule}).
          Chosen temporal model: {b.chosen.experiment_b.config}, {b.chosen.experiment_b.epochs} epochs ({b.chosen.experiment_b.rule}).</p>
        {best.length > 0 && (
          <div className="overflow-x-auto">
            <table className="num mt-3 w-full min-w-[760px] text-sm">
              <caption className="pb-1 text-left text-muted">Experiment A at this cutoff: mean over the three folds (spread across folds in brackets). Weighted models are recalibrated on the natural class mix.</caption>
              <thead><tr><th className={TH}>Features</th><th className={TH}>Count</th><th className={TH}>Man weight</th><th className={TH}>Log loss</th><th className={TH}>Balanced acc.</th><th className={TH}>Man recall</th><th className={TH}>Man precision</th></tr></thead>
              <tbody>
                {best.map((r) => <tr key={r.features + r.man_weight} className={`border-t border-line ${r.features === b.chosen.experiment_a.features && r.man_weight === b.chosen.experiment_a.man_weight ? "bg-surface" : ""}`}>
                  <td className="py-1 pr-3">{r.features}</td><td>{r.n_features}</td><td>{r.man_weight}</td><td>{r.mean.log_loss.toFixed(4)} ({r.fold_sd.log_loss.toFixed(4)})</td><td>{pct(r.mean.balanced_accuracy, 1)}</td><td>{r.mean.man_recall.toFixed(3)}</td><td>{r.mean.man_precision.toFixed(3)}</td></tr>)}
              </tbody>
            </table>
          </div>
        )}
        {cvB && (
          <div className="overflow-x-auto">
            <table className="num mt-5 w-full min-w-[760px] text-sm">
              <caption className="pb-1 text-left text-muted">Experiment B: temporal models, validation loss averaged over the four cutoffs. {cvB.device_note}</caption>
              <thead><tr><th className={TH}>Model</th><th className={TH}>Mean validation loss (spread)</th><th className={TH}>Median best epoch</th><th className={TH}>+1.5s log loss</th><th className={TH}>+1.5s man recall</th><th className={TH}>Seconds per epoch</th><th className={TH}>Parameters</th></tr></thead>
              <tbody>
                {cvB.results.map((r) => <tr key={r.config} className={`border-t border-line ${r.config === b.chosen.experiment_b.config ? "bg-surface" : ""}`}><td className="py-1 pr-3">{r.config}</td><td>{r.mean_val_loss.toFixed(4)} ({r.val_loss_sd.toFixed(4)})</td><td>{r.median_best_epoch}</td>
                  <td>{r.mean_by_horizon.post_1_5s.log_loss.toFixed(4)}</td><td>{r.mean_by_horizon.post_1_5s.man_recall.toFixed(3)}</td><td>{r.folds[0].seconds_per_epoch}</td><td>{r.folds[0].parameters.toLocaleString()}</td></tr>)}
              </tbody>
            </table>
            <p className="mt-2 text-xs text-muted">Mechanical checks on random inputs: reordering players changes the output by at most {Number((cvB.structural_checks.gru as Record<string, number>).player_order_max_abs_diff).toExponential(1)}; rewriting later frames changes earlier outputs by {(cvB.structural_checks.gru as Record<string, number>).future_frames_max_abs_diff_on_earlier_outputs}; filling masked player slots with garbage changes the output by {(cvB.structural_checks.gru as Record<string, number>).masked_slots_max_abs_diff}.</p>
          </div>
        )}
      </section>

      <section aria-labelledby="e-sens">
        <h2 id="e-sens" className={H2}>The selected-player limit, measured</h2>
        <p className="mt-1 max-w-[72ch] text-sm text-muted">{b.selected_player_sensitivity.note}</p>
        <div className="overflow-x-auto">
          <table className="num mt-2 w-full min-w-[640px] text-sm">
            <thead><tr><th className={TH}>Tracked coverage defenders</th><th className={TH}>Plays</th><th className={TH}>Released label man</th><th className={TH}>Accuracy</th><th className={TH}>Man recall</th></tr></thead>
            <tbody>{(b.selected_player_sensitivity.by_tracked_defenders[model === "v2_temporal_3seed" ? "v2_temporal" : model] ?? []).map((r) => <tr key={r.tracked_defenders} className="border-t border-line"><td className="py-1 pr-3">{r.tracked_defenders}</td><td>{r.plays}</td><td>{pct(r.man_share)}</td><td>{pct(r.accuracy, 1)}</td><td>{r.man_recall.toFixed(3)}</td></tr>)}</tbody>
          </table>
        </div>
      </section>

      {hold && (
        <section aria-labelledby="e-hold">
          <h2 id="e-hold" className={H2}>Defenses the v2 models never saw</h2>
          <p className="mt-1 max-w-[72ch] text-sm text-muted">{hold.design} Four folds of eight defenses, model input up to +1.5s.</p>
          <div className="overflow-x-auto">
            <table className="num mt-2 w-full min-w-[720px] text-sm">
              <thead><tr><th className={TH}>Model</th><th className={TH}>Plays</th><th className={TH}>Accuracy (95% range)</th><th className={TH}>By fold</th><th className={TH}>Macro F1</th><th className={TH}>Log loss</th><th className={TH}>Man recall / precision</th></tr></thead>
              <tbody>{Object.entries(hold.models).map(([m, v]) => <tr key={m} className="border-t border-line"><td className="py-1 pr-3">{b.labels[m] ?? m}</td><td>{v.n}</td><td>{pct(v.accuracy, 1)} ({pct(v.game_bootstrap_95.accuracy[0], 1)} to {pct(v.game_bootstrap_95.accuracy[1], 1)})</td><td>{v.accuracy_by_fold.map((a) => pct(a)).join(", ")}</td><td>{v.macro_f1.toFixed(3)}</td><td>{v.log_loss.toFixed(3)}</td><td>{v.man_recall.toFixed(3)} / {v.man_precision.toFixed(3)}</td></tr>)}</tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-muted">Lower than the benchmark numbers above, as expected when a defense is new to the model. Held-out groups: {hold.folds.map((f) => f.held_out.join(" ")).join(" / ")}.</p>
        </section>
      )}

      {queue && (
        <section aria-labelledby="e-queue">
          <h2 id="e-queue" className={H2}>Error-review queue (development weeks)</h2>
          <p className="mt-1 max-w-[72ch] text-sm text-muted">{queue.purpose} {queue.disagreements_available} of {queue.plays_available} development plays disagree with the temporal model.</p>
          <div className="mt-2 grid gap-4 md:grid-cols-2">
            {Object.entries(queue.queues).map(([name, rows]) => (
              <table key={name} className="num w-full text-xs">
                <caption className="pb-1 text-left text-sm text-muted">{name.replaceAll("_", " ")} ({rows.length})</caption>
                <tbody>{rows.slice(0, 6).map((r) => <tr key={r.key} className="border-t border-line"><td className="py-1 pr-2">{r.key}</td><td className="pr-2">wk {r.week} {r.defense}</td><td className="pr-2">label {r.released_label}</td><td>{pct(r.p_man)} man</td></tr>)}</tbody>
              </table>
            ))}
          </div>
          <p className="mt-2 text-xs text-muted">First six of each queue shown; the full lists are in the repository report. No reviewer annotations exist yet.</p>
        </section>
      )}

      {fam && (
        <section aria-labelledby="e-fam">
          <h2 id="e-fam" className={H2}>Coverage family: a separate task, development results only</h2>
          <p className="mt-1 max-w-[72ch] border-l-2 border-amber pl-3 text-sm text-muted">{fam.status} {fam.hierarchy}. Excluded: {fam.excluded.map((c) => TYPE_LABELS[c] ?? c).join(", ")} (too few plays).</p>
          {(["post_1s", "post_1_5s"] as const).filter((k) => fam.horizons[k]).map((k) => (
            <div key={k} className="overflow-x-auto">
              <table className="num mt-3 w-full min-w-[760px] text-sm">
                <caption className="pb-1 text-left text-muted">{HORIZONS.find((w) => w.key === k)!.label}, {fam.horizons[k].dev_plays} development plays. Per-class recall on the right.</caption>
                <thead><tr><th className={TH}>Model</th><th className={TH}>Accuracy</th><th className={TH}>Balanced acc.</th><th className={TH}>Macro F1</th><th className={TH}>Log loss</th>{fam.classes.map((c) => <th key={c} className={TH}>{(TYPE_LABELS[c] ?? c).replace(/ \(.*\)/, "")}</th>)}</tr></thead>
                <tbody>{Object.entries(fam.horizons[k].models).map(([m, v]) => <tr key={m} className="border-t border-line"><td className="py-1 pr-3">{m.replaceAll("_", " ")}</td><td>{pct(v.accuracy, 1)}</td><td>{pct(v.balanced_accuracy, 1)}</td><td>{v.macro_f1.toFixed(3)}</td><td>{v.log_loss.toFixed(3)}</td>{fam.classes.map((c) => <td key={c}>{v.per_class[c].recall.toFixed(2)}</td>)}</tr>)}</tbody>
              </table>
            </div>
          ))}
        </section>
      )}

      <section aria-labelledby="e-v1">
        <h2 id="e-v1" className={H2}>Earlier studies on the v1 benchmark model</h2>
        <p className="mt-1 text-sm text-muted">Play-count funnel, man-recall breakdown, unseen defenses and the first family attempt, computed for the v1 models at the three v1 cutoffs.</p>
      </section>
      <CoverageStudies windowKey={h === "post_0_5s" ? "post_1s" : h} model={asked === "v1_logit" ? "geometry_logit" : "geometry_gbm"} />
      <p className="text-xs text-muted">{b.decision_rule}</p>
    </div>
  );
}
