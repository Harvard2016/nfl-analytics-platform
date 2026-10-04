"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { type HorizonKey, TYPE_LABELS, loadDataset, loadJson, pct } from "@/lib/demo";

type WindowKey = HorizonKey;
const MODEL_LABELS: Record<string, string> = { class_prior: "Class prior", context_logit: "Team, down and distance", geometry_logit: "Logistic regression, v1 geometry", geometry_gbm: "Boosted trees, v1 geometry" };
const V1_KEY: Record<string, string> = { geometry_gbm: "v1_gbm", geometry_logit: "v1_logit" };

type Rate = { group: string; plays: number; correct: number; rate: number; interval_95: [number, number] | null };
type Slim = { n?: number; accuracy: number; balanced_accuracy: number; macro_f1: number; log_loss: number; brier?: number;
  per_class: Record<string, { precision: number; recall: number; support: number }>; accuracy_by_fold?: number[];
  game_bootstrap_95?: { accuracy: [number, number] } };
type Studies = {
  funnel: { steps: { step: string; plays: number; removed?: number; removed_labels?: Record<string, number> }[]; evaluated_plays: number; rule: string;
    by_split: Record<string, { plays: number; Man: number; Zone: number }> };
  short_throws: { note: string; groups: { group: string; plays: number; man: number; zone: number; models: Record<string, Slim> }[] };
  error_analysis: { note: string; windows: Record<string, Record<string, {
    man_recall: number; man_plays: number; missed_man: number; zone_recall: number; zone_plays: number;
    man_recall_by_released_coverage_type: Rate[]; zone_recall_by_released_coverage_type: Rate[]; man_recall_by_down: Rate[]; man_recall_by_distance: Rate[];
    missed_man_median_p_man: number; missed_man_share_near_the_line: number;
    missed_vs_caught_man_feature_gaps: { feature: string; missed_man_mean: number; caught_man_mean: number; zone_mean: number }[] }>> };
  team_holdout: { design: string; folds: { held_out: string[] }[]; windows: Record<string, Record<string, Slim>> };
  multiclass_dev: { status: string; classes: string[]; excluded_classes: Record<string, number>; exclusion_rule: string;
    windows: Record<string, { train_plays: number; dev_plays: number; models: Record<string, Slim & { per_class: Record<string, { precision: number; recall: number; f1: number; support: number }>;
      confusion: { labels: string[]; rows_released_cols_predicted: number[][] } }> }> };
};

const H2 = "narrow text-lg font-semibold";
const TH = "py-1.5 pr-3 text-left font-normal text-muted";

function RateTable({ rows, caption, label }: { rows: Rate[]; caption: string; label: (g: string) => string }) {
  return (
    <table className="num mt-2 w-full text-sm">
      <caption className="pb-1 text-left text-muted">{caption}</caption>
      <thead><tr><th className={TH}>Group</th><th className={TH}>Found</th><th className={TH}>Of</th><th className={TH}>Rate (95% range)</th></tr></thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.group} className="border-t border-line">
            <td className="py-1.5 pr-3">{label(r.group)}</td><td>{r.correct}</td><td>{r.plays}</td>
            <td>{pct(r.rate)}{r.interval_95 ? ` (${pct(r.interval_95[0])} to ${pct(r.interval_95[1])})` : ""}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function CoverageStudies({ windowKey, model }: { windowKey: WindowKey; model: string }) {
  const [s, setS] = useState<Studies | null | undefined>(undefined);
  useEffect(() => { loadDataset().then((ds) => (ds ? loadJson<Studies>(ds, "studies.json") : null)).then(setS); }, []);
  if (!s) return s === null ? <p className="text-sm text-muted">Extra studies have not been exported for this dataset.</p> : null;

  const errModel = model === "geometry_logit" ? "geometry_logit" : "geometry_gbm";
  const err = s.error_analysis.windows[windowKey]?.[errModel];
  const hold = s.team_holdout.windows[windowKey];
  const multi = s.multiclass_dev.windows[windowKey];
  const mm = multi?.models.geometry_gbm;
  const type = (g: string) => TYPE_LABELS[g] ?? g;

  return (
    <>
      {err && (
        <section aria-labelledby="h-err">
          <h2 id="h-err" className={H2}>Where it misses man coverage</h2>
          <p className="mt-2 max-w-[70ch] text-sm text-muted">
            {MODEL_LABELS[errModel]} finds {err.man_plays - err.missed_man} of {err.man_plays} test plays the release labels man ({pct(err.man_recall)}),
            against {pct(err.zone_recall)} of {err.zone_plays} zone plays. The {err.missed_man} missed man plays were mostly confident misses:
            their median man probability was {pct(err.missed_man_median_p_man)}, and only {pct(err.missed_man_share_near_the_line)} sat between 35% and 50%.
          </p>
          <div className="mt-3 grid gap-x-8 gap-y-5 md:grid-cols-2">
            <RateTable rows={err.man_recall_by_released_coverage_type} caption="Man plays found, by released coverage type" label={type} />
            <RateTable rows={err.man_recall_by_down} caption="Man plays found, by down" label={(g) => g} />
            <RateTable rows={err.zone_recall_by_released_coverage_type} caption="Zone plays found, by released coverage type" label={type} />
            <div>
              <table className="num mt-2 w-full text-sm">
                <caption className="pb-1 text-left text-muted">Why: missed man plays look like zone on these measurements</caption>
                <thead><tr><th className={TH}>Measurement (mean)</th><th className={TH}>Missed man</th><th className={TH}>Found man</th><th className={TH}>Zone</th></tr></thead>
                <tbody>
                  {err.missed_vs_caught_man_feature_gaps.slice(0, 5).map((g) => (
                    <tr key={g.feature} className="border-t border-line"><td className="py-1.5 pr-3">{g.feature}</td><td>{g.missed_man_mean}</td><td>{g.caught_man_mean}</td><td>{g.zone_mean}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <p className="mt-3 text-xs text-muted">{s.error_analysis.note} <Link className="text-teal underline" href={`/coverage/errors?model=${V1_KEY[errModel]}&window=${windowKey}`}>Inspect {MODEL_LABELS[errModel].toLowerCase()} errors at this window</Link></p>
        </section>
      )}

      <section aria-labelledby="h-funnel">
        <h2 id="h-funnel" className={H2}>Every play accounted for</h2>
        <table className="num mt-2 w-full text-sm">
          <thead><tr><th className={TH}>Step</th><th className={TH}>Removed</th><th className={TH}>Plays left</th></tr></thead>
          <tbody>
            {s.funnel.steps.map((st) => (
              <tr key={st.step} className="border-t border-line align-top">
                <td className="py-1.5 pr-3">{st.step}{st.removed_labels ? ` (${st.removed_labels.Man} man, ${st.removed_labels.Zone} zone)` : ""}</td>
                <td>{st.removed ?? ""}</td><td>{st.plays}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-2 max-w-[70ch] text-sm text-muted">{s.funnel.rule} The {s.funnel.evaluated_plays} remaining plays split into{" "}
          {Object.entries(s.funnel.by_split).map(([k, v]) => `${k === "dev" ? "calibration" : k} ${v.plays}`).join(", ")}.</p>
        {s.short_throws.groups.filter((g) => Object.keys(g.models).length > 0).map((g) => (
          <p key={g.group} className="mt-2 max-w-[70ch] text-sm text-muted">
            Left out of the test comparison for being thrown early: {g.plays} test-week plays ({g.man} man, {g.zone} zone). {g.group.split(": ")[1]}, boosted trees
            get {pct(g.models.geometry_gbm.accuracy)} of them. Too few plays for a firm number.
          </p>
        ))}
      </section>

      {hold && (
        <section aria-labelledby="h-hold">
          <h2 id="h-hold" className={H2}>Defenses the model has never seen</h2>
          <p className="mt-2 max-w-[70ch] text-sm text-muted">{s.team_holdout.design}</p>
          <div className="overflow-x-auto">
            <table className="num mt-2 w-full min-w-[640px] text-sm">
              <thead><tr><th className={TH}>Model</th><th className={TH}>Plays</th><th className={TH}>Accuracy (95% range)</th><th className={TH}>By fold</th><th className={TH}>Macro F1</th><th className={TH}>Log loss</th><th className={TH}>Man recall</th></tr></thead>
              <tbody>
                {Object.entries(hold).map(([k, v]) => (
                  <tr key={k} className="border-t border-line">
                    <td className="py-1.5 pr-3">{MODEL_LABELS[k] ?? k}</td><td>{v.n}</td>
                    <td>{pct(v.accuracy, 1)}{v.game_bootstrap_95 ? ` (${pct(v.game_bootstrap_95.accuracy[0], 1)} to ${pct(v.game_bootstrap_95.accuracy[1], 1)})` : ""}</td>
                    <td>{v.accuracy_by_fold?.map((a) => pct(a)).join(", ")}</td><td>{v.macro_f1.toFixed(3)}</td><td>{v.log_loss.toFixed(3)}</td><td>{pct(v.per_class.Man.recall)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-muted">Held-out groups: {s.team_holdout.folds.map((f) => f.held_out.join(" ")).join(" / ")}.</p>
        </section>
      )}

      {multi && mm && (
        <section aria-labelledby="h-multi">
          <h2 id="h-multi" className={H2}>Coverage type: development results only</h2>
          <p className="mt-2 max-w-[70ch] border-l-2 border-amber pl-3 text-sm text-muted">{s.multiclass_dev.status}</p>
          <div className="overflow-x-auto">
            <table className="num mt-3 w-full min-w-[520px] text-sm">
              <thead><tr><th className={TH}>Model ({multi.dev_plays} development plays)</th><th className={TH}>Accuracy</th><th className={TH}>Balanced accuracy</th><th className={TH}>Macro F1</th><th className={TH}>Log loss</th></tr></thead>
              <tbody>
                {Object.entries(multi.models).map(([k, v]) => (
                  <tr key={k} className="border-t border-line"><td className="py-1.5 pr-3">{MODEL_LABELS[k] ?? k}</td><td>{pct(v.accuracy, 1)}</td><td>{pct(v.balanced_accuracy, 1)}</td><td>{v.macro_f1.toFixed(3)}</td><td>{v.log_loss.toFixed(3)}</td></tr>
                ))}
              </tbody>
            </table>
            <table className="num mt-4 w-full min-w-[520px] text-sm">
              <caption className="pb-1 text-left text-muted">Boosted trees by class. Rows are the released label, columns the prediction.</caption>
              <thead><tr><th className={TH}>Released label</th>{mm.confusion.labels.map((l) => <th key={l} className={TH}>{type(l).replace(/ \(.*\)/, "")}</th>)}<th className={TH}>Recall</th><th className={TH}>Precision</th></tr></thead>
              <tbody>
                {mm.confusion.rows_released_cols_predicted.map((row, i) => {
                  const l = mm.confusion.labels[i];
                  return (
                    <tr key={l} className="border-t border-line"><td className="py-1.5 pr-3">{type(l)}</td>
                      {row.map((n, j) => <td key={j} className={i === j ? "text-ink" : n > 0 ? "text-coral" : "text-muted"}>{n}</td>)}
                      <td>{pct(mm.per_class[l].recall)}</td><td>{pct(mm.per_class[l].precision)}</td></tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-muted">Left out for {s.multiclass_dev.exclusion_rule}: {Object.entries(s.multiclass_dev.excluded_classes).map(([k, n]) => `${type(k)} (${n})`).join(", ")}. Probabilities are not calibrated yet.</p>
        </section>
      )}
    </>
  );
}
