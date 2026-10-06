"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Row = { mean_val_loss: number; parameters: number; pooled_clean: Record<string, { log_loss: number; accuracy: number; man_recall: number; macro_f1: number }>; pooled_degraded: Record<string, { log_loss: number }>;
  vs_control_post_1_5s?: { mean_difference: number; interval_95: [number, number] }; passes_rule?: boolean };
type Cov = { split_role: string; models: Record<string, Row>; decision: { statement: string }; E3_ensembles_and_calibration?: Record<string, { log_loss?: number; brier?: number }>;
  E4_family?: Record<string, { mean: { accuracy: number; macro_f1: number; log_loss: number } }>; E4_note?: string };
type Fresh = { plays: number; label_counts: { man: number; zone: number }; limits: string[]; results: Record<string, Record<string, { n: number; accuracy: number; accuracy_interval_95_wilson: [number, number]; man_recall: number; man_precision: number; log_loss: number; always_zone_accuracy: number }>> };
type HlEval = { oracle: { three_minutes: number }; test: { v1_selector: Record<string, Record<string, { output_s: number; precision: number; recall_of_labelled_time: number; max_output_s: number }>>;
  strict_decoder: Record<string, Record<string, { output_s: number; precision: number; recall_of_labelled_time: number; max_output_s: number; recall_interval_95_over_games: [number, number] }>> }; split_roles: Record<string, string> };
type HlExp = { split_role: string; rankers: Record<string, { mean_average_precision: number } & Record<string, unknown>>; segment_decoding_at_strict_180s: Record<string, { recall_of_labelled_time: number; precision: number }>; decision: { statement: string } };
type HlCand = { results: Record<string, Record<"validation" | "test", { mean_average_precision: number; games_better?: number; vs_shipped?: { interval_95_over_games: [number, number] } }>>; decision: { statement: string } };
type Pg = { nested_development: { seasons: [number, number]; "v3 nested": { games: number; log_loss: number; vs_elo: { mean_log_loss_difference: number; interval_95: [number, number] }; vs_v2_elo_offset: { mean_log_loss_difference: number; interval_95: [number, number] } };
  controls: Record<string, { log_loss: number }> }; decision: { statement: string }; configurations_scored: number; frozen_v3_choice: { key: string };
  previously_examined_benchmark: { "v3 final choice": { log_loss: number; vs_elo: { mean_log_loss_difference: number; interval_95: [number, number] } } } };

const get = <T,>(u: string) => fetch(u).then((r) => (r.ok ? (r.json() as Promise<T>) : null)).catch(() => null);
const TH = "py-1.5 pr-3 text-left font-normal text-muted";
const iv = (x: [number, number], d = 4) => `${x[0].toFixed(d)} to ${x[1] >= 0 ? "+" : ""}${x[1].toFixed(d)}`;
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

// Third round of experiments. Every table is read from the saved report; unsuccessful results stay in the list.
export default function V3Results() {
  const [cov, setCov] = useState<Cov | null>(null);
  const [he, setHe] = useState<HlEval | null>(null);
  const [hx, setHx] = useState<HlExp | null>(null);
  const [pg, setPg] = useState<Pg | null>(null);
  const [hc, setHc] = useState<HlCand | null>(null);
  const [fr, setFr] = useState<Fresh | null>(null);
  useEffect(() => {
    get<Cov>("/demo/real/coverage/experiments_v3.json").then(setCov);
    get<HlEval>("/demo/highlights/eval_v3.json").then(setHe);
    get<HlExp>("/demo/highlights/experiments_v3.json").then(setHx);
    get<Pg>("/demo/pregame/experiments_v3.json").then(setPg);
    get<HlCand>("/demo/highlights/candidates_v3.json").then(setHc);
    get<Fresh>("/demo/real/coverage/fresh_2024.json").then(setFr);
  }, []);
  const H3 = "H3 temporal fusion";
  return (
    <section id="round-three" aria-labelledby="r-v3" className="scroll-mt-24 border-t border-line" data-testid="v3-results">
      <div className="mx-auto max-w-[1500px] px-4 py-12 lg:px-8">
        <p className="kicker">Round three · registered before running</p>
        <h2 id="r-v3" className="display mt-1 text-4xl sm:text-5xl">What the next experiments showed</h2>
        <p className="mt-3 max-w-[72ch] text-sm text-muted">Each list of experiments and its selection rule was written down before the first run. All of this is development data unless a table says otherwise; the previously examined benchmarks were not used to choose anything.</p>

        <div className="mt-8 grid gap-10 xl:grid-cols-3">
          <div className="min-w-0">
            <h3 className="narrow text-2xl font-semibold">Coverage</h3>
            {cov ? (<>
              <p className="mt-1 text-xs text-muted">{cov.split_role}</p>
              <div className="overflow-x-auto"><table className="num mt-2 w-full min-w-[420px] text-sm">
                <thead><tr><th className={TH}>Model (out-of-fold)</th><th className={TH}>Log loss +1.5 s</th><th className={TH}>Degraded input</th><th className={TH}>vs control</th></tr></thead>
                <tbody>{Object.entries(cov.models).map(([k, v]) => (
                  <tr key={k} className="border-t border-line align-top"><td className="py-1.5 pr-3">{k.replace("|seed", ", seed ")}</td><td>{v.pooled_clean.post_1_5s.log_loss.toFixed(4)}</td><td>{v.pooled_degraded.post_1_5s.log_loss.toFixed(4)}</td>
                    <td>{v.vs_control_post_1_5s ? iv(v.vs_control_post_1_5s.interval_95) : "control"}</td></tr>))}</tbody>
              </table></div>
              {cov.E3_ensembles_and_calibration && <p className="num mt-2 text-xs text-muted">Calibration and ensembles at +1.5 s (log loss): {Object.entries(cov.E3_ensembles_and_calibration).filter(([, v]) => typeof v.log_loss === "number").map(([k, v]) => `${k} ${v.log_loss!.toFixed(4)}`).join(" · ")}</p>}
              {cov.E4_family && <p className="num mt-2 text-xs text-muted">Coverage family, 7 classes: {Object.entries(cov.E4_family).map(([k, v]) => `${k.replace("|seed42", "")} accuracy ${pct(v.mean.accuracy)}, macro F1 ${v.mean.macro_f1.toFixed(3)}`).join(" · ")}. {cov.E4_note}</p>}
              <p className="mt-2 border-l-2 border-line pl-3 text-sm">{cov.decision.statement}</p>
              {fr && (() => { const b = fr.results["primary: seed 42"].post_1_5s; return (
                <div className="mt-4 border border-line p-3" data-testid="fresh-2024">
                  <p className="kicker">A small fresh sample: 2024 season, one look</p>
                  <p className="mt-1"><span className="display text-4xl num">{pct(b.accuracy)}</span> <span className="text-sm text-muted">agreement with the released label at +1.5 s on {b.n} plays from 3 games (95% range {pct(b.accuracy_interval_95_wilson[0])} to {pct(b.accuracy_interval_95_wilson[1])}). Man recall {b.man_recall.toFixed(2)}, log loss {b.log_loss.toFixed(3)}. Always answering zone would score {pct(b.always_zone_accuracy)}.</span></p>
                  <p className="mt-2 text-xs text-muted">Tracking for these plays came from a different release than their labels, and no model had seen any 2024 play. The protocol was written down before scoring. Three late-season games cannot stand for a season, and the tracked players were selected the same way as in 2023. The 95.2% figure on 2023 remains a previously examined benchmark; this sample sits beside it and is now examined too.</p>
                </div>); })()}
              <p className="mt-2 text-sm"><Link className="text-teal underline underline-offset-4" href="/coverage/review">Error slices and the review queue</Link></p>
            </>) : <p className="mt-2 text-sm text-muted">Coverage round-three report not exported yet.</p>}
          </div>

          <div className="min-w-0">
            <h3 className="narrow text-2xl font-semibold">Highlights</h3>
            {he && (<>
              <p className="mt-1 text-xs text-muted">Reel decoder on the 6 test games ({he.split_roles.reported}).</p>
              <div className="overflow-x-auto"><table className="num mt-2 w-full min-w-[400px] text-sm">
                <thead><tr><th className={TH}>3-minute reel</th><th className={TH}>Longest output</th><th className={TH}>Precision</th><th className={TH}>Recall of labelled time</th></tr></thead>
                <tbody>
                  <tr className="border-t border-line"><td className="py-1.5 pr-3">First selector (preserved)</td><td className="text-coral">{he.test.v1_selector[H3]["3 minutes"].max_output_s} s</td><td>{pct(he.test.v1_selector[H3]["3 minutes"].precision)}</td><td>{pct(he.test.v1_selector[H3]["3 minutes"].recall_of_labelled_time)}</td></tr>
                  <tr className="border-t border-line"><td className="py-1.5 pr-3">Strict-budget decoder</td><td>{he.test.strict_decoder[H3]["3 minutes"].max_output_s} s</td><td>{pct(he.test.strict_decoder[H3]["3 minutes"].precision)}</td><td>{pct(he.test.strict_decoder[H3]["3 minutes"].recall_of_labelled_time)} ({iv(he.test.strict_decoder[H3]["3 minutes"].recall_interval_95_over_games, 3)})</td></tr>
                  <tr className="border-t border-line text-muted"><td className="py-1.5 pr-3">Upper bound for any 180 s reel</td><td>180 s</td><td /><td>{pct(he.oracle.three_minutes)}</td></tr>
                </tbody></table></div>
              <p className="mt-2 text-xs text-muted">The upper bound is what a perfect 3-minute selection could cover, since the editors&apos; reels are much longer than 3 minutes. The strict decoder fixes the overshoot; it does not find more highlights.</p>
            </>)}
            {hx ? (<>
              <div className="overflow-x-auto"><table className="num mt-4 w-full min-w-[400px] text-sm">
                <thead><tr><th className={TH}>Ranker (development folds)</th><th className={TH}>Mean average precision</th></tr></thead>
                <tbody>{Object.entries(hx.rankers).map(([k, v]) => <tr key={k} className="border-t border-line"><td className="py-1.5 pr-3">{k}</td><td>{v.mean_average_precision.toFixed(4)}</td></tr>)}</tbody></table></div>
              <p className="num mt-2 text-xs text-muted">Segment-aware decoding, recall at a strict 180 s: {Object.entries(hx.segment_decoding_at_strict_180s).map(([k, v]) => `${k} ${pct(v.recall_of_labelled_time)}`).join(" · ")}</p>
              <p className="mt-2 border-l-2 border-line pl-3 text-sm">{hx.decision.statement}</p>
              {hc && (<>
                <div className="overflow-x-auto"><table className="num mt-4 w-full min-w-[400px] text-sm">
                  <thead><tr><th className={TH}>Rebuilt on all training games</th><th className={TH}>Validation games</th><th className={TH}>Examined test games</th></tr></thead>
                  <tbody>{Object.entries(hc.results).map(([k, v]) => <tr key={k} className="border-t border-line align-top"><td className="py-1.5 pr-3">{k}</td>
                    <td>{v.validation.mean_average_precision.toFixed(4)}{v.validation.vs_shipped ? ` (${iv(v.validation.vs_shipped.interval_95_over_games, 3)})` : ""}</td>
                    <td>{v.test.mean_average_precision.toFixed(4)}{v.test.vs_shipped ? ` (${iv(v.test.vs_shipped.interval_95_over_games, 3)})` : ""}</td></tr>)}</tbody></table></div>
                <p className="mt-1 text-xs text-muted">Mean average precision; brackets are the range of the difference from the shipped ranker over 6 games. Both sets of games were used before.</p>
                <p className="mt-2 border-l-2 border-line pl-3 text-sm">{hc.decision.statement}</p>
              </>)}
            </>) : <p className="mt-2 text-sm text-muted">Highlights round-three experiments not exported yet.</p>}
          </div>

          <div className="min-w-0">
            <h3 className="narrow text-2xl font-semibold">Game prediction</h3>
            {pg ? (<>
              <p className="mt-1 text-xs text-muted">{pg.configurations_scored} configurations, chosen season by season from earlier seasons only ({pg.nested_development.seasons[0]}–{pg.nested_development.seasons[1]}, {pg.nested_development["v3 nested"].games} games).</p>
              <div className="overflow-x-auto"><table className="num mt-2 w-full min-w-[380px] text-sm">
                <thead><tr><th className={TH}>Model</th><th className={TH}>Log loss</th></tr></thead>
                <tbody>
                  <tr className="border-t border-line"><td className="py-1.5 pr-3">Opponent-adjusted ratings (round three)</td><td>{pg.nested_development["v3 nested"].log_loss.toFixed(4)}</td></tr>
                  {Object.entries(pg.nested_development.controls).map(([k, v]) => <tr key={k} className="border-t border-line"><td className="py-1.5 pr-3">{k}</td><td>{v.log_loss.toFixed(4)}</td></tr>)}
                </tbody></table></div>
              <p className="num mt-2 text-xs text-muted">Difference from Elo {iv(pg.nested_development["v3 nested"].vs_elo.interval_95)}; from the frozen v2 model {iv(pg.nested_development["v3 nested"].vs_v2_elo_offset.interval_95)}. On the previously examined 2023–2025 seasons, against Elo: {iv(pg.previously_examined_benchmark["v3 final choice"].vs_elo.interval_95)}.</p>
              <p className="mt-2 border-l-2 border-line pl-3 text-sm">{pg.decision.statement}</p>
              <p className="mt-2 text-sm"><Link className="text-teal underline underline-offset-4" href="/predictions/forecasts">2026 forecast log</Link></p>
            </>) : <p className="mt-2 text-sm text-muted">Game prediction round-three report not exported yet.</p>}
          </div>
        </div>
      </div>
    </section>
  );
}
