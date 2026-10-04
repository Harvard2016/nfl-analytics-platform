"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { pct } from "@/lib/demo";
import { type Block, type Performance, PMODELS, loadPregame, signed } from "@/lib/pregame";

const TH = "py-1.5 pr-3 text-left font-normal text-muted";

function Scores({ block, labels, caption }: { block: Block; labels: Performance["labels"]; caption: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="num w-full min-w-[760px] text-sm">
        <caption className="pb-1 text-left text-muted">{caption}</caption>
        <thead><tr><th className={TH}>Model</th><th className={TH}>Games</th><th className={TH}>Log loss</th><th className={TH}>Brier</th><th className={TH}>Accuracy</th><th className={TH}>Log loss vs Elo (95% range)</th><th className={TH}>Resamples better than Elo</th></tr></thead>
        <tbody>
          {PMODELS.map((m) => {
            const v = block[m];
            return (
              <tr key={m} className="border-t border-line">
                <td className="py-1.5 pr-3">{labels[m]}</td><td>{v.games}</td><td>{v.log_loss.toFixed(4)}</td><td>{v.brier.toFixed(4)}</td><td>{pct(v.accuracy, 1)}</td>
                <td>{v.vs_elo ? `${signed(v.vs_elo.mean_log_loss_difference)} (${signed(v.vs_elo.interval_95[0])} to ${signed(v.vs_elo.interval_95[1])})` : ""}</td>
                <td>{v.vs_elo ? pct(v.vs_elo.share_of_resamples_better) : ""}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Reliability({ rows }: { rows: Block["elo"]["calibration"] }) {
  // Reliability diagram: predicted probability (x) against how often the home team won (y). The diagonal is perfect calibration.
  const S = 220, P = 26;
  const xy = (p: number, o: number) => [P + p * (S - 2 * P), S - P - o * (S - 2 * P)];
  const max = Math.max(...rows.map((r) => r.games));
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="h-56 w-56" role="img" aria-label="Reliability diagram. The table beside it gives the same numbers.">
      <rect x={P} y={P} width={S - 2 * P} height={S - 2 * P} fill="none" stroke="var(--color-line)" />
      <line x1={P} y1={S - P} x2={S - P} y2={P} stroke="var(--color-muted)" strokeDasharray="3 3" />
      {rows.map((r) => { const [x, y] = xy(r.mean_predicted, r.home_won); return <circle key={r.lo} cx={x} cy={y} r={2.5 + 5 * Math.sqrt(r.games / max)} fill="var(--color-teal)" fillOpacity={0.75} />; })}
      <text x={S / 2} y={S - 6} textAnchor="middle" fontSize={10} fill="var(--color-muted)">Predicted home win</text>
      <text x={10} y={S / 2} textAnchor="middle" fontSize={10} fill="var(--color-muted)" transform={`rotate(-90 10 ${S / 2})`}>Home team won</text>
    </svg>
  );
}

export default function PredictionPerformance() {
  const [r, setR] = useState<Performance | null | undefined>(undefined);
  useEffect(() => { loadPregame<Performance>("v2_performance.json").then(setR); }, []);
  if (r === undefined) return <p className="p-6 text-muted">Loading scores…</p>;
  if (r === null) return <p className="p-6 text-muted">No backtest exported yet. Run bin/pregame-lens backtest, then reload.</p>;
  const B = r.previously_examined_benchmark, D = r.development, eo = B.models.elo_offset.vs_elo!, mg = B.models.margin.vs_elo!;
  const seasons = [...Object.entries(D.by_season), ...Object.entries(B.by_season)];
  const roll = r.rolling_100_games;
  const lo = Math.min(...roll.flatMap((x) => [x.elo_log_loss, x.elo_offset_log_loss])), hi = Math.max(...roll.flatMap((x) => [x.elo_log_loss, x.elo_offset_log_loss]));
  const line = (k: "elo_log_loss" | "elo_offset_log_loss") => roll.map((x, i) => `${i === 0 ? "M" : "L"}${(i / (roll.length - 1)) * 600},${120 - ((x[k] - lo) / (hi - lo)) * 110}`).join("");

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-8 p-4 lg:p-6">
      <header>
        <h1 className="display text-4xl">Game predictor: is anything better than Elo?</h1>
        <p className="mt-3 max-w-[72ch] text-muted">
          Not established. After trying {r.configurations_tried} configurations on {D.seasons[0]}–{D.seasons[1]} and freezing the best, the Elo-offset model scores
          {" "}{B.models.elo_offset.log_loss.toFixed(4)} log loss on {B.seasons[0]}–{B.seasons[1]} against {B.models.elo.log_loss.toFixed(4)} for plain Elo: a difference of
          {" "}{signed(eo.mean_log_loss_difference)} with a 95% range of {signed(eo.interval_95[0])} to {signed(eo.interval_95[1])}. The point-margin model is {signed(mg.mean_log_loss_difference)}
          {" "}({signed(mg.interval_95[0])} to {signed(mg.interval_95[1])}). Both ranges include zero.
        </p>
        <p className="mt-3 max-w-[72ch] text-sm text-muted">Target: {r.target}. Cutoff: {r.cutoff} Games breaking that rule: {r.cutoff_violations}. {r.reconstructed} {r.qb_assumption} Not used: {r.not_used.join("; ")}.</p>
      </header>

      <section aria-labelledby="pp-dev">
        <h2 id="pp-dev" className="narrow text-lg font-semibold">Development seasons {D.seasons[0]}–{D.seasons[1]}: where every choice was made</h2>
        <p className="mt-1 max-w-[72ch] text-sm text-muted">Each season is predicted by a model fitted on every earlier season since 2002. {r.selection.rule}. {r.selection.note}</p>
        <div className="mt-3"><Scores block={D.models} labels={r.labels} caption={`${D.models.elo.games} games. Negative differences favour the model; ranges resample whole season-weeks.`} /></div>
        <table className="num mt-5 w-full max-w-3xl text-sm">
          <caption className="pb-1 text-left text-muted">Which feature groups help the Elo-offset model (development seasons, same penalty and window as the chosen model)</caption>
          <thead><tr><th className={TH}>Feature groups added to Elo</th><th className={TH}>Log loss</th><th className={TH}>vs Elo</th><th className={TH}>Seasons better than Elo (of {D.seasons[1] - D.seasons[0] + 1})</th></tr></thead>
          <tbody>
            {r.ablations.map((a) => <tr key={a.groups} className={`border-t border-line ${a.groups === r.selection.chosen.elo_offset.groups ? "bg-surface" : ""}`}><td className="py-1.5 pr-3">{a.groups}</td><td>{a.log_loss.toFixed(4)}</td><td>{signed(a.vs_elo)}</td><td>{a.seasons_better_than_elo}</td></tr>)}
          </tbody>
        </table>
        <p className="mt-2 max-w-[72ch] text-xs text-muted">Rest and recent-form features add nothing measurable on their own. Efficiency, rate and quarterback features each help by about a thousandth of a log-loss point. Chosen: {r.selection.chosen.elo_offset.groups}, {r.selection.chosen.elo_offset.window} window, penalty {r.selection.chosen.elo_offset.penalty}; blend weight {r.selection.blend.weight_elo_offset} on the Elo-offset model.</p>
      </section>

      <section aria-labelledby="pp-bench">
        <h2 id="pp-bench" className="narrow text-lg font-semibold">Previously examined benchmark, {B.seasons[0]}–{B.seasons[1]}</h2>
        <p className="mt-1 max-w-[72ch] text-sm text-muted">{B.note}</p>
        <div className="mt-3"><Scores block={B.models} labels={r.labels} caption={`${B.models.elo.games} games, regular season and playoffs`} /></div>
        <div className="overflow-x-auto">
          <table className="num mt-5 w-full min-w-[560px] max-w-3xl text-sm">
            <caption className="pb-1 text-left text-muted">Log loss by season. Development seasons first, then the previously examined ones.</caption>
            <thead><tr><th className={TH}>Season</th><th className={TH}>Games</th><th className={TH}>Elo</th><th className={TH}>Elo-offset</th><th className={TH}>Point margin</th><th className={TH}>Blend</th></tr></thead>
            <tbody>
              {seasons.map(([s, b]) => (
                <tr key={s} className={`border-t border-line ${Number(s) >= B.seasons[0] ? "bg-surface" : ""}`}><td className="py-1.5 pr-3">{s}</td><td>{b.elo.games}</td>
                  {(["elo", "elo_offset", "margin", "blend"] as const).map((m) => <td key={m} className={m !== "elo" && b[m].log_loss < b.elo.log_loss ? "text-teal" : ""}>{b[m].log_loss.toFixed(4)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-muted">Green marks seasons where a model beat Elo. The winner changes from season to season.</p>
      </section>

      <section aria-labelledby="pp-cal" className="grid gap-6 md:grid-cols-[auto_1fr]">
        <div>
          <h2 id="pp-cal" className="narrow text-lg font-semibold">Is the probability honest?</h2>
          <Reliability rows={B.models.elo_offset.calibration} />
        </div>
        <div>
          <table className="num mt-9 w-full max-w-md text-sm">
            <caption className="pb-1 text-left text-muted">Elo-offset model, {B.seasons[0]}–{B.seasons[1]}, ten equal-width bins. Circle size follows the number of games.</caption>
            <thead><tr><th className={TH}>Predicted home win</th><th className={TH}>Home team won</th><th className={TH}>Games</th></tr></thead>
            <tbody>{B.models.elo_offset.calibration.map((c) => <tr key={c.lo} className="border-t border-line"><td className="py-1 pr-3">{pct(c.mean_predicted)}</td><td>{pct(c.home_won)}</td><td>{c.games}</td></tr>)}</tbody>
          </table>
          <p className="mt-2 text-xs text-muted">Calibration slope {B.models.elo_offset.calibration_slope.toFixed(2)} (1 is ideal), intercept {signed(B.models.elo_offset.calibration_intercept, 2)}. No calibrator is applied.</p>
        </div>
      </section>

      <section aria-labelledby="pp-roll">
        <h2 id="pp-roll" className="narrow text-lg font-semibold">Rolling log loss, last 100 games</h2>
        <svg viewBox="0 0 600 130" className="mt-2 w-full max-w-3xl" role="img" aria-label={`Rolling 100-game log loss from ${roll[0].through} to ${roll.at(-1)!.through}; Elo and the Elo-offset model track each other closely, between ${lo.toFixed(2)} and ${hi.toFixed(2)}.`}>
          <path d={line("elo_log_loss")} fill="none" stroke="var(--color-muted)" strokeWidth={1.5} />
          <path d={line("elo_offset_log_loss")} fill="none" stroke="var(--color-teal)" strokeWidth={1.5} />
        </svg>
        <p className="num text-xs text-muted"><span className="text-muted">Grey: Elo.</span> <span className="text-teal">Teal: Elo-offset.</span> From {roll[0].through} to {roll.at(-1)!.through}; vertical range {lo.toFixed(3)} to {hi.toFixed(3)}. The two lines move together: hard stretches are hard for both.</p>
      </section>

      {r.closing_market_reference && (
        <section aria-labelledby="pp-mkt">
          <h2 id="pp-mkt" className="narrow text-lg font-semibold">Later-information reference: closing betting prices</h2>
          <p className="mt-1 max-w-[72ch] text-sm text-muted">{r.closing_market_reference.note} On {r.closing_market_reference.games} games: market {r.closing_market_reference.market_log_loss.toFixed(4)},
            Elo {r.closing_market_reference.elo_log_loss_same_games.toFixed(4)}, Elo-offset {r.closing_market_reference.elo_offset_log_loss_same_games.toFixed(4)}. The market is far ahead of both, as expected.</p>
        </section>
      )}
      {r.in_progress && (
        <section aria-labelledby="pp-now">
          <h2 id="pp-now" className="narrow text-lg font-semibold">{r.in_progress.seasons.join(", ")} so far, through {r.in_progress.through}</h2>
          <div className="mt-3"><Scores block={r.in_progress.models} labels={r.labels} caption="Reconstructed like the rest. Too few games for a firm reading." /></div>
        </section>
      )}
      <p className="text-sm"><Link className="text-teal underline" href="/predictions">Browse individual backtests</Link>{" · "}<Link className="text-teal underline" href="/predictions/forecasts">Live forecast log</Link></p>
      <p className="break-all text-xs text-muted">Model {r.version}. Run {r.run_at}.</p>
    </div>
  );
}
