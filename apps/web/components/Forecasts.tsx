"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Badge } from "@/components/ui";
import { pct } from "@/lib/demo";
import { type Forecasts as Data, loadPregame, signed } from "@/lib/pregame";

const TH = "py-1.5 pr-3 text-left font-normal text-muted";
const when = (iso: string) => iso.replace("T", " ").slice(0, 16);

export default function Forecasts() {
  const [d, setD] = useState<Data | null | undefined>(undefined);
  const [open, setOpen] = useState<string | null>(null);
  useEffect(() => { loadPregame<Data>("forecasts.json").then(setD); }, []);
  if (d === undefined) return <p className="p-6 text-muted">Loading forecast log…</p>;
  if (d === null || d.records.length === 0) return <p className="p-6 text-muted">No live forecasts recorded yet. Run bin/pregame-lens forecast before the next games kick off.</p>;
  const scored = d.records.filter((r) => r.outcome_attached && !r.outcome_attached.tie);
  const rec = d.records.find((r) => r.record_id === open) ?? d.records[0];

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-4 lg:p-6">
      <header>
        <h1 className="display text-4xl">Live forecast log</h1>
        <p className="mt-3 max-w-[72ch] text-muted">{d.note} These are separate from the reconstructed backtests: each record keeps the time it was actually created, the data it had and the quarterbacks it assumed.</p>
        <p className="num mt-2 text-sm text-muted">{d.records.length} records. {scored.length} have a result attached{scored.length ? `; the favourite won ${scored.filter((r) => (r.probabilities.elo_offset >= 0.5) === r.outcome_attached!.home_won).length} of them` : ""}. Too few to score yet.</p>
      </header>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="overflow-x-auto">
          <table className="num w-full min-w-[720px] text-sm">
            <thead><tr><th className={TH}>Game</th><th className={TH}>Kickoff (Eastern)</th><th className={TH}>Created (UTC)</th><th className={TH}>Timing</th><th className={TH}>Elo</th><th className={TH}>Elo-offset</th><th className={TH}>Result</th></tr></thead>
            <tbody>
              {d.records.map((r) => (
                <tr key={r.record_id} className={`border-t border-line ${r.record_id === rec.record_id ? "bg-surface" : ""}`}>
                  <td className="py-1.5 pr-3"><button className="underline decoration-line underline-offset-4 hover:decoration-teal" onClick={() => setOpen(r.record_id)}>{r.away} at {r.home}</button></td>
                  <td>{when(r.kickoff_eastern)}</td><td>{when(r.created_at_utc)}</td>
                  <td className={r.created_before_cutoff ? "text-teal" : "text-amber"}>{r.created_before_cutoff ? "Before the 24-hour cutoff" : "Inside 24 hours"}</td>
                  <td>{pct(r.probabilities.elo)}</td><td>{pct(r.probabilities.elo_offset)}</td>
                  <td>{r.outcome_attached ? `${r.outcome_attached.away_score}–${r.outcome_attached.home_score}` : "Not played"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <aside aria-label="Forecast record" className="text-sm">
          <h2 className="narrow text-lg font-semibold">{rec.away} at {rec.home}</h2>
          <p className="mt-1"><Badge kind="Predicted" /> <span className="display text-3xl num">{pct(rec.probabilities.elo_offset)}</span> <span className="text-muted">{rec.home} win</span></p>
          <dl className="num mt-3 space-y-1.5 text-xs text-muted">
            <div><dt className="inline text-ink">Cutoff: </dt><dd className="inline">{when(rec.cutoff_eastern)} Eastern (24 hours before kickoff)</dd></div>
            <div><dt className="inline text-ink">Created: </dt><dd className="inline">{when(rec.created_at_utc)} UTC. {rec.timing_note ?? "Created before the cutoff."}</dd></div>
            <div><dt className="inline text-ink">Data it had: </dt><dd className="inline">{rec.data_snapshot.completed_games.toLocaleString()} completed games, last on {rec.data_snapshot.last_completed_game}; schedule file {rec.data_snapshot.games_csv_sha256.slice(0, 10)}</dd></div>
            <div><dt className="inline text-ink">Quarterback scenario: </dt><dd className="inline">{rec.qb_assumption.home ?? "unknown"} for {rec.home}, {rec.qb_assumption.away ?? "unknown"} for {rec.away}. {rec.qb_assumption.basis}. The forecast is conditional on this.</dd></div>
            <div><dt className="inline text-ink">Model: </dt><dd className="inline">{rec.model_version}, fitted on {rec.trained_on_games.toLocaleString()} games; calibrator: {rec.calibrator}</dd></div>
            <div><dt className="inline text-ink">Other models: </dt><dd className="inline">Elo {pct(rec.probabilities.elo, 1)}, point margin {pct(rec.probabilities.margin, 1)} (expected margin {signed(rec.predicted_margin, 1)}), blend {pct(rec.probabilities.blend, 1)}</dd></div>
          </dl>
          <table className="num mt-3 w-full text-xs">
            <caption className="pb-1 text-left text-muted">From the Elo baseline ({signed(rec.explanation.elo_logit, 3)}) in {rec.explanation.unit}</caption>
            <tbody>
              <tr className="border-t border-line"><td className="py-1.5 pr-2">Model intercept</td><td className="text-right">{signed(rec.explanation.intercept, 3)}</td></tr>
              {rec.explanation.terms.filter((t) => Math.abs(t.log_odds) >= 0.005).slice(0, 8).map((t) => <tr key={t.feature} className="border-t border-line align-top"><td className="py-1.5 pr-2">{t.description}</td><td className={`text-right ${t.log_odds >= 0 ? "text-teal" : "text-defense"}`}>{signed(t.log_odds, 3)}</td></tr>)}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-muted">Contributions describe the model, not causes. <Link className="text-teal underline" href="/predictions/performance">How this model has scored</Link></p>
        </aside>
      </div>
    </div>
  );
}
