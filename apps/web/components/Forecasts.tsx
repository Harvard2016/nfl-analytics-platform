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
  const ROLE_ORDER = ["official", "early preview", "late", "legacy (not scored)"];
  const records = [...d.records].sort((a, b) => ROLE_ORDER.indexOf(a.role ?? "legacy (not scored)") - ROLE_ORDER.indexOf(b.role ?? "legacy (not scored)") || a.kickoff_eastern.localeCompare(b.kickoff_eastern));
  const rec = records.find((r) => r.record_id === open) ?? records[0];
  const board = d.leaderboard;
  const ROLE_CLASS: Record<string, string> = { official: "text-teal", "early preview": "text-ink", late: "text-amber", "legacy (not scored)": "text-muted" };

  return (
    <div className="mx-auto flex max-w-[1500px] flex-col gap-6 px-4 py-8 lg:px-8">
      <header>
        <p className="kicker">Game prediction · prospective record</p>
        <h1 className="display mt-1 text-5xl sm:text-6xl">2026 forecast log<span className="text-teal">.</span></h1>
        <p className="mt-3 max-w-[76ch] text-muted">{d.note} These are separate from the reconstructed backtests: each record keeps the time it was actually created, the preserved source files it was built from and the quarterbacks it assumed.</p>
        {d.counts && (
          <p className="num mt-2 text-sm text-muted" data-testid="forecast-counts">
            <strong className="font-semibold text-ink">{d.counts.official} official</strong> (created and sourced before the 24-hour cutoff), {d.counts.early_preview} superseded early previews, <span className="text-amber">{d.counts.late} late</span>,{" "}
            {d.counts.legacy_limited_provenance} legacy records with limited provenance that are never scored. {d.counts.official_with_outcome} official forecasts have a result.
          </p>
        )}
      </header>

      <section aria-labelledby="f-board" className="border border-line p-4 text-sm" data-testid="forecast-leaderboard">
        <h2 id="f-board" className="narrow text-2xl font-semibold">2026 prospective leaderboard</h2>
        {board?.models ? (
          <table className="num mt-2 w-full max-w-xl text-sm">
            <thead><tr><th className={TH}>Model</th><th className={TH}>Log loss</th><th className={TH}>Brier</th><th className={TH}>Accuracy</th></tr></thead>
            <tbody>{Object.entries(board.models).map(([m, v]) => <tr key={m} className="border-t border-line"><td className="py-1.5 pr-3">{m.replace("_", "-")}</td><td>{v.log_loss.toFixed(4)}</td><td>{v.brier.toFixed(4)}</td><td>{pct(v.accuracy, 1)}</td></tr>)}</tbody>
          </table>
        ) : <p className="mt-1 text-muted">No official forecast has a result yet, so there is nothing to score. Games played before these records were written are backtests and do not count here.</p>}
        <p className="mt-2 max-w-[76ch] text-xs text-muted">{board ? `${board.games_scored} games scored. ${board.caution}` : ""} All models are issued on the same games at the same cutoff. The forecasting model is the frozen v2 choice; Elo is the control.</p>
        {d.policy && <details className="mt-3 border-t border-line pt-2"><summary className="narrow font-semibold">Rules fixed before any result</summary>
          <dl className="mt-2 space-y-2 text-xs text-muted">{Object.entries(d.policy).filter(([, v]) => v.length > 20).map(([k, v]) => <div key={k}><dt className="kicker">{k.replace(/_/g, " ")}</dt><dd>{v}</dd></div>)}</dl></details>}
      </section>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        <div className="overflow-x-auto">
          <table className="num w-full min-w-[820px] text-sm">
            <thead><tr><th className={TH}>Game</th><th className={TH}>Kickoff (Eastern)</th><th className={TH}>Created (UTC)</th><th className={TH}>Role</th><th className={TH}>Elo</th><th className={TH}>Elo-offset</th><th className={TH}>Result</th></tr></thead>
            <tbody>
              {records.map((r) => (
                <tr key={r.record_id} className={`border-t border-line ${r.record_id === rec.record_id ? "bg-surface" : ""}`} data-role={r.role}>
                  <td className="py-1.5 pr-3"><button className="underline decoration-line underline-offset-4 hover:decoration-teal" onClick={() => setOpen(r.record_id)}>{r.away} at {r.home}</button></td>
                  <td>{when(r.kickoff_eastern)}</td><td>{when(r.created_at_utc)}</td>
                  <td className={ROLE_CLASS[r.role ?? "legacy (not scored)"]}>{r.role === "official" ? "Official" : r.role === "late" ? "Late: after the cutoff" : r.role === "early preview" ? "Early preview" : "Legacy, not scored"}</td>
                  <td>{pct(r.probabilities.elo)}</td><td>{pct(r.probabilities.elo_offset)}</td>
                  <td>{r.outcome_attached ? `${r.outcome_attached.away_score}–${r.outcome_attached.home_score}` : "Not played"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <aside aria-label="Forecast record" className="min-w-0 text-sm">
          <h2 className="narrow text-2xl font-semibold">{rec.away} at {rec.home}</h2>
          <p className="mt-1"><Badge kind="Predicted" /> <span className="display text-4xl num">{pct(rec.probabilities.elo_offset, 1)}</span> <span className="text-muted">{rec.home} win</span></p>
          <dl className="num mt-3 space-y-1.5 text-xs text-muted">
            <div><dt className="inline text-ink">Role: </dt><dd className="inline">{rec.role ?? "legacy"}. {rec.provenance}</dd></div>
            <div><dt className="inline text-ink">Cutoff: </dt><dd className="inline">{when(rec.cutoff_eastern)} Eastern (24 hours before kickoff){rec.cutoff_utc ? `; ${when(rec.cutoff_utc)} UTC` : ""}</dd></div>
            <div><dt className="inline text-ink">Created: </dt><dd className="inline">{when(rec.created_at_utc)} UTC. {rec.timing_note ?? "Created before the cutoff."}</dd></div>
            {rec.source_snapshot ? (
              <div><dt className="inline text-ink">Source snapshot: </dt><dd className="inline break-all">{rec.source_snapshot.snapshot_id}, fetched {when(rec.source_snapshot.created_at_utc)} UTC; schedule through {rec.source_snapshot.coverage.schedule_last_completed_game}, play-by-play through {rec.source_snapshot.coverage.pbp_latest_game_date} (week {rec.source_snapshot.coverage.pbp_latest_week}); content hash {rec.source_snapshot.content_sha256.slice(0, 12)}</dd></div>
            ) : <div><dt className="inline text-ink">Data it had: </dt><dd className="inline">{rec.data_snapshot.completed_games.toLocaleString()} completed games, last on {rec.data_snapshot.last_completed_game}. Only the schedule file was hashed.</dd></div>}
            <div><dt className="inline text-ink">Quarterback scenario: </dt><dd className="inline">{rec.qb_assumption.home ?? "unknown"} for {rec.home}, {rec.qb_assumption.away ?? "unknown"} for {rec.away}. {rec.qb_assumption.basis}. The forecast is conditional on this.</dd></div>
            <div><dt className="inline text-ink">Model: </dt><dd className="inline break-all">{rec.model_version}, fitted on {rec.trained_on_games.toLocaleString()} games{rec.training ? ` through ${rec.training.through}` : ""}; calibrator: {rec.calibrator}{rec.bundle_sha256 ? `; bundle ${rec.bundle_sha256.slice(0, 12)}` : ""}</dd></div>
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
