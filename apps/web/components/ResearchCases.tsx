"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import Reveal from "@/components/Reveal";

type Cell = { benchmark: { accuracy: number; man_recall: number; n: number } };
type Bench = { labels: Record<string, string>; models: Record<string, Record<string, Cell>> };
type HL = { shipped: string; clips: number; positive_share: number; results: Record<string, { test: { games: number; mean_average_precision: number; chance_average_precision: number; "3 minutes": { precision: number; recall_of_highlight_time: number } } }> };
type PG = { configurations_tried: number; labels: Record<string, string>;
  previously_examined_benchmark: { seasons: [number, number]; models: Record<string, { games: number; log_loss: number; vs_elo?: { mean_log_loss_difference: number; interval_95: [number, number] } }> } };
type Run = { module: string; run_id: string; name: string; started_at: string; train_seconds: number; peak_memory_mb: number; latest: boolean };

const get = <T,>(url: string) => fetch(url).then((r) => (r.ok ? (r.json() as Promise<T>) : null)).catch(() => null);
const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
const HORIZONS = [["at_snap", "Snap"], ["post_0_5s", "+0.5 s"], ["post_1s", "+1.0 s"], ["post_1_5s", "+1.5 s"]] as const;
const MODELS = [["v1_gbm", "var(--color-muted)"], ["v2_gbm_rel", "var(--color-defense)"], ["v2_temporal", "var(--color-ink)"]] as const;
const A = "text-teal underline underline-offset-4";

function Case({ id, n, title, question, figure, children }: { id: string; n: string; title: string; question: string; figure: React.ReactNode; children: React.ReactNode }) {
  return (
    <Reveal as="section" className="border-t border-line py-12 lg:py-16">
      <div id={id} className="mx-auto grid max-w-[1500px] scroll-mt-24 gap-8 px-4 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] lg:gap-14 lg:px-8">
        <div className="min-w-0">
          <p className="display text-7xl leading-none text-line" aria-hidden="true">{n}</p>
          <h2 className="display mt-2 text-4xl sm:text-5xl">{title}</h2>
          <p className="narrow mt-3 max-w-[40ch] text-2xl">{question}</p>
          <div className="mt-4 max-w-[62ch] space-y-3 text-sm text-muted">{children}</div>
        </div>
        <div className="min-w-0">{figure}</div>
      </div>
    </Reveal>
  );
}

export default function ResearchCases() {
  const [bench, setBench] = useState<Bench | null>(null);
  const [hl, setHl] = useState<HL | null>(null);
  const [pg, setPg] = useState<PG | null>(null);
  useEffect(() => {
    get<Bench>("/demo/real/coverage/benchmark_v2.json").then(setBench);
    get<HL>("/demo/highlights/index.json").then(setHl);
    get<PG>("/demo/pregame/v2_performance.json").then(setPg);
  }, []);
  const acc = (m: string, h: string) => bench?.models[m]?.[h]?.benchmark.accuracy;
  const y = (v: number) => 150 - ((v - 0.8) / 0.2) * 140;
  const x = (i: number) => 40 + i * 110;
  const t = bench?.models.v2_temporal?.post_1_5s.benchmark;
  const B = pg?.previously_examined_benchmark;
  const hres = hl ? Object.entries(hl.results).filter(([k]) => !k.startsWith("H3 without")) : [];
  const ship = hl ? hl.results[hl.shipped].test : null;
  const pm = ["home_prior", "elo", "v1_logistic", "elo_offset", "margin", "blend"].filter((m) => B?.models[m]);
  const lo = -0.014, hi = 0.008, px = (v: number) => ((v - lo) / (hi - lo)) * 100;

  return (
    <>
      <Case id="coverage" n="01" title="Coverage" question="Does reading every defender and receiver over time beat summary geometry?" figure={
        <figure>
          {bench ? (
            <svg viewBox="0 0 400 180" className="block w-full" role="img" aria-label={`Accuracy against the released coverage label by input cutoff. At 1.5 seconds after the snap: ${MODELS.map(([m]) => `${bench.labels?.[m] ?? m} ${pct(acc(m, "post_1_5s") ?? 0)}`).join(", ")}.`}>
              {[0.8, 0.85, 0.9, 0.95, 1].map((v) => <g key={v}><line x1={40} x2={390} y1={y(v)} y2={y(v)} stroke="var(--color-line)" /><text x={34} y={y(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-muted)">{Math.round(v * 100)}%</text></g>)}
              {HORIZONS.map(([, l], i) => <text key={l} x={x(i)} y={172} textAnchor="middle" fontSize={9} fill="var(--color-muted)">{l}</text>)}
              {MODELS.map(([m, c]) => {
                const pts = HORIZONS.map(([h], i) => [x(i), acc(m, h)] as const).filter((p): p is readonly [number, number] => p[1] != null);
                return <g key={m}><path d={pts.map(([px0, v], i) => `${i ? "L" : "M"}${px0},${y(v)}`).join("")} fill="none" stroke={c} strokeWidth={m === "v2_temporal" ? 2.5 : 1.5} />
                  {pts.map(([px0, v]) => <circle key={px0} cx={px0} cy={y(v)} r={m === "v2_temporal" ? 3.5 : 2.5} fill={c} />)}</g>;
              })}
            </svg>
          ) : <div className="h-56" />}
          <figcaption className="mt-2 text-xs text-muted">
            <span className="kicker block">Accuracy by how much of the play the model sees · {t?.n.toLocaleString()} plays, weeks 15–18</span>
            {bench && MODELS.map(([m, c]) => <span key={m} className="mr-4 inline-flex items-center gap-1.5"><span className="inline-block h-0.5 w-5" style={{ background: c }} />{bench.labels?.[m] ?? m}</span>)}
            <span className="block">The first benchmark model has no +0.5 s point. Axis starts at 80%.</span>
          </figcaption>
        </figure>}>
        <p><strong className="font-semibold text-ink">Result.</strong> {t ? <>The temporal model agrees with the released coverage label on {pct(t.accuracy)} of plays at +1.5 s, with man recall {t.man_recall.toFixed(2)}.</> : "Loading the saved benchmark…"} Weeks 15–18 had already been examined, so this is a comparison on a previously examined benchmark, not fresh data.</p>
        <p><strong className="font-semibold text-ink">Method.</strong> Three model families on the same plays: boosted trees on shell geometry, boosted trees with defender-to-receiver measurements, and a small temporal model reading every defender–receiver pair frame by frame. All choices were made on chronological folds inside weeks 1–12.</p>
        <p><strong className="font-semibold text-ink">Limitation.</strong> The release tracks only the passer, route runners and the defenders it marks as coverage players, chosen with knowledge of the play. These are not snap-time-only or full-field predictions.</p>
        <p className="flex flex-wrap gap-x-5 gap-y-1"><Link className={A} href="/coverage/evaluation?model=v2_temporal">Results, paired differences, calibration</Link><Link className={A} href="/coverage/errors">Error gallery</Link><Link className={A} href="/coverage?model=v2_temporal">Film room</Link></p>
      </Case>

      <Case id="highlights" n="02" title="Highlights" question="Which signals recover an editor's highlight reel from a full broadcast?" figure={
        <figure>
          {hl ? (
            <table className="num w-full text-sm">
              <caption className="kicker pb-2 text-left">Test mean average precision · {ship?.games} held-out games</caption>
              <tbody>
                {hres.map(([n, r]) => (
                  <tr key={n} className="border-t border-line"><th scope="row" className={`w-[46%] py-1.5 pr-3 text-left font-normal ${n === hl.shipped ? "font-semibold text-ink" : "text-muted"}`}>{n}</th>
                    <td><span className="block h-2.5" style={{ width: `${(r.test.mean_average_precision / 0.7) * 100}%`, background: n === hl.shipped ? "var(--color-amber)" : "var(--color-muted)" }} /></td>
                    <td className="w-14 pl-3 text-right">{r.test.mean_average_precision.toFixed(3)}</td></tr>
                ))}
                <tr className="border-t border-line text-muted"><th scope="row" className="py-1.5 pr-3 text-left font-normal">Random ranking (expected)</th><td><span className="block h-2.5 bg-line" style={{ width: `${((ship?.chance_average_precision ?? 0) / 0.7) * 100}%` }} /></td><td className="pl-3 text-right">{ship?.chance_average_precision.toFixed(3)}</td></tr>
              </tbody>
            </table>
          ) : <div className="h-56" />}
        </figure>}>
        <p><strong className="font-semibold text-ink">Result.</strong> {ship ? <>Temporal fusion of audio and visual embeddings reaches {ship.mean_average_precision.toFixed(3)} mean average precision against {ship.chance_average_precision.toFixed(3)} for a random ranking.</> : "Loading the saved ranking record…"}</p>
        <p><strong className="font-semibold text-ink">Method.</strong> Target: whether each 2-second clip of a full broadcast appears in the official highlight video (40 NFL games, SVHighlights). Games are split before any scoring; the test games were scored once.</p>
        <p><strong className="font-semibold text-ink">Limitation.</strong> {ship ? <>A 3-minute reel is {pct(ship["3 minutes"].precision, 0)} editorial highlights but covers only {pct(ship["3 minutes"].recall_of_highlight_time, 0)} of highlight time.</> : null} The label is an editor&apos;s selection, not detection of big plays. Touchdown, interception, sack and fumble are defined but not measured: that needs a verified mapping from video time to plays and footage cleared for review.</p>
        <p className="flex flex-wrap gap-x-5 gap-y-1"><Link className={A} href="/highlights">Timeline and full benchmark table</Link><Link className={A} href="/highlights/review">Review tool</Link></p>
      </Case>

      <Case id="predictions" n="03" title="Game prediction" question="Does anything beat plain Elo once selection is done honestly?" figure={
        <figure>
          {B ? (
            <table className="num w-full text-sm">
              <caption className="kicker pb-2 text-left">Log loss minus Elo, with 95% range · {B.seasons[0]}–{B.seasons[1]}, {B.models.elo.games} games</caption>
              <tbody>
                {pm.map((m) => { const d = B.models[m].vs_elo; return (
                  <tr key={m} className="border-t border-line"><th scope="row" className="w-[40%] py-2 pr-3 text-left font-normal text-muted">{pg!.labels[m]}</th>
                    <td className="relative">
                      <span className="absolute inset-y-0 w-px bg-ink" style={{ left: `${px(0)}%` }} />
                      {d && <><span className="absolute top-1/2 h-px bg-ink" style={{ left: `${px(d.interval_95[0])}%`, width: `${px(d.interval_95[1]) - px(d.interval_95[0])}%` }} /><span className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 bg-ink" style={{ left: `${px(d.mean_log_loss_difference)}%` }} /></>}
                    </td>
                    <td className="w-16 pl-3 text-right">{B.models[m].log_loss.toFixed(4)}</td></tr>
                ); })}
              </tbody>
            </table>
          ) : <div className="h-56" />}
          <figcaption className="mt-2 text-xs text-muted">Vertical line: Elo. Left of it is better. Every range crosses the line. Right column: log loss.</figcaption>
        </figure>}>
        <p><strong className="font-semibold text-ink">Result.</strong> No established gain over Elo. {pg ? <>{pg.configurations_tried} configurations were tried on 2012–2022;</> : null} the gains on the previously examined {B?.seasons[0]}–{B?.seasons[1]} seasons are a few thousandths of log loss with ranges that include zero.</p>
        <p><strong className="font-semibold text-ink">Method.</strong> Target: home win, from information available 24 hours before kickoff. Features use only games that kicked off at least 28 hours earlier; tests rewrite a target game&apos;s score, box score and starter and confirm its features do not move.</p>
        <p><strong className="font-semibold text-ink">Limitation.</strong> Backtests are not live forecasts. The append-only forecast log is the only place a prediction is recorded before its game. Nothing here is betting advice.</p>
        <p className="flex flex-wrap gap-x-5 gap-y-1"><Link className={A} href="/predictions/performance">The evidence</Link><Link className={A} href="/predictions/forecasts">Live forecast log</Link></p>
      </Case>
    </>
  );
}

// Latest run records for the research page. Reads the same frozen export as the full notebook.
export function NotebookPreview() {
  const [runs, setRuns] = useState<Run[] | null | undefined>(undefined);
  useEffect(() => { get<{ runs: Run[] }>("/demo/research/runs.json").then((d) => setRuns(d?.runs ?? null)); }, []);
  if (runs === undefined) return <p className="mt-6 text-muted">Loading run records…</p>;
  if (runs === null) return <p className="mt-6 text-muted">No run records exported yet.</p>;
  return (
    <div className="mt-6 overflow-x-auto">
      <table className="num w-full min-w-[640px] text-sm">
        <thead><tr className="text-left text-muted">{["Run", "Module", "Started (UTC)", "Seconds", "Peak memory", "Status"].map((h) => <th key={h} className="kicker py-2 pr-3 font-normal">{h}</th>)}</tr></thead>
        <tbody>
          {[...runs].sort((a, b) => b.started_at.localeCompare(a.started_at)).map((r) => (
            <tr key={r.run_id} className="border-t border-line">
              <td className="py-2 pr-3"><Link className="underline decoration-line underline-offset-4 hover:decoration-teal" href={`/research/experiments?run=${r.run_id}`}>{r.name}</Link></td>
              <td className="pr-3">{r.module}</td><td className="pr-3">{r.started_at.slice(0, 16).replace("T", " ")}</td><td className="pr-3">{r.train_seconds}</td><td className="pr-3">{r.peak_memory_mb} MB</td>
              <td className={r.latest ? "text-teal" : "text-muted"}>{r.latest ? "Current" : "Superseded"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
