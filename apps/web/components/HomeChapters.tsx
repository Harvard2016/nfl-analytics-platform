"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import HomeField from "@/components/HomeField";
import Reveal from "@/components/Reveal";

type Bench = { models: Record<string, Record<string, { benchmark: { accuracy: number; man_recall: number; n: number } }>> };
type HL = { shipped: string; results: Record<string, { test: { mean_average_precision: number; chance_average_precision: number; "3 minutes": { precision: number; recall_of_highlight_time: number } } }>;
  games: { id: number; title: string }[] };
type HLGame = { clips: number; label: number[]; scores: Record<string, number[]>; shipped_model: string; selected: Record<string, number[]> };
type PG = { previously_examined_benchmark: { seasons: [number, number]; models: Record<string, { games: number; log_loss: number; vs_elo?: { mean_log_loss_difference: number; interval_95: [number, number] } }> }; labels: Record<string, string> };

const get = <T,>(url: string) => fetch(url).then((r) => (r.ok ? (r.json() as Promise<T>) : null)).catch(() => null);
const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
const runs = (a: number[]) => { const out: [number, number][] = []; let s = -1; a.forEach((v, i) => { if (v && s < 0) s = i; if (!v && s >= 0) { out.push([s, i]); s = -1; } }); if (s >= 0) out.push([s, a.length]); return out; };

function Chapter({ n, title, children, figure, flip = false }: { n: string; title: string; children: React.ReactNode; figure: React.ReactNode; flip?: boolean }) {
  return (
    <Reveal as="section" className="border-t border-line py-12 lg:py-20">
      <div className={`mx-auto grid max-w-[1500px] items-center gap-8 px-4 lg:grid-cols-2 lg:gap-14 lg:px-8 ${flip ? "lg:[&>*:first-child]:order-2" : ""}`}>
        <div className="flex gap-5">
          <span className="display text-[5.5rem] leading-[0.8] text-line sm:text-[8rem]" aria-hidden="true">{n}</span>
          <div className="min-w-0">
            <h2 className="display text-4xl sm:text-6xl">{title}</h2>
            {children}
          </div>
        </div>
        <div className="min-w-0">{figure}</div>
      </div>
    </Reveal>
  );
}

export default function HomeChapters() {
  const [bench, setBench] = useState<Bench | null>(null);
  const [hl, setHl] = useState<HL | null>(null);
  const [game, setGame] = useState<HLGame | null>(null);
  const [pg, setPg] = useState<PG | null>(null);
  useEffect(() => {
    get<Bench>("/demo/real/coverage/benchmark_v2.json").then(setBench);
    get<PG>("/demo/pregame/v2_performance.json").then(setPg);
    get<HL>("/demo/highlights/index.json").then((d) => { setHl(d); if (d?.games[0]) get<HLGame>(`/demo/highlights/games/${d.games[0].id}.json`).then(setGame); });
  }, []);
  const temporal = bench?.models.v2_temporal?.post_1_5s.benchmark, v1 = bench?.models.v1_gbm?.post_1_5s.benchmark;
  const ship = hl ? hl.results[hl.shipped].test : null;
  const B = pg?.previously_examined_benchmark;
  const max = B ? Math.max(...Object.values(B.models).map((m) => m.log_loss)) : 1;

  return (
    <div className="paper">
      <Chapter n="01" title="Coverage Intelligence" figure={
        <figure>
          <div className="bg-[#101713] p-2"><HomeField /></div>
          <figcaption className="kicker mt-2">Observed tracking from one held-out play. Passer, route runners and selected coverage defenders, snap to throw.</figcaption>
        </figure>}>
        <p className="mt-4 max-w-[52ch] text-lg text-muted">Watch a play from player tracking, compare the released coverage label with a model&apos;s call, and see which players and measurements moved it.</p>
        {temporal && v1 ? (
          <p className="mt-5 flex flex-wrap items-baseline gap-x-4 gap-y-1">
            <span className="display text-6xl num">{pct(temporal.accuracy)}</span>
            <span className="max-w-[34ch] text-sm text-muted">accuracy at +1.5 s for the temporal model on {temporal.n.toLocaleString()} plays, against {pct(v1.accuracy)} for the first benchmark model. <strong className="font-semibold text-ink">Previously examined benchmark</strong>, not fresh data.</span>
          </p>
        ) : <p className="mt-5 text-sm text-muted">Loading the saved benchmark record…</p>}
        <p className="mt-6 flex flex-wrap items-center gap-5"><Link href="/coverage?model=v2_temporal" className="btn-primary">Enter the film room <span aria-hidden="true">→</span></Link><Link href="/coverage/evaluation?model=v2_temporal" className="btn-quiet">How it was tested</Link></p>
      </Chapter>

      <Chapter n="02" title="Highlight Intelligence" flip figure={
        <figure>
          <div className="bg-[#101713] p-4">
            {game ? (
              <svg viewBox={`0 0 ${game.clips} 100`} preserveAspectRatio="none" className="block h-44 w-full" role="img" aria-label="One game's timeline: editorial highlight segments, the model's three-minute reel and the model's score over the broadcast.">
                {runs(game.label).map(([a, b]) => <rect key={`l${a}`} x={a} y={0} width={b - a} height={12} fill="#8ebaf1" />)}
                {runs(game.selected["3 minutes"]).map(([a, b]) => <rect key={`s${a}`} x={a} y={16} width={b - a} height={12} fill="#e8ad66" />)}
                <path d={game.scores[game.shipped_model].map((v, i) => `${i ? "L" : "M"}${i},${100 - v * 0.66}`).join("")} fill="none" stroke="#f2efe5" strokeWidth={1} vectorEffect="non-scaling-stroke" opacity={0.85} />
              </svg>
            ) : <div className="h-44" />}
          </div>
          <figcaption className="kicker mt-2">Blue: editorial highlight label. Amber: the model&apos;s 3-minute reel. Line: ranking score. {hl?.games[0]?.title.slice(0, 48)}</figcaption>
        </figure>}>
        <p className="mt-4 max-w-[52ch] text-lg text-muted">Rank every two seconds of a full broadcast by how likely the editors were to put it in the official highlight reel. A ranking of editorial agreement, not event detection.</p>
        {ship ? (
          <p className="mt-5 flex flex-wrap items-baseline gap-x-4 gap-y-1">
            <span className="display text-6xl num">{ship.mean_average_precision.toFixed(2)}</span>
            <span className="max-w-[36ch] text-sm text-muted">mean average precision on 6 held-out games (random ranking: {ship.chance_average_precision.toFixed(2)}). <strong className="font-semibold text-ink">Limit:</strong> a 3-minute reel is {pct(ship["3 minutes"].precision, 0)} highlights but covers only {pct(ship["3 minutes"].recall_of_highlight_time, 0)} of highlight time.</span>
          </p>
        ) : <p className="mt-5 text-sm text-muted">Loading the saved ranking record…</p>}
        <p className="mt-6 flex flex-wrap items-center gap-5"><Link href="/highlights" className="btn-primary">Open the timeline <span aria-hidden="true">→</span></Link><Link href="/research#highlights" className="btn-quiet">Method and limits</Link></p>
      </Chapter>

      <Chapter n="03" title="Game Prediction" figure={
        <figure>
          <div className="bg-[#101713] p-5">
            {B ? (
              <table className="num w-full text-sm text-[#f2efe5]">
                <caption className="sr-only">Log loss by model on the previously examined seasons; lower is better.</caption>
                <tbody>
                  {(["home_prior", "elo", "elo_offset", "margin"] as const).map((m) => (
                    <tr key={m}><th scope="row" className="narrow w-44 py-2 pr-3 text-left text-base font-semibold">{pg!.labels[m]}</th>
                      <td><span className="block h-3" style={{ width: `${((B.models[m].log_loss - 0.6) / (max - 0.6)) * 100}%`, background: m === "elo" ? "#f2efe5" : m === "home_prior" ? "#9ba89d" : "#8ebaf1" }} /></td>
                      <td className="w-16 pl-3 text-right">{B.models[m].log_loss.toFixed(4)}</td></tr>
                  ))}
                </tbody>
              </table>
            ) : <div className="h-40" />}
          </div>
          <figcaption className="kicker mt-2">Log loss, {B?.seasons[0]}–{B?.seasons[1]}, {B?.models.elo.games} games. Lower is better. Bars start at 0.60.</figcaption>
        </figure>}>
        <p className="mt-4 max-w-[52ch] text-lg text-muted">Pregame win probabilities built only from what was known 24 hours before kickoff, scored against a plain Elo rating season by season.</p>
        {B?.models.elo_offset.vs_elo ? (
          <p className="mt-5 flex flex-wrap items-baseline gap-x-4 gap-y-1">
            <span className="display text-4xl sm:text-5xl">No established gain over Elo</span>
            <span className="max-w-[40ch] text-sm text-muted">Best model minus Elo: {B.models.elo_offset.vs_elo.mean_log_loss_difference.toFixed(4)} log loss, 95% range {B.models.elo_offset.vs_elo.interval_95[0].toFixed(4)} to +{B.models.elo_offset.vs_elo.interval_95[1].toFixed(4)}. The range includes zero.</span>
          </p>
        ) : <p className="mt-5 text-sm text-muted">Loading the saved backtest record…</p>}
        <p className="mt-6 flex flex-wrap items-center gap-5"><Link href="/predictions" className="btn-primary">See a matchup <span aria-hidden="true">→</span></Link><Link href="/predictions/performance" className="btn-quiet">The evidence</Link></p>
      </Chapter>
    </div>
  );
}
