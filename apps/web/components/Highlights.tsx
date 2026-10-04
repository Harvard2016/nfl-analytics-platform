"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui";
import { pct } from "@/lib/demo";
import StadiumBackdrop from "@/components/StadiumBackdrop";
import HighlightPlayer, { type PlaybackRequest } from "@/components/HighlightPlayer";
import ClipEvidence from "@/components/ClipEvidence";
import { sourceVideoLink, formatClock as clock } from "@/lib/highlightTime";

type Budget = "1 minute" | "3 minutes" | "5 minutes";
type Summary = { games: number; mean_average_precision: number; average_precision_range: [number, number]; hit_at_1: number; chance_average_precision: number }
  & Record<Budget, { precision: number; recall_of_highlight_time: number; temporal_iou: number; segments_missed_share: number }>;
type Index = {
  task: string; scope: string; label: string; score_meaning: string; calibration: string; captions: string; budget_note: string; selection_rule: string;
  shipped: string; clips: number; positive_share: number; attribution: string; source: string; revision: string; run_id: string;
  split: { train: number[]; validation: number[]; test: number[]; rule: string };
  results: Record<string, { info: { inputs: string | string[]; method?: string; probes?: { case: string; text: string; score: number }[]; top_positive_terms?: string[]; probe_note?: string }; validation: Summary; test: Summary }>;
  games: { id: number; title: string; split: string; clips: number; average_precision: number; chance: number; three_minute_precision: number; file: string }[];
  event_extension: { status: string; needs: string; what_exists: string };
};
type Candidate = { start_clip: number; end_clip: number; start_s: number; end_s: number; source_time_s: number; score_rank: number; loudness_above_background: number;
  loudness_rank: number; commentary_terms: string[]; excerpt: string; editorial_overlap: number; in_editorial_highlights: boolean };
type Game = {
  id: number; title: string; split: string; clips: number; label: number[]; loudness_above_background: number[]; scores: Record<string, number[]>; shipped_model: string;
  selected: Record<Budget, number[]>; candidates: Candidate[];
  missed_highlights: { start_s: number; end_s: number; source_time_s: number; seconds: number; best_score_rank: number; excerpt: string }[]; missed_highlight_count: number;
  commentary: { start_s: number; end_s: number; excerpt: string }[];
  source: { full_video: string; official_highlights: string; trim_start_s: number; time_note: string };
  metrics: { average_precision: number; chance_average_precision: number } & Record<Budget, { precision: number; recall_of_highlight_time: number; highlight_segments: number; segments_missed_entirely: number }>;
};

type Recap = { mode: string; source: string; game_id: string; final: string; alignment: string; definitions: Record<string, string>; counts: Record<string, number>;
  events: { play_id: number; quarter: number; game_clock: string; offense: string; events: string[]; yards: number | null; epa: number | null; description: string }[] };

const BUDGETS: Budget[] = ["1 minute", "3 minutes", "5 minutes"];
const TH = "py-1.5 pr-3 text-left font-normal text-muted";
const runs = (a: number[]) => { const out: [number, number][] = []; let s = -1; a.forEach((v, i) => { if (v && s < 0) s = i; if (!v && s >= 0) { out.push([s, i]); s = -1; } }); if (s >= 0) out.push([s, a.length]); return out; };
const sourceLink = (g: Game, t: number) => sourceVideoLink(g.source, t);

function path(values: number[], h: number, max: number, min = 0) {
  return values.map((v, i) => `${i === 0 ? "M" : "L"}${i},${(h - ((Math.max(min, Math.min(max, v)) - min) / (max - min)) * h).toFixed(1)}`).join("");
}

export default function Highlights() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const [index, setIndex] = useState<Index | null | undefined>(undefined);
  const [game, setGame] = useState<Game | null>(null);
  const [cursor, setCursor] = useState<number | null>(null);
  const [recap, setRecap] = useState<Recap | null>(null);
  const [tab, setTab] = useState<"evidence" | "missed" | "review">("evidence");
  const [playback, setPlayback] = useState<PlaybackRequest | null>(null);
  const selectMoment = (start: number, end: number | null, label = "ranked interval") => {
    setCursor(Math.floor(start / 2));
    setPlayback((prev) => ({ start, end, label, sequence: (prev?.sequence ?? 0) + 1 }));
  };
  const followVideo = useCallback((time: number) => setCursor((prev) => {
    const clip = Math.max(0, Math.floor(time / 2));
    return clip === prev ? prev : clip;
  }), []);

  useEffect(() => { fetch("/demo/highlights/index.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then(setIndex).catch(() => setIndex(null)); }, []);
  const gid = index ? (index.games.find((g) => String(g.id) === params.get("game")) ?? index.games[0])?.id : null;
  const budget = (BUDGETS.find((b) => b === params.get("budget")) ?? "3 minutes") as Budget;
  const model = params.get("model") && game?.scores[params.get("model")!] ? params.get("model")! : game?.shipped_model ?? "";
  useEffect(() => {
    if (gid == null) return;
    let live = true;
    fetch(`/demo/highlights/games/${gid}.json`).then((r) => r.json()).then((g) => { if (live) { setGame(g); setCursor(null); setPlayback(null); } });
    fetch(`/demo/highlights/recaps/${gid}.json`).then((r) => (r.ok ? r.json() : null)).then((x) => { if (live) setRecap(x); }).catch(() => { if (live) setRecap(null); });
    return () => { live = false; };
  }, [gid]);
  const set = (k: string, v: string | null) => { const n = new URLSearchParams(params.toString()); if (v == null) n.delete(k); else n.set(k, v); router.replace(`${pathname}?${n.toString()}`, { scroll: false }); };

  const bands = useMemo(() => (game && game.id === gid ? { label: runs(game.label), selected: runs(game.selected[budget]) } : null), [game, gid, budget]);
  if (index === undefined) return <p className="p-6 text-muted">Loading highlight benchmark…</p>;
  if (index === null) return <p className="p-6 text-muted">No highlight export found. Run bin/highlights-lens run and bin/highlights-lens export, then reload.</p>;

  const g = game && game.id === gid ? game : null;
  const N = g?.clips ?? 1;
  const cur = g && cursor != null ? Math.max(0,Math.min(g.clips-1,cursor)) : g?.candidates[0]?.start_clip ?? 0;
  const cand = g?.candidates.find((c) => cur >= c.start_clip && cur < c.end_clip);
  const talk = g?.commentary.find((c) => cur * 2 >= c.start_s && cur * 2 < c.end_s);
  const shipped = index.results[index.shipped];
  const main = Object.entries(index.results).filter(([n]) => !n.startsWith("H3 without"));
  const ablation = Object.entries(index.results).filter(([n]) => n.startsWith("H3 without") || n === "H3 temporal fusion");
  const lo = Math.max(0, cur - 45), hi = Math.min(N, cur + 46);
  const nearby = g ? g.commentary.filter((c) => c.end_s >= cur * 2 - 40 && c.start_s <= cur * 2 + 40).slice(0, 4) : [];
  const selectedCandidate = g?.candidates.find(c => c.start_s === playback?.start) ?? cand ?? g?.candidates[0];
  const selectedStart = playback?.start ?? selectedCandidate?.start_s ?? cur*2;
  const selectedEnd = playback ? playback.end ?? Math.min(N*2,selectedStart+20) : selectedCandidate?.end_s ?? Math.min(N*2,selectedStart+2);
  const nextCandidateIndex = g && selectedCandidate ? g.candidates.indexOf(selectedCandidate)+1 : 0;

  return (
    <div>
      <header className="highlight-header cinematic-band border-b border-line">
        <StadiumBackdrop />
        <div className="mx-auto grid max-w-[1500px] gap-6 px-4 py-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:items-end lg:px-8">
          <div>
            <p className="kicker">Highlight intelligence</p>
            <h1 className="display mt-1 text-6xl sm:text-8xl">Find the <span className="text-amber">moment.</span></h1>
          </div>
          <div className="max-w-[58ch] text-sm text-muted">
            <p className="text-lg text-ink">Select a moment. Watch the source. Inspect the signals.</p>
            <p className="mt-2">Ranks editorial highlight selection. Scores are ranks, not probabilities.</p>
            <details className="mt-4 border-t border-line pt-2"><summary>Dataset and evaluation</summary><p className="mt-2">{index.scope}. Editorial label: {index.label} On {shipped.test.games} held-out games: {shipped.test.mean_average_precision.toFixed(3)} mAP (random: {shipped.test.chance_average_precision.toFixed(3)}). A 3-minute reel is {pct(shipped.test["3 minutes"].precision)} editorial highlights but covers only {pct(shipped.test["3 minutes"].recall_of_highlight_time)} of highlight time. Source playback uses YouTube; no footage is hosted.</p></details>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-[1500px] px-4 py-4 lg:px-8">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border border-line px-4 py-2 text-sm">
          <label className="flex min-w-0 items-center gap-2 text-muted">Game
            <select value={gid ?? ""} onChange={(e) => set("game", e.target.value)} className="max-w-[62vw] border border-line bg-surface px-2 text-ink sm:max-w-md">
              {index.games.map((x) => <option key={x.id} value={x.id}>{x.title.slice(0, 64)} ({x.split})</option>)}
            </select>
          </label>
          {g && (
            <label className="flex items-center gap-2 text-muted">Score from
              <select value={model} onChange={(e) => set("model", e.target.value)} className="border border-line bg-surface px-2 text-ink">{Object.keys(g.scores).map((n) => <option key={n}>{n}</option>)}</select>
            </label>
          )}
          <div role="group" aria-label="Reel length" className="flex border border-line">
            {BUDGETS.map((b) => <button key={b} aria-pressed={b === budget} onClick={() => set("budget", b === "3 minutes" ? null : b)} className={`narrow px-3 text-base transition-colors duration-150 ${b === budget ? "bg-amber font-bold text-bg" : "text-muted hover:text-ink"}`}>{b}</button>)}
          </div>
          {g && <span className="mono num text-xs text-muted">{g.split.toUpperCase()} GAME · AP {g.metrics.average_precision.toFixed(3)} · CHANCE {g.metrics.chance_average_precision.toFixed(3)}</span>}
        </div>

        {!g ? <p className="py-6 text-muted">Loading game…</p> : (
          <div key={g.id} className="fade-swap mt-4 grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
            <section aria-label="Timeline" className="min-w-0">
              <HighlightPlayer key={g.id} source={g.source} title={g.title} duration={g.clips*2} request={playback} onTime={followVideo}
                onReplay={() => selectMoment(selectedStart,selectedEnd,playback?.label ?? "ranked interval")}
                onNext={() => { const next = g.candidates[nextCandidateIndex]; if (next) selectMoment(next.start_s,next.end_s); }} hasNext={nextCandidateIndex<g.candidates.length} />
              <ClipEvidence scores={g.scores[model]} loudness={g.loudness_above_background} label={g.label} commentary={g.commentary} start={selectedStart} end={selectedEnd} cursor={cur} model={model} />
              <p className="kicker mb-2 mt-6">Full-game overview / click to seek</p>
              <div className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-x-2">
                <ul className="mono relative text-[10px] leading-tight text-muted" aria-hidden="true">
                  <li className="absolute top-[10%]">MODEL RANK</li><li className="absolute top-[41%] text-defense">EDITORIAL</li><li className="absolute top-[52%] text-amber">MODEL REEL</li>
                  <li className="absolute top-[68%]">AUDIO</li><li className="absolute top-[88%]">COMMENTARY</li>
                </ul>
                <div className="border border-line bg-surface">
                  <svg viewBox={`0 0 ${N} 150`} preserveAspectRatio="none" className="block h-32 w-full cursor-crosshair" role="img"
                    aria-label={`Timeline of ${g.title}. Rows: model score rank, editorial highlight segments, the model's ${budget} reel, loudness above background, and commentary activity. Use the slider below to move the cursor.`}
                    onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); const clip=Math.max(0, Math.min(N-1,Math.floor(((e.clientX-r.left)/r.width)*N))); const candidate=g.candidates.find(c=>clip>=c.start_clip&&clip<c.end_clip); selectMoment(clip*2,candidate?.end_s ?? null,candidate ? "ranked interval" : "full-video position"); }}>
                    <path d={path(g.scores[model], 50, 100)} transform="translate(0 4)" fill="none" stroke="#f2efe5" strokeWidth={1} vectorEffect="non-scaling-stroke" />
                    {bands!.label.map(([a, b]) => <rect key={`l${a}`} x={a} y={60} width={b - a} height={11} fill="#8ebaf1" />)}
                    {bands!.selected.map(([a, b]) => <rect key={`s${a}`} x={a} y={76} width={b - a} height={11} fill="#e8ad66" />)}
                    <path d={path(g.loudness_above_background, 30, 20, -10)} transform="translate(0 94)" fill="none" stroke="#e8ad66" strokeWidth={1} vectorEffect="non-scaling-stroke" opacity={0.9} />
                    {g.commentary.map((c) => <rect key={c.start_s} x={c.start_s / 2} y={132} width={Math.max(1, (c.end_s - c.start_s) / 2 - 1)} height={10} fill="#9ba89d" opacity={0.55} />)}
                    {cand && <rect x={cand.start_clip} y={0} width={cand.end_clip - cand.start_clip} height={150} fill="#e8ad66" opacity={0.18} />}
                    <line x1={cur} x2={cur} y1={0} y2={150} stroke="#e8ad66" strokeWidth={2} vectorEffect="non-scaling-stroke" />
                  </svg>
                </div>
                <span />
                <div className="mono num mt-1 flex justify-between text-[10px] text-muted"><span>0:00:00</span><span className="text-amber">{clock(cur * 2)}</span><span>{clock(N * 2)} TRIMMED BROADCAST</span></div>
                <span />
                <input type="range" min={0} max={N - 1} value={cur} onChange={(e) => selectMoment(Number(e.target.value)*2,null,"full-video position")} className="w-full" style={{ accentColor: "#e8ad66" }} aria-label="Timeline position" />
              </div>
              <p className="mt-1 text-xs text-muted">Every row is measured data for this game: model score rank, the released editorial label, the model&apos;s {budget} reel, loudness above the local background, and machine-transcribed commentary segments. No play diagram is shown because no tracking exists for these broadcasts.</p>

              <h2 className="kicker mt-6">Candidate moments in the 3-minute reel, best first</h2>
              <ul className="mt-2 flex gap-2 overflow-x-auto pb-2">
                {g.candidates.slice(0, 24).map((c) => (
                  <li key={c.start_clip}>
                    <button onClick={() => selectMoment(c.start_s,c.end_s)} aria-current={selectedCandidate === c ? "true" : undefined}
                      className={`candidate-button w-44 shrink-0 border px-3 py-3 text-left transition-colors duration-150 ${selectedCandidate === c ? "border-amber bg-amber/10" : "border-line hover:border-ink"}`}>
                      <span className="mb-2 flex items-center justify-between text-[10px] text-amber"><span>▶ PLAY INTERVAL</span><span>{c.end_s-c.start_s}s</span></span>
                      <span className="mono num block text-sm">{clock(c.start_s)} <span className="text-muted">RANK {c.score_rank}</span></span>
                      <span className={`block text-xs ${c.in_editorial_highlights ? "text-defense" : "text-coral"}`}>{c.in_editorial_highlights ? "In the editors' reel" : "Not in the editors' reel"}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>

            <aside aria-label="Selected moment" className="min-w-0 text-sm">
              <div role="tablist" aria-label="Moment panels" className="flex border-b border-line">
                {(["evidence", "missed", "review"] as const).map((k) => (
                  <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)} className={`narrow -mb-px border-b-2 px-4 py-2 text-lg transition-colors duration-150 ${tab === k ? "border-amber text-amber" : "border-transparent text-muted hover:text-ink"}`}>
                    {k === "evidence" ? "Evidence" : k === "missed" ? "Missed moments" : "Review"}</button>
                ))}
              </div>
              {tab === "evidence" && (
                <div role="tabpanel" className="fade-swap pt-4">
                  <div className="flex items-start justify-between gap-3">
                    <div><p className="kicker">Selected moment</p><p className="display text-6xl num">Rank {g.scores[model][cur]}</p><p className="text-xs text-muted">Ranking score percentile within this game (100 = highest). Not a probability.</p></div>
                    <p className="text-right"><span className="kicker block">Source time</span><span className="mono num text-xl text-amber">{clock(g.source.trim_start_s + cur * 2)}</span></p>
                  </div>
                  <p className="mono num mt-2 text-xs text-muted">TIMELINE {clock(cur * 2)} FROM THE TRIMMED START. <a className="text-teal underline" href={sourceLink(g, cur * 2)} target="_blank" rel="noreferrer">Open the source video at this moment</a></p>
                  <table className="num mt-4 w-full text-xs">
                    <caption className="kicker pb-1 text-left"><Badge kind="Predicted" /> signals at this moment</caption>
                    <tbody>
                      {Object.entries(g.scores).map(([n, s]) => <tr key={n} className="border-t border-line"><td className="py-1.5 pr-2">{n} rank</td><td className="text-right">{s[cur]}</td></tr>)}
                      <tr className="border-t border-line"><td className="py-1.5 pr-2">Loudness above local background</td><td className="text-right">{g.loudness_above_background[cur].toFixed(1)} dB</td></tr>
                      {cand && <tr className="border-t border-line"><td className="py-1.5 pr-2">Commentary words the word model weighted</td><td className="text-right">{cand.commentary_terms.length ? cand.commentary_terms.join(", ") : "none"}</td></tr>}
                    </tbody>
                  </table>
                  <div className="mt-4 border border-line p-3">
                    <p className="kicker"><Badge kind="Released label" /> evaluation only</p>
                    <p className="narrow mt-1 text-2xl">{g.label[cur] ? "In the editors' reel" : "Not in the editors' reel"}</p>
                    <p className="text-xs text-muted">The editorial label is used to score the ranking afterwards. It is never an input and is not a reason the model ranked this moment.</p>
                  </div>
                </div>
              )}
              {tab === "missed" && (
                <div role="tabpanel" className="fade-swap pt-4">
                  <p className="text-xs text-muted">{g.missed_highlight_count} of {g.metrics["3 minutes"].highlight_segments} labelled segments have no overlap with the 3-minute reel. Its clips are {pct(g.metrics["3 minutes"].precision)} editorial highlights, yet it covers only {pct(g.metrics["3 minutes"].recall_of_highlight_time)} of highlight time. Longest first.</p>
                  <ul className="mt-2 max-h-[26rem] overflow-y-auto border-t border-line">
                    {g.missed_highlights.map((m) => (
                      <li key={m.start_s}>
                        <button onClick={() => { selectMoment(m.start_s,m.end_s,"missed editorial segment"); setTab("evidence"); }} className="w-full border-b border-line px-2 py-2 text-left hover:bg-surface">
                          <span className="mono num flex justify-between gap-2 text-xs"><span>{clock(m.start_s)} · {m.seconds}S</span><span className="text-muted">BEST RANK {m.best_score_rank}</span></span>
                          <span className="block text-xs text-muted">{m.excerpt || "No commentary transcribed here"}</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {tab === "review" && (
                <div role="tabpanel" className="fade-swap pt-4 text-muted">
                  <p>Record event labels, the evidence you had, how sure you are, the event moment and better clip boundaries for this game&apos;s candidates. Annotations stay in your browser until you save them as a file, and never change the benchmark labels.</p>
                  <p className="mt-3"><Link className="btn-quiet text-ink" href={`/highlights/review?game=${g.id}`}>Open the review tool</Link></p>
                  <p className="mt-4 text-xs">{index.event_extension.needs}</p>
                </div>
              )}
            </aside>
          </div>
        )}
      </div>

      <div className="paper">
        <div className="mx-auto flex max-w-[1500px] flex-col gap-10 px-4 py-12 lg:px-8">
          {g && (
            <section aria-labelledby="h-closer" className="grid gap-8 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)_minmax(0,1fr)]">
              <div><p className="kicker">Why this moment?</p><h2 id="h-closer" className="display mt-1 text-5xl sm:text-6xl">A closer look<span className="text-amber">.</span></h2>
                <p className="mt-3 text-sm text-muted">Measured signals within 90 seconds of {clock(cur * 2)}. Nothing here is generated: the audio curve is the released loudness series and the text is the machine transcript.</p></div>
              <figure>
                <figcaption className="kicker"><Badge kind="Observed" /> audio evidence: loudness above background</figcaption>
                <svg viewBox={`0 0 ${hi - lo} 60`} preserveAspectRatio="none" className="mt-2 block h-32 w-full bg-surface" role="img" aria-label={`Loudness from ${clock(lo * 2)} to ${clock(hi * 2)}; ${g.loudness_above_background[cur]} decibels above background at the cursor.`}>
                  {g.loudness_above_background.slice(lo, hi).map((v, i) => { const h = Math.max(1, Math.min(28, Math.abs(v) * 1.6)); return <rect key={i} x={i + 0.15} y={30 - h} width={0.7} height={h * 2} fill={lo + i === cur ? "#101713" : "#99601b"} opacity={lo + i === cur ? 1 : 0.75} />; })}
                </svg>
                <p className="mono num mt-1 flex justify-between text-[10px] text-muted"><span>{clock(lo * 2)}</span><span>{clock(hi * 2)}</span></p>
              </figure>
              <div>
                <p className="kicker"><Badge kind="Observed" /> transcript evidence (machine transcript, short excerpts)</p>
                {nearby.length === 0 ? <p className="mt-2 text-sm text-muted">No commentary was transcribed near this moment.</p> : (
                  <ul className="mt-2 text-sm">
                    {nearby.map((c) => <li key={c.start_s} className={`grid grid-cols-[4.5rem_1fr] gap-2 border-t border-line py-1.5 ${c === talk ? "bg-surface" : ""}`}><span className="mono num text-xs text-muted">{clock(c.start_s)}</span><span>{c.excerpt}</span></li>)}
                  </ul>
                )}
              </div>
            </section>
          )}

      {g && recap && recap.game_id && (
        <section aria-labelledby="h-recap">
          <h2 id="h-recap" className="display text-3xl sm:text-4xl">Metadata-led recap: what play-by-play says happened</h2>
          <p className="mt-1 max-w-[75ch] border-l-2 border-amber pl-3 text-sm text-muted">A separate product from the timeline above. These events come straight from {recap.source} for game {recap.game_id} ({recap.final}). They are not detections by the media models and they are {recap.alignment.toLowerCase()} Turnover = {recap.definitions.turnover}; big play = {recap.definitions["big play"]}.</p>
          <p className="num mt-2 text-sm text-muted">{Object.entries(recap.counts).map(([k, v]) => `${v} ${k}`).join(", ")}.</p>
          <div className="mt-2 max-h-80 overflow-auto border-t border-line">
            <table className="num w-full min-w-[640px] text-sm">
              <thead><tr><th className={TH}>Quarter, clock</th><th className={TH}>Offense</th><th className={TH}>Events</th><th className={TH}>Yards</th><th className={TH}>Play-by-play text (excerpt)</th></tr></thead>
              <tbody>{recap.events.map((e) => <tr key={e.play_id} className="border-t border-line align-top"><td className="py-1.5 pr-3">Q{e.quarter} {e.game_clock}</td><td className="pr-3">{e.offense}</td><td className="pr-3">{e.events.join(", ")}</td><td className="pr-3">{e.yards}</td><td className="text-muted">{e.description}</td></tr>)}</tbody>
            </table>
          </div>
          <p className="mt-2 text-sm"><Link className="text-teal underline" href={`/highlights/review?game=${g.id}`}>Review this game&apos;s candidates and record event labels</Link></p>
        </section>
      )}

      <section aria-labelledby="h-bench">
        <h2 id="h-bench" className="display text-3xl sm:text-4xl">Every experiment on the same games</h2>
        <p className="mt-1 max-w-[75ch] text-sm text-muted">{index.clips.toLocaleString()} two-second clips, {pct(index.positive_share, 1)} labelled highlight. {index.split.rule}. {index.selection_rule}; the test games were scored once. {index.budget_note}</p>
        <div className="overflow-x-auto">
          <table className="num mt-2 w-full min-w-[760px] text-sm">
            <thead><tr><th className={TH}>Experiment</th><th className={TH}>Validation mAP</th><th className={TH}>Test mAP (range over games)</th><th className={TH}>Top clip is a highlight</th><th className={TH}>3-min precision</th><th className={TH}>3-min recall</th><th className={TH}>Segments missed</th></tr></thead>
            <tbody>
              {main.map(([n, r]) => (
                <tr key={n} className={`border-t border-line ${n === index.shipped ? "bg-surface" : ""}`}>
                  <td className="py-1.5 pr-3">{n}{n === index.shipped ? " (shipped)" : ""}</td><td>{r.validation.mean_average_precision.toFixed(3)}</td>
                  <td>{r.test.mean_average_precision.toFixed(3)} ({r.test.average_precision_range[0].toFixed(2)} to {r.test.average_precision_range[1].toFixed(2)})</td>
                  <td>{pct(r.test.hit_at_1)}</td><td>{pct(r.test["3 minutes"].precision)}</td><td>{pct(r.test["3 minutes"].recall_of_highlight_time)}</td><td>{pct(r.test["3 minutes"].segments_missed_share)}</td>
                </tr>
              ))}
              <tr className="border-t border-line text-muted"><td className="py-1.5 pr-3">Random ranking (expected)</td><td>{shipped.validation.chance_average_precision.toFixed(3)}</td><td>{shipped.test.chance_average_precision.toFixed(3)}</td><td /><td>{pct(shipped.test.chance_average_precision)}</td><td /><td /></tr>
            </tbody>
          </table>
        </div>
        <div className="mt-5 grid gap-6 md:grid-cols-2">
          <table className="num w-full text-sm">
            <caption className="pb-1 text-left text-muted">Temporal model with one input stream masked at scoring time (test mAP)</caption>
            <tbody>{ablation.map(([n, r]) => <tr key={n} className="border-t border-line"><td className="py-1.5 pr-3">{n}</td><td>{r.test.mean_average_precision.toFixed(3)}</td></tr>)}</tbody>
          </table>
          {index.results["H1 commentary"]?.info.probes && (
            <table className="num w-full text-sm">
              <caption className="pb-1 text-left text-muted">Commentary model on short test phrases. {index.results["H1 commentary"].info.probe_note}</caption>
              <thead><tr><th className={TH}>Case</th><th className={TH}>Phrase</th><th className={TH}>Score</th></tr></thead>
              <tbody>{index.results["H1 commentary"].info.probes!.map((p) => <tr key={p.case} className="border-t border-line"><td className="py-1.5 pr-3">{p.case}</td><td className="pr-3">{p.text}</td><td>{p.score.toFixed(2)}</td></tr>)}</tbody>
            </table>
          )}
        </div>
      </section>

      <section aria-labelledby="h-how">
        <h2 id="h-how" className="display text-3xl sm:text-4xl">How was this calculated?</h2>
        <ol className="mt-2 max-w-[75ch] list-decimal space-y-2 pl-5 text-sm text-muted marker:text-ink">
          <li><span className="text-ink">Inputs.</span> Per 2-second clip: loudness, machine-transcribed commentary words with timestamps, and released audio and visual embeddings. All obtainable from the full game alone.</li>
          <li><span className="text-ink">Context.</span> Processing is offline, so a clip&apos;s features can include commentary and neighbouring clips that come after it.</li>
          <li><span className="text-ink">Representation.</span> Each embedding is reduced to 64 components with a transform fitted on the 28 training games only.</li>
          <li><span className="text-ink">Models.</span> Loudness above background; word model; per-clip classifiers; a small temporal-convolution model with modality masks.</li>
          <li><span className="text-ink">Output.</span> {index.score_meaning} {index.calibration}</li>
          <li><span className="text-ink">Label comparison.</span> {index.label}</li>
          <li><span className="text-ink">Left out on purpose.</span> {index.captions} Labels, frame-match files and the highlight videos themselves are never model inputs.</li>
          <li><span className="text-ink">Event detection.</span> {index.event_extension.needs}</li>
        </ol>
        <p className="mt-3 break-all text-xs text-muted">{index.attribution} Source {index.source}, revision {index.revision.slice(0, 8)}. Run {index.run_id}.</p>
      </section>
        </div>
      </div>
    </div>
  );
}
