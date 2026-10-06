"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Field, { type Highlight } from "@/components/Field";
import ModuleTour from "@/components/ModuleTour";
import StadiumBackdrop from "@/components/StadiumBackdrop";
import { Badge, Bar } from "@/components/ui";
import {
  type Ablation, type Dataset, type HorizonKey, type ModelKey, type Play, type PlaySummary, type Similar, type Status, type Term, DEFAULT_HORIZON, DEFAULT_MODEL, HORIZONS, MODELS,
  STATUS_CLASS, STATUS_TEXT, TYPE_LABELS, horizonFrom, loadDataset, loadPlay, modelFrom, ordinal, pct, statusOf,
} from "@/lib/demo";

type View = "sample" | "errors" | "quick";
type Filter = "all" | Status;
type Tab = "evidence" | "similar" | "errors";
const FILTERS: Filter[] = ["all", "correct", "incorrect", "abstained"];
const SPEEDS = [0.5, 1, 2];
const VIEWS: { key: View; label: string; href: string }[] = [
  { key: "sample", label: "Random sample", href: "/coverage" },
  { key: "quick", label: "Quick throws", href: "/coverage/quick" },
  { key: "errors", label: "Error gallery", href: "/coverage/errors" },
];
const TABS: { key: Tab; label: string }[] = [{ key: "evidence", label: "Evidence" }, { key: "similar", label: "Similar plays" }, { key: "errors", label: "Errors" }];
const SCOPE = "Tracking covers the passer, route runners, and selected coverage defenders from snap to throw.";

function MiniField({ play }: { play: Similar }) {
  // Static paths of a reference play, snap to +1.5 s, in the same orientation as the main field.
  const pts = (tr: number[][]) => tr.map(([x, y]) => `${(26 - y).toFixed(1)},${(22 - x).toFixed(1)}`).join(" ");
  return (
    <svg viewBox="0 0 52 34" className="w-full bg-turf" role="img" aria-label={`Paths of ${play.offense} against ${play.defense}, snap to 1.5 seconds`}>
      <line x1={0} x2={52} y1={22} y2={22} stroke="#d8f16e" strokeWidth={0.25} opacity={0.8} />
      {play.tracks.offense.map((t, i) => <g key={`o${i}`}><polyline points={pts(t)} fill="none" stroke="#f2efe5" strokeWidth={0.35} opacity={0.6} /><circle cx={26 - t.at(-1)![1]} cy={22 - t.at(-1)![0]} r={0.7} fill="#13241b" stroke="#f2efe5" strokeWidth={0.25} /></g>)}
      {play.tracks.defense.map((t, i) => <g key={`d${i}`}><polyline points={pts(t)} fill="none" stroke="#8ebaf1" strokeWidth={0.35} opacity={0.7} /><circle cx={26 - t.at(-1)![1]} cy={22 - t.at(-1)![0]} r={0.6} fill="#8ebaf1" /></g>)}
    </svg>
  );
}

// What each saved model is for. The preserved benchmark stays the page default; the champion is the best v2 model.
const ROLE: Record<ModelKey, string> = { v2_temporal: "Champion", v1_gbm: "Preserved benchmark", v2_gbm_rel: "Comparison", v1_logit: "Comparison" };

export default function CoverageExplorer({ view }: { view: View }) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const [ds, setDs] = useState<Dataset | null | undefined>(undefined);
  const [loaded, setLoaded] = useState<Play | null>(null);
  const [failed, setFailed] = useState<{ id: string; message: string } | null>(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [selected, setSelected] = useState<string | null>(null);
  const [trails, setTrails] = useState(true);
  const [labels, setLabels] = useState(false);
  const [evidence, setEvidence] = useState<Highlight>(null);
  const [tab, setTab] = useState<Tab>("evidence");
  const [query, setQuery] = useState("");
  const [libOpen, setLibOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const tRef = useRef(0);
  const expandBtn = useRef<HTMLButtonElement>(null);
  const closeBtn = useRef<HTMLButtonElement>(null);

  const model = modelFrom(params.get("model"));
  const wanted = horizonFrom(params.get("window") ?? params.get("horizon"));
  const filter = (FILTERS.includes(params.get("show") as Filter) ? params.get("show") : "all") as Filter;
  const playId = params.get("play");
  const down = Number(params.get("down") ?? 0);
  const team = params.get("defense") ?? "";
  const modelHorizons = ds?.index.models[model].horizons ?? [];
  const horizon: HorizonKey = modelHorizons.includes(wanted) || modelHorizons.length === 0 ? wanted : (modelHorizons.at(-1) as HorizonKey);

  const setParam = useCallback((changes: Record<string, string | null>) => {
    const next = new URLSearchParams(params.toString());
    for (const [k, v] of Object.entries(changes)) { if (v == null) next.delete(k); else next.set(k, v); }
    router.replace(`${pathname}?${next.toString()}`, { scroll: false });
  }, [params, pathname, router]);
  const carry = `model=${model}&window=${horizon}`;

  useEffect(() => { loadDataset().then(setDs); }, []);

  const base = useMemo(() => {
    if (!ds) return [] as PlaySummary[];
    const ids = new Set(view === "sample" ? ds.index.sample.ids : view === "quick" ? ds.index.quick_throws.ids : ds.index.errors.ids[model][horizon] ?? []);
    return ds.index.plays.filter((p) => ids.has(p.id));
  }, [ds, view, model, horizon]);
  const leanOf = useCallback((p: PlaySummary) => p.predictions[model]?.[horizon] ?? null, [model, horizon]);
  const status = useCallback((p: PlaySummary): Status | null => { const l = leanOf(p); return l ? statusOf(l, p.released.manZone) : null; }, [leanOf]);
  const counts = useMemo(() => {
    const c = { correct: 0, incorrect: 0, abstained: 0, unavailable: 0 };
    for (const p of base) { const s = status(p); if (s) c[s] += 1; else c.unavailable += 1; }
    return c;
  }, [base, status]);
  const plays = useMemo(() => {
    const q = query.trim().toLowerCase();
    return base.filter((p) => (filter === "all" || status(p) === filter) && (down === 0 || p.down === down) && (team === "" || p.defense === team)
      && (q === "" || `${p.offense} ${p.defense} week ${p.week} wk ${p.week} ${p.released.manZone} ${p.released.coverage ?? ""}`.toLowerCase().includes(q)));
  }, [base, filter, status, down, team, query]);
  const teams = useMemo(() => [...new Set(base.map((p) => p.defense))].sort(), [base]);
  const current = plays.find((p) => p.id === playId) ?? plays[0] ?? null;
  const play = loaded && current && loaded.id === current.id ? loaded : null;
  const error = failed && current && failed.id === current.id ? failed.message : null;

  useEffect(() => {
    if (!ds || !current) return;
    let live = true;
    const id = current.id;
    loadPlay(ds, current.file)
      .then((p) => { if (live) { setLoaded(p); setT(p.frames[0]); tRef.current = p.frames[0]; setSelected(null); setPlaying(false); setEvidence(null); } })
      .catch((e) => live && setFailed({ id, message: String(e.message ?? e) }));
    return () => { live = false; };
  }, [ds, current?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Playback follows real time between observed frames (10 per second). It only moves the picture: the model's input cutoff is a separate control.
  useEffect(() => {
    if (!playing || !play) return;
    const end = play.frames[play.frames.length - 1];
    let raf = 0, last = performance.now();
    const tick = (now: number) => {
      tRef.current = Math.min(end, tRef.current + ((now - last) / 100) * speed);
      last = now;
      setT(tRef.current);
      if (tRef.current >= end) setPlaying(false); else raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, play, speed]);

  const seek = useCallback((v: number) => {
    if (!play) return;
    const c = Math.max(play.frames[0], Math.min(play.frames[play.frames.length - 1], v));
    tRef.current = c; setT(c);
  }, [play]);
  const toggle = useCallback(() => {
    if (!play) return;
    if (!playing && tRef.current >= play.frames[play.frames.length - 1]) seek(play.frames[0]);
    setPlaying((p) => !p);
  }, [play, playing, seek]);
  const closeExpanded = useCallback(() => { setExpanded(false); requestAnimationFrame(() => expandBtn.current?.focus()); }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && expanded) { closeExpanded(); return; }
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "BUTTON" || tag === "A" || tag === "SUMMARY") return;
      if (e.key === " ") { e.preventDefault(); toggle(); }
      if (e.key === "ArrowRight") { setPlaying(false); seek(Math.floor(tRef.current) + 1); }
      if (e.key === "ArrowLeft") { setPlaying(false); seek(Math.ceil(tRef.current) - 1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, seek, expanded, closeExpanded]);
  useEffect(() => { if (expanded) closeBtn.current?.focus(); }, [expanded]);

  if (ds === undefined) return <p className="p-6 text-muted">Loading plays…</p>;
  if (ds === null)
    return (
      <div className="max-w-xl p-6">
        <h1 className="display text-3xl">No coverage export found</h1>
        <p className="mt-3 text-muted">Run the pipeline to create one, then reload this page.</p>
        <pre className="mono mt-4 overflow-x-auto border border-line bg-surface p-3 text-sm">bin/coverage-lens export-v3</pre>
      </div>
    );

  const spec = ds.index.models[model];
  const cutoff = ds.index.cutoffs[model][horizon];
  const hz = play?.horizons[horizon] ?? null;
  const main = hz?.predictions[model] ?? null;
  const playStatus = current ? status(current) : null;
  const sel = play?.entities.find((e) => e.id === selected);
  const frameIdx = play ? Math.round(t) - play.frames[0] : 0;
  const first = play?.frames[0] ?? 0, lastF = play?.frames[play.frames.length - 1] ?? 0;
  const hLabel = HORIZONS.find((w) => w.key === horizon)!;
  const available = ds.index.errors.available[model][horizon] ?? 0;
  const expl = hz?.explanations[model];
  const nameOf = (id: string) => { const e = play?.entities.find((x) => x.id === id); return e ? `${e.name ?? id} (${e.position ?? "?"})` : id; };
  const showEvidence = (kind: string, text: string) => {
    if (!hz) return;
    if (kind === "pairs") setEvidence({ players: hz.evidence.pairs.flatMap((p) => [p.defender, p.receiver]), links: hz.evidence.pairs.map((p) => [p.defender, p.receiver]), note: text });
    else if (kind === "nearest") setEvidence({ players: hz.evidence.nearest.flatMap((p) => [p.defender, p.receiver]), links: hz.evidence.nearest.map((p) => [p.defender, p.receiver]), note: text });
    else if (kind === "deepest") setEvidence({ players: hz.evidence.deepest, links: [], note: text });
    else setEvidence({ players: play!.entities.filter((e) => e.side === "defense").map((e) => e.id!), links: [], note: text });
  };
  const setNote = view === "sample" ? ds.index.sample.note : view === "quick" ? ds.index.quick_throws.note
    : `${ds.index.errors.note} Shown here: ${base.length} of the ${available} plays where ${spec.short} at ${hLabel.label} leans against the released label.`;
  const errorIds = ds.index.errors.ids[model][horizon] ?? [];
  const errorPlays = ds.index.plays.filter((p) => errorIds.includes(p.id)).slice(0, 8);

  const controls = play && (
    <>
      <div className="mt-3 flex flex-wrap items-center gap-2 sm:gap-3">
        <button onClick={toggle} className="narrow min-w-20 bg-teal px-4 py-2 text-lg font-bold text-bg transition-transform duration-150 active:scale-95">{playing ? "Pause" : "Play"}</button>
        <button onClick={() => { setPlaying(false); seek(Math.ceil(t) - 1); }} aria-label="Back one frame" className="border border-line px-3 py-2 hover:border-ink">◀</button>
        <button onClick={() => { setPlaying(false); seek(Math.floor(t) + 1); }} aria-label="Forward one frame" className="border border-line px-3 py-2 hover:border-ink">▶</button>
        <button onClick={() => { setPlaying(false); seek(hz?.latest_frame ?? 0); }} className="border border-line px-3 py-2 text-sm hover:border-ink">Go to model cutoff</button>
        <label className="flex items-center gap-1 text-sm text-muted">Speed
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="border border-line bg-surface px-1.5 text-ink">{SPEEDS.map((s) => <option key={s} value={s}>{s}x</option>)}</select>
        </label>
        <label className="flex items-center gap-1.5 text-sm text-muted"><input type="checkbox" checked={trails} onChange={(e) => setTrails(e.target.checked)} /> Trails</label>
        <label className="flex items-center gap-1.5 text-sm text-muted"><input type="checkbox" checked={labels} onChange={(e) => setLabels(e.target.checked)} /> Positions</label>
        <span className="mono num ml-auto text-sm">+{(t / 10).toFixed(1)}s after snap</span>
      </div>
      <div className="relative mt-1">
        <input type="range" aria-label="Playback position (does not change the model input)" min={first} max={lastF} step={1} value={Math.round(t)} onChange={(e) => { setPlaying(false); seek(Number(e.target.value)); }} className="w-full" />
        <div className="mono relative h-5 text-[10px] text-muted" aria-hidden>
          <span className="absolute" style={{ left: 0 }}>SNAP</span>
          {hz && hz.latest_frame > 0 && <span className="absolute -translate-x-1/2 text-amber" style={{ left: `${((hz.latest_frame - first) / Math.max(1, lastF - first)) * 100}%` }}>MODEL CUTOFF</span>}
          <span className="absolute right-0">THROW</span>
        </div>
      </div>
    </>
  );

  return (
    <div>
      <header className="cinematic-band border-b border-line">
        <StadiumBackdrop />
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-end justify-between gap-x-10 gap-y-3 px-4 py-6 lg:px-8">
          <div>
            <div className="flex items-center gap-4"><p className="kicker">Coverage / film room</p><ModuleTour module="coverage" ready={!!play} /></div>
            <h1 className="display mt-1 text-5xl sm:text-7xl">Read the defense<span className="text-teal">.</span></h1>
          </div>
          <p className="max-w-[58ch] text-sm text-muted">
            <strong className="font-semibold text-ink">Selected-player tracking.</strong> {SCOPE} The release chose those defenders after the play, so this is not a snap-time or live system.{" "}
            <Link className="text-teal underline" href={`/coverage/evaluation?${carry}#e-sens`}>Data scope and limits</Link>{" · "}
            <Link className="text-teal underline" href="/coverage/analyze">Analyze your play</Link>
          </p>
        </div>
      </header>

      <div className="mx-auto flex max-w-[1500px] flex-col gap-4 px-4 py-4 lg:px-8">
        <section data-tour="coverage-settings" aria-label="What is shown" className="border border-line">
          <div className="flex flex-wrap items-stretch">
            <nav aria-label="Play set" className="flex text-sm">
              {VIEWS.map((v) => (
                <Link key={v.key} href={`${v.href}?${carry}`} aria-current={view === v.key ? "page" : undefined}
                  className={`narrow flex items-center px-4 py-2.5 text-base transition-colors duration-150 ${view === v.key ? "bg-ink text-bg" : "text-muted hover:text-ink"}`}>{v.label}</Link>
              ))}
            </nav>
            <div className="flex min-w-0 flex-1 flex-wrap items-center gap-x-5 gap-y-2 border-l border-line px-4 py-2 text-sm">
              <label className="flex min-w-0 max-w-full items-center gap-2 text-muted">Model
                <select value={model} onChange={(e) => setParam({ model: e.target.value === DEFAULT_MODEL ? null : e.target.value, play: view === "errors" ? null : playId })} className="min-w-0 max-w-full border border-line bg-surface px-2 text-ink">
                  {MODELS.map((m) => <option key={m} value={m}>{ROLE[m]}: {ds.index.models[m].short}</option>)}
                </select>
              </label>
              <div role="group" aria-label="Model input cutoff" className="flex items-center border border-line">
                <span className="px-2 text-muted">Model sees up to</span>
                {HORIZONS.map((w) => {
                  const off = !modelHorizons.includes(w.key) || (play != null && play.horizons[w.key] == null);
                  return (
                    <button key={w.key} aria-pressed={w.key === horizon} disabled={off}
                      title={!modelHorizons.includes(w.key) ? "This model was not trained for this cutoff" : off ? "The ball was thrown before this cutoff on this play" : undefined}
                      onClick={() => setParam({ window: w.key === DEFAULT_HORIZON ? null : w.key, horizon: null, play: view === "errors" ? null : playId })}
                      className={`narrow px-3 text-base transition-colors duration-150 ${w.key === horizon ? "bg-teal font-bold text-bg" : off ? "cursor-not-allowed text-line" : "text-muted hover:text-ink"}`}>{w.label}</button>
                  );
                })}
              </div>
              <p className="num text-ink" data-testid="settings"><span className="text-muted">Showing</span> {spec.short}, {hLabel.label}, confidence cutoff <strong className="font-semibold">{cutoff != null ? pct(cutoff) : "n/a"}</strong></p>
              <p className="text-xs text-muted" data-testid="model-role">{ROLE[model]} model.{model !== "v2_temporal" && <> The champion is the temporal model (v2): <button className="text-teal underline" onClick={() => setParam({ model: "v2_temporal", play: view === "errors" ? null : playId })}>switch to it</button>.</>} Which model is shown first is a display choice, not a model improvement.</p>
            </div>
          </div>
          <p className="border-t border-line px-4 py-2 text-xs text-muted" data-testid="set-note">{setNote} The model sees {hLabel.detail}. Playback only moves the picture; it never changes what the model was given.</p>
        </section>

        <div className="grid gap-5 lg:grid-cols-[280px_minmax(0,1fr)_380px]">
          {/* play library: a drawer on small screens, a column on large ones */}
          <aside data-tour="coverage-library" aria-label="Play library" className="order-3 lg:order-1">
            <button className="narrow w-full border border-line px-3 text-left text-base lg:hidden" aria-expanded={libOpen} aria-controls="library" onClick={() => setLibOpen((o) => !o)}>
              {libOpen ? "Hide" : "Show"} play library ({plays.length})
            </button>
            <div id="library" className={`${libOpen ? "block" : "hidden"} mt-3 lg:mt-0 lg:block`}>
              <a href="#play" className="sr-only focus:not-sr-only focus:mb-2 focus:block focus:bg-teal focus:px-3 focus:py-2 focus:text-bg">Skip the play list</a>
              <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search team, week or label" aria-label="Search plays" className="w-full border border-line bg-surface px-3 py-2 text-sm placeholder:text-muted" />
              <div className="mt-2 flex flex-wrap gap-2 text-sm">
                <select aria-label="Show" value={filter} onChange={(e) => setParam({ show: e.target.value === "all" ? null : e.target.value, play: null })} className="border border-line bg-surface px-2">
                  <option value="all">All plays</option>
                  <option value="correct">{STATUS_TEXT.correct}</option><option value="incorrect">{STATUS_TEXT.incorrect}</option><option value="abstained">{STATUS_TEXT.abstained}</option>
                </select>
                <select aria-label="Down" value={down} onChange={(e) => setParam({ down: e.target.value === "0" ? null : e.target.value, play: null })} className="border border-line bg-surface px-2">
                  <option value={0}>Any down</option>{[1, 2, 3, 4].map((d) => <option key={d} value={d}>{ordinal(d)} down</option>)}
                </select>
                <select aria-label="Defense" value={team} onChange={(e) => setParam({ defense: e.target.value || null, play: null })} className="border border-line bg-surface px-2">
                  <option value="">Any defense</option>{teams.map((x) => <option key={x}>{x}</option>)}
                </select>
              </div>
              <p className="num mt-2 text-xs" data-testid="counts">
                {base.length} plays: <span className={STATUS_CLASS.correct}>{counts.correct} accepted and agree</span>, <span className={STATUS_CLASS.incorrect}>{counts.incorrect} accepted and disagree</span>,{" "}
                <span className={STATUS_CLASS.abstained}>{counts.abstained} abstained</span>{counts.unavailable > 0 && <span className="text-muted">, {counts.unavailable} thrown before this cutoff</span>}
              </p>
              <p className="num text-xs text-muted" data-testid="shown">Showing {plays.length} of {base.length}</p>
              {plays.length === 0 ? <p className="mt-3 border border-line p-3 text-sm text-muted">No plays match. Clear the search or a filter to see more.</p> : (
                <ul className="mt-2 max-h-[62vh] overflow-y-auto border-t border-line">
                  {plays.map((p) => {
                    const lean = leanOf(p), st = status(p), active = current?.id === p.id;
                    return (
                      <li key={p.id}>
                        <button onClick={() => { setParam({ play: p.id }); setLibOpen(false); }} aria-current={active ? "true" : undefined} data-status={st ?? "unavailable"}
                          className={`w-full border-b border-l-2 border-b-line px-2 py-2 text-left text-sm transition-colors duration-150 hover:bg-surface ${active ? "border-l-teal bg-surface" : "border-l-transparent"}`}>
                          <span className="flex justify-between gap-2"><span className="narrow text-base">{p.offense} at {p.defense}</span><span className={`text-xs ${st ? STATUS_CLASS[st] : "text-muted"}`}>{st ? STATUS_TEXT[st] : "No prediction"}</span></span>
                          <span className="mono num block text-[11px] text-muted">WK {p.week} · Q{p.quarter} · {ordinal(p.down).toUpperCase()} &amp; {p.yardsToGo} · THROWN {(p.frames_observed / 10).toFixed(1)}S</span>
                          <span className="num block text-xs text-muted">Label {p.released.manZone}. {lean ? `Leans ${lean.predicted}, ${pct(lean.p_man)} man` : "Thrown before this cutoff"}</span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          </aside>

          <section data-tour="coverage-field" id="play" tabIndex={-1} aria-label="Play" className="order-1 min-w-0 lg:order-2">
            {error && <p className="border border-coral/60 p-3 text-sm text-coral">This play could not be loaded: {error}. Pick another play or re-run the export.</p>}
            {!play && !error && <p className="text-muted">{current ? "Loading tracking…" : "Pick a play from the library."}</p>}
            {play && (
              <div key={play.id} className="fade-swap">
                <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                  <h2 className="display text-3xl sm:text-4xl">{play.offense} <span className="text-muted">vs</span> {play.defense}</h2>
                  <p className="mono num text-xs text-muted">WEEK {play.week} · Q{play.quarter} · {ordinal(play.down).toUpperCase()} &amp; {play.yardsToGo} · {play.yardsToGoal} YARDS TO GOAL</p>
                </div>
                <div className="relative">
                  <Field entities={play.entities} frames={play.frames} t={t} losX={play.los_x} cutoff={hz?.latest_frame ?? 0} selected={selected} onSelect={setSelected} showTrails={trails} highlight={evidence} labels={labels} />
                  <button ref={expandBtn} onClick={() => setExpanded(true)} className="narrow absolute right-2 top-2 border border-ink/40 bg-bg/70 px-3 text-sm backdrop-blur hover:border-ink">Expand field</button>
                </div>
                {evidence && <p className="mt-2 flex items-start justify-between gap-3 border border-amber/50 px-3 py-1.5 text-xs text-amber"><span>Highlighted: {evidence.note}. Amber paths run from the snap to the model&apos;s cutoff; dashed lines join the pairs at the cutoff.</span><button className="underline" onClick={() => setEvidence(null)}>Clear</button></p>}
                {controls}
                <p className="mt-1 text-xs text-muted">
                  <span className="mr-3">○ route runner</span><span className="mr-3">● passer</span><span className="mr-3 text-defense">✕ coverage defender</span>
                  <Badge kind="Observed" /> {SCOPE} No ball, linemen, rushers or pre-snap frames exist in this source, and none are drawn. Space plays, arrow keys step a frame.
                </p>
                <label className="mt-2 flex items-center gap-2 text-sm text-muted">Inspect a player
                  <select value={selected ?? ""} onChange={(e) => setSelected(e.target.value || null)} className="border border-line bg-surface px-2 text-ink">
                    <option value="">None</option>
                    {play.entities.filter((e) => e.id).map((e) => <option key={e.id} value={e.id!}>{e.side === "defense" ? "Defense" : "Offense"}: {e.name ?? e.id} ({e.position ?? "?"})</option>)}
                  </select>
                </label>
                {sel && <p className="num mt-2 border border-line bg-surface px-3 py-2 text-sm"><Badge kind="Observed" /> <strong>{sel.name ?? "Unnamed player"}</strong> {sel.position}, {sel.side}. {sel.s[frameIdx] != null ? `${sel.s[frameIdx]!.toFixed(1)} yards per second at this frame.` : "No observation at this frame."}</p>}
              </div>
            )}
          </section>

          <aside data-tour="coverage-evidence" aria-label="Prediction and evidence" className="order-2 min-w-0 lg:order-3">
            <div role="tablist" aria-label="Evidence panels" className="flex border-b border-line">
              {TABS.map((x) => (
                <button key={x.key} role="tab" id={`tab-${x.key}`} aria-selected={tab === x.key} aria-controls={`panel-${x.key}`} tabIndex={tab === x.key ? 0 : -1} onClick={() => setTab(x.key)}
                  onKeyDown={(e) => { const i = TABS.findIndex((y) => y.key === tab); if (e.key === "ArrowRight") { const n = TABS[(i + 1) % 3].key; setTab(n); document.getElementById(`tab-${n}`)?.focus(); } if (e.key === "ArrowLeft") { const n = TABS[(i + 2) % 3].key; setTab(n); document.getElementById(`tab-${n}`)?.focus(); } }}
                  className={`narrow -mb-px border-b-2 px-4 py-2 text-lg transition-colors duration-150 ${tab === x.key ? "border-teal text-teal" : "border-transparent text-muted hover:text-ink"}`}>{x.label}</button>
              ))}
            </div>

            {tab === "evidence" && (
              <div role="tabpanel" id="panel-evidence" aria-labelledby="tab-evidence" className="fade-swap pt-4">
                {play && !hz && <p className="border border-line p-3 text-sm text-muted">The ball was thrown {(play.frames_observed / 10).toFixed(1)} seconds after the snap, before the {hLabel.label} cutoff. There is no prediction at this cutoff and nothing is extrapolated. Choose an earlier cutoff.</p>}
                {play && hz && !main && <p className="border border-line p-3 text-sm text-muted">{spec.short} has no saved prediction for this play at {hLabel.label}.</p>}
                {play && hz && main && playStatus && (
                  <>
                    <p className="sr-only-live" aria-live="polite">{play.offense} offense against {play.defense}. Released label {play.released.manZone}. {spec.short} at {hLabel.label}: {STATUS_TEXT[playStatus]}, leans {main.predicted}, {pct(main.p_man)} man.</p>
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="kicker"><Badge kind="Predicted" /> model prediction</p>
                        <p className="display mt-1 text-6xl" data-testid="lean">Leans {main.predicted}</p>
                        <p className={`mt-1 text-sm font-semibold ${STATUS_CLASS[playStatus]}`} data-testid="status">{STATUS_TEXT[playStatus]}</p>
                      </div>
                      <p className="text-right"><span className="display block text-4xl num">{pct(main.confidence)}</span><span className="kicker">confidence</span></p>
                    </div>
                    <p className="num mt-1 text-xs text-muted" data-testid="confidence">Confidence {pct(main.confidence, 1)}, cutoff {pct(main.confidence_threshold)}. {main.accepted ? "At or above the cutoff, so the lean counts as a prediction." : "Below the cutoff: the model abstains and the lean is not counted."}</p>
                    <div className="num mt-3 space-y-2 text-sm" data-testid="probs">
                      {(["Man", "Zone"] as const).map((c) => { const v = c === "Man" ? main.p_man : main.p_zone; return <div key={c} className="grid grid-cols-[3rem_1fr_3.5rem] items-center gap-2"><span>{c}</span><Bar value={v} tone={c === "Man" ? "teal" : "defense"} /><span className="text-right">{pct(v, 1)}</span></div>; })}
                      <p className="text-xs text-muted">Calibrated probability that the released label is man or zone, from {spec.label}. Shown whether or not the model abstains.</p>
                    </div>
                    <div className="mt-4 border border-line p-3">
                      <p className="kicker"><Badge kind="Released label" /></p>
                      <p className="narrow mt-1 text-2xl" ><span data-testid="label">{play.released.manZone}</span>{play.released.coverage ? <span className="text-muted"> · {TYPE_LABELS[play.released.coverage] ?? play.released.coverage}</span> : null}</p>
                      <p className="text-xs text-muted">{play.released.source}</p>
                    </div>

                    <section aria-labelledby="h-time" className="mt-5">
                      <h2 id="h-time" className="narrow text-lg">Prediction change as the model sees more</h2>
                      <table className="num mt-1 w-full text-xs" data-testid="prefix">
                        <thead><tr className="text-left text-muted"><th className="py-1 font-normal">Model input cutoff</th><th className="font-normal">Man probability</th><th className="font-normal">State</th></tr></thead>
                        <tbody>
                          {HORIZONS.filter((w) => modelHorizons.includes(w.key)).map((w) => {
                            const p = play.horizons[w.key]?.predictions[model];
                            return (
                              <tr key={w.key} className={`border-t border-line ${w.key === horizon ? "bg-surface" : ""}`}>
                                <td className="py-1.5">{w.label}</td>
                                <td>{p ? <span className="flex items-center gap-2"><span className="inline-block h-2 bg-teal transition-[width] duration-200" style={{ width: `${p.p_man * 60}px` }} />{pct(p.p_man, 1)}</span> : "thrown before this cutoff"}</td>
                                <td className={p ? STATUS_CLASS[statusOf(p, play.released.manZone)] : "text-muted"}>{p ? STATUS_TEXT[statusOf(p, play.released.manZone)] : ""}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                      <p className="mt-1 text-xs text-muted">Each row is a separately saved output with its own calibrator. Nothing is filled in between cutoffs.</p>
                    </section>

                    <section aria-labelledby="h-expl" className="mt-5">
                      <h2 id="h-expl" className="narrow text-lg"><Badge kind="Derived" /> Model explanation</h2>
                      <p className="text-xs text-muted">Select a row to highlight the players it is measured from.</p>
                      {Array.isArray(expl) ? (
                        <table className="num mt-2 w-full text-xs" data-testid="explanation" data-model={model}>
                          <thead><tr className="text-left text-muted"><th className="py-1 pr-2 font-normal">Measurement</th><th className="pr-2 text-right font-normal">Measured</th><th className="text-right font-normal">Sensitivity</th></tr></thead>
                          <tbody>
                            {(expl as Term[]).slice(0, 7).map((c) => {
                              const d = ds.index.feature_descriptions[c.feature] ?? { text: c.feature, evidence: "all" as const };
                              return (
                                <tr key={c.feature} className="border-t border-line align-top">
                                  <td className="py-1.5 pr-2"><button className="min-h-0 text-left text-ink underline decoration-line underline-offset-2 hover:decoration-amber" onClick={() => showEvidence(d.evidence, d.text)}>{d.text}</button></td>
                                  <td className="py-1.5 pr-2 text-right text-ink">{c.value ?? "n/a"}</td>
                                  <td className={`py-1.5 text-right ${c.log_odds_toward_man >= 0 ? "text-teal" : "text-defense"}`}>{c.log_odds_toward_man >= 0 ? "Man" : "Zone"} {Math.abs(c.log_odds_toward_man).toFixed(2)}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      ) : expl ? (
                        <div data-testid="explanation" data-model={model}>
                          <table className="num mt-2 w-full text-xs">
                            <thead><tr className="text-left text-muted"><th className="py-1 pr-2 font-normal">Input group replaced by its average</th><th className="text-right font-normal">Sensitivity</th></tr></thead>
                            <tbody>{(expl as Ablation).groups.map((g) => <tr key={g.group} className="border-t border-line"><td className="py-1.5 pr-2 text-ink">{g.group}</td><td className={`text-right ${g.log_odds_toward_man >= 0 ? "text-teal" : "text-defense"}`}>{g.log_odds_toward_man >= 0 ? "Man" : "Zone"} {Math.abs(g.log_odds_toward_man).toFixed(2)}</td></tr>)}</tbody>
                          </table>
                          <table className="num mt-3 w-full text-xs">
                            <thead><tr className="text-left text-muted"><th className="py-1 pr-2 font-normal">Player removed from the input</th><th className="text-right font-normal">Sensitivity</th></tr></thead>
                            <tbody>
                              {[...(expl as Ablation).players].sort((a, b) => Math.abs(b.log_odds_toward_man) - Math.abs(a.log_odds_toward_man)).slice(0, 5).map((p) => (
                                <tr key={p.id} className="border-t border-line"><td className="py-1.5 pr-2"><button className="min-h-0 text-left text-ink underline decoration-line underline-offset-2 hover:decoration-amber" onClick={() => setEvidence({ players: [p.id], links: [], note: nameOf(p.id) })}>{nameOf(p.id)}</button></td>
                                  <td className={`text-right ${p.log_odds_toward_man >= 0 ? "text-teal" : "text-defense"}`}>{p.log_odds_toward_man >= 0 ? "Man" : "Zone"} {Math.abs(p.log_odds_toward_man).toFixed(2)}</td></tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : <p className="mt-1 text-xs">No explanation exported for this model at this cutoff.</p>}
                      <p className="mt-1 text-xs text-muted">Sensitivity is in log-odds toward man, before calibration. {spec.explanation} It describes the model, not what the defense intended.</p>
                    </section>

                    <section aria-labelledby="h-pairs" className="mt-5">
                      <h2 id="h-pairs" className="narrow text-lg"><Badge kind="Observed" /> Pairings at the cutoff</h2>
                      <table className="num mt-1 w-full text-xs">
                        <thead><tr className="text-left text-muted"><th className="py-1 pr-2 font-normal">Closest one-to-one pairing</th><th className="pr-2 text-right font-normal">Yards apart</th><th className="text-right font-normal">At snap</th></tr></thead>
                        <tbody>
                          {hz.evidence.pairs.map((p) => (
                            <tr key={p.defender + p.receiver} className="border-t border-line"><td className="py-1.5 pr-2"><button className="min-h-0 text-left text-ink underline decoration-line underline-offset-2 hover:decoration-amber" onClick={() => setEvidence({ players: [p.defender, p.receiver], links: [[p.defender, p.receiver]], note: `${nameOf(p.defender)} and ${nameOf(p.receiver)}` })}>{nameOf(p.defender)} with {nameOf(p.receiver)}</button></td>
                              <td className="pr-2 text-right">{p.separation.toFixed(1)}</td><td className="text-right">{p.separation_at_snap?.toFixed(1)}</td></tr>
                          ))}
                        </tbody>
                      </table>
                      <p className="mt-1 text-xs text-muted">A geometric pairing that minimizes total distance. Not a confirmed matchup or coverage responsibility.</p>
                    </section>

                    <details className="mt-5 border border-line p-3">
                      <summary className="narrow text-lg">How was this calculated?</summary>
                      <dl className="mt-3 space-y-2.5 text-sm text-muted">
                        <div><dt className="kicker">Input</dt><dd>Released tracking for {play.entities.length} players, 10 frames per second, frames 0 to {hz.latest_frame} ({hLabel.detail}).</dd></div>
                        <div><dt className="kicker">Representation</dt><dd>The play is rotated so the offense moves up the field; positions are measured from the line of scrimmage and the formation centre. {model === "v2_temporal" ? "Every defender-receiver pair is described frame by frame and read by a small recurrent model." : model === "v2_gbm_rel" ? "Shell geometry plus defender-to-receiver relational measurements, summarized over players." : "Shell geometry summarized over players."}</dd></div>
                        <div><dt className="kicker">Model version</dt><dd className="mono break-all text-xs" data-testid="version">{main.version}</dd></div>
                        <div><dt className="kicker">Probabilities</dt><dd>Man {pct(main.p_man, 1)}, zone {pct(main.p_zone, 1)}, from a Platt calibrator fitted on weeks 13–14 for this model and cutoff.</dd></div>
                        <div><dt className="kicker">Policy</dt><dd>{ds.index.decision_rule}</dd></div>
                        <div><dt className="kicker">Same play, same cutoff, every model</dt>
                          <dd><table className="num mt-1 w-full text-xs"><tbody>
                            {MODELS.map((m) => { const p = hz.predictions[m]; return <tr key={m} className={`border-t border-line ${m === model ? "text-ink" : ""}`}><td className="py-1.5">{ds.index.models[m].short}{m === model ? " (shown)" : ""}</td><td className="text-right">{p ? `${pct(p.p_man, 1)} man` : "not trained for this cutoff"}</td></tr>; })}
                          </tbody></table></dd></div>
                        <div><dt className="kicker">Evaluation context</dt><dd><Link className="text-teal underline" href={`/coverage/evaluation?${carry}`}>Results for this model and cutoff</Link>. {ds.index.benchmark_note}</dd></div>
                      </dl>
                    </details>
                  </>
                )}
              </div>
            )}

            {tab === "similar" && (
              <div role="tabpanel" id="panel-similar" aria-labelledby="tab-similar" className="fade-swap pt-4">
                {!play || play.similar.plays.length === 0 ? <p className="text-sm text-muted">No similar plays for this play: the comparison needs 1.5 seconds of tracking.</p> : (
                  <>
                    <p className="text-xs text-muted">Pool: {play.similar.pool}. Measured by the {play.similar.basis}. {play.similar.note}</p>
                    <div className="mt-3 space-y-4">
                      {play.similar.plays.map((s) => (
                        <figure key={s.id}>
                          <MiniField play={s} />
                          <figcaption className="num mt-1 text-xs text-muted"><span className="text-ink">{s.offense} at {s.defense}</span>, week {s.week}, {ordinal(s.down)} and {s.yardsToGo}. <Badge kind="Released label" /> {s.released.manZone}{s.released.coverage ? `, ${TYPE_LABELS[s.released.coverage] ?? s.released.coverage}` : ""}. Distance {s.distance.toFixed(2)}</figcaption>
                        </figure>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}

            {tab === "errors" && (
              <div role="tabpanel" id="panel-errors" aria-labelledby="tab-errors" className="fade-swap pt-4 text-sm">
                <p className="text-muted">{spec.short} at {hLabel.label} leans against the released label on <strong className="text-ink">{available}</strong> of {ds.index.eligible_test_plays.toLocaleString()} benchmark plays. In this view: <span className={STATUS_CLASS.incorrect}>{counts.incorrect} accepted and disagree</span>, <span className={STATUS_CLASS.abstained}>{counts.abstained} abstained</span>.</p>
                <ul className="mt-3 border-t border-line">
                  {errorPlays.map((p) => { const l = p.predictions[model]?.[horizon]; return (
                    <li key={p.id} className="border-b border-line"><Link className="block px-1 py-2 hover:bg-surface" href={`/coverage/errors?${carry}&play=${p.id}`}>
                      <span className="narrow text-base">{p.offense} at {p.defense}</span> <span className="mono text-[11px] text-muted">WK {p.week}</span>
                      <span className="num block text-xs text-muted">Label {p.released.manZone}{l ? `, model leans ${l.predicted} (${pct(l.p_man)} man)` : ""}</span></Link></li>); })}
                </ul>
                <p className="mt-3"><Link className="btn-quiet" href={`/coverage/errors?${carry}`}>Open the error gallery</Link></p>
                <p className="mt-3 text-xs text-muted">{ds.index.errors.note}</p>
              </div>
            )}
          </aside>
        </div>
      </div>

      {expanded && play && (
        <div role="dialog" aria-modal="true" aria-label="Expanded field" className="fade-swap fixed inset-0 z-50 overflow-y-auto bg-bg p-4">
          <div className="mx-auto max-w-[1100px]">
            <div className="mb-2 flex items-center justify-between gap-3">
              <p className="display text-2xl">{play.offense} vs {play.defense} <span className="mono text-xs normal-case text-muted">model input up to {hLabel.label}</span></p>
              <button ref={closeBtn} onClick={closeExpanded} className="narrow border border-line px-3 text-base hover:border-ink">Close</button>
            </div>
            <Field entities={play.entities} frames={play.frames} t={t} losX={play.los_x} cutoff={hz?.latest_frame ?? 0} selected={selected} onSelect={setSelected} showTrails={trails} highlight={evidence} labels={labels} />
            {controls}
          </div>
        </div>
      )}
    </div>
  );
}
