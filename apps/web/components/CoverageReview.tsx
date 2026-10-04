"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import Field from "@/components/Field";
import { Badge } from "@/components/ui";
import { type Entity, HORIZONS, type HorizonKey, TYPE_LABELS, pct } from "@/lib/demo";

type Play = { key: string; queue: string; week: number; defense: string; released_label: string; released_coverage_type: string | null; p_man: Record<HorizonKey, number | null>;
  lean_at_last_cutoff: string; suggested_look: string[]; geometry: Record<string, number | boolean>; frames: number[]; los_x: number; entities: Entity[] };
type Review = { purpose: string; model: string; status: string; verdicts: string[]; suggestion_note: string; display_note: string; queues: Record<string, number>; plays: Play[] };
type Slice = { slice: string; plays: number; error_rate: number; log_loss: number; man_recall: number | null };
type Slices = { predictions: string; split_role: string; plays: number; note: string; overall_by_cutoff: Record<HorizonKey, { plays: number; error_rate: number }>; slices_at_last_available_cutoff: Record<string, Slice[]> };
type Note = { verdict: string; note: string; at: string };

const KEY = "gridiron-coverage-review:v3";
const QUEUE_TEXT: Record<string, string> = { confident_disagreements: "Confident disagreements", uncertain: "Near 50/50", minority_class_misses: "Missed man coverage", random_control: "Random control" };
const get = <T,>(u: string) => fetch(u).then((r) => (r.ok ? (r.json() as Promise<T>) : null)).catch(() => null);

export default function CoverageReview() {
  const [d, setD] = useState<Review | null | undefined>(undefined);
  const [sl, setSl] = useState<Slices | null>(null);
  const [queue, setQueue] = useState("confident_disagreements");
  const [open, setOpen] = useState<string | null>(null);
  const [t, setT] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const [notes, setNotes] = useState<Record<string, Note>>(() => {
    if (typeof window === "undefined") return {};
    try { return JSON.parse(localStorage.getItem(KEY) ?? "{}"); } catch { return {}; }           // storage unavailable: annotations stay in memory
  });
  const [table, setTable] = useState("released family");
  useEffect(() => {
    get<Review>("/demo/real/coverage/review_v3.json").then((x) => setD(x));
    get<Slices>("/demo/real/coverage/error_slices_v3.json").then(setSl);
  }, []);
  const list = useMemo(() => d?.plays.filter((p) => p.queue === queue) ?? [], [d, queue]);
  const play = list.find((p) => p.key === open) ?? list[0];
  function save(next: Record<string, Note>) {
    setNotes(next);
    try { localStorage.setItem(KEY, JSON.stringify(next)); } catch { /* keep in memory */ }
  }
  function exportJson() {
    const body = { exported_at: new Date().toISOString(), queue_version: "coverage-review-v3", note: "Reviewer annotations. They never change the released labels or the benchmark.", annotations: notes };
    const url = URL.createObjectURL(new Blob([JSON.stringify(body, null, 1)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url; a.download = "coverage_review_annotations.json"; a.click();
    URL.revokeObjectURL(url);
  }
  if (d === undefined) return <p className="p-6 text-muted">Loading the review queue…</p>;
  if (d === null) return <p className="p-6 text-muted">No review export found. Run the coverage review export.</p>;
  const last = play ? [...HORIZONS].reverse().find((h) => play.p_man[h.key] != null) : undefined;
  const done = Object.keys(notes).length;

  return (
    <div className="mx-auto flex max-w-[1500px] flex-col gap-6 px-4 py-8 lg:px-8">
      <header>
        <p className="kicker">Coverage / error review</p>
        <h1 className="display mt-1 text-5xl sm:text-6xl">Second look<span className="text-teal">.</span></h1>
        <p className="mt-3 max-w-[78ch] text-muted">{d.purpose}</p>
        <p className="mt-2 max-w-[78ch] border-l-2 border-amber pl-3 text-sm text-amber" data-testid="review-status">
          {done === 0 ? `This queue is ${d.status}.` : `${done} of ${d.plays.length} plays annotated in this browser.`} Annotations here are yours: they stay in this browser until you export them, and they are not published results.
        </p>
        <p className="mt-2 text-sm"><Link className="text-teal underline" href="/coverage/evaluation">Back to the evaluation</Link></p>
      </header>

      <div className="grid gap-5 lg:grid-cols-[260px_minmax(0,1fr)_340px]">
        <nav aria-label="Review queue" className="min-w-0 text-sm">
          <label className="block"><span className="kicker">Queue</span>
            <select value={queue} onChange={(e) => { setQueue(e.target.value); setOpen(null); setT(0); }} className="mt-1 block w-full border border-line bg-surface p-2">
              {Object.entries(d.queues).map(([k, n]) => <option key={k} value={k}>{QUEUE_TEXT[k] ?? k} ({n})</option>)}</select></label>
          <ul className="mt-2 max-h-[32rem] overflow-y-auto border-t border-line">
            {list.map((p) => (
              <li key={p.key}><button onClick={() => { setOpen(p.key); setT(0); setSelected(null); }} aria-current={play?.key === p.key ? "true" : undefined}
                className={`w-full border-b border-line px-2 py-2 text-left hover:bg-surface ${play?.key === p.key ? "bg-surface" : ""}`}>
                <span className="narrow block text-lg">{p.defense} defense · wk {p.week}</span>
                <span className="mono block text-[11px] uppercase text-muted">Label {p.released_label} · leans {p.lean_at_last_cutoff}</span>
                <span className={`block text-xs ${notes[p.key] ? "text-teal" : "text-muted"}`}>{notes[p.key]?.verdict ?? "Not reviewed"}</span>
              </button></li>))}
          </ul>
        </nav>

        {play && (
          <section aria-label="Play" className="min-w-0">
            <Field entities={play.entities} frames={play.frames} t={Math.min(t, play.frames.length - 1)} losX={play.los_x} cutoff={play.frames.length - 1} selected={selected} onSelect={setSelected} showTrails />
            <label className="mt-2 block text-sm text-muted">Frame <span className="mono num text-ink">+{(t / 10).toFixed(1)} s</span>
              <input type="range" min={0} max={play.frames.length - 1} value={Math.min(t, play.frames.length - 1)} onChange={(e) => setT(Number(e.target.value))} className="block w-full" /></label>
            <p className="text-xs text-muted"><Badge kind="Observed" /> {d.display_note} Tracking covers the route runners and the coverage defenders the release selected after the play.</p>
          </section>
        )}

        {play && (
          <aside aria-label="Evidence and annotation" className="min-w-0 text-sm">
            <div className="grid grid-cols-2 gap-3">
              <div className="border border-line p-3"><p className="kicker"><Badge kind="Released label" /></p><p className="narrow mt-1 text-2xl">{play.released_label}</p>
                <p className="text-xs text-muted">{play.released_coverage_type ? TYPE_LABELS[play.released_coverage_type] ?? play.released_coverage_type : "family not given"}</p></div>
              <div className="border border-line p-3"><p className="kicker"><Badge kind="Predicted" /></p><p className="narrow mt-1 text-2xl">Leans {play.lean_at_last_cutoff}</p>
                <p className="num text-xs text-muted">{last ? `${pct(play.p_man[last.key]!, 1)} man at ${last.label}` : ""}</p></div>
            </div>
            <table className="num mt-3 w-full text-xs"><caption className="kicker pb-1 text-left">Man probability as the model sees more</caption><tbody>
              {HORIZONS.map((h) => <tr key={h.key} className="border-t border-line"><td className="py-1">{h.label}</td><td className="text-right">{play.p_man[h.key] == null ? "play ended before this cutoff" : pct(play.p_man[h.key]!, 1)}</td></tr>)}
            </tbody></table>
            <p className="kicker mt-3"><Badge kind="Derived" /> automated things to look at</p>
            <ul className="mt-1 list-disc space-y-1 pl-5 text-xs text-muted">{play.suggested_look.map((s) => <li key={s}>{s}</li>)}</ul>
            <p className="mt-1 text-xs text-muted">{d.suggestion_note}</p>

            <fieldset className="mt-4 border border-line p-3" data-testid="review-form"><legend className="narrow px-1 text-lg font-semibold">Your judgement</legend>
              {d.verdicts.map((v) => (
                <label key={v} className="mt-1 flex gap-2"><input type="radio" name="verdict" checked={notes[play.key]?.verdict === v}
                  onChange={() => save({ ...notes, [play.key]: { verdict: v, note: notes[play.key]?.note ?? "", at: new Date().toISOString() } })} /><span className="first-letter:uppercase">{v}</span></label>))}
              <label className="mt-2 block"><span className="kicker">Notes: what you saw, how sure you are</span>
                <textarea rows={3} value={notes[play.key]?.note ?? ""} disabled={!notes[play.key]} placeholder={notes[play.key] ? "" : "Choose a judgement first"}
                  onChange={(e) => save({ ...notes, [play.key]: { ...notes[play.key], note: e.target.value, at: new Date().toISOString() } })} className="mt-1 block w-full border border-line bg-surface p-2 text-sm" /></label>
              {notes[play.key] && <button className="btn-quiet mt-2" onClick={() => { const n = { ...notes }; delete n[play.key]; save(n); }}>Clear this annotation</button>}
            </fieldset>
            <p className="mt-3 flex flex-wrap gap-4"><button className="btn-quiet" onClick={exportJson} disabled={done === 0}>Export annotations as JSON</button></p>
            <p className="mt-2 text-xs text-muted">A disagreement is not proof that the label is wrong. The released labels and benchmark are never changed from here.</p>
          </aside>
        )}
      </div>

      {sl && (
        <section aria-labelledby="rv-slices" className="border-t border-line pt-6" data-testid="error-slices">
          <h2 id="rv-slices" className="display text-3xl sm:text-4xl">Where the development errors are</h2>
          <p className="mt-2 max-w-[78ch] text-sm text-muted">{sl.plays.toLocaleString()} plays. {sl.predictions}. {sl.split_role} {sl.note}</p>
          <p className="num mt-2 text-sm text-muted">Error rate by cutoff: {HORIZONS.map((h) => `${h.label} ${pct(sl.overall_by_cutoff[h.key].error_rate, 1)} of ${sl.overall_by_cutoff[h.key].plays.toLocaleString()}`).join(" · ")}</p>
          <label className="mt-3 flex w-fit items-center gap-2 text-sm text-muted">Slice by
            <select value={table} onChange={(e) => setTable(e.target.value)} className="border border-line bg-surface px-2 text-ink">{Object.keys(sl.slices_at_last_available_cutoff).map((k) => <option key={k}>{k}</option>)}</select></label>
          <div className="mt-2 overflow-x-auto"><table className="num w-full min-w-[520px] max-w-3xl text-sm">
            <thead><tr className="text-left text-muted"><th className="py-1.5 pr-3 font-normal">{table}</th><th className="pr-3 font-normal">Plays</th><th className="pr-3 font-normal">Error rate</th><th className="pr-3 font-normal">Log loss</th><th className="font-normal">Man recall</th></tr></thead>
            <tbody>{sl.slices_at_last_available_cutoff[table].map((r) => (
              <tr key={r.slice} className="border-t border-line"><td className="py-1.5 pr-3">{TYPE_LABELS[r.slice] ?? (r.slice === "True" ? "Yes" : r.slice === "False" ? "No" : r.slice)}</td><td className="pr-3">{r.plays.toLocaleString()}</td>
                <td className="pr-3">{pct(r.error_rate, 1)}</td><td className="pr-3">{r.log_loss.toFixed(3)}</td><td>{r.man_recall == null ? "n/a" : pct(r.man_recall, 1)}</td></tr>))}</tbody>
          </table></div>
        </section>
      )}
    </div>
  );
}
