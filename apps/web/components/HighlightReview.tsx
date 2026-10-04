"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

// Local review tool. Annotations stay in this browser (localStorage) until exported as a file.
// Without authorized video, a reviewer can only record what transcripts, loudness and play-by-play support,
// so every annotation carries its evidence basis and an uncertainty level. Nothing here is fed back into the benchmark labels.

type Candidate = { start_clip: number; end_clip: number; start_s: number; end_s: number; source_time_s: number; score_rank: number; excerpt: string; in_editorial_highlights: boolean };
type Game = { id: number; title: string; candidates: Candidate[]; source: { full_video: string; trim_start_s: number } };
type Note = { events: string[]; basis: string; uncertainty: string; replay: string; event_moment_s: string; clip_start_s: string; clip_end_s: string; comment: string };
const EVENTS = ["touchdown", "interception", "sack", "fumble", "turnover", "big play", "none of these"];
const BASIS = ["transcript only", "transcript and play-by-play", "authorized video viewed", "not reviewable"];
const KEY = "gridiron-lens-highlight-review-v1";
const clock = (s: number) => `${Math.floor(s / 3600)}:${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const blank = (c: Candidate): Note => ({ events: [], basis: "transcript only", uncertainty: "unsure", replay: "unknown", event_moment_s: "", clip_start_s: String(c.start_s), clip_end_s: String(c.end_s), comment: "" });

export default function HighlightReview() {
  const params = useSearchParams();
  const [games, setGames] = useState<{ id: number; title: string }[]>([]);
  const [gid, setGid] = useState<number | null>(params.get("game") ? Number(params.get("game")) : null);
  const [game, setGame] = useState<Game | null>(null);
  const [notes, setNotes] = useState<Record<string, Note>>({});
  const [ready, setReady] = useState(false);

  useEffect(() => {
    fetch("/demo/highlights/index.json").then((r) => r.json()).then((d) => {
      setGames(d.games);
      setGid((g) => g ?? d.games[0]?.id ?? null);
      try { const raw = window.localStorage.getItem(KEY); if (raw) setNotes(JSON.parse(raw)); } catch { /* storage unavailable: annotations last for this visit only */ }
      setReady(true);
    });
  }, []);
  useEffect(() => { if (gid != null) fetch(`/demo/highlights/games/${gid}.json`).then((r) => r.json()).then(setGame); }, [gid]);
  useEffect(() => { if (ready) { try { window.localStorage.setItem(KEY, JSON.stringify(notes)); } catch { /* ignore */ } } }, [notes, ready]);

  const total = games.length * 40;
  const done = Object.values(notes).filter((n) => n.events.length > 0).length;
  const update = (k: string, c: Candidate, patch: Partial<Note>) => setNotes((prev) => ({ ...prev, [k]: { ...(prev[k] ?? blank(c)), ...patch } }));
  const download = () => {
    const blob = new Blob([JSON.stringify({ annotation_version: "review-v1", exported_at: new Date().toISOString(), note: "Reviewer annotations. Separate from the benchmark's editorial labels, which are never changed.", annotations: notes }, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "highlight_review_annotations.json"; a.click(); URL.revokeObjectURL(a.href);
  };

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-5 p-4 lg:p-6">
      <header>
        <h1 className="display text-4xl">Candidate review</h1>
        <p className="mt-3 max-w-[72ch] text-muted">Up to 40 model-selected intervals per game across {games.length} games ({total} candidates), including ones the editors left out. Record event labels, the evidence you had, how sure you are, the event moment and better clip boundaries. Annotations are stored in this browser and can be saved as a file.</p>
        <p className="mt-2 max-w-[72ch] border-l-2 border-amber pl-3 text-sm text-muted">No video is available here. Labels made from transcripts and play-by-play alone must say so, and they do not count as visually verified. The benchmark&apos;s editorial labels are never edited.</p>
      </header>
      <div className="flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2 text-muted">Game
          <select value={gid ?? ""} onChange={(e) => setGid(Number(e.target.value))} className="max-w-[70vw] border border-line bg-surface px-2 text-ink">{games.map((g) => <option key={g.id} value={g.id}>{g.title.slice(0, 70)}</option>)}</select>
        </label>
        <span className="num text-muted">{done} of {total} candidates labelled</span>
        <button onClick={download} className="narrow bg-teal px-4 font-semibold text-bg">Save annotations as a file</button>
      </div>
      {!game ? <p className="text-muted">Loading candidates…</p> : (
        <ol className="border-t border-line">
          {game.candidates.slice(0, 40).map((c) => {
            const k = `${game.id}:${c.start_clip}`;
            const n = notes[k] ?? blank(c);
            return (
              <li key={k} className="grid gap-3 border-b border-line py-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
                <div className="text-sm">
                  <p className="num"><span className="font-semibold">{clock(c.start_s)} to {clock(c.end_s)}</span> <span className="text-muted">score rank {c.score_rank}; {c.in_editorial_highlights ? "in the editors' reel" : "not in the editors' reel"}</span></p>
                  <p className="mt-1 text-muted">{c.excerpt ? `“${c.excerpt}”` : "No commentary transcribed here."}</p>
                  <a className="mt-1 inline-block text-teal underline" href={`${game.source.full_video}&t=${Math.floor(c.source_time_s)}s`} target="_blank" rel="noreferrer">Open the public source video at {clock(c.source_time_s)}</a>
                </div>
                <fieldset className="grid gap-2 text-sm">
                  <legend className="sr-only">Annotation for the interval at {clock(c.start_s)}</legend>
                  <div className="flex flex-wrap gap-x-3 gap-y-1">
                    {EVENTS.map((ev) => (
                      <label key={ev} className="flex items-center gap-1.5"><input type="checkbox" checked={n.events.includes(ev)} onChange={(e) => update(k, c, { events: e.target.checked ? [...n.events, ev] : n.events.filter((x) => x !== ev) })} />{ev}</label>
                    ))}
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <label className="flex items-center gap-1 text-muted">Evidence<select value={n.basis} onChange={(e) => update(k, c, { basis: e.target.value })} className="border border-line bg-surface px-1 text-ink">{BASIS.map((b) => <option key={b}>{b}</option>)}</select></label>
                    <label className="flex items-center gap-1 text-muted">Certainty<select value={n.uncertainty} onChange={(e) => update(k, c, { uncertainty: e.target.value })} className="border border-line bg-surface px-1 text-ink">{["sure", "fairly sure", "unsure"].map((b) => <option key={b}>{b}</option>)}</select></label>
                    <label className="flex items-center gap-1 text-muted">Replay<select value={n.replay} onChange={(e) => update(k, c, { replay: e.target.value })} className="border border-line bg-surface px-1 text-ink">{["live play", "replay", "unknown"].map((b) => <option key={b}>{b}</option>)}</select></label>
                  </div>
                  <div className="flex flex-wrap gap-2 text-muted">
                    <label className="flex items-center gap-1">Event at (s)<input value={n.event_moment_s} onChange={(e) => update(k, c, { event_moment_s: e.target.value })} inputMode="numeric" className="w-20 border border-line bg-surface px-1 py-1.5 text-ink" /></label>
                    <label className="flex items-center gap-1">Clip start (s)<input value={n.clip_start_s} onChange={(e) => update(k, c, { clip_start_s: e.target.value })} inputMode="numeric" className="w-20 border border-line bg-surface px-1 py-1.5 text-ink" /></label>
                    <label className="flex items-center gap-1">Clip end (s)<input value={n.clip_end_s} onChange={(e) => update(k, c, { clip_end_s: e.target.value })} inputMode="numeric" className="w-20 border border-line bg-surface px-1 py-1.5 text-ink" /></label>
                  </div>
                  <label className="flex items-center gap-1 text-muted">Note<input value={n.comment} onChange={(e) => update(k, c, { comment: e.target.value })} className="flex-1 border border-line bg-surface px-2 py-1.5 text-ink" /></label>
                </fieldset>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
