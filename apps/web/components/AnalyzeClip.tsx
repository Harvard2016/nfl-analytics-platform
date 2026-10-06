"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import ModuleTour from "@/components/ModuleTour";
import { BackendNotice, JobStatus, useBackend, useJob } from "@/components/UploadShell";
import { Badge } from "@/components/ui";
import { mediaUrl, result, submit } from "@/lib/inference";
import { type ClipEdit, initialClipEdits, reviewedCandidates } from "@/lib/clipReview";
import { type AudioTimeline, audioExtent, binBounds, timelineSeek } from "@/lib/clipTimeline";

type Cand = { asset: string; rank_in_reel: number; candidate_moment_s: number; clip_start_s: number; clip_end_s: number; ffprobe_duration_s: number; loudness_above_background_db: number };
type Result = {
  mode: string; mode_note: string; score_meaning: string; domain_shift: string; time_origin: string; clip_seconds: number; event_types: string; evidence_streams_used: string[];
  media: { duration_s: number; has_video: boolean };
  timeline: AudioTimeline & { loudness_above_background_db: number[]; rank_within_file: number[]; ranked_by?: string };
  commentary?: { model: string; word_model: string; words: number; note: string; transcription_seconds: number; segments: { start_s: number; end_s: number; text: string }[]; sounds: { start_s: number; end_s: number; label: string }[] } | null;
  candidates: Cand[]; reel: { asset: string; ffprobe_duration_s: number; budget_s: number; decoder_output_s: number } | null;
  measured: { seconds: number; peak_memory_mb: number };
};

const MODE_TEXT: Record<string, string> = { loudness_baseline: "Loudness baseline", commentary_experimental: "Commentary words", trained_multimodal: "Trained multimodal ranker" };
const MODE_NOTE: Record<string, string> = { loudness_baseline: "Ranks clips by how loud they are against their surroundings.", commentary_experimental: "Transcribes speech locally, then ranks clips with a word model trained on NFL broadcast commentary. Shows the transcript beside the video." };
const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}.${Math.floor((s % 1) * 10)}`;

export default function AnalyzeClip() {
  const backend = useBackend();
  const [file, setFile] = useState<File | null>(null);
  const [reel, setReel] = useState(60);
  const [mode, setMode] = useState("loudness_baseline");
  const [res, setRes] = useState<Result | null>(null);
  const [edits, setEdits] = useState<Record<string, ClipEdit>>({});
  const [now, setNow] = useState(0);
  const [active, setActive] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const stopAt = useRef<number | null>(null);
  const selection = useRef(0);
  const job = useJob(async (tk, isCurrent) => {
    const r = await result<Result>(tk);
    if (!isCurrent()) return;
    setRes(r);
    setEdits(initialClipEdits(r.candidates));
  });

  // Local preview straight from the chosen file; the object URL is released when the file changes or the page closes.
  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const ready = backend.state === "ready" && backend.caps.modules.highlights.modes.loudness_baseline.ready;
  const dur = res?.media.duration_s ?? 0;
  const extent = res ? audioExtent(res.timeline, res.clip_seconds) : [0, 0];
  const path = useMemo(() => {
    if (!res) return { loud: "", rank: "" };
    const l = res.timeline.loudness_above_background_db, lo = Math.min(...l), hi = Math.max(...l);
    const centre = (i: number) => { const [a, b] = binBounds(res.timeline, i, res.clip_seconds); return (a + b) / 2; };
    return { loud: l.map((v, i) => `${i ? "L" : "M"}${centre(i)},${56 - ((v - lo) / (hi - lo || 1)) * 22}`).join(""),
             rank: res.timeline.rank_within_file.map((v, i) => `${i ? "L" : "M"}${centre(i)},${28 - (v / 100) * 24}`).join("") };
  }, [res]);

  const near = (res?.commentary?.segments ?? []).filter((g) => g.end_s >= now - 20 && g.start_s <= now + 20).slice(0, 8);

  function seek(s: number, until: number | null = null, play = false) {
    const v = video.current;
    if (!v) return;
    v.currentTime = Math.max(0, Math.min(s, v.duration || s));
    stopAt.current = until;
    if (play) v.play().catch(() => { /* the browser may require a click on the player first */ });
  }
  function onTime() {
    const v = video.current;
    if (!v) return;
    setNow(v.currentTime);
    if (stopAt.current != null && v.currentTime >= stopAt.current) { v.pause(); stopAt.current = null; }
  }
  async function go(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    const version = selection.current;
    void job.stop().catch(() => { /* selection is reset even if cancellation is unreachable */ }); job.reset();
    setBusy(true); setRes(null);
    try {
      const tk = await submit("highlights", mode, { reel_seconds: reel, lead_s: 2, tail_s: 2 }, file);
      if (selection.current === version) job.start(tk);
    } catch (err) { if (selection.current === version) job.setError(String((err as Error).message)); }
    if (selection.current === version) setBusy(false);
  }
  function exportJson() {
    if (!res) return;
    const body = { exported_at: new Date().toISOString(), mode: res.mode, time_origin: res.time_origin, score_meaning: res.score_meaning,
      candidates: reviewedCandidates(res.candidates, edits) };
    const url = URL.createObjectURL(new Blob([JSON.stringify(body, null, 1)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url; a.download = "highlight_review.json"; a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div>
      <header className="cinematic-band border-b border-line">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-end justify-between gap-x-10 gap-y-3 px-4 py-6 lg:px-8">
          <div>
            <div className="flex items-center gap-4"><p className="kicker">Highlights / clip ranking</p>{ready && <ModuleTour module="highlights-upload" ready />}</div>
            <h1 className="display mt-1 text-5xl sm:text-7xl">Analyze your <span className="text-amber">clip.</span></h1>
          </div>
          <p className="max-w-[58ch] text-sm text-muted">
            <strong className="font-semibold text-ink">Your file, played beside its evidence.</strong> Use only footage you have permission to process. The working mode is the loudness baseline; the trained multimodal model cannot score new files yet.{" "}
            <Link className="text-teal underline" href="/highlights">Back to the benchmark games</Link>
          </p>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1500px] gap-6 px-4 py-6 lg:grid-cols-[minmax(0,360px)_minmax(0,1fr)] lg:px-8">
        <div className="flex min-w-0 flex-col gap-4">
          <BackendNotice backend={backend} what="Clip ranking" />
          {backend.state === "ready" && (
            <form onSubmit={go} className="border border-line p-4 text-sm" data-tour="upload-form">
              <h2 className="narrow text-2xl font-semibold">Clip</h2>
              <p className="mt-1 text-xs text-muted">Audio or video, {backend.caps.modules.highlights.limits.min_duration_s} seconds to {backend.caps.modules.highlights.limits.max_duration_s / 60} minutes, up to {Math.round(backend.caps.modules.highlights.limits.max_bytes / 2 ** 20)} MB. Full games run from the command line.</p>
              <label className="mt-3 block"><span className="kicker">File</span>
                <input type="file" accept="video/*,audio/*" required disabled={busy} onChange={(e) => {
                  selection.current++; void job.stop().catch(() => { /* selection is reset even if cancellation is unreachable */ }); job.reset(); setFile(e.target.files?.[0] ?? null); setRes(null); setEdits({}); setActive(null); setNow(0); stopAt.current = null;
                }} className="mt-1 block w-full border border-line bg-surface p-2 text-sm" /></label>
              <label className="mt-3 block"><span className="kicker">Reel length (seconds of final output)</span>
                <input type="number" min={4} max={600} step={1} value={reel} onChange={(e) => setReel(Number(e.target.value))} className="num mt-1 block w-28 border border-line bg-surface p-2" /></label>
              <label className="mt-3 flex gap-2 text-xs text-muted"><input type="checkbox" required className="mt-0.5" /><span>I have permission to process this file.</span></label>
              <button className="btn-primary mt-4" style={{ background: "var(--color-amber)" }} disabled={!file || busy || !ready}>Analyze the clip</button>
              <fieldset className="mt-4 border-t border-line pt-3 text-xs" data-testid="clip-modes"><legend className="kicker">Ranking mode</legend>
                {Object.entries(backend.caps.modules.highlights.modes).map(([k, m]) => (
                  <label key={k} className={`mt-2 flex gap-2 ${m.ready ? "" : "opacity-60"}`}><input type="radio" name="mode" value={k} disabled={!m.ready} checked={mode === k} onChange={() => setMode(k)} className="mt-0.5" />
                    <span><span className="text-ink">{MODE_TEXT[k] ?? k.replace(/_/g, " ")}</span> <span className={m.ready ? "text-teal" : "text-muted"}>{m.ready ? (m.experimental ? "experimental" : "working") : "not available"}</span>
                      <span className="block text-muted">{m.reason ?? MODE_NOTE[k]}</span></span></label>))}
              </fieldset>
            </form>
          )}
          <JobStatus job={job.job} error={job.error} onCancel={job.stop} onDelete={async () => { if (await job.discard()) setRes(null); }} />
          {backend.state === "ready" && <p className="text-xs text-muted">{backend.caps.privacy}</p>}
        </div>

        <section aria-label="Player and evidence" className="min-w-0" data-tour="upload-result">
          {preview ? (
            <video ref={video} src={preview} controls playsInline onTimeUpdate={onTime} className="aspect-video w-full bg-black" aria-label="Your uploaded clip" data-testid="clip-player" />
          ) : ready ? <div className="grid aspect-video w-full place-items-center border border-line p-4 text-center text-sm text-muted">Choose a file to preview it here. The preview never leaves your browser.</div> : (
            <div className="border border-line p-6 text-sm text-muted"><h2 className="narrow text-2xl font-semibold text-ink">What it does when it runs</h2>
              <ul className="mt-2 list-disc space-y-1 pl-5"><li>Plays your clip above its loudness and rank tracks, with the playhead following the video.</li><li>Finds the loudest moments and cuts clips within an exact reel length.</li><li>Lets you adjust bounds, mark replays, remove candidates and export the result.</li></ul>
              <p className="mt-3">Saved examples from the benchmark games are on the <Link className="text-teal underline" href="/highlights">Highlights page</Link>. They are not an analysis of your file.</p></div>
          )}

          {res && (
            <div className="fade-swap mt-3">
              <p className="mono text-xs uppercase text-amber" data-testid="clip-mode">{res.mode}</p>
              <p className="text-xs text-muted">{res.mode_note} {res.score_meaning}</p>
              <div className="mt-2 grid grid-cols-[5.5rem_minmax(0,1fr)] gap-x-2">
                <ul className="mono relative text-[10px] text-muted" aria-hidden="true"><li className="absolute top-[12%]">RANK</li><li className="absolute top-[44%] text-amber">LOUDNESS</li><li className="absolute top-[70%] text-defense">SPEECH</li><li className="absolute top-[88%]">CLIPS</li></ul>
                <svg viewBox={`0 0 ${dur || 1} 82`} preserveAspectRatio="none" className="block h-40 w-full cursor-crosshair border border-line bg-surface" role="img" data-testid="clip-timeline"
                  aria-label="Timeline of your clip: rank within the file, loudness above background, spoken segments from the transcript when available, and the selected clips. Click to seek the player."
                  onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); seek(timelineSeek((e.clientX - r.left) / r.width, dur)); }}>
                  {extent[0] > 0 && <rect x={0} y={0} width={extent[0]} height={58} fill="var(--color-line)" opacity={0.4}><title>No analysed audio in this interval</title></rect>}
                  {extent[1] < dur && <rect x={extent[1]} y={0} width={dur - extent[1]} height={58} fill="var(--color-line)" opacity={0.4} data-testid="missing-audio"><title>No analysed audio in this interval</title></rect>}
                  <path d={path.rank} fill="none" stroke="var(--color-ink)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
                  <path d={path.loud} fill="none" stroke="var(--color-amber)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
                  {res.candidates.filter((c) => !edits[c.asset]?.removed).map((c) => { const e = edits[c.asset]; return (
                    <g key={c.asset}><rect x={e.start} y={72} width={e.end - e.start} height={8} fill="var(--color-amber)" opacity={active === c.asset ? 1 : 0.55} />
                      <line x1={c.candidate_moment_s} x2={c.candidate_moment_s} y1={70} y2={82} stroke="var(--color-ink)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" /></g>); })}
                  {res.commentary?.segments.map((g) => <rect key={g.start_s} x={g.start_s} y={60} width={g.end_s - g.start_s} height={6} fill="var(--color-defense)" opacity={0.8} />)}
                  <line x1={now} x2={now} y1={0} y2={82} stroke="var(--color-teal)" strokeWidth={2} vectorEffect="non-scaling-stroke" data-testid="clip-playhead" />
                </svg>
                <span /><p className="mono num mt-1 flex justify-between text-[10px] text-muted"><span>0:00</span><span className="text-teal">{clock(now)}</span><span>{clock(dur)}</span></p>
                <span /><input type="range" min={0} max={dur} step={0.1} value={now} onChange={(e) => seek(Number(e.target.value))} className="w-full" style={{ accentColor: "var(--color-amber)" }} aria-label="Seek the clip" />
              </div>
              <p className="mt-1 text-xs text-muted"><Badge kind="Observed" /> Loudness is measured audio{res.commentary ? "; speech bars are where the machine transcript found words" : ""}. <Badge kind="Derived" /> Rank orders the clips of this file by {res.timeline.ranked_by ?? "loudness above background"}. Bars are padded clips; ticks are the candidate moments. Shaded regions have no analysed audio, not silent audio. {res.domain_shift}</p>
              {res.commentary && (
                <div className="mt-3 border border-line p-3" data-testid="clip-transcript">
                  <p className="kicker"><Badge kind="Predicted" /> heard near the playhead (machine transcript)</p>
                  {near.length === 0 ? <p className="mt-1 text-sm text-muted">No speech was transcribed within 20 seconds of {clock(now)}.</p> : (
                    <ul className="mt-1 text-sm">{near.map((g) => (
                      <li key={g.start_s} className={`grid grid-cols-[4rem_1fr] gap-2 border-t border-line py-1 ${now >= g.start_s && now <= g.end_s ? "bg-surface" : ""}`}>
                        <button className="mono num text-left text-xs text-teal underline" onClick={() => seek(g.start_s)} aria-label={`Seek to ${clock(g.start_s)}`}>{clock(g.start_s)}</button><span>{g.text}</span></li>))}</ul>)}
                  <p className="mt-2 text-xs text-muted">{res.commentary.words} words from {res.commentary.model} in {res.commentary.transcription_seconds} s. {res.commentary.note} Words are evidence of what was said, not of what happened.</p>
                </div>)}

              <h2 className="kicker mt-5">Candidates ({res.candidates.filter((c) => !edits[c.asset]?.removed).length} kept, reel {res.reel ? `${res.reel.decoder_output_s.toFixed(1)} of ${res.reel.budget_s} s selected; rendered file ${res.reel.ffprobe_duration_s.toFixed(2)} s` : "empty"})</h2>
              <ul className="mt-2 grid gap-2 md:grid-cols-2" data-testid="clip-candidates">
                {res.candidates.map((c) => { const e = edits[c.asset]; return (
                  <li key={c.asset} className={`border p-3 text-sm ${active === c.asset ? "border-amber" : "border-line"} ${e.removed ? "opacity-50" : ""}`}>
                    <p className="mono num flex justify-between text-xs"><span>#{c.rank_in_reel} · MOMENT {clock(c.candidate_moment_s)}</span><span className="text-muted">+{c.loudness_above_background_db} dB</span></p>
                    <p className="mt-2 flex flex-wrap items-end gap-2">
                      <label className="text-xs text-muted">Start<input type="number" step={0.5} min={0} max={e.end - 1} value={e.start} onChange={(ev) => setEdits({ ...edits, [c.asset]: { ...e, start: Number(ev.target.value), reviewed: false } })} className="num ml-1 w-20 border border-line bg-surface px-1" /></label>
                      <label className="text-xs text-muted">End<input type="number" step={0.5} min={e.start + 1} max={dur} value={e.end} onChange={(ev) => setEdits({ ...edits, [c.asset]: { ...e, end: Number(ev.target.value), reviewed: false } })} className="num ml-1 w-20 border border-line bg-surface px-1" /></label>
                      <label className="text-xs text-muted">Replay?<select value={e.replay} onChange={(ev) => setEdits({ ...edits, [c.asset]: { ...e, replay: ev.target.value as ClipEdit["replay"], reviewed: false } })} className="ml-1 border border-line bg-surface px-1"><option>unknown</option><option>live action</option><option>replay</option></select></label>
                    </p>
                    <p className="mt-2 flex flex-wrap gap-3">
                      <button className="btn-quiet" onClick={() => { setActive(c.asset); seek(e.start, e.end, true); }}>Play this clip</button>
                      <button className="btn-quiet" onClick={() => setEdits({ ...edits, [c.asset]: { ...e, removed: !e.removed, reviewed: false } })}>{e.removed ? "Restore" : "Remove"}</button>
                      {job.ticket && <a className="btn-quiet" href={mediaUrl(job.ticket, c.asset)} download>Download the cut</a>}
                    </p>
                    <label className="mt-2 flex items-start gap-2 text-xs text-muted"><input type="checkbox" checked={e.reviewed} onChange={(ev) => setEdits({ ...edits, [c.asset]: { ...e, reviewed: ev.target.checked } })} className="mt-0.5" />I watched this clip and reviewed these settings</label>
                  </li>); })}
              </ul>
              <p className="mt-3 flex flex-wrap items-center gap-4 text-sm">
                <button className="btn-quiet" onClick={exportJson}>Export review as JSON</button>
                {job.ticket && res.reel && <a className="btn-quiet" href={mediaUrl(job.ticket, res.reel.asset)} download>Download the reel ({res.reel.ffprobe_duration_s.toFixed(1)} s)</a>}
                <span className="text-xs text-muted">Downloaded cuts use the model&apos;s bounds; your edited bounds are saved in the JSON. Event types: {res.event_types}.</span>
              </p>
              <p className="mono mt-2 text-[10px] uppercase text-muted">Processed in {res.measured.seconds} s · peak memory {res.measured.peak_memory_mb} MB · streams used: {res.evidence_streams_used.join(", ")}</p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
