"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { type Capabilities, type Job, type Ticket, STATE_TEXT, TERMINAL, cancel, capabilities, configured, remove, status } from "@/lib/inference";

export type Backend = { state: "unconfigured" } | { state: "checking" } | { state: "unreachable"; error: string } | { state: "ready"; caps: Capabilities };

export function useBackend(): Backend {
  const [b, setB] = useState<Backend>(configured ? { state: "checking" } : { state: "unconfigured" });
  useEffect(() => {
    if (!configured) return;
    const ac = new AbortController();
    capabilities(ac.signal).then((caps) => setB({ state: "ready", caps })).catch((e) => { if (!ac.signal.aborted) setB({ state: "unreachable", error: String(e.message ?? e) }); });
    return () => ac.abort();
  }, []);
  return b;
}

// Polls a submitted job until it reaches a final state. Stages are reported as they are; no percentage is invented.
export function useJob(onDone: (t: Ticket) => void) {
  const [ticket, setTicket] = useState<Ticket | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const done = useRef(onDone);
  useEffect(() => { done.current = onDone; });
  useEffect(() => {
    if (!ticket) return;
    let live = true, timer = 0;
    const tick = async () => {
      try {
        const j = await status(ticket);
        if (!live) return;
        setJob(j);
        if (j.state === "complete") done.current(ticket);
        if (!TERMINAL.includes(j.state)) timer = window.setTimeout(tick, 700);
      } catch (e) { if (live) setError(String((e as Error).message)); }
    };
    tick();
    return () => { live = false; window.clearTimeout(timer); };
  }, [ticket]);
  const start = useCallback((t: Ticket) => { setError(null); setJob(null); setTicket(t); }, []);
  const stop = useCallback(async () => { if (ticket) await cancel(ticket); }, [ticket]);
  const discard = useCallback(async () => { if (ticket && (await remove(ticket))) { setTicket(null); setJob(null); return true; } return false; }, [ticket]);
  return { ticket, job, error, setError, start, stop, discard };
}

export function BackendNotice({ backend, what }: { backend: Backend; what: string }) {
  if (backend.state === "ready") return null;
  if (backend.state === "checking") return <p role="status" className="border border-line p-4 text-sm text-muted">Looking for the local inference service…</p>;
  return (
    <section aria-labelledby="up-local" className="border border-amber/60 bg-surface p-5 text-sm" data-testid="backend-notice">
      <h2 id="up-local" className="narrow text-2xl font-semibold">{backend.state === "unconfigured" ? "This runs on your own machine" : "The local service is not answering"}</h2>
      <p className="mt-2 max-w-[70ch] text-muted">
        {backend.state === "unconfigured"
          ? `${what} is done by a small Python service that runs locally, next to the models. This public site has no such service behind it, so it does not accept files and does not pretend to analyse them.`
          : `The site is configured to use a local service, but it could not be reached (${backend.error}). Start it and reload.`}
      </p>
      <ol className="mono mt-3 list-decimal space-y-1 pl-5 text-xs text-muted">
        <li>git clone the repository; uv sync --extra service; brew install ffmpeg</li>
        <li>bin/gridiron-api <span className="text-ink">(serves http://127.0.0.1:8765, loopback only)</span></li>
        <li>cd apps/web &amp;&amp; NEXT_PUBLIC_GRIDIRON_API=http://127.0.0.1:8765 npm run dev</li>
      </ol>
      <p className="mt-3 max-w-[70ch] text-xs text-muted">Trained weights and datasets are not in the repository, so a fresh clone needs the pipelines run first (see the README). Nothing on your computer is contacted by this page unless you configure that address yourself.</p>
    </section>
  );
}

export function JobStatus({ job, error, onCancel, onDelete }: { job: Job | null; error: string | null; onCancel: () => void; onDelete: () => void }) {
  if (error) return <p role="alert" className="border border-coral/60 p-3 text-sm text-coral">{error}</p>;
  if (!job) return null;
  const final = TERMINAL.includes(job.state);
  return (
    <div className="border border-line p-3 text-sm" data-testid="job-status" data-state={job.state}>
      <p role="status" aria-live="polite"><span className="kicker mr-2">Job</span><strong className={`font-semibold ${job.state === "failed" ? "text-coral" : job.state === "complete" ? "text-teal" : "text-ink"}`}>{STATE_TEXT[job.state]}</strong>
        {job.stage && <span className="text-muted"> · {job.stage}</span>}</p>
      {!final && <p className="mt-1 text-xs text-muted">Stages are reported as they happen. No percentage is shown because none is measured.</p>}
      {job.error && <p role="alert" className="mt-2 text-coral">{job.error}</p>}
      {job.warnings.length > 0 && <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-amber">{job.warnings.map((w) => <li key={w}>{w}</li>)}</ul>}
      <p className="mt-3 flex flex-wrap gap-3">
        {!final && <button className="btn-quiet" onClick={onCancel}>Cancel</button>}
        {final && <button className="btn-quiet" onClick={onDelete}>Delete this job and its files</button>}
        <span className="text-xs text-muted">Kept privately until {job.expires_at.slice(0, 16).replace("T", " ")} UTC, then deleted.</span>
      </p>
    </div>
  );
}
