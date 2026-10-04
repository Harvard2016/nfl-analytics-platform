"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import Field from "@/components/Field";
import ModuleTour from "@/components/ModuleTour";
import { BackendNotice, JobStatus, useBackend, useJob } from "@/components/UploadShell";
import { Badge } from "@/components/ui";
import { type Entity, HORIZONS, type HorizonKey, pct } from "@/lib/demo";
import { result, submit, templateUrl } from "@/lib/inference";

type Hz = { available: boolean; reason?: string; frame?: number; p_man?: number; p_zone?: number; lean?: string; confidence?: number; reliable_prediction_cutoff?: number;
  accepted_under_reliable_policy?: boolean; observed?: { nearest_defender_to_each_route_runner_yards: number[]; mean_defender_depth_yards: number } };
type Result = {
  play_id: string; mode: string; scope: string; explanation_note: string; horizons: Record<HorizonKey, Hz>;
  model: { version: string; bundle_sha256: string; input_schema: string; calibration: string };
  input_quality: { defenders: number; route_runners: number; frames_supplied: number; frames_usable: number; observed_orientation: boolean; warnings: string[] };
  comparison_label: { value: string; note: string } | null;
  playback: { frames: number[]; los_x: number; entities: Entity[]; note: string };
};

export default function AnalyzePlay() {
  const backend = useBackend();
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState("release-compatible");
  const [res, setRes] = useState<Result | null>(null);
  const [hz, setHz] = useState<HorizonKey>("post_1_5s");
  const [t, setT] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const heading = useRef<HTMLHeadingElement>(null);
  const job = useJob(async (tk) => {
    const r = await result<Result>(tk);
    setRes(r);
    const last = [...HORIZONS].reverse().find((h) => r.horizons[h.key].available);
    if (last) { setHz(last.key); setT(r.horizons[last.key].frame ?? 0); }
  });
  useEffect(() => { if (res) heading.current?.focus(); }, [res]);
  const ready = backend.state === "ready" && backend.caps.modules.coverage.ready;
  const cur = res?.horizons[hz];

  async function go(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setBusy(true); setRes(null);
    try { job.start(await submit("coverage", mode, {}, file)); } catch (err) { job.setError(String((err as Error).message)); }
    setBusy(false);
  }

  return (
    <div>
      <header className="cinematic-band border-b border-line">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-end justify-between gap-x-10 gap-y-3 px-4 py-6 lg:px-8">
          <div>
            <div className="flex items-center gap-4"><p className="kicker">Coverage / tracking classification</p><ModuleTour module="coverage-upload" ready={ready} /></div>
            <h1 className="display mt-1 text-5xl sm:text-7xl">Analyze your play<span className="text-teal">.</span></h1>
          </div>
          <p className="max-w-[58ch] text-sm text-muted">
            <strong className="font-semibold text-ink">Tracking in, probabilities out.</strong> Upload player tracking for one pass play and the saved temporal model reads it exactly as it reads the benchmark plays.
            It does not read video. <Link className="text-teal underline" href="/coverage">Back to the film room</Link>
          </p>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1500px] gap-6 px-4 py-6 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)] lg:px-8">
        <div className="flex min-w-0 flex-col gap-4">
          <BackendNotice backend={backend} what="Tracking classification" />
          {backend.state === "ready" && (
            <form onSubmit={go} className="border border-line p-4 text-sm" data-tour="upload-form">
              <h2 className="narrow text-2xl font-semibold">Tracking file</h2>
              <p className="mt-1 text-xs text-muted">One play, CSV or JSON, 10 frames per second, yards on a 120 by 53.3 field. <a className="text-teal underline" href={templateUrl}>Download the CSV template</a></p>
              <label className="mt-3 block"><span className="kicker">File</span>
                <input type="file" accept=".csv,.json,text/csv,application/json" required onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="mt-1 block w-full border border-line bg-surface p-2 text-sm" /></label>
              <fieldset className="mt-3"><legend className="kicker">Input kind</legend>
                {[["release-compatible", "Release-compatible", "Route runners and coverage defenders only, with observed orientation, like the benchmark data."],
                  ["broader", "Broader tracking (experimental)", "Other player sets or no orientation. A domain shift: no benchmark accuracy applies."]].map(([v, l, d]) => (
                  <label key={v} className="mt-2 flex gap-2"><input type="radio" name="mode" value={v} checked={mode === v} onChange={() => setMode(v)} className="mt-1" /><span><span className="text-ink">{l}</span><span className="block text-xs text-muted">{d}</span></span></label>
                ))}
              </fieldset>
              <p className="mt-3 text-xs text-muted">A coverage label column, if present, is ignored by the model and shown only for comparison.</p>
              <button className="btn-primary mt-4" disabled={!file || busy || !ready}>Run the model</button>
              <details className="mt-4 border-t border-line pt-3"><summary className="narrow font-semibold">What must the file contain?</summary>
                <dl className="mt-2 space-y-2 text-xs text-muted">
                  {Object.entries(backend.caps.modules.coverage.input.required).map(([k, v]) => <div key={k}><dt className="mono inline text-ink">{k}</dt>: <dd className="inline">{v}</dd></div>)}
                  <div><dt className="inline text-ink">Motion</dt>: <dd className="inline">{backend.caps.modules.coverage.input.motion}</dd></div>
                  <div><dt className="inline text-ink">Orientation</dt>: <dd className="inline">{backend.caps.modules.coverage.input.orientation}</dd></div>
                  {Object.entries(backend.caps.modules.coverage.input.optional).map(([k, v]) => <div key={k}><dt className="mono inline text-ink">{k}</dt>: <dd className="inline">{v}</dd></div>)}
                </dl>
              </details>
            </form>
          )}
          <JobStatus job={job.job} error={job.error} onCancel={job.stop} onDelete={async () => { if (await job.discard()) setRes(null); }} />
          {backend.state === "ready" && <p className="text-xs text-muted">{backend.caps.privacy}</p>}
        </div>

        <section aria-label="Result" className="min-w-0" data-tour="upload-result">
          {!res ? (
            <div className="border border-line p-6 text-sm text-muted">
              <h2 className="narrow text-2xl font-semibold text-ink">What you will get</h2>
              <ul className="mt-2 list-disc space-y-1 pl-5">
                <li>Man and Zone probabilities at each cutoff the play actually reaches: snap, +0.5, +1.0 and +1.5 seconds.</li>
                <li>Field playback of the uploaded players with the cutoff marked.</li>
                <li>Input checks, model version and how the result was calculated.</li>
              </ul>
              <p className="mt-3">No accuracy badge is shown for an upload: one play, usually with no label, cannot measure accuracy.</p>
            </div>
          ) : (
            <div className="fade-swap grid gap-5 xl:grid-cols-[minmax(0,1fr)_340px]">
              <div className="min-w-0">
                <h2 ref={heading} tabIndex={-1} className="display text-4xl outline-none">Play {res.play_id}</h2>
                <p className="mono text-xs uppercase text-muted">{res.mode} · {res.input_quality.defenders} defenders · {res.input_quality.route_runners} route runners · {res.input_quality.frames_usable} usable frames</p>
                <div className="mt-2"><Field entities={res.playback.entities} frames={res.playback.frames} t={Math.min(t, res.playback.frames.length - 1)} losX={res.playback.los_x} cutoff={cur?.frame ?? 0} selected={selected} onSelect={setSelected} showTrails /></div>
                <label className="mt-2 block text-sm text-muted">Playback frame <span className="mono num text-ink">+{(t / 10).toFixed(1)} s</span>
                  <input type="range" min={0} max={res.playback.frames.length - 1} value={t} onChange={(e) => setT(Number(e.target.value))} className="block w-full" /></label>
                <p className="text-xs text-muted"><Badge kind="Observed" /> {res.playback.note} Playback only moves the picture; the model saw the frames up to the chosen cutoff.</p>
              </div>
              <div className="min-w-0 text-sm">
                <div role="group" aria-label="Model sees up to" className="flex flex-wrap border border-line" data-testid="upload-cutoffs">
                  {HORIZONS.map((h) => { const a = res.horizons[h.key].available; return (
                    <button key={h.key} aria-pressed={h.key === hz} disabled={!a} title={a ? undefined : res.horizons[h.key].reason}
                      onClick={() => { setHz(h.key); setT(res.horizons[h.key].frame ?? 0); }}
                      className={`narrow flex-1 px-2 py-1 text-lg ${h.key === hz ? "bg-teal font-bold text-bg" : a ? "text-muted hover:text-ink" : "text-muted/40 line-through"}`}>{h.label}</button>); })}
                </div>
                {cur?.available ? (
                  <div className="mt-3" data-testid="upload-prediction">
                    <p className="kicker"><Badge kind="Predicted" /> model output</p>
                    <p className="display text-6xl">Leans {cur.lean}</p>
                    <div className="num mt-2 space-y-1">
                      {([["Man", cur.p_man!], ["Zone", cur.p_zone!]] as const).map(([n, v]) => (
                        <div key={n} className="grid grid-cols-[3rem_1fr_3.5rem] items-center gap-2"><span>{n}</span><span className="h-2 bg-line"><span className={`block h-2 ${n === "Man" ? "bg-teal" : "bg-defense"}`} style={{ width: `${v * 100}%` }} /></span><span className="text-right">{pct(v, 1)}</span></div>))}
                    </div>
                    <p className="mt-2 text-xs text-muted">Confidence {pct(cur.confidence!, 1)}; the reliable-predictions policy accepts at {pct(cur.reliable_prediction_cutoff!)} or above, so this one is {cur.accepted_under_reliable_policy ? "accepted" : "an abstention"}. A policy changes which leans count, not the model.</p>
                    <table className="num mt-3 w-full text-xs"><caption className="kicker pb-1 text-left"><Badge kind="Observed" /> geometry at this cutoff</caption><tbody>
                      <tr className="border-t border-line"><td className="py-1 pr-2">Nearest defender to each route runner (yards)</td><td className="text-right">{cur.observed!.nearest_defender_to_each_route_runner_yards.join(", ")}</td></tr>
                      <tr className="border-t border-line"><td className="py-1 pr-2">Mean defender depth (yards)</td><td className="text-right">{cur.observed!.mean_defender_depth_yards}</td></tr>
                    </tbody></table>
                  </div>
                ) : <p className="mt-3 text-muted">{cur?.reason}</p>}
                <div className="mt-3 border border-line p-3"><p className="kicker"><Badge kind="Released label" /> comparison</p>
                  <p className="narrow text-xl">{res.comparison_label ? res.comparison_label.value : "No label supplied"}</p>
                  <p className="text-xs text-muted">{res.comparison_label?.note ?? "Without a label there is nothing to compare with, and no accuracy is shown."}</p></div>
                <p className={`mt-3 border-l-2 pl-3 text-xs ${res.mode === "release-compatible" ? "border-line text-muted" : "border-amber text-amber"}`}>{res.scope}</p>
                {res.input_quality.warnings.length > 0 && <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-amber">{res.input_quality.warnings.map((w) => <li key={w}>{w}</li>)}</ul>}
                <details className="mt-3 border border-line p-3"><summary className="narrow font-semibold">How was this calculated?</summary>
                  <dl className="mt-2 space-y-2 text-xs text-muted">
                    <div><dt className="kicker">Model</dt><dd className="mono break-all">{res.model.version} · sha256 {res.model.bundle_sha256.slice(0, 16)}</dd></div>
                    <div><dt className="kicker">Input schema</dt><dd>{res.model.input_schema}: per player and frame, position relative to the line of scrimmage and formation centre, velocity and facing; every defender-receiver pair is compared.</dd></div>
                    <div><dt className="kicker">Calibration</dt><dd>{res.model.calibration}</dd></div>
                    <div><dt className="kicker">Reading it</dt><dd>{res.explanation_note}</dd></div>
                  </dl></details>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
