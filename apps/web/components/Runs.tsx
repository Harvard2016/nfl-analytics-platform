"use client";

import { useEffect, useState } from "react";

type Run = {
  module: string; run_id: string; name: string; started_at: string; git_commit: string; seed: number | null; latest: boolean;
  environment: { python: string; machine: string; device: string; packages: Record<string, string> };
  target: string | null; population: string | null; exclusions: unknown; input_cutoff: unknown; split: unknown; calibration: string | null;
  hyperparameters: unknown; metrics: unknown; ablations: unknown; outputs: Record<string, { path: string; sha256: string | null }>;
  data: Record<string, { path: string; sha256: string | null }>; train_seconds: number; peak_memory_mb: number; evidence_kinds: Record<string, string>;
};
const TH = "py-1.5 pr-3 text-left font-normal text-muted";
const show = (v: unknown) => (typeof v === "string" ? v : JSON.stringify(v, null, 1));

export default function Runs() {
  const [runs, setRuns] = useState<Run[] | null | undefined>(undefined);
  const [open, setOpen] = useState<string | null>(() => (typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("run")));
  const [module, setModule] = useState("all");
  useEffect(() => { fetch("/demo/research/runs.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then((d) => setRuns(d?.runs ?? null)).catch(() => setRuns(null)); }, []);
  if (runs === undefined) return <p className="p-6 text-muted">Loading run records…</p>;
  if (runs === null) return <p className="p-6 text-muted">No run records exported yet. Run the experiments, then the research export.</p>;
  const list = runs.filter((r) => module === "all" || r.module === module).sort((a, b) => b.started_at.localeCompare(a.started_at));
  const r = list.find((x) => x.run_id === open) ?? list[0];

  return (
    <div className="mx-auto flex max-w-[1500px] flex-col gap-6 px-4 py-10 lg:px-8">
      <header>
        <p className="kicker">Research · frozen run records</p>
        <h1 className="display mt-1 text-6xl sm:text-7xl">Experiment notebook<span className="text-teal">.</span></h1>
        <p className="mt-3 max-w-[72ch] text-muted">Every training and evaluation run writes one record: what data, which split, what cutoff, which settings, how long it took and where its outputs are. These are frozen exports; the site does not query a live tracker. Failed and negative runs stay in the list.</p>
      </header>
      <label className="flex w-fit items-center gap-2 text-sm text-muted">Module
        <select value={module} onChange={(e) => setModule(e.target.value)} className="border border-line bg-surface px-2 text-ink"><option value="all">All</option><option value="coverage">Coverage</option><option value="pregame">Game predictor</option><option value="highlights">Highlights</option></select>
      </label>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="overflow-x-auto">
          <table className="num w-full min-w-[520px] text-sm">
            <thead><tr><th className={TH}>Run</th><th className={TH}>Module</th><th className={TH}>Started (UTC)</th><th className={TH}>Seconds</th><th className={TH}>Peak memory</th></tr></thead>
            <tbody>
              {list.map((x) => (
                <tr key={x.run_id} className={`border-t border-line ${x.run_id === r?.run_id ? "bg-surface" : ""}`}>
                  <td className="py-1.5 pr-3"><button className="text-left underline decoration-line underline-offset-4 hover:decoration-teal" onClick={() => setOpen(x.run_id)}>{x.name}</button>{x.latest ? "" : <span className="text-muted"> (superseded)</span>}</td>
                  <td>{x.module}</td><td>{x.started_at.slice(0, 16).replace("T", " ")}</td><td>{x.train_seconds}</td><td>{x.peak_memory_mb} MB</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {r && (
          <section aria-label="Run record" className="min-w-0 text-sm">
            <h2 className="narrow break-all text-lg font-semibold">{r.run_id}</h2>
            <dl className="mt-2 space-y-2 text-muted">
              <div><dt className="inline text-ink">Code: </dt><dd className="inline break-all">{r.git_commit}; seed {r.seed ?? "n/a"}; {r.environment.device}, {r.environment.machine}, Python {r.environment.python}</dd></div>
              <div><dt className="inline text-ink">Packages: </dt><dd className="inline">{Object.entries(r.environment.packages).map(([k, v]) => `${k} ${v}`).join(", ")}</dd></div>
              <div><dt className="inline text-ink">Target (released label): </dt><dd className="inline">{r.target ?? "n/a"}</dd></div>
              <div><dt className="inline text-ink">Population: </dt><dd className="inline">{r.population ?? "n/a"}{r.exclusions ? `; excluded: ${show(r.exclusions)}` : ""}</dd></div>
              <div><dt className="inline text-ink">Input cutoff (observed inputs): </dt><dd className="inline">{show(r.input_cutoff)}</dd></div>
              <div><dt className="inline text-ink">Calibration: </dt><dd className="inline">{r.calibration ?? "n/a"}</dd></div>
            </dl>
            {(["split", "hyperparameters", "metrics", "ablations"] as const).map((k) => r[k] && Object.keys(r[k] as object).length > 0 ? (
              <details key={k} className="mt-3 border-t border-line pt-2" open={k === "metrics"}>
                <summary className="narrow font-semibold">{k === "metrics" ? "Metrics (model predictions against released labels)" : k[0].toUpperCase() + k.slice(1)}</summary>
                <pre className="num mt-2 max-h-80 overflow-auto bg-surface p-3 text-xs">{show(r[k])}</pre>
              </details>
            ) : null)}
            <details className="mt-3 border-t border-line pt-2">
              <summary className="narrow font-semibold">Saved outputs and data</summary>
              <ul className="num mt-2 space-y-1 text-xs text-muted">
                {Object.entries({ ...r.data, ...r.outputs }).map(([k, v]) => <li key={k} className="break-all"><span className="text-ink">{k}</span>: {v.path}{v.sha256 ? `, sha256 ${v.sha256.slice(0, 12)}` : ""}</li>)}
              </ul>
            </details>
          </section>
        )}
      </div>
    </div>
  );
}
