// Architecture figure: three separate pipelines, versioned outputs, one read-only site. Plain HTML so it reflows on small screens.
const PIPES = [
  { name: "Coverage", tone: "border-teal text-teal", data: "Big Data Bowl 2026 tracking, 2023 pass plays", steps: ["Normalize + audit", "Shell, relational and frame features", "Trees and a temporal model", "Evaluation by cutoff, team and tracked-defender count"], cmd: "bin/coverage-lens", out: "Per-play tracking, predictions, benchmark JSON" },
  { name: "Highlights", tone: "border-amber text-amber", data: "SVHighlights football subset, 40 broadcasts", steps: ["Audit + game split", "Loudness, transcript and embedding features", "Per-clip and temporal fusion rankers", "Ranking metrics by reel length"], cmd: "bin/highlights-lens", out: "Per-game timelines, candidates, benchmark JSON" },
  { name: "Game prediction", tone: "border-defense text-defense", data: "nflverse schedules and play-by-play, 2002–2025", steps: ["28-hour feature cutoff", "Elo, Elo-offset and point-margin models", "Walk-forward backtest", "Append-only forecast log"], cmd: "bin/pregame-lens", out: "Per-game forecasts, performance JSON, forecast log" },
];

export default function Architecture() {
  return (
    <figure aria-label="Architecture: three independent pipelines write versioned outputs that one read-only website displays">
      <ol className="grid gap-4 lg:grid-cols-3">
        {PIPES.map((p) => (
          <li key={p.name} className={`border-t-2 bg-surface p-4 ${p.tone.split(" ")[0]}`}>
            <p className={`display text-3xl ${p.tone.split(" ")[1]}`}>{p.name}</p>
            <p className="mono mt-1 text-[11px] uppercase text-muted">{p.data}</p>
            <ol className="mt-3 text-sm">
              {p.steps.map((s, i) => <li key={s} className="flex gap-3 border-t border-line py-1.5"><span className="mono num text-xs text-muted">{String(i + 1).padStart(2, "0")}</span><span>{s}</span></li>)}
            </ol>
            <p className="mono mt-3 text-xs text-muted">{p.cmd}</p>
            <p className="mt-3 border-t border-line pt-2 text-xs text-muted"><span className="kicker block">Writes</span>{p.out}</p>
          </li>
        ))}
      </ol>
      <div className="flex justify-around py-1 text-2xl text-muted" aria-hidden="true"><span>↓</span><span className="hidden lg:inline">↓</span><span className="hidden lg:inline">↓</span></div>
      <div className="border border-line p-4">
        <p className="kicker">Versioned outputs</p>
        <p className="mt-1 text-sm text-muted">Parquet tables, saved models and one JSON run record per run (commit, seed, data hashes, split, cutoff, settings, metrics). <span className="mono text-xs">reports/v1</span> is preserved; new work goes to <span className="mono text-xs">reports/v2</span>. An export step writes compact JSON for display.</p>
      </div>
      <div className="py-1 text-center text-2xl text-muted" aria-hidden="true">↓</div>
      <div className="border border-ink p-4">
        <p className="kicker">Shared website</p>
        <p className="mt-1 text-sm text-muted">One Next.js app that only reads the frozen exports in the browser. No server-side model, database or queue. The pipelines share identifiers, the run-record contract and interface components, and never share features, models or evaluation.</p>
      </div>
      <figcaption className="mt-2 text-xs text-muted">The game predictor never reads coverage or highlight outputs. Training and evaluation run locally; the site is a static deployment of the frozen exports.</figcaption>
    </figure>
  );
}
