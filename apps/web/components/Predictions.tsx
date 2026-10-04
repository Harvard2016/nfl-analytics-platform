"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui";
import { pct } from "@/lib/demo";
import { type Games, PMODELS, loadPregame, signed } from "@/lib/pregame";
import ModuleTour from "@/components/ModuleTour";
import StadiumBackdrop from "@/components/StadiumBackdrop";
import TeamPortrait from "@/components/TeamPortrait";

const sig = (x: number) => 1 / (1 + Math.exp(-x));

export default function Predictions() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const [data, setData] = useState<Games | null | undefined>(undefined);
  useEffect(() => { loadPregame<Games>("v2_games.json").then(setData); }, []);
  const seasons = useMemo(() => (data ? [...new Set(data.games.map((g) => g.season))].sort((a, b) => b - a) : []), [data]);
  if (data === undefined) return <p className="p-6 text-muted">Loading backtests…</p>;
  if (data === null) return <p className="p-6 text-muted">No backtest exported yet. Run bin/pregame-lens backtest, then reload.</p>;

  const set = (changes: Record<string, string | null>) => {
    const next = new URLSearchParams(params.toString());
    for (const [k, v] of Object.entries(changes)) { if (v == null) next.delete(k); else next.set(k, v); }
    router.replace(`${pathname}?${next.toString()}`, { scroll: false });
  };
  const byId = data.games.find((g) => g.id === params.get("game"));
  const season = byId?.season ?? (Number(params.get("season")) || seasons.find((s) => s <= 2025) || seasons[0]);
  const show = params.get("show") === "misses" ? "misses" : "all";
  const inSeason = data.games.filter((g) => g.season === season);
  const weeks = [...new Set(inSeason.map((g) => g.week))].sort((a, b) => a - b);
  const week = byId?.week ?? (weeks.includes(Number(params.get("week"))) ? Number(params.get("week")) : weeks[0]);
  // "misses": games where the favourite by the v2 model lost, most confident first
  const list = show === "misses"
    ? inSeason.filter((g) => (g.p.elo_offset >= 0.5) !== g.home_won).sort((a, b) => Math.abs(b.p.elo_offset - 0.5) - Math.abs(a.p.elo_offset - 0.5)).slice(0, 24)
    : inSeason.filter((g) => g.week === week);
  const game = byId ?? list[0] ?? inSeason[0];
  const p = game.p.elo_offset;
  const steps = [{ description: "Model intercept", log_odds: game.intercept }, ...game.terms.filter((t) => Math.abs(t.log_odds) >= 0.005)]
    .reduce<{ description: string; log_odds: number; total: number }[]>((acc, t) => [...acc, { ...t, total: (acc.at(-1)?.total ?? game.elo_logit) + t.log_odds }], []);
  const span = Math.max(0.05, ...steps.map((s) => Math.abs(s.log_odds)));

  return (
    <div>
      <header data-tour="prediction-selectors" className="prediction-header cinematic-band border-b border-line">
        <StadiumBackdrop />
        <div className="mx-auto grid max-w-[1500px] gap-6 px-4 py-8 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end lg:px-8">
          <div>
            <div className="flex items-center gap-4"><p className="kicker">Game prediction</p><ModuleTour module="predictions" /></div>
            <h1 className="display mt-1 text-6xl sm:text-8xl">Before kickoff<span className="text-teal">.</span></h1>
            <p className="mt-3 max-w-[60ch] text-muted">Win probabilities built from team and quarterback performance known 24 hours before kickoff. What you browse here are <strong className="font-semibold text-ink">reconstructed backtests</strong>; forecasts recorded before games are in the <Link className="text-teal underline" href="/predictions/forecasts">live forecast log</Link>.</p>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <label className="flex items-center gap-1 text-muted"><span className="kicker">Season</span>
              <select value={season} onChange={(e) => set({ season: e.target.value, week: null, game: null })} className="border border-line bg-surface px-2 text-ink">{seasons.map((s) => <option key={s}>{s}</option>)}</select>
            </label>
            {show === "all" && (
              <label className="flex items-center gap-1 text-muted"><span className="kicker">Week</span>
                <select value={week} onChange={(e) => set({ season: String(season), week: e.target.value, game: null })} className="border border-line bg-surface px-2 text-ink">{weeks.map((w) => <option key={w}>{w}</option>)}</select>
              </label>
            )}
            <select aria-label="Show" value={show} onChange={(e) => set({ show: e.target.value === "all" ? null : "misses", game: null })} className="max-w-full border border-line bg-surface px-2">
              <option value="all">By week</option>
              <option value="misses">Favourite lost</option>
            </select>
          </div>
        </div>
        <nav aria-label="Games" className="mx-auto max-w-[1500px] overflow-x-auto px-4 pb-4 lg:px-8">
          <ul className="flex gap-2">
            {list.map((g) => (
              <li key={g.id}>
                <button onClick={() => set({ game: g.id })} aria-current={g.id === game.id ? "true" : undefined}
                  className={`w-28 shrink-0 border px-2 py-1.5 text-left transition-colors duration-150 ${g.id === game.id ? "border-teal text-ink" : "border-line text-muted hover:border-ink hover:text-ink"}`}>
                  <span className="narrow block text-base leading-tight">{g.away} at {g.home}</span>
                  <span className="mono num block text-[10px]">{g.date.slice(5)} · {pct(g.p.elo_offset)} {g.home}</span>
                </button>
              </li>
            ))}
          </ul>
        </nav>
      </header>

      <section key={game.id} aria-label="Selected matchup" className="matchup-stage fade-swap border-b border-line">
        <div className="matchup-grid mx-auto max-w-[1500px]">
          <TeamPortrait team={game.away} side="away" />
          <div data-tour="prediction-probabilities" className="matchup-probabilities text-center">
            <p className="mono mx-auto w-fit border border-line px-3 py-1 text-[11px] tracking-[0.2em] text-muted">HISTORICAL BACKTEST · 24 H INPUT CUTOFF</p>
            <p className="kicker mt-5">Win probability, Elo-offset model</p>
            <div className="mt-2 grid grid-cols-[auto_1fr_auto] items-center gap-3">
              <span className="display text-4xl num">{pct(1 - p)}</span>
              <span className="flex h-4 w-full" role="img" aria-label={`${game.away} ${pct(1 - p)}, ${game.home} ${pct(p)}`}>
                <span className="h-4 bg-defense transition-[width] duration-300" style={{ width: `${(1 - p) * 100}%` }} /><span className="h-4 flex-1 bg-teal" />
              </span>
              <span className="display text-5xl text-teal num">{pct(p)}</span>
            </div>
            <p className="mono mt-1 flex justify-between text-[11px] text-muted"><span>{game.away} WIN</span><span>{game.home} WIN</span></p>
            <p className="mono num mt-4 text-xs text-muted">ELO BASELINE &nbsp; {pct(1 - game.p.elo)} &nbsp;|&nbsp; {pct(game.p.elo)}</p>
            <p className="mt-4"><Link href="/predictions/performance" className="narrow inline-block border border-line px-4 py-2 text-base hover:border-ink">No established gain over Elo. See the evidence</Link></p>
            <p className="mono num mt-3 text-[11px] text-muted">{game.date} · {game.season} WEEK {game.week}{game.type !== "REG" ? ` · ${game.type}` : ""}</p>
          </div>
          <TeamPortrait team={game.home} side="home" />
        </div>
        <p className="matchup-art-note">Original decorative helmet artwork · not official uniform imagery</p>
      </section>

      <div className="paper">
        <div className="mx-auto grid max-w-[1500px] grid-cols-[minmax(0,1fr)] gap-10 px-4 py-12 lg:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)] lg:px-8">
          <section data-tour="prediction-breakdown" aria-labelledby="pg-how">
            <p className="kicker">Model breakdown</p>
            <h2 id="pg-how" className="display mt-1 text-5xl sm:text-6xl">From Elo to the forecast<span className="text-teal">.</span></h2>
            <p className="mt-3 max-w-[64ch] text-muted">The model starts at the Elo baseline and adds one term per feature. Each bar is that term in <strong className="font-semibold text-ink">log-odds of a {game.home} win</strong>: a coefficient times a standardized feature. The terms add up on the log-odds scale, not in percentage points, so the running probability is shown beside each step.</p>
            <div className="mt-5 overflow-x-auto">
              <table className="num w-full min-w-[560px] text-sm" data-testid="waterfall">
                <thead><tr className="text-left text-muted"><th className="py-1.5 pr-3 font-normal">Step</th><th className="w-[34%] font-normal">Change in log-odds</th><th className="pr-3 text-right font-normal">Running total</th><th className="text-right font-normal">{game.home} win</th></tr></thead>
                <tbody>
                  <tr className="border-t border-line"><td className="py-2 pr-3 font-semibold">Elo baseline</td><td /><td className="pr-3 text-right">{signed(game.elo_logit, 3)}</td><td className="text-right">{pct(sig(game.elo_logit), 1)}</td></tr>
                  {steps.map((t) => (
                    <tr key={t.description} className="border-t border-line align-middle">
                      <td className="py-2 pr-3">{t.description}</td>
                      <td>
                        <span className="grid grid-cols-2 items-center" aria-hidden="true">
                          <span className="flex justify-end">{t.log_odds < 0 && <span className="h-3 bg-defense transition-[width] duration-300" style={{ width: `${(Math.abs(t.log_odds) / span) * 100}%` }} />}</span>
                          <span className="flex border-l border-ink/40">{t.log_odds >= 0 && <span className="h-3 bg-teal transition-[width] duration-300" style={{ width: `${(t.log_odds / span) * 100}%` }} />}</span>
                        </span>
                        <span className="mono block text-center text-[11px] text-muted">{signed(t.log_odds, 3)} {t.log_odds >= 0 ? `toward ${game.home}` : `toward ${game.away}`}</span>
                      </td>
                      <td className="pr-3 text-right">{signed(t.total, 3)}</td><td className="text-right">{pct(sig(t.total), 1)}</td>
                    </tr>
                  ))}
                  <tr className="border-t-2 border-ink"><td className="py-2 pr-3 font-semibold">Final forecast (with terms under 0.005)</td><td /><td /><td className="text-right text-lg font-semibold">{pct(p, 1)}</td></tr>
                </tbody>
              </table>
            </div>
            <p className="mt-2 max-w-[64ch] text-xs text-muted">Contributions describe the fitted model. They do not show that a factor caused the result.</p>

            <details data-tour="prediction-method" className="mt-6 border border-line p-4" open>
              <summary className="narrow text-xl">How was this calculated?</summary>
              <dl className="mt-3 grid gap-3 text-sm text-muted sm:grid-cols-2">
                <div><dt className="kicker">Inputs</dt><dd>Every game that kicked off at least 28 hours before this one. Elo before the game: {game.home} {game.elo[0]}, {game.away} {game.elo[1]}.</dd></div>
                <div><dt className="kicker">Quarterback assumption</dt><dd>Projected starters are each team&apos;s primary passer in its previous game: {game.qb.home ?? "unknown"} ({game.qb.home_dropbacks} recent dropbacks) for {game.home}, {game.qb.away ?? "unknown"} ({game.qb.away_dropbacks}) for {game.away}. A scenario, not a confirmed lineup.{game.missing_inputs.length > 0 && <span className="text-amber"> Missing: {game.missing_inputs.join(", ")}; the league average is used.</span>}</dd></div>
                <div><dt className="kicker">Model</dt><dd>{data.version}, refit before the {game.season} season on {game.trained_on_games.toLocaleString()} earlier games. No calibrator applied.</dd></div>
                <div><dt className="kicker">Cutoff</dt><dd>{data.cutoff}</dd></div>
              </dl>
            </details>
          </section>

          <aside data-tour="prediction-results" aria-label="Observed result and comparisons">
            <p className="kicker"><Badge kind="Observed" /> result</p>
            <table className="mt-2 w-full">
              <caption className="mono pb-1 text-right text-xs text-muted">FINAL</caption>
              <tbody>
                <tr className="border-t border-line"><th scope="row" className="display py-2 text-left text-4xl">{game.away}</th><td className={`display text-right text-4xl num ${game.home_won ? "text-muted" : ""}`}>{game.away_score}</td></tr>
                <tr className="border-t border-line"><th scope="row" className="display py-2 text-left text-4xl">{game.home}</th><td className={`display text-right text-4xl num ${game.home_won ? "" : "text-muted"}`}>{game.home_score}</td></tr>
              </tbody>
            </table>
            <p className="mt-2 text-sm text-muted">{game.home_won ? game.home : game.away} won. Attached after the forecast; never an input.</p>

            <h3 className="narrow mt-8 text-xl">Every model on this game</h3>
            <table className="num mt-1 w-full text-sm"><tbody>
              {PMODELS.map((m) => <tr key={m} className="border-t border-line"><td className="py-1.5">{data.labels[m]}</td><td className="text-right">{pct(game.p[m], 1)} {game.home}</td></tr>)}
              <tr className="border-t border-line"><td className="py-1.5">Point-margin model&apos;s expected margin</td><td className="text-right">{game.home} {signed(game.predicted_margin, 1)}</td></tr>
            </tbody></table>

            <h3 className="narrow mt-8 text-xl">Backtests, not live forecasts</h3>
            <p className="mt-1 text-sm text-muted">{data.reconstructed}</p>
            <p className="mt-4 flex flex-wrap gap-5"><Link className="btn-quiet" href="/predictions/performance">Calibration and history</Link><Link className="btn-quiet" href="/predictions/forecasts">Live forecast log</Link></p>
          </aside>
        </div>
      </div>
    </div>
  );
}
