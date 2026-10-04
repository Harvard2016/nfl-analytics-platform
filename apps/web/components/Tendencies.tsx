"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { SelectionNote } from "@/components/ui";
import { TYPE_LABELS, loadDataset, loadJson, pct } from "@/lib/demo";

type Cell = { plays: number; man: number; man_rate: number | null; interval_95: [number, number] | null };
type Block = { all: Cell; by_situation: (Cell & { down: string; distance: string })[]; coverage_types: Record<string, number> };
type Pred = { plays: number; released_man_rate: number | null; predicted_man_rate: number | null; abstained: number; agreement: number | null };
type Data = {
  synthetic: boolean;
  sample: { season: number[]; weeks: [number, number]; dates: { first_game: string; last_game: string } | null; plays: number; scope: string };
  released: { league: Block; teams: Record<string, Block> };
  predicted: { scope: string; model: string; confidence_bar: number; league: Pred; teams: Record<string, Pred> };
};

const DOWNS = ["1st", "2nd", "3rd/4th"];
const DISTS = ["1-3", "4-7", "8+"];
const DOWN_PARAM: Record<string, string> = { "1st": "1", "2nd": "2", "3rd/4th": "3" };
// The predicted section of this page is computed with this model and window; links carry them into the explorer.
const MODEL = "v1_gbm";
const WINDOW = "post_1_5s";
const MIN_PLAYS = 20;
const TH = "py-1.5 pr-3 text-left font-normal text-muted";

// Cell shade encodes man rate; the number and the play count are always printed, so colour is never the only signal.
function shade(rate: number | null) {
  return rate == null ? undefined : { backgroundColor: `rgba(69, 214, 182, ${Math.min(0.55, rate * 0.9).toFixed(2)})` };
}

function Grid({ block, team }: { block: Block; team: string }) {
  return (
    <table className="num w-full text-sm">
      <caption className="pb-1 text-left text-muted">Share of plays the release labels man, by down and yards to go. Each cell shows the rate and the number of plays behind it.</caption>
      <thead><tr><th className={TH}>Down</th>{DISTS.map((d) => <th key={d} className={TH}>{d} to go</th>)}</tr></thead>
      <tbody>
        {DOWNS.map((dn) => (
          <tr key={dn} className="border-t border-line">
            <th className="py-2 pr-3 text-left font-normal">{dn}</th>
            {DISTS.map((ds) => {
              const c = block.by_situation.find((x) => x.down === dn && x.distance === ds)!;
              const thin = c.plays < MIN_PLAYS;
              return (
                <td key={ds} className="border border-line p-0">
                  <Link href={`/coverage?model=${MODEL}&window=${WINDOW}&${team ? `defense=${team}&` : ""}down=${DOWN_PARAM[dn]}`} className="block px-2 py-2 hover:outline hover:outline-1 hover:outline-teal"
                    style={thin ? undefined : shade(c.man_rate)}>
                    <span className="block text-base">{c.plays === 0 ? "none" : thin ? "too few" : pct(c.man_rate!)}</span>
                    <span className="block text-xs text-muted">{c.man} of {c.plays} plays{c.interval_95 && !thin ? `, ${pct(c.interval_95[0])} to ${pct(c.interval_95[1])}` : ""}</span>
                  </Link>
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function Tendencies() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const [d, setD] = useState<Data | null | undefined>(undefined);
  useEffect(() => { loadDataset().then((ds) => (ds ? loadJson<Data>(ds, "tendencies.json") : null)).then(setD); }, []);

  if (d === undefined) return <p className="p-6 text-muted">Loading tendencies…</p>;
  if (d === null) return <p className="p-6 text-muted">Tendencies have not been exported for this dataset. Run the studies and export commands, then reload.</p>;

  const teams = Object.keys(d.released.teams);
  const team = teams.includes(params.get("team") ?? "") ? params.get("team")! : "";
  const block = team ? d.released.teams[team] : d.released.league;
  const pred = team ? d.predicted.teams[team] : d.predicted.league;
  const types = Object.entries(block.coverage_types);

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-8 p-4 lg:p-6">
      <header>
        <h1 className="display text-4xl">How often each defense plays man</h1>
        <p className="mt-3 max-w-[70ch] text-muted">
          {d.sample.plays.toLocaleString()} pass plays from the {d.sample.season.join(", ")} season, weeks {d.sample.weeks[0]} to {d.sample.weeks[1]}
          {d.sample.dates ? ` (games from ${d.sample.dates.first_game} to ${d.sample.dates.last_game})` : ""}. {d.sample.scope}
        </p>
      </header>

      <label className="flex w-fit items-center gap-2 text-sm text-muted">Defense
        <select value={team} onChange={(e) => router.replace(e.target.value ? `${pathname}?team=${e.target.value}` : pathname, { scroll: false })}
          className="border border-line bg-surface px-2 py-1.5 text-ink">
          <option value="">Whole league</option>
          {teams.map((t) => <option key={t}>{t}</option>)}
        </select>
      </label>

      <section aria-labelledby="h-rel">
        <h2 id="h-rel" className="narrow text-lg font-semibold">From the released labels: {team || "league"}</h2>
        <p className="num mt-1 text-sm text-muted">
          Man on {block.all.man} of {block.all.plays} plays: {pct(block.all.man_rate ?? 0, 1)}
          {block.all.interval_95 ? ` (95% range ${pct(block.all.interval_95[0], 1)} to ${pct(block.all.interval_95[1], 1)})` : ""}.
          {team ? ` League: ${pct(d.released.league.all.man_rate ?? 0, 1)}.` : ""}
        </p>
        <div className="mt-3 overflow-x-auto"><Grid block={block} team={team} /></div>
        <p className="mt-2 text-xs text-muted">Cells with fewer than {MIN_PLAYS} plays show the count but no rate. Select a cell to open the explorer filtered to that defense and down. The explorer holds a random sample of 120 test-week plays, so a cell can open an empty list.</p>
        <table className="num mt-5 w-full max-w-md text-sm">
          <caption className="pb-1 text-left text-muted">Released coverage types on these plays</caption>
          <tbody>
            {types.map(([k, n]) => (
              <tr key={k} className="border-t border-line"><td className="py-1.5 pr-3">{TYPE_LABELS[k] ?? k}</td><td>{n}</td><td>{pct(n / block.all.plays, 1)}</td></tr>
            ))}
          </tbody>
        </table>
      </section>

      <section aria-labelledby="h-pred">
        <h2 id="h-pred" className="narrow text-lg font-semibold">From the model&apos;s predictions, kept separate: {team || "league"}</h2>
        <SelectionNote />
        <p className="mt-3 max-w-[70ch] text-sm text-muted">{d.predicted.scope} The comparison below uses the same plays for both columns.</p>
        <table className="num mt-2 w-full max-w-xl text-sm">
          <thead><tr><th className={TH}>Plays</th><th className={TH}>Released label: man</th><th className={TH}>Model predicts man</th><th className={TH}>Agree play by play</th><th className={TH}>Model abstained</th></tr></thead>
          <tbody>
            <tr className="border-t border-line">
              <td className="py-1.5 pr-3">{pred.plays}</td><td>{pred.released_man_rate == null ? "n/a" : pct(pred.released_man_rate, 1)}</td>
              <td>{pred.predicted_man_rate == null ? "n/a" : pct(pred.predicted_man_rate, 1)}</td><td>{pred.agreement == null ? "n/a" : pct(pred.agreement, 1)}</td><td>{pred.abstained}</td>
            </tr>
          </tbody>
        </table>
        <p className="mt-2 max-w-[70ch] text-xs text-muted">
          The model under-calls man, so a predicted rate is not a substitute for the released rate. With {pred.plays} plays for one defense the gap is mostly noise.
          Model {d.predicted.model}, snap + 1.5s window. <Link className="text-teal underline" data-testid="errors-link" href={`/coverage/errors?model=${MODEL}&window=${WINDOW}`}>Inspect this model&apos;s errors</Link>
        </p>
      </section>
    </div>
  );
}
