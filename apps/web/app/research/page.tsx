import type { Metadata } from "next";
import Link from "next/link";
import ResearchCases, { NotebookPreview } from "@/components/ResearchCases";

export const metadata: Metadata = { title: "Research" };

const H2 = "display text-3xl sm:text-4xl";

export default function Page() {
  return (
    <>
      <article className="paper">
        <header className="research-masthead mx-auto max-w-[1500px] px-4 pb-10 pt-12 lg:px-8 lg:pt-20">
          <p className="kicker">Research</p>
          <h1 className="display mt-2 text-7xl sm:text-9xl">The playbook<span className="text-teal">.</span></h1>
          <p className="mt-5 max-w-[62ch] text-lg text-muted">What was tried, what held up and what did not. Three independent systems, each with its own data, target, split and limits. Every number below is read from a frozen run record, so the site and the reports cannot drift apart.</p>
          <nav aria-label="Case studies" className="narrow mt-6 flex flex-wrap gap-x-6 gap-y-1 text-xl">
            <a className="underline decoration-line underline-offset-4 hover:decoration-teal" href="#coverage">01 Coverage</a>
            <a className="underline decoration-line underline-offset-4 hover:decoration-teal" href="#highlights">02 Highlights</a>
            <a className="underline decoration-line underline-offset-4 hover:decoration-teal" href="#predictions">03 Game prediction</a>
            <a className="underline decoration-line underline-offset-4 hover:decoration-teal" href="#notebook">Experiment notebook</a>
          </nav>
        </header>

        <ResearchCases />

        <div className="mx-auto grid max-w-[1500px] gap-10 border-t border-line px-4 py-12 lg:grid-cols-2 lg:gap-14 lg:px-8">
          <section aria-labelledby="r-null">
            <h2 id="r-null" className={H2}>Negative and null results</h2>
            <ul className="mt-4 max-w-[62ch] list-disc space-y-2 pl-5 text-sm text-muted">
              <li>Game predictor: nothing reliably beats Elo, including after adding efficiency, rate and quarterback features.</li>
              <li>Game predictor: rest, bye and recent-form features add nothing measurable on their own.</li>
              <li>Coverage: team, down and distance alone barely beat the class prior.</li>
              <li>Coverage: counting tracked defenders helped the snap model but depends on a post-play choice by the release, so it was removed.</li>
              <li>Coverage: Apple GPU (MPS) training was slower than CPU for the small temporal model.</li>
              <li>Highlights: adding commentary to the temporal model lowered test mean average precision; see the benchmark table.</li>
            </ul>
          </section>
          <section aria-labelledby="r-limits">
            <h2 id="r-limits" className={H2}>Limits that will not change</h2>
            <p className="mt-4 max-w-[62ch] text-sm text-muted">Scores measure agreement with released or editorial labels, not ground truth. Feature contributions and sensitivities describe a fitted model, not cause and effect. Coverage tendencies are counts over tracked pass plays, not every snap. Nothing here is betting advice.</p>
            <p className="mt-3 max-w-[62ch] text-sm text-muted">Sources, licences, what may be displayed and what was never used are listed on the <Link className="text-teal underline underline-offset-4" href="/engineering">engineering page</Link>. The first benchmark is frozen in a manifest with data checksums, split ids, model hashes and metrics, and is never overwritten.</p>
          </section>
        </div>
      </article>

      <section id="notebook" aria-labelledby="r-notebook" className="grain scroll-mt-20 border-t border-line">
        <div className="mx-auto max-w-[1500px] px-4 py-14 lg:px-8">
          <p className="kicker">Frozen run records</p>
          <h2 id="r-notebook" className="display mt-1 text-5xl sm:text-6xl">Experiment notebook<span className="text-teal">.</span></h2>
          <p className="mt-3 max-w-[62ch] text-muted">Every training and evaluation run writes one record: data, split, cutoff, settings, time and outputs. Superseded and negative runs stay in the list. The site reads a frozen export; it does not query a live tracker.</p>
          <NotebookPreview />
          <p className="mt-6"><Link className="btn-quiet" href="/research/experiments">Open every run record</Link></p>
        </div>
      </section>
    </>
  );
}
