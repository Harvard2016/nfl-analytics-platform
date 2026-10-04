import Link from "next/link";
import HomeChapters from "@/components/HomeChapters";
import Parallax from "@/components/Parallax";
import StadiumScene from "@/components/StadiumScene";

const d = (ms: number) => ({ "--d": `${ms}ms` }) as React.CSSProperties;

export default function Home() {
  return (
    <>
      <section className="relative isolate overflow-hidden border-b border-line">
        <Parallax className="absolute inset-0 -z-10"><StadiumScene className="h-full w-full" /></Parallax>
        <div className="mx-auto flex min-h-[min(78vh,760px)] max-w-[1500px] flex-col justify-center px-4 py-16 lg:px-8">
          <p className="kicker rise" style={d(0)}>NFL analytics / three independent systems</p>
          <h1 className="display mt-4 text-[5.2rem] leading-[0.82] sm:text-[9rem] lg:text-[12.5rem]">
            <span className="rise block" style={d(80)}>Read</span>
            <span className="rise block" style={d(200)}>the <span className="outline-text">field</span><span className="text-teal">.</span></span>
          </h1>
          <p className="rise mt-6 max-w-[34ch] text-xl sm:text-2xl" style={d(360)}>Three systems. One place to inspect the evidence.</p>
          <p className="rise mt-8 flex flex-wrap items-center gap-6" style={d(480)}>
            <Link href="/coverage?model=v2_temporal" className="btn-primary">Enter the film room <span aria-hidden="true">→</span></Link>
            <Link href="/research" className="btn-quiet">Explore the research</Link>
          </p>
          <p className="kicker rise mt-12 max-w-[60ch] normal-case tracking-normal" style={d(600)}>Illustration, not footage. Every number on this site comes from a saved model record with its limits stated beside it.</p>
        </div>
      </section>

      <HomeChapters />

      <section className="grain border-t border-line">
        <div className="mx-auto grid max-w-[1500px] gap-10 px-4 py-16 md:grid-cols-2 lg:px-8">
          <div>
            <p className="kicker">Research</p>
            <h2 className="display mt-3 text-5xl sm:text-6xl">The playbook.</h2>
            <p className="mt-4 max-w-[48ch] text-muted">What was tried, what held up and what did not: development folds, paired comparisons, calibration, error galleries and negative results, all read from frozen run records.</p>
            <p className="mt-6"><Link href="/research" className="btn-quiet">Read the case studies</Link></p>
          </div>
          <div>
            <p className="kicker">Engineering</p>
            <h2 className="display mt-3 text-5xl sm:text-6xl">Built to be inspected.</h2>
            <p className="mt-4 max-w-[48ch] text-muted">One product, three separate pipelines. Local Python training, versioned outputs, JSON run records, export checks and a site that only reads saved files.</p>
            <p className="mt-6"><Link href="/engineering" className="btn-quiet">See how it is built</Link></p>
          </div>
        </div>
      </section>
    </>
  );
}
