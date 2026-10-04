import Link from "next/link";
import HomeChapters from "@/components/HomeChapters";
import Parallax from "@/components/Parallax";
import StadiumBackdrop, { ChalkRoutes } from "@/components/StadiumBackdrop";

const d = (ms: number) => ({ "--d": `${ms}ms` }) as React.CSSProperties;

export default function Home() {
  return (
    <>
      <section className="home-hero relative isolate overflow-hidden border-b border-line">
        <Parallax amount={0.08} className="absolute inset-0 -z-10"><StadiumBackdrop hero /></Parallax>
        <ChalkRoutes />
        <div className="hero-copy mx-auto flex max-w-[1500px] flex-col justify-center px-5 lg:px-10">
          <p className="kicker rise" style={d(0)}>NFL analytics / three independent systems</p>
          <h1 className="display hero-title mt-5">
            <span className="rise block" style={d(80)}>Read</span>
            <span className="rise block" style={d(200)}>the <span className="outline-text">field</span><span className="text-teal">.</span></span>
          </h1>
          <p className="rise mt-6 max-w-[34ch] text-xl sm:text-2xl" style={d(360)}>Three systems. One place to inspect the evidence.</p>
          <p className="rise mt-8 flex flex-wrap items-center gap-6" style={d(480)}>
            <Link href="/coverage?model=v2_temporal" className="btn-primary">Enter the film room <span aria-hidden="true">→</span></Link>
            <Link href="/research" className="btn-quiet">Explore the research</Link>
          </p>
          <p className="hero-caption rise" style={d(600)}>Original stadium artwork · illustrative routes</p>
        </div>
        <div className="hero-rail" aria-hidden="true"><span>Different perspective.<br />Same game.</span><span className="hero-scroll">Explore ↓</span></div>
      </section>
      <div className="platform-strip"><span className="kicker">Inside the lens</span><Link href="/coverage?model=v2_temporal">01 <span>Read the defense</span> ↗</Link><Link href="/highlights">02 <span>Find the moment</span> ↗</Link><Link href="/predictions">03 <span>Before kickoff</span> ↗</Link></div>

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
