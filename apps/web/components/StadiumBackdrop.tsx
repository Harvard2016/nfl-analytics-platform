import Image from "next/image";

/** Original, generated stadium artwork. Decoration only, never game evidence. */
export default function StadiumBackdrop({ hero = false }: { hero?: boolean }) {
  return (
    <div className={`stadium-backdrop ${hero ? "stadium-backdrop--hero" : ""}`} aria-hidden="true">
      <Image src="/art/stadium-dusk.webp" alt="" fill sizes="100vw" preload={hero} className="stadium-photo" />
      <div className="stadium-shade" />
    </div>
  );
}

export function ChalkRoutes() {
  return (
    <svg className="hero-routes" viewBox="0 0 900 400" fill="none" aria-hidden="true">
      <defs><marker id="chalk-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="currentColor" /></marker></defs>
      <g stroke="currentColor" strokeWidth="2.5" className="chalk-offense">
        <path className="route" pathLength="1" markerEnd="url(#chalk-arrow)" d="M160 300C180 250 240 210 290 130L355 80" />
        <path className="route" pathLength="1" markerEnd="url(#chalk-arrow)" d="M285 320C300 220 480 230 620 140" style={{ "--d": "850ms" } as React.CSSProperties} />
        <path className="route" pathLength="1" markerEnd="url(#chalk-arrow)" d="M420 340L450 240Q465 210 540 220L790 245" style={{ "--d": "1000ms" } as React.CSSProperties} />
        {[ [160,300], [285,320], [420,340] ].map(([x,y]) => <circle key={x} className="marker" cx={x} cy={y} r="8" />)}
      </g>
      <g stroke="#8ebaf1" strokeWidth="3" opacity=".85">
        {[ [250,180], [400,190], [610,110], [740,260] ].map(([x,y]) => <path key={x} className="marker" d={`M${x-6} ${y-6}l12 12m0-12l-12 12`} />)}
        <path d="M400 190Q420 100 520 105" className="route" pathLength="1" />
      </g>
      <g stroke="#d8f16e" strokeWidth="3"><path className="route" pathLength="1" d="M550 325Q620 235 590 150L560 80" markerEnd="url(#chalk-arrow)" /><circle cx="550" cy="325" r="9" className="marker" /></g>
    </svg>
  );
}
