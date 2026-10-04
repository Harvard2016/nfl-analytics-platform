// Original decorative illustration drawn in SVG: a night stadium, a field in perspective and a few route lines.
// It is not a photograph, not broadcast footage and not tracking data. Hidden from assistive technology.
export default function StadiumScene({ className = "" }: { className?: string }) {
  const lights = [[720, 70], [930, 52], [1130, 78], [1310, 130]];
  return (
    <svg viewBox="0 0 1440 720" preserveAspectRatio="xMidYMid slice" className={className} aria-hidden="true" focusable="false">
      <defs>
        <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#0b120f" /><stop offset="0.3" stopColor="#16221c" /><stop offset="0.52" stopColor="#6b4424" /><stop offset="0.66" stopColor="#101713" />
        </linearGradient>
        <radialGradient id="glow" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stopColor="#f2efe5" stopOpacity="0.55" /><stop offset="1" stopColor="#f2efe5" stopOpacity="0" /></radialGradient>
        <linearGradient id="turf" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#1c3527" /><stop offset="1" stopColor="#0f1c15" /></linearGradient>
        <linearGradient id="fade" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stopColor="#101713" stopOpacity="0.94" /><stop offset="0.42" stopColor="#101713" stopOpacity="0.55" /><stop offset="0.75" stopColor="#101713" stopOpacity="0" /></linearGradient>
      </defs>
      <rect width="1440" height="720" fill="url(#sky)" />
      {/* stands */}
      <path d="M0 330 L260 250 L620 215 L1000 240 L1440 320 L1440 470 L0 470 Z" fill="#0d1511" />
      {Array.from({ length: 7 }, (_, i) => <path key={i} d={`M0 ${350 + i * 16} L300 ${284 + i * 16} L640 ${252 + i * 16} L1010 ${276 + i * 16} L1440 ${352 + i * 16}`} fill="none" stroke="#f2efe5" strokeOpacity={0.05} />)}
      <path d="M0 370 L300 300 L640 268 L1010 292 L1440 372 L1440 470 L0 470 Z" fill="#131c17" />
      {Array.from({ length: 90 }, (_, i) => <circle key={i} cx={(i * 163) % 1440} cy={300 + ((i * 71) % 150)} r={1.1} fill="#e8ad66" opacity={0.18 + ((i * 7) % 10) / 40} />)}
      {/* floodlight banks */}
      {lights.map(([x, y], i) => (
        <g key={i}>
          <circle cx={x} cy={y} r={150} fill="url(#glow)" opacity={0.6} />
          <line x1={x} y1={y + 14} x2={x} y2={y + 190} stroke="#0d1511" strokeWidth={6} />
          {Array.from({ length: 12 }, (_, k) => <circle key={k} cx={x - 33 + (k % 6) * 13} cy={y - 6 + Math.floor(k / 6) * 12} r={4} fill="#f2efe5" />)}
        </g>
      ))}
      {/* field in perspective */}
      <path d="M-200 720 L330 450 L1110 450 L1640 720 Z" fill="url(#turf)" />
      {Array.from({ length: 11 }, (_, i) => { const t = i / 10; return <line key={i} x1={330 + t * 780} y1={450} x2={-200 + t * 1840} y2={720} stroke="#f2efe5" strokeOpacity={0.22} strokeWidth={1.5} />; })}
      {[0.25, 0.55, 0.85].map((t) => <line key={t} x1={330 - t * 530} y1={450 + t * 270} x2={1110 + t * 530} y2={450 + t * 270} stroke="#f2efe5" strokeOpacity={0.1} />)}
      {/* route lines and markers, drawn in after the headline */}
      <g fill="none" strokeLinecap="round" strokeWidth={3}>
        <path className="route" pathLength={1} style={{ "--d": "700ms" } as React.CSSProperties} d="M840 640 C 860 560, 900 520, 980 500" stroke="#f2efe5" />
        <path className="route" pathLength={1} style={{ "--d": "900ms" } as React.CSSProperties} d="M1010 660 C 1030 590, 1120 560, 1210 520" stroke="#d8f16e" />
        <path className="route" pathLength={1} style={{ "--d": "1100ms" } as React.CSSProperties} d="M700 650 C 690 590, 640 560, 590 540" stroke="#f2efe5" />
        <path className="route" pathLength={1} style={{ "--d": "1250ms" } as React.CSSProperties} d="M930 500 C 1000 530, 1060 520, 1100 500" stroke="#8ebaf1" strokeDasharray="0.02 0.02" />
      </g>
      {[[840, 640], [1010, 660], [700, 650]].map(([x, y], i) => <circle key={i} className="marker" style={{ "--d": `${500 + i * 120}ms` } as React.CSSProperties} cx={x} cy={y} r={9} fill="#101713" stroke="#f2efe5" strokeWidth={3} />)}
      {[[930, 500], [1180, 560], [620, 520], [1090, 610]].map(([x, y], i) => (
        <g key={i} className="marker" style={{ "--d": `${800 + i * 120}ms` } as React.CSSProperties} stroke="#8ebaf1" strokeWidth={4} strokeLinecap="round">
          <line x1={x - 8} y1={y - 8} x2={x + 8} y2={y + 8} /><line x1={x - 8} y1={y + 8} x2={x + 8} y2={y - 8} />
        </g>
      ))}
      <rect width="1440" height="720" fill="url(#fade)" />
    </svg>
  );
}
