import { formatClock } from "@/lib/highlightTime";

type Props = { scores: number[]; loudness: number[]; label: number[]; commentary: { start_s: number; end_s: number }[]; start: number; end: number; cursor: number; model: string };
export default function ClipEvidence({ scores, loudness, label, commentary, start, end, cursor, model }: Props) {
  const a = Math.max(0, Math.floor(start / 2) - 8), b = Math.min(scores.length, Math.ceil(end / 2) + 9);
  const length = Math.max(1, b - a - 1);
  const W = 900;
  const x = (clip: number) => Math.max(0, Math.min(W, (clip - a) / length * W));
  const curve = (values: number[], top: number, height: number, min: number, max: number) => values.slice(a,b).map((v,i) => `${i ? "L" : "M"}${x(a+i).toFixed(2)},${(top+height-(Math.min(max,Math.max(min,v))-min)/(max-min)*height).toFixed(2)}`).join(" ");
  const lmin = Math.min(-2,...loudness.slice(a,b)), lmax = Math.max(6,...loudness.slice(a,b));
  const rank = scores[Math.max(0,Math.min(scores.length-1,cursor))];
  return (
    <figure className="clip-evidence">
      <figcaption className="clip-graph-title"><span className="kicker">Signals at this moment</span><span className="mono num">{formatClock(start)} — {formatClock(end)} · <span className="text-amber">rank {rank}</span></span></figcaption>
      <div className="clip-graph-grid">
        <div className="clip-lane-labels" aria-hidden="true"><span>MODEL RANK<small>0–100</small></span><span>AUDIO<small>above background</small></span><span>LABEL<small>evaluation only</small></span><span>SPEECH<small>transcript activity</small></span></div>
        <svg viewBox="0 0 900 190" preserveAspectRatio="none" className="clip-graph" role="img" aria-label={`${model} percentile ranks and measured audio loudness near the selected interval. Current rank ${rank} out of 100. Audio ${loudness[cursor]?.toFixed(1) ?? 'unavailable'} decibels above background. Blue labels are evaluation only.`}>
          {[14,39,64,90,119,143,165,185].map(y => <line key={y} x1="0" x2={W} y1={y} y2={y} stroke="#2b3a30" />)}
          <rect x={x(start/2)} y="0" width={Math.max(1,x(end/2)-x(start/2))} height="190" fill="#e8ad66" opacity=".08" />
          <path d={`${curve(scores,14,50,0,100)} L${x(b-1)},64 L0,64 Z`} fill="#d8f16e" opacity=".1" />
          <path d={curve(scores,14,50,0,100)} fill="none" stroke="#d8f16e" strokeWidth="2" vectorEffect="non-scaling-stroke" />
          <path d={curve(loudness,90,40,lmin,lmax)} fill="none" stroke="#e8ad66" strokeWidth="1.8" vectorEffect="non-scaling-stroke" />
          {label.slice(a,b).map((v,i)=>v ? <rect key={i} x={x(a+i)} y="145" width={W/length} height="13" fill="#8ebaf1" /> : null)}
          {commentary.filter(c=>c.end_s>=a*2 && c.start_s<=b*2).map(c=><rect key={c.start_s} x={x(c.start_s/2)} y="171" width={Math.max(1,x(c.end_s/2)-x(c.start_s/2))} height="11" fill="#9ba89d" opacity=".6" />)}
          <line x1={x(cursor)} x2={x(cursor)} y1="0" y2="190" stroke="#f2efe5" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
          <circle cx={x(cursor)} cy="6" r="4" fill="#f2efe5" />
        </svg>
      </div>
      <p className="clip-axis"><span>{formatClock(a*2)}</span><span className="text-amber">{formatClock(cursor*2)}</span><span>{formatClock((b-1)*2)}</span></p>
      <p className="clip-note">Saved measurements, not live audio analysis. Model ranks and loudness are separate signals; this plot does not establish feature attribution.</p>
    </figure>
  );
}
