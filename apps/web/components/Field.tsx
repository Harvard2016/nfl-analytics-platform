"use client";

import type { Entity } from "@/lib/demo";

// Playbook orientation: offense at the bottom moving up the screen, sideline to sideline across.
// Data is in yards with the offense moving toward +x; +y is the offense's left.
const WIDTH = 53.3;
const BEHIND = 12;
const AHEAD = 30;
const HASH = [23.58, 29.72];
const TRAIL = 15;

export type Highlight = { players: string[]; links: [string, string][]; note: string } | null;

type Props = {
  entities: Entity[];
  frames: number[];
  t: number; // playback time in frames, may be fractional; positions between frames are interpolated for display only
  losX: number;
  cutoff: number; // last frame the selected model was given; independent of playback
  selected: string | null;
  onSelect: (id: string | null) => void;
  showTrails: boolean;
  highlight?: Highlight;
  labels?: boolean; // position abbreviation under every marker
};

function at(values: (number | null)[], i: number, frac: number): number | null {
  const a = values[i];
  const b = values[Math.min(i + 1, values.length - 1)];
  if (a == null) return null;
  if (b == null || frac === 0) return a;
  return a + (b - a) * frac;
}

export default function Field({ entities, frames, t, losX, cutoff, selected, onSelect, showTrails, highlight, labels = false }: Props) {
  const x0 = Math.max(0, losX - BEHIND);
  const x1 = Math.min(120, losX + AHEAD);
  const H = x1 - x0;
  const sx = (y: number) => WIDTH - y;
  const sy = (x: number) => x1 - x;
  const pos = Math.max(0, Math.min(frames.length - 1, t - frames[0]));
  const i = Math.floor(pos);
  const frac = pos - i;
  const hidden = t > cutoff;
  const hot = new Set(highlight?.players ?? []);
  const byId = new Map(entities.map((e) => [e.id, e]));
  const cutIdx = Math.max(0, Math.min(frames.length - 1, cutoff - frames[0]));
  const lines: number[] = [];
  for (let x = Math.ceil(x0 / 5) * 5; x <= x1; x += 5) if (x >= 10 && x <= 110) lines.push(x);
  const observed = entities.filter((e) => at(e.x, i, frac) != null).length;

  return (
    <svg viewBox={`0 0 ${WIDTH} ${H}`} className="block h-auto w-full select-none bg-turf" role="img"
      aria-label={`Field view at ${(t / 10).toFixed(1)} seconds after the snap. ${observed} of ${entities.length} tracked players observed at this frame.${highlight ? ` Highlighted: ${highlight.note}` : ""}`}
      onClick={() => onSelect(null)}>
      {x1 > 110 && <rect x={0} y={sy(x1)} width={WIDTH} height={x1 - 110} fill="#16291f" />}
      {x0 < 10 && <rect x={0} y={sy(10)} width={WIDTH} height={10 - x0} fill="#16291f" />}
      {lines.filter((x) => (x / 5) % 2 === 0).map((x) => <rect key={`b${x}`} x={0} y={sy(x + 5)} width={WIDTH} height={5} fill="#f2efe5" opacity={0.035} />)}
      {lines.map((x) => (
        <g key={x}>
          <line x1={0} x2={WIDTH} y1={sy(x)} y2={sy(x)} stroke="#f2efe5" strokeOpacity={x % 10 === 0 ? 0.42 : 0.2} strokeWidth={x % 10 === 0 ? 0.16 : 0.09} />
          {x % 10 === 0 && x > 10 && x < 110 && [3.2, WIDTH - 3.2].map((nx) => <text key={nx} x={nx} y={sy(x) - 0.6} fontSize={2.6} textAnchor="middle" fill="#f2efe5" fillOpacity={0.45} className="display num">{50 - Math.abs(x - 60)}</text>)}
        </g>
      ))}
      {HASH.map((h) => Array.from({ length: Math.floor(H) + 1 }, (_, k) => Math.ceil(x0) + k).filter((x) => x > 10 && x < 110 && x <= x1)
        .map((x) => <line key={`${h}-${x}`} x1={sx(h) - 0.35} x2={sx(h) + 0.35} y1={sy(x)} y2={sy(x)} stroke="#f2efe5" strokeOpacity={0.3} strokeWidth={0.1} />))}
      <line x1={0} x2={WIDTH} y1={sy(losX)} y2={sy(losX)} stroke="#d8f16e" strokeWidth={0.26} opacity={0.95} />
      <text x={WIDTH / 2} y={sy(losX) + 1.9} fontSize={1.4} fill="#d8f16e" textAnchor="middle" className="narrow">Line of scrimmage</text>

      {/* evidence: the observed path of each highlighted player from the snap to the model's cutoff, and the pair links at the cutoff */}
      {highlight && highlight.players.map((id) => {
        const e = byId.get(id);
        if (!e) return null;
        const pts: string[] = [];
        for (let k = 0; k <= cutIdx; k++) { const x = e.x[k], y = e.y[k]; if (x != null && y != null) pts.push(`${sx(y).toFixed(2)},${sy(x).toFixed(2)}`); }
        return pts.length > 1 ? <polyline key={`ev-${id}`} points={pts.join(" ")} fill="none" stroke="#e8ad66" strokeWidth={0.45} strokeLinecap="round" opacity={0.9} /> : null;
      })}
      {highlight && highlight.links.map(([a, b]) => {
        const d = byId.get(a), r = byId.get(b);
        const dx = d?.x[cutIdx], dy = d?.y[cutIdx], rx = r?.x[cutIdx], ry = r?.y[cutIdx];
        return dx != null && dy != null && rx != null && ry != null
          ? <line key={`ln-${a}-${b}`} x1={sx(dy)} y1={sy(dx)} x2={sx(ry)} y2={sy(rx)} stroke="#e8ad66" strokeWidth={0.2} strokeDasharray="0.6 0.5" /> : null;
      })}

      {showTrails && entities.map((e) => {
        if (e.side === "ball") return null;
        const pts: string[] = [];
        for (let k = Math.max(0, i - TRAIL); k <= i; k++) { const x = e.x[k], y = e.y[k]; if (x != null && y != null) pts.push(`${sx(y).toFixed(2)},${sy(x).toFixed(2)}`); }
        const cx = at(e.x, i, frac), cy = at(e.y, i, frac);
        if (cx != null && cy != null) pts.push(`${sx(cy).toFixed(2)},${sy(cx).toFixed(2)}`);
        if (pts.length < 2) return null;
        return <polyline key={`trail-${e.id}`} points={pts.join(" ")} fill="none" strokeWidth={0.22} strokeLinecap="round" stroke={e.side === "defense" ? "#8ebaf1" : "#f2efe5"} opacity={0.3} />;
      })}

      {entities.map((e, k) => {
        const x = at(e.x, i, frac), y = at(e.y, i, frac);
        if (x == null || y == null) return null; // no observation: draw nothing rather than a guess
        const px = sx(y), py = sy(x);
        if (py < -1 || py > H + 1) return null;
        const isSel = selected != null && e.id === selected;
        const isHot = e.id != null && hot.has(e.id);
        const dim = highlight != null && !isHot && !isSel ? 0.35 : 1;
        const o = e.o[i];
        const rad = o == null ? null : (o * Math.PI) / 180;
        return (
          <g key={e.id ?? k} transform={`translate(${px} ${py})`} className="cursor-pointer" opacity={dim} onClick={(ev) => { ev.stopPropagation(); onSelect(e.id); }}>
            {(isSel || isHot) && <circle r={1.7} fill="none" stroke="#e8ad66" strokeWidth={isSel ? 0.3 : 0.18} />}
            {rad != null && <line x1={0} y1={0} x2={-Math.cos(rad) * 1.5} y2={-Math.sin(rad) * 1.5} stroke={e.side === "defense" ? "#8ebaf1" : "#f2efe5"} strokeWidth={0.16} opacity={0.7} />}
            {e.side === "defense" ? (
              <g stroke="#8ebaf1" strokeWidth={0.34} strokeLinecap="round"><line x1={-0.62} y1={-0.62} x2={0.62} y2={0.62} /><line x1={-0.62} y1={0.62} x2={0.62} y2={-0.62} /></g>
            ) : (
              <circle r={0.78} fill={e.passer ? "#f2efe5" : "#13241b"} stroke="#f2efe5" strokeWidth={0.26} />
            )}
            {labels && !isSel && !isHot && <text y={2.45} fontSize={1.05} textAnchor="middle" fill="#f2efe5" fillOpacity={0.7} className="mono">{e.position ?? ""}</text>}
            {(isSel || isHot) && <text y={-2.1} fontSize={1.4} textAnchor="middle" fill="#e8ad66" className="narrow num" fontWeight={600}>{e.position ?? ""}</text>}
            <circle r={1.4} fill="transparent" />
          </g>
        );
      })}
      {hidden && <text x={0.8} y={1.9} fontSize={1.35} fill="#e8ad66" className="narrow">Past the model&apos;s input cutoff: this prediction never saw these frames</text>}
    </svg>
  );
}
