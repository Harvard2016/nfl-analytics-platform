"use client";

import { useEffect, useRef, useState } from "react";
import Field from "@/components/Field";
import { type Play, loadDataset, loadPlay } from "@/lib/demo";

// One play on a loop. Stays on the snap frame when the visitor prefers reduced motion.
export default function HomeField() {
  const [play, setPlay] = useState<Play | null>(null);
  const [missing, setMissing] = useState(false);
  const [t, setT] = useState(0);
  const tRef = useRef(0);

  useEffect(() => {
    loadDataset().then(async (ds) => {
      if (!ds || ds.index.plays.length === 0) return setMissing(true);
      const first = ds.index.plays.find((p) => p.id === ds.index.sample.ids[0]) ?? ds.index.plays[0];
      setPlay(await loadPlay(ds, first.file));
    });
  }, []);

  useEffect(() => {
    if (!play || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const first = play.frames[0], end = play.frames[play.frames.length - 1];
    tRef.current = first;
    let raf = 0, last = performance.now(), hold = 0;
    const tick = (now: number) => {
      const dt = now - last; last = now;
      if (tRef.current >= end) { hold += dt; if (hold > 1200) { tRef.current = first; hold = 0; } }
      else tRef.current = Math.min(end, tRef.current + dt / 100);
      setT(tRef.current);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [play]);

  if (missing) return <p className="border border-line p-4 text-sm text-muted">No exported plays yet. Run the coverage pipeline to see a play here.</p>;
  if (!play) return <div className="aspect-[53/42] w-full bg-turf" />;
  return (
    <div className="flex flex-col gap-2">
      <Field entities={play.entities} frames={play.frames} t={t} losX={play.los_x} cutoff={Infinity} selected={null} onSelect={() => {}} showTrails />
    </div>
  );
}
