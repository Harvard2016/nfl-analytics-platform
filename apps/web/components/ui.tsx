import type { ReactNode } from "react";

const BADGE: Record<string, string> = {
  Observed: "border-ink/40 text-ink",
  "Released label": "border-defense/60 text-defense",
  Predicted: "border-teal/60 text-teal",
  Derived: "border-muted/50 text-muted",
  Synthetic: "border-amber text-amber",
};

// Says where a value came from. Never rely on colour alone: the word is always shown.
export function Badge({ kind }: { kind: keyof typeof BADGE | string }) {
  return (
    <span className={`narrow inline-block border px-1.5 py-px text-[11px] leading-4 ${BADGE[kind] ?? BADGE.Derived}`}>{kind}</span>
  );
}

export function SyntheticBanner({ children }: { children?: ReactNode }) {
  return (
    <div role="note" className="border border-amber/60 bg-amber/10 px-4 py-2.5 text-sm text-amber">
      <strong className="font-semibold">Synthetic fixture.</strong>{" "}
      {children ?? "These plays, labels and numbers are invented to exercise the interface. They are not NFL data and not model results."}
    </div>
  );
}

// The source limit that has to travel with every coverage result.
export function SelectionNote() {
  return (
    <p role="note" className="border-l-2 border-amber bg-surface px-3 py-2 text-sm text-muted">
      <strong className="font-semibold text-ink">Read this first.</strong> These are predictions from the released tracking, which
      covers only the passer, the route runners and the defenders the release marks as coverage players. That choice of defenders
      was made with knowledge of how the play went, and every measurement here is computed from that chosen group. So this is not a
      system that knows only what was visible at the snap, and it is not ready to run on live games.
    </p>
  );
}

export function Bar({ value, tone = "teal" }: { value: number; tone?: "teal" | "defense" | "muted" }) {
  const color = { teal: "bg-teal", defense: "bg-defense", muted: "bg-muted" }[tone];
  return (
    <div className="h-2 w-full bg-line">
      <div className={`h-2 ${color}`} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}
