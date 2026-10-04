// Read contract for exported coverage demo files (schema "coverage-demo-v3").
// The site only presents saved outputs; all model computation happens in the Python pipeline.

export type HorizonKey = "at_snap" | "post_0_5s" | "post_1s" | "post_1_5s";
export const HORIZONS: { key: HorizonKey; label: string; detail: string }[] = [
  { key: "at_snap", label: "Snap", detail: "the released players' positions at the snap frame only" },
  { key: "post_0_5s", label: "+0.5s", detail: "tracking up to 0.5 seconds after the snap" },
  { key: "post_1s", label: "+1.0s", detail: "tracking up to 1 second after the snap" },
  { key: "post_1_5s", label: "+1.5s", detail: "tracking up to 1.5 seconds after the snap" },
];
export const DEFAULT_HORIZON: HorizonKey = "post_1_5s";
export type ModelKey = "v1_gbm" | "v1_logit" | "v2_gbm_rel" | "v2_temporal";
export const MODELS: ModelKey[] = ["v1_gbm", "v2_gbm_rel", "v2_temporal", "v1_logit"];
export const DEFAULT_MODEL: ModelKey = "v1_gbm";
const ALIAS: Record<string, string> = { geometry_gbm: "v1_gbm", geometry_logit: "v1_logit" };       // links written before v3

export const horizonFrom = (v: string | null): HorizonKey => HORIZONS.find((w) => w.key === v)?.key ?? DEFAULT_HORIZON;
export const modelFrom = (v: string | null): ModelKey => MODELS.find((m) => m === (ALIAS[v ?? ""] ?? v)) ?? DEFAULT_MODEL;

export type Label = "Man" | "Zone";
export type Lean = { p_man: number; predicted: Label; accepted: boolean };

// One definition used by every badge, count, filter and colour:
//   abstained = confidence below the saved cutoff (whatever the lean)
//   correct   = accepted and the lean equals the released label
//   incorrect = accepted and the lean differs from the released label
export type Status = "correct" | "incorrect" | "abstained";
export const statusOf = (lean: Lean, label: Label): Status => (!lean.accepted ? "abstained" : lean.predicted === label ? "correct" : "incorrect");
export const STATUS_TEXT: Record<Status, string> = { correct: "Accepted, agrees", incorrect: "Accepted, disagrees", abstained: "Abstained" };
export const STATUS_CLASS: Record<Status, string> = { correct: "text-teal", incorrect: "text-coral", abstained: "text-muted" };

export type PlaySummary = {
  id: string; file: string; week: number; season: number; offense: string; defense: string;
  quarter: number; down: number; yardsToGo: number; yardsToGoal: number; frames_observed: number;
  released: { manZone: Label; coverage: string | null; source: string };
  predictions: Record<ModelKey, Partial<Record<HorizonKey, Lean | null>>>;
  split: string;
};

export type Entity = {
  id: string | null; side: "offense" | "defense" | "ball" | "unknown"; passer?: boolean;
  name: string | null; position: string | null; jersey: number | null;
  x: (number | null)[]; y: (number | null)[]; s: (number | null)[]; o: (number | null)[];
};

export type Prediction = { version: string; p_man: number; p_zone: number; predicted: Label; confidence: number; confidence_threshold: number; accepted: boolean };
export type Term = { feature: string; value: number | null; log_odds_toward_man: number };
export type Ablation = { groups: { group: string; log_odds_toward_man: number }[]; players: { id: string; log_odds_toward_man: number }[] };
export type Pair = { defender: string; receiver: string; separation: number; closing_speed?: number; separation_at_snap?: number };
export type HorizonData = {
  latest_frame: number; predictions: Partial<Record<ModelKey, Prediction>>;
  explanations: Partial<Record<ModelKey, Term[] | Ablation>>;
  evidence: { pairs: Pair[]; nearest: Pair[]; deepest: string[] };
};
export type Similar = { id: string; week: number; offense: string; defense: string; down: number; yardsToGo: number; released: { manZone: Label; coverage: string | null };
  distance: number; tracks: { defense: number[][][]; offense: number[][][] } };

export type Play = PlaySummary & {
  description: string | null; los_x: number; ref_y: number; frames: number[]; entities: Entity[];
  horizons: Record<HorizonKey, HorizonData | null>;
  similar: { pool: string; basis: string; note: string; plays: Similar[] };
};

export type DemoIndex = {
  synthetic: boolean; generated_at: string; source: string; display_rights: string; eligible_test_plays: number;
  models: Record<ModelKey, { label: string; short: string; horizons: HorizonKey[]; explain: string; explanation: string }>;
  default_model: ModelKey; default_horizon: HorizonKey;
  cutoffs: Record<ModelKey, Partial<Record<HorizonKey, number>>>; versions: Record<ModelKey, Partial<Record<HorizonKey, string>>>;
  decision_rule: string; benchmark_note: string;
  sample: { ids: string[]; seed: number; note: string };
  quick_throws: { ids: string[]; seed: number; available: number; note: string };
  errors: { ids: Record<ModelKey, Partial<Record<HorizonKey, string[]>>>; available: Record<ModelKey, Partial<Record<HorizonKey, number>>>; seed: number; note: string };
  feature_descriptions: Record<string, { text: string; evidence: "pairs" | "nearest" | "deepest" | "all" }>;
  plays: PlaySummary[];
};

export type Dataset = { base: string; index: DemoIndex };

export async function loadDataset(): Promise<Dataset | null> {
  try {
    const r = await fetch("/demo/real/coverage/index.json", { cache: "no-store" });
    if (r.ok) return { base: "/demo/real/coverage", index: (await r.json()) as DemoIndex };
  } catch { /* fall through */ }
  return null;
}

export async function loadPlay(ds: Dataset, file: string): Promise<Play> {
  const r = await fetch(`${ds.base}/${file}`);
  if (!r.ok) throw new Error(`Play file missing: ${file}`);
  return (await r.json()) as Play;
}

export async function loadJson<T>(ds: Dataset, name: string): Promise<T | null> {
  try {
    const r = await fetch(`${ds.base}/${name}`, { cache: "no-store" });
    return r.ok ? ((await r.json()) as T) : null;
  } catch { return null; }
}

export const TYPE_LABELS: Record<string, string> = {
  COVER_0_MAN: "Cover 0 (man)", COVER_1_MAN: "Cover 1 (man)", COVER_2_MAN: "Cover 2 man", COVER_2_ZONE: "Cover 2 (zone)",
  COVER_3_ZONE: "Cover 3 (zone)", COVER_4_ZONE: "Cover 4 (zone)", COVER_6_ZONE: "Cover 6 (zone)", PREVENT: "Prevent",
};

export const ordinal = (n: number) => `${n}${["th", "st", "nd", "rd"][n % 10 > 3 || (n % 100 > 10 && n % 100 < 14) ? 0 : n % 10]}`;
export const pct = (v: number, d = 0) => `${(v * 100).toFixed(d)}%`;
export const signed = (v: number, d = 3) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
