// Read contract for the pregame v2 export. Independent of the coverage and highlight files.
export type PModel = "home_prior" | "elo" | "v1_logistic" | "v1_boosted_trees" | "elo_offset" | "margin" | "blend";
export const PMODELS: PModel[] = ["home_prior", "elo", "v1_logistic", "v1_boosted_trees", "elo_offset", "margin", "blend"];

export type Term = { feature: string; description: string; standardized?: number; log_odds: number };
export type Game = {
  id: string; season: number; week: number; type: string; date: string; home: string; away: string;
  home_score: number; away_score: number; home_won: boolean; p: Record<PModel, number>; elo: [number, number];
  elo_logit: number; intercept: number; terms: Term[];
  qb: { home: string | null; away: string | null; home_dropbacks: number; away_dropbacks: number };
  predicted_margin: number; trained_on_games: number; missing_inputs: string[];
};
export type Games = { version: string; kind: string; cutoff: string; qb_assumption: string; reconstructed: string; labels: Record<PModel, string>; games: Game[] };

export type Score = {
  games: number; log_loss: number; brier: number; accuracy: number; home_win_rate: number; calibration_slope: number; calibration_intercept: number;
  calibration: { lo: number; hi: number; games: number; mean_predicted: number; home_won: number }[];
  vs_elo?: { mean_log_loss_difference: number; interval_95: [number, number]; share_of_resamples_better: number; blocks: number };
};
export type Block = Record<PModel, Score>;
export type Performance = {
  version: string; run_at: string; target: string; cutoff: string; cutoff_violations: number; reconstructed: string; qb_assumption: string; not_used: string[];
  labels: Record<PModel, string>; configurations_tried: number; features: Record<string, string>;
  selection: { development_seasons: [number, number]; games: number; elo_log_loss: number; rule: string; note: string;
    chosen: Record<"elo_offset" | "margin", { groups: string; window: string | null; penalty: number; log_loss: number; vs_elo: number; seasons_better_than_elo: number }>;
    blend: { weight_elo_offset: number; log_loss: number } };
  ablations: { groups: string; window: string | null; penalty: number; log_loss: number; vs_elo: number; seasons_better_than_elo: number }[];
  development: { seasons: [number, number]; models: Block; by_season: Record<string, Block> };
  previously_examined_benchmark: { seasons: [number, number]; note: string; models: Block; by_season: Record<string, Block> };
  in_progress: { seasons: number[]; through: string; models: Block } | null;
  closing_market_reference: { note: string; games: number; market_log_loss: number; elo_log_loss_same_games: number; elo_offset_log_loss_same_games: number } | null;
  rolling_100_games: { through: string; games: number; elo_log_loss: number; elo_brier: number; elo_offset_log_loss: number; elo_offset_brier: number }[];
};

export type Forecast = {
  record_id: string; kind: string; game_id: string; season: number; week: number; home: string; away: string; kickoff_eastern: string; cutoff_eastern: string;
  created_at_utc: string; created_before_cutoff: boolean; timing_note: string | null; model_version: string; trained_on_games: number; calibrator: string;
  data_snapshot: { games_csv_sha256: string; games_csv_modified: string; last_completed_game: string; completed_games: number };
  qb_assumption: { home: string | null; away: string | null; basis: string };
  features: Record<string, number | null>; probabilities: Record<"elo" | "elo_offset" | "margin" | "blend", number>;
  explanation: { unit: string; elo_logit: number; intercept: number; terms: Term[] }; predicted_margin: number;
  outcome_attached: { home_score: number; away_score: number; home_won: boolean; tie: boolean; attached_at_utc: string } | null;
};
export type Forecasts = { generated_at: string; note: string; records: Forecast[] };

export async function loadPregame<T>(name: string): Promise<T | null> {
  try {
    const r = await fetch(`/demo/pregame/${name}`, { cache: "no-store" });
    return r.ok ? ((await r.json()) as T) : null;
  } catch { return null; }
}

export const signed = (v: number, d = 4) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
