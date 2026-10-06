export type ClipCandidate = { asset: string; candidate_moment_s: number; clip_start_s: number; clip_end_s: number };
export type ClipEdit = { start: number; end: number; removed: boolean; replay: "unknown" | "live action" | "replay"; reviewed: boolean };

export function initialClipEdits(candidates: ClipCandidate[]): Record<string, ClipEdit> {
  return Object.fromEntries(candidates.map((c) => [c.asset, { start: c.clip_start_s, end: c.clip_end_s, removed: false, replay: "unknown", reviewed: false }]));
}

export function reviewedCandidates(candidates: ClipCandidate[], edits: Record<string, ClipEdit>) {
  return candidates.map((c) => {
    const e = edits[c.asset];
    return { candidate_moment_s: c.candidate_moment_s, model_clip: [c.clip_start_s, c.clip_end_s],
      edited_clip: [e.start, e.end], removed: e.removed, replay: e.replay, event_labels: [] as string[], reviewed_by_a_person: e.reviewed };
  });
}
