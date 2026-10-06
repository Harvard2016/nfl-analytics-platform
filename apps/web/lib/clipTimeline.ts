export type AudioTimeline = { audio_start_s?: number; audio_end_s?: number; bin_start_s?: number[]; bin_end_s?: number[]; loudness_db: number[] };

// Plot and pointer coordinates are uploaded-file seconds, never a fraction of the audio length.
export function timelineSeek(fraction: number, mediaSeconds: number): number {
  return Math.max(0, Math.min(1, fraction)) * mediaSeconds;
}

export function binBounds(timeline: AudioTimeline, i: number, step: number): [number, number] {
  const start = timeline.bin_start_s?.[i] ?? (timeline.audio_start_s ?? 0) + i * step;
  const end = timeline.bin_end_s?.[i] ?? Math.min(start + step, timeline.audio_end_s ?? Infinity);
  return [start, end];
}

export function audioExtent(timeline: AudioTimeline, step: number): [number, number] {
  if (!timeline.loudness_db.length) return [0, 0];
  return [binBounds(timeline, 0, step)[0], binBounds(timeline, timeline.loudness_db.length - 1, step)[1]];
}
