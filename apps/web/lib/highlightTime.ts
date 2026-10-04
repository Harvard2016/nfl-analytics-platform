export type VideoSource = { full_video: string; trim_start_s: number };

export function youtubeId(url: string): string | null {
  try {
    const u = new URL(url);
    const host = u.hostname.toLowerCase();
    const id = host === "youtu.be" ? u.pathname.slice(1).split("/")[0]
      : ["youtube.com", "www.youtube.com", "youtube-nocookie.com", "www.youtube-nocookie.com"].includes(host)
        ? u.searchParams.get("v") ?? (u.pathname.match(/^\/(?:embed|shorts)\/([^/]+)/)?.[1] ?? "") : "";
    return id && /^[\w-]{11}$/.test(id) ? id : null;
  } catch { return null; }
}

// The current exports describe an additive trim mapping. Keep this in one place;
// a future release with piecewise anchors must replace this contract explicitly.
export function toSourceTime(source: VideoSource, timelineSeconds: number): number {
  if (!Number.isFinite(source.trim_start_s) || source.trim_start_s < 0 || !Number.isFinite(timelineSeconds)) throw new Error("Invalid video-time mapping");
  return source.trim_start_s + Math.max(0, timelineSeconds);
}
export function toTimelineTime(source: VideoSource, sourceSeconds: number, duration: number): number {
  if (!Number.isFinite(sourceSeconds) || !Number.isFinite(duration) || duration < 0) throw new Error("Invalid playback time");
  return Math.max(0, Math.min(duration, sourceSeconds - toSourceTime(source, 0)));
}
export function sourceVideoLink(source: VideoSource, timelineSeconds: number): string {
  const url = new URL(source.full_video);
  url.searchParams.set("t", `${Math.floor(toSourceTime(source, timelineSeconds))}s`);
  return url.toString();
}
export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 3600)}:${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}
