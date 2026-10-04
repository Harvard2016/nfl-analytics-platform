"use client";

import { useEffect, useRef, useState } from "react";
import { formatClock, sourceVideoLink, toSourceTime, toTimelineTime, youtubeId, youtubeEmbedUrl, youtubeErrorMessage, type VideoSource } from "@/lib/highlightTime";

type Player = {
  loadVideoById(options: { videoId: string; startSeconds: number; endSeconds?: number }): void;
  seekTo(time: number, seekAhead: boolean): void;
  playVideo(): void; pauseVideo(): void; getCurrentTime(): number; destroy(): void;
};
type PlayerEvent = { target: Player; data: number };
type YouTubeAPI = { Player: new (element: HTMLElement, config: {
  events: { onReady(event: PlayerEvent): void; onStateChange(event: PlayerEvent): void; onError(event: PlayerEvent): void; onAutoplayBlocked(): void };
}) => Player };
declare global { interface Window { YT?: YouTubeAPI; onYouTubeIframeAPIReady?: () => void } }
let apiPromise: Promise<YouTubeAPI> | null = null;
function loadYouTube(): Promise<YouTubeAPI> {
  if (window.YT?.Player) return Promise.resolve(window.YT);
  if (apiPromise) return apiPromise;
  apiPromise = new Promise<YouTubeAPI>((resolve, reject) => {
    const prior = window.onYouTubeIframeAPIReady;
    const timer = window.setTimeout(() => reject(new Error("Player connection timed out")), 15000);
    window.onYouTubeIframeAPIReady = () => { prior?.(); window.clearTimeout(timer); if (window.YT) resolve(window.YT); };
    const script = document.createElement("script");
    script.src = "https://www.youtube.com/iframe_api";
    script.onerror = () => { window.clearTimeout(timer); reject(new Error("Player connection unavailable")); };
    document.head.appendChild(script);
  }).catch((e) => { apiPromise = null; throw e; });
  return apiPromise;
}

export type PlaybackRequest = { start: number; end: number | null; sequence: number; label: string };
type Props = { source: VideoSource; title: string; duration: number; request: PlaybackRequest | null; onTime(time: number): void; onReplay(): void; onNext(): void; hasNext: boolean };

export default function HighlightPlayer({ source, title, duration, request, onTime, onReplay, onNext, hasNext }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const player = useRef<Player | null>(null);
  const [ready, setReady] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [errorCode, setErrorCode] = useState<number | null>(null);
  const pendingStart = useRef<number | null>(null);
  const [status, setStatus] = useState("Select a candidate to watch the source interval.");
  const [failed, setFailed] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const latest = useRef({ source, duration, request, onTime });
  const previousTime = useRef<number | null>(null);
  useEffect(() => { latest.current = { source, duration, request, onTime }; }, [source, duration, request, onTime]);
  const videoId = youtubeId(source.full_video);
  const enabled = request !== null;

  useEffect(() => {
    if (!enabled || !videoId || !host.current) return;
    const container = host.current;
    let disposed = false;
    let instance: Player | null = null;
    let readyTimer: number | undefined;
    const frame = document.createElement("iframe");
    const r = latest.current.request;
    frame.src = youtubeEmbedUrl(latest.current.source, window.location.origin, r?.start ?? 0, r?.end ?? null);
    frame.title = `Source video: ${title}`;
    frame.allow = "autoplay; encrypted-media; fullscreen; picture-in-picture";
    frame.allowFullscreen = true;
    frame.referrerPolicy = "strict-origin-when-cross-origin";
    container.appendChild(frame);
    loadYouTube().then((api) => {
      if (disposed) return;
      readyTimer = window.setTimeout(() => {
        if (!disposed) { setFailed(true); setStatus("YouTube did not finish opening this video. Retry the player or use the source link."); }
      }, 15000);
      // Create the iframe ourselves so permission and referrer attributes are present on its first request.
      instance = new api.Player(frame, {
        events: {
          onReady: (event) => {
            if (disposed) return;
            player.current = event.target;
            window.clearTimeout(readyTimer);
            setReady(true); setStatus("Ready. Play the selected interval.");
          },
          onStateChange: (event) => {
            if (disposed) return;
            setPlaying(event.data === 1);
            if (event.data === 1) {
              if (pendingStart.current !== null && Math.abs(event.target.getCurrentTime() - pendingStart.current) <= 3) pendingStart.current = null;
              setStatus("Playing source video");
            }
            else if (event.data === 2) setStatus("Paused");
            else if (event.data === 3) setStatus("Buffering…");
            else if (event.data === 0 && latest.current.request?.end != null) {
              setStatus("Interval finished. Replay it or choose the next candidate.");
              latest.current.onTime(Math.max(latest.current.request.start, latest.current.request.end - .01));
            }
          },
          onError: (event) => { if (!disposed) { window.clearTimeout(readyTimer); setFailed(true); setPlaying(false); setErrorCode(event.data); setStatus(youtubeErrorMessage(event.data)); } },
          onAutoplayBlocked: () => { if (!disposed) { setPlaying(false); setStatus("Your browser blocked automatic playback. Press Play interval or the play button inside the video."); } },
        },
      });
    }).catch(() => { if (!disposed) { setFailed(true); setStatus("Could not connect the player. The source link is still available."); } });
    return () => { disposed = true; window.clearTimeout(readyTimer); player.current = null; instance?.destroy(); container.replaceChildren(); };
  }, [enabled, videoId, attempt, title]);

  useEffect(() => {
    if (!ready || !request || !player.current || failed) return;
    const startSeconds = toSourceTime(source, request.start);
    pendingStart.current = startSeconds;
    previousTime.current = null;
    // Loading an interval avoids racing seekTo/playVideo against an asynchronous cueVideoById.
    player.current.loadVideoById({ videoId: videoId!, startSeconds, ...(request.end != null ? { endSeconds: toSourceTime(source, request.end) } : {}) });
  }, [ready, request, source, failed, videoId]);

  useEffect(() => {
    if (!ready || failed) return;
    const poll = window.setInterval(() => {
      const p = player.current;
      if (!p) return;
      const current = p.getCurrentTime();
      if (!Number.isFinite(current)) return;
      const { source: s, duration: d, request: r, onTime: notify } = latest.current;
      // Ignore the old interval's clock until the player has reached the newly requested start.
      if (pendingStart.current !== null) {
        if (Math.abs(current - pendingStart.current) > 3) return;
        pendingStart.current = null;
      }
      const timeline = toTimelineTime(s, current, d);
      setTime(timeline);
      if (playing || (previousTime.current !== null && Math.abs(current-previousTime.current) > .3)) notify(timeline);
      previousTime.current = current;
      if (playing && r?.end != null && current >= toSourceTime(s, r.end)) {
        p.pauseVideo(); setPlaying(false); setStatus("Interval finished. Replay it or choose the next candidate.");
        notify(Math.max(r.start, r.end - 0.01));
      }
    }, 150);
    return () => window.clearInterval(poll);
  }, [ready, playing, failed]);

  const start = request?.start ?? 0;
  return (
    <div data-tour="highlight-player" className="highlight-player">
      <div className="player-titlebar"><span className="kicker">Source film / {request?.label ?? "candidate preview"}</span><a href={sourceVideoLink(source, start)} target="_blank" rel="noopener">Watch on YouTube ↗</a></div>
      <div className={`video-stage${failed ? " video-stage--error" : ""}`}>
        <div ref={host} className="youtube-host" hidden={!enabled || failed} />
        {(!enabled || failed || !videoId) && <div className="video-placeholder">
          <span className="player-play-symbol" aria-hidden="true">▶</span>
          <p className="display text-4xl sm:text-5xl">Watch the moment.</p>
          <p className="mt-2 max-w-[42ch] text-sm text-muted">{failed ? status : "Select a candidate below to play its ranked interval with synchronized model and audio evidence."}</p>
          {!failed && videoId && <button className="btn-primary mt-5" onClick={onReplay}>Play selected candidate →</button>}
          {(failed || !videoId) && <a className="btn-primary mt-5" href={sourceVideoLink(source, start)} target="_blank" rel="noopener">Open source video ↗</a>}
          {failed && videoId && <button className="mt-3 text-sm underline" onClick={() => { setReady(false); setFailed(false); setErrorCode(null); setStatus("Retrying YouTube…"); setAttempt(n => n + 1); }}>Retry player</button>}
          <span className="video-source-name">{title}</span>
        </div>}
      </div>
      <div className="player-transport">
        <div className="flex flex-wrap items-center gap-2">
          <button className="transport-primary" onClick={() => {
            if (!request || !ready) { onReplay(); return; }
            if (playing) player.current?.pauseVideo();
            else {
              if (request.end != null && time >= request.end - .2) onReplay();
              else player.current?.playVideo();
            }
          }} disabled={failed || !videoId}>{playing ? "Pause" : "Play interval"}</button>
          <button onClick={onReplay} disabled={failed || !videoId}>Replay ↺</button>
          <button onClick={onNext} disabled={!hasNext}>Next →</button>
        </div>
        <span className="mono num text-xs text-muted">{formatClock(start)}{request?.end != null ? ` — ${formatClock(request.end)}` : " · source preview"}</span>
      </div>
      <p className="player-status" role="status">{status}{errorCode !== null && <span className="mono block" data-testid="youtube-error">YouTube error {errorCode}</span>}</p>
      <p className="px-4 pb-3 text-[10px] leading-relaxed text-muted">Official YouTube player when embedding is permitted. Timing uses the dataset&apos;s trim mapping, not independent frame verification. Candidate intervals are selected ranking windows, not full-play boundaries.</p>
    </div>
  );
}
