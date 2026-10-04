"use client";

import { useEffect, useRef, useState } from "react";
import { formatClock, sourceVideoLink, toSourceTime, toTimelineTime, youtubeId, type VideoSource } from "@/lib/highlightTime";

type Player = {
  cueVideoById(options: { videoId: string; startSeconds: number }): void;
  seekTo(time: number, seekAhead: boolean): void;
  playVideo(): void; pauseVideo(): void; getCurrentTime(): number; destroy(): void;
};
type PlayerEvent = { target: Player; data: number };
type YouTubeAPI = { Player: new (element: HTMLElement, config: {
  videoId: string; width: string; height: string; host: string;
  playerVars: Record<string, string | number>;
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
    loadYouTube().then((api) => {
      if (disposed) return;
      const mount = document.createElement("div");
      container.appendChild(mount);
      instance = new api.Player(mount, {
        videoId, width: "100%", height: "100%", host: "https://www.youtube-nocookie.com",
        playerVars: { origin: window.location.origin, playsinline: 1, controls: 1, autoplay: 0 },
        events: {
          onReady: (event) => {
            if (disposed) return;
            player.current = event.target;
            const r = latest.current.request;
            event.target.cueVideoById({ videoId, startSeconds: toSourceTime(latest.current.source, r?.start ?? 0) });
            setReady(true); setStatus("Ready. Play the selected interval.");
          },
          onStateChange: (event) => {
            if (disposed) return;
            setPlaying(event.data === 1);
            if (event.data === 1) setStatus("Playing source video");
            else if (event.data === 2) setStatus("Paused");
            else if (event.data === 3) setStatus("Buffering…");
          },
          onError: () => { if (!disposed) { setFailed(true); setStatus("This source cannot play here. Watch the selected moment on YouTube."); } },
          onAutoplayBlocked: () => { if (!disposed) setStatus("Your browser requires another click. Choose Play interval."); },
        },
      });
    }).catch(() => { if (!disposed) { setFailed(true); setStatus("Could not connect the player. The source link is still available."); } });
    return () => { disposed = true; player.current = null; instance?.destroy(); container.replaceChildren(); };
  }, [enabled, videoId]);

  useEffect(() => {
    if (!ready || !request || !player.current || failed) return;
    player.current.seekTo(toSourceTime(source, request.start), true);
    player.current.playVideo();
  }, [ready, request, source, failed]);

  useEffect(() => {
    if (!ready || failed) return;
    const poll = window.setInterval(() => {
      const p = player.current;
      if (!p) return;
      const current = p.getCurrentTime();
      if (!Number.isFinite(current)) return;
      const { source: s, duration: d, request: r, onTime: notify } = latest.current;
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
    <div className="highlight-player">
      <div className="player-titlebar"><span className="kicker">Source film / {request?.label ?? "candidate preview"}</span><a href={sourceVideoLink(source, start)} target="_blank" rel="noreferrer">Watch on YouTube ↗</a></div>
      <div className="video-stage">
        <div ref={host} className="youtube-host" hidden={!enabled || failed} />
        {(!enabled || failed || !videoId) && <div className="video-placeholder">
          <span className="player-play-symbol" aria-hidden="true">▶</span>
          <p className="display text-4xl sm:text-5xl">Watch the moment.</p>
          <p className="mt-2 max-w-[42ch] text-sm text-muted">{failed ? status : "Select a candidate below to play its ranked interval with synchronized model and audio evidence."}</p>
          {!failed && videoId && <button className="btn-primary mt-5" onClick={onReplay}>Play selected candidate →</button>}
          {(failed || !videoId) && <a className="btn-primary mt-5" href={sourceVideoLink(source, start)} target="_blank" rel="noreferrer">Open source video ↗</a>}
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
      <p className="player-status" role="status">{status}</p>
      <p className="px-4 pb-3 text-[10px] leading-relaxed text-muted">Official YouTube player when embedding is permitted. Timing uses the dataset&apos;s trim mapping, not independent frame verification. Candidate intervals are selected ranking windows, not full-play boundaries.</p>
    </div>
  );
}
