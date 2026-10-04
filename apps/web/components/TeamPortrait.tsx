import Image from "next/image";
import { teamVisual } from "@/lib/teams";

export default function TeamPortrait({ team, side }: { team: string; side: "away" | "home" }) {
  const { name, color } = teamVisual(team);
  return (
    <div className={`team-portrait team-portrait--${side}`} style={{ "--team-color": color } as React.CSSProperties}>
      <div className="team-helmet" aria-hidden="true">
        <Image src="/art/helmet-silver.webp" alt="" fill sizes="(max-width: 767px) 45vw, 38vw" className="helmet-image" />
        <span className="helmet-tint" />
      </div>
      <div className="team-wordmark">
        <p className="kicker">{side} <span className="team-rule" /></p>
        <p className="team-name">{name}</p>
        <p className="display team-abbreviation">{team}</p>
      </div>
    </div>
  );
}
