"""Event extension, the part that works without video: verified game identity and a metadata-led recap.

Two product modes are kept apart:
- media-led detection: candidates come from media-derived signals (the ranking models). That is what the timeline shows.
- metadata-led recap: the event list below comes straight from play-by-play. It is a metadata product. It is not
  evidence that the media models detected those plays, and it is not aligned to video time.

Event definitions (labels can overlap on one play):
  touchdown, interception, sack, fumble lost: the nflverse play-by-play flags.
  turnover: derived, interception or fumble lost.
  big play: pass of 20+ yards or rush of 10+ yards.

Identity mapping is strict: a source video maps to a schedule row only when the two teams named in its title and its
season, week, round or Super Bowl number leave exactly one game. Anything else stays unmapped with the reason.
No event label or event metric is produced for the media models: that needs a verified video-time to play alignment.
"""
from __future__ import annotations

import re

import polars as pl

from ..pregame import pipeline as schedule
from ..shared import config
from ..shared.provenance import write_json
from . import pipeline as P

NICK = {"Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF", "Panthers": "CAR", "Bears": "CHI", "Bengals": "CIN", "Browns": "CLE", "Cowboys": "DAL",
        "Broncos": "DEN", "Lions": "DET", "Packers": "GB", "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC", "Raiders": "LV", "Chargers": "LAC", "Rams": "LA",
        "Dolphins": "MIA", "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG", "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "49ers": "SF",
        "Seahawks": "SEA", "Buccaneers": "TB", "Titans": "TEN", "Redskins": "WAS", "Commanders": "WAS"}
ROMAN = {"LIII": 53, "LIV": 54, "LV": 55, "LVI": 56, "LVII": 57, "LVIII": 58, "LIX": 59}
ROUND = {"Wild Card": "WC", "Divisional": "DIV", "Championship": "CON", "Super Bowl": "SB"}
BIG_PASS, BIG_RUSH = 20, 10


def map_games() -> list[dict]:
    games = schedule.load_games().filter(pl.col("result").is_not_null())
    out = []
    for m in P.metadata():
        title = m["title"]
        teams = sorted({abbr for nick, abbr in NICK.items() if re.search(rf"\b{nick}\b", title)})
        row = {"id": m["id"], "source_id": m["source_id"], "title": title, "teams_in_title": teams, "game_id": None, "reason": None}
        if len(teams) != 2:
            out.append(row | {"reason": f"expected two team names in the title, found {len(teams)}"})
            continue
        c = games.filter(pl.col("home").is_in(teams) & pl.col("away").is_in(teams))
        sb = re.search(r"Super Bowl (\d+|[LXVI]+)\b", title)
        year = re.search(r"\b(20\d\d)\b", title)
        week = re.search(r"Week (\d+)", title)
        rnd = next((code for word, code in ROUND.items() if word in title), None)
        if sb:
            n = int(sb.group(1)) if sb.group(1).isdigit() else ROMAN.get(sb.group(1))
            c = c.filter((pl.col("game_type") == "SB") & (pl.col("season") == n + 1965)) if n else c.filter(pl.col("game_type") == "SB")
        else:
            if rnd:
                c = c.filter(pl.col("game_type") == rnd)
            if year:
                c = c.filter(pl.col("season") == int(year.group(1)))
            if week:
                c = c.filter(pl.col("week") == int(week.group(1)))
        if c.height == 1:
            g = c.row(0, named=True)
            row |= {"game_id": g["game_id"], "season": g["season"], "week": g["week"], "game_type": g["game_type"], "home": g["home_team"], "away": g["away_team"],
                    "final": f"{g['away_team']} {g['away_score']}, {g['home_team']} {g['home_score']}", "reason": "unique match on teams plus season, week, round or Super Bowl number in the title"}
        else:
            row["reason"] = f"{c.height} schedule rows match the title; left unmapped rather than guessed"
        out.append(row)
    return out


def recap(game_id: str, season: int) -> list[dict]:
    """Metadata-led event list for one mapped game, from play-by-play only."""
    cols = ["game_id", "play_id", "qtr", "time", "posteam", "desc", "touchdown", "interception", "sack", "fumble_lost", "pass", "rush", "yards_gained", "epa"]
    pbp = pl.read_parquet(config.RAW / "nflverse" / "pbp" / f"play_by_play_{season}.parquet", columns=cols).filter(pl.col("game_id") == game_id)
    out = []
    for r in pbp.iter_rows(named=True):
        tags = [t for t, on in (("touchdown", r["touchdown"] == 1), ("interception", r["interception"] == 1), ("sack", r["sack"] == 1), ("fumble lost", r["fumble_lost"] == 1),
                                ("big play", (r["pass"] == 1 and (r["yards_gained"] or 0) >= BIG_PASS) or (r["rush"] == 1 and (r["yards_gained"] or 0) >= BIG_RUSH))) if on]
        if r["interception"] == 1 or r["fumble_lost"] == 1:
            tags.append("turnover")
        if tags:
            desc = r["desc"] or ""
            out.append({"play_id": int(r["play_id"]), "quarter": r["qtr"], "game_clock": r["time"], "offense": r["posteam"], "events": tags, "yards": r["yards_gained"],
                        "epa": None if r["epa"] is None else round(float(r["epa"]), 2), "description": desc if len(desc) <= 100 else desc[:99].rsplit(" ", 1)[0] + "…"})
    return out


def run() -> dict:
    mapped = map_games()
    write_json(config.MANIFESTS / "svhighlights_game_map.json", {"rule": __doc__.split("Identity mapping is strict:")[1].split("No event label")[0].strip(), "games": mapped})
    web = config.WEB_DEMO / "highlights" / "recaps"
    web.mkdir(parents=True, exist_ok=True)
    n = 0
    for g in mapped:
        if g["game_id"] and (config.WEB_DEMO / "highlights" / "games" / f"{g['id']}.json").exists():
            ev = recap(g["game_id"], g["season"])
            write_json(web / f"{g['id']}.json", {"mode": "metadata-led recap", "source": "nflverse play-by-play", "game_id": g["game_id"], "final": g["final"],
                                                 "alignment": "Not aligned to video time. Quarter and game clock only.",
                                                 "definitions": {"turnover": "interception or fumble lost", "big play": f"pass of {BIG_PASS}+ yards or rush of {BIG_RUSH}+ yards"},
                                                 "counts": {t: sum(t in e["events"] for e in ev) for t in ("touchdown", "interception", "sack", "fumble lost", "turnover", "big play")}, "events": ev})
            n += 1
    return {"videos": len(mapped), "mapped": sum(g["game_id"] is not None for g in mapped), "unmapped": [(g["id"], g["title"][:50], g["reason"]) for g in mapped if not g["game_id"]], "recaps_written": n}
