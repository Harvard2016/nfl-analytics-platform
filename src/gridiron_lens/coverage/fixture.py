"""SYNTHETIC fixture in the BDB 2025 file layout.

Invented geometry, invented teams, invented labels. It exists so the pipeline, tests and UI can be
exercised before the real files are on disk. Nothing produced from it is a model result, and every
artifact derived from it carries synthetic=true. Game ids start with 2099 and clubs are SY*.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ..shared import config
from . import schema

TEAMS = ["SYA", "SYB", "SYC", "SYD"]
OFF_SPOTS = [  # (depth behind LOS, lateral offset from ball, role)
    (-0.5, -3.0, "T"), (-0.5, -1.5, "G"), (-0.5, 0.0, "C"), (-0.5, 1.5, "G"), (-0.5, 3.0, "T"),
    (-5.0, 0.0, "QB"), (-6.5, 1.5, "RB"), (-0.7, 5.0, "TE"),
    (-0.7, -18.0, "WR"), (-1.0, 16.0, "WR"), (-1.0, 10.0, "WR"),
]
RECEIVERS = [7, 8, 9, 10, 6]  # indices into OFF_SPOTS that run routes


def _facing_deg(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Degrees clockwise from +y, the BDB angle convention."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def _play(rng: np.random.Generator, man: bool, pre: int, post: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions in the canonical frame: (frames, 22, 2) in (x_rel, lateral-from-ball), plus facing targets."""
    t = np.arange(-pre, post + 1)
    after = np.clip(t, 0, None)[:, None] / 10.0  # seconds since snap
    off = np.array([[d, l] for d, l, _ in OFF_SPOTS], float) + rng.normal(0, 0.25, (11, 2))
    off_xy = np.repeat(off[None], len(t), 0)
    if rng.random() < 0.4:  # pre-snap motion by the slot receiver, finished 5 frames before the snap
        progress = np.clip((t + pre) / max(pre - 5, 1), 0, 1)
        off_xy[:, 10, 1] += rng.uniform(-7, 7) * (progress - 1)
    routes = {i: (rng.uniform(4, 7), rng.uniform(-2.5, 2.5)) for i in RECEIVERS}
    for i, (vx, vy) in routes.items():
        off_xy[:, i, 0] += vx * after[:, 0]
        off_xy[:, i, 1] += vy * after[:, 0]
    off_xy[:, 5, 0] -= 2.0 * np.minimum(after[:, 0], 1.5)  # QB drop
    d = np.zeros((len(t), 11, 2))
    line = np.array([[1.0, -4.5], [1.0, -1.5], [1.0, 1.5], [1.0, 4.5]])
    d[:, 0:4] = line[None] + after[:, :, None] * np.array([-1.2, 0.0])
    facing_target = np.zeros((len(t), 11, 2))
    qb = off_xy[:, 5]
    for k in range(4):
        facing_target[:, k] = qb
    if man:
        for k, r in enumerate(RECEIVERS):  # defenders 4..8 trail a receiver each
            cushion = np.array([rng.uniform(1.5, 7.0), rng.normal(0, 0.8)])
            lag = rng.uniform(0.75, 0.95)
            d[:, 4 + k] = off_xy[0, r] + cushion + (off_xy[:, r] - off_xy[0, r]) * lag
            facing_target[:, 4 + k] = off_xy[:, r]
        deep = np.array([[rng.uniform(11, 15), rng.normal(0, 2)], [rng.uniform(4, 9), rng.normal(0, 4)]])
        d[:, 9:11] = deep[None] + after[:, :, None] * np.array([1.5, 0.0])
        facing_target[:, 9:11] = qb[:, None]
    else:
        spots = np.array([[5, -14], [5, -5], [5, 4], [5, 13], [7, 0], [rng.uniform(11, 15), -9], [rng.uniform(11, 15), 9]], float)
        spots += rng.normal(0, 1.2, spots.shape)
        drops = np.array([[2.0, -1.0], [1.5, 0], [1.5, 0], [2.0, 1.0], [2.5, 0], [3.0, -1.0], [3.0, 1.0]])
        d[:, 4:11] = spots[None] + after[:, :, None] * drops[None]
        facing_target[:, 4:11] = qb[:, None]
    d += rng.normal(0, 0.08, d.shape)
    # label noise in the geometry itself: some plays look like the other family
    return np.concatenate([off_xy, d], 1), np.concatenate([np.repeat(qb[:, None], 11, 1) + [8, 0], facing_target], 1), t


def generate(out_dir: Path, plays_per_game: int = 30, games_per_week: int = 1, seed: int = 7) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SYNTHETIC").write_text("Synthetic fixture. Not NFL data. Not a model result.\n")
    rng = np.random.default_rng(seed)
    games, plays, players, nfl_id = [], [], [], 900000
    roster: dict[str, list[int]] = {}
    for team in TEAMS:
        roster[team] = []
        for unit in ("O", "D"):
            for j in range(11):
                nfl_id += 1
                role = OFF_SPOTS[j][2] if unit == "O" else ["DE", "DT", "DT", "DE", "CB", "CB", "CB", "LB", "LB", "FS", "SS"][j]
                players.append({"nflId": nfl_id, "position": role, "displayName": f"Synthetic {team} {unit}{j + 1}"})
                roster[team].append(nfl_id)
    L, W = config.FIELD_LENGTH, config.FIELD_WIDTH
    for week in schema.WEEKS_2025:
        rows = []
        for g in range(games_per_week):
            gid = 2099000000 + week * 100 + g
            home, away = rng.choice(TEAMS, 2, replace=False)
            games.append({"gameId": gid, "season": 2099, "week": week, "homeTeamAbbr": home, "visitorTeamAbbr": away})
            for n in range(plays_per_game):
                pid = 50 + n * 37
                off_team, def_team = (home, away) if rng.random() < 0.5 else (away, home)
                down = int(rng.integers(1, 5))
                man = rng.random() < (0.28 + 0.06 * (down >= 3) + (0.08 if def_team == "SYA" else 0))
                xy, face_at, t = _play(rng, man if rng.random() > 0.12 else not man, int(rng.integers(12, 40)), int(rng.integers(22, 45)))
                left = rng.random() < 0.5
                los = float(rng.uniform(25, 95))  # canonical x of the line of scrimmage
                ball_y = float(rng.uniform(20, 33))
                x, y = xy[..., 0] + los, xy[..., 1] + ball_y
                to = face_at - xy
                o = _facing_deg(to[..., 0], to[..., 1]) + rng.normal(0, 12, to.shape[:2])
                vel = np.gradient(np.stack([x, y], -1), axis=0) * 10.0
                s = np.linalg.norm(vel, axis=-1)
                dr = _facing_deg(vel[..., 0], vel[..., 1])
                if left:  # write raw coordinates as the source would for a leftward play
                    x, y, o, dr = L - x, W - y, o + 180.0, dr + 180.0
                ids = roster[off_team][:11] + roster[def_team][11:]
                clubs = [off_team] * 11 + [def_team] * 11
                snap_i = int(np.where(t == 0)[0][0])
                for fi, rel in enumerate(t):
                    ftype = "BEFORE_SNAP" if rel < 0 else "SNAP" if rel == 0 else "AFTER_SNAP"
                    event = "ball_snap" if rel == 0 else None
                    for k in range(22):
                        rows.append((gid, pid, ids[k], fi + 1, ftype, k + 1, clubs[k], "left" if left else "right",
                                     round(float(x[fi, k]), 2), round(float(y[fi, k]), 2), round(float(s[fi, k]), 2), 0.0,
                                     round(float(o[fi, k] % 360), 2), round(float(dr[fi, k] % 360), 2), event))
                    c = 2 if rel <= 0 else 5  # ball sits at the centre, then goes back to the QB
                    rows.append((gid, pid, None, fi + 1, ftype, None, "football", "left" if left else "right",
                                 round(float(x[fi, c]), 2), round(float(y[fi, c]), 2), 0.0, 0.0, None, None, event))
                assert snap_i >= 0
                plays.append({
                    "gameId": gid, "playId": pid, "playDescription": "Synthetic play", "quarter": int(rng.integers(1, 5)),
                    "down": down, "yardsToGo": int(rng.integers(1, 15)), "possessionTeam": off_team,
                    "defensiveTeam": def_team, "gameClock": "10:00",
                    "preSnapHomeScore": int(rng.integers(0, 28)), "preSnapVisitorScore": int(rng.integers(0, 28)),
                    "absoluteYardlineNumber": round(L - los if left else los, 2),
                    "pff_passCoverage": "Cover-1" if man else "Cover-3", "pff_manZone": "Man" if man else "Zone",
                })
        cols = ["gameId", "playId", "nflId", "frameId", "frameType", "jerseyNumber", "club", "playDirection",
                "x", "y", "s", "a", "o", "dir", "event"]
        pl.DataFrame(rows, schema=cols, orient="row", infer_schema_length=None).write_csv(
            out_dir / schema.tracking_file(week), null_value="NA")
    pl.DataFrame(games).write_csv(out_dir / "games.csv")
    pl.DataFrame(plays).write_csv(out_dir / "plays.csv")
    pl.DataFrame(players).write_csv(out_dir / "players.csv")
    return out_dir
