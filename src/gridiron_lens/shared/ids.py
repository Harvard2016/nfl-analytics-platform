"""Source-aware string identifiers. BDB numeric IDs are never assumed equal to nflverse IDs."""
from __future__ import annotations


def game_key(source: str, game_id: int | str) -> str:
    return f"{source}:{game_id}"


def play_key(source: str, game_id: int | str, play_id: int | str) -> str:
    # playId alone is not global: a play is identified by game plus source play id.
    return f"{source}:{game_id}:{play_id}"


def split_play_key(key: str) -> tuple[str, str, str]:
    source, game_id, play_id = key.split(":")
    return source, game_id, play_id
