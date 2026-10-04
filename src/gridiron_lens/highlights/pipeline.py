"""Highlight Intelligence, first benchmark: ranking agreement with editorial highlight selection.

Independent of the coverage and pregame modules.

Data: the American-football subset of SVHighlights (Lee, Ki, Kang, Kim; KDD 2026; CC BY-NC 4.0), pinned to one
Hugging Face revision. Forty full-game broadcasts, verified here as 40 distinct NFL games (2016-2024 seasons) from the
NFL's own channel. The release contains features and annotations, not video.

Target: each 2-second clip is labelled 1 if it appears in that game's official highlight video (found by the dataset
authors through frame matching), else 0. So a score here means "agrees with the editors' selection", not "a major
play happened". Official edits contain replays and omit some big plays.

Kept out of model inputs: labels, alignment outputs, highlight-video paths, frame-match indices, and the
model-generated segment captions. The released `txt_clip` file is a single query embedding, not per-clip text, and is
not used. Inputs are things obtainable from the full game alone: loudness, commentary words, audio and visual embeddings.

Processing is offline: features for a clip may use commentary and context that come after it. That is unlike the
coverage module, whose inputs are causal.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re

import numpy as np

from ..shared import config
from ..shared.provenance import now_utc, write_json

REVISION = "fdedf750ddc2524093eda9bca1fbe24d1210724a"
SOURCE = "https://huggingface.co/datasets/idong1004/SVHighlights"
RAW = config.RAW / "svhighlights"
PROC = config.PROCESSED / "svhighlights"
FEAT = config.FEATURES / "highlights"
OUT = config.REPORTS / "v2"
CLIP_SECONDS = 2.0
SPLIT_SEED = 20261004
N_VAL, N_TEST = 6, 6
MODALITIES = {"vid_clip": ("npz", "features"), "vid_slowfast": ("npz", "features"), "aud_pann": ("npy", None)}
TEAM_WORDS = r"(?:vs\.?|at)"


def video_ids() -> list[int]:
    return sorted(int(p.stem.split("_")[-1]) for p in (PROC / "whisper").glob("american_football_*.json"))


def load_labels() -> dict[int, np.ndarray]:
    lab = json.loads((RAW / "annotations" / "label.json").read_text())
    return {int(x["vid"].split("_")[-1]): np.array(x["saliency_scores"], np.int8) for x in lab if x["vid"].startswith("american_football_")}


def load_volume(align: bool = True) -> dict[int, np.ndarray]:
    """Loudness per clip. Nine games carry one extra trailing value (a partial last clip); aligned arrays are cut to the label length."""
    vol = json.loads((RAW / "annotations" / "volume.json").read_text())
    out = {int(k.split("_")[-1]): np.array(v, np.float32) for k, v in vol.items() if k.startswith("american_football_")}
    if align:
        n = {v: len(y) for v, y in load_labels().items()}
        out = {v: a[: n[v]] for v, a in out.items()}
    return out


def load_modality(vid: int, name: str, n: int | None = None) -> np.ndarray:
    """Embedding per clip, cut to `n` clips when given (drops the partial trailing clip some files carry)."""
    kind, key = MODALITIES[name]
    path = PROC / "american_football" / name / f"{vid}.{kind}"
    arr = np.load(path)
    arr = (arr[key] if key else arr).astype(np.float32)
    return arr if n is None else arr[:n]


def metadata() -> list[dict]:
    """Source rows joined with the looked-up titles. Parses season, week and teams from the title text."""
    with open(RAW / "metadata" / "video_list.csv") as fh:
        rows = {r["vid"]: r for r in csv.DictReader(fh) if r["vid"].startswith("american_football_")}
    titles = {v["vid"]: v for v in json.loads((PROC / "video_metadata.json").read_text())["videos"]}
    out = []
    for vid in sorted(rows, key=lambda s: int(s.split("_")[-1])):
        r, t = rows[vid], titles.get(vid, {})
        title = t.get("full_link_title") or ""
        year = re.search(r"(20\d\d)", title)
        week = re.search(r"Week (\d+)", title)
        out.append({"id": int(vid.split("_")[-1]), "source_id": vid, "title": title, "channel": t.get("full_link_channel"),
                    "league": "NFL" if t.get("full_link_channel") == "NFL" else "unverified",
                    "season_in_title": int(year.group(1)) if year else None, "week_in_title": int(week.group(1)) if week else None,
                    "full_link": r["full_link"], "highlight_link": r["hl_link"], "full_length_s": float(r["full_length"]),
                    "trim_start_s": float(r["full_start"]), "trim_end_s": float(r["full_end"]), "highlight_length_s": float(r["hl_length"])})
    return out


def audit() -> dict:
    """What is actually on disk: counts, alignment between labels, loudness and features, duplicates, missing files."""
    labels, vol, meta = load_labels(), load_volume(align=False), metadata()
    rows, problems = [], []
    for m in meta:
        v = m["id"]
        n = len(labels[v])
        shapes = {}
        for name in MODALITIES:
            try:
                shapes[name] = list(load_modality(v, name).shape)
            except FileNotFoundError:
                shapes[name] = None
                problems.append(f"video {v}: {name} missing")
        words = json.loads((PROC / "whisper" / f"american_football_{v}.json").read_text())
        trimmed = m["trim_end_s"] - m["trim_start_s"]
        row = {**m, "clips": n, "positive_clips": int(labels[v].sum()), "positive_share": round(float(labels[v].mean()), 4), "loudness_values": len(vol[v]),
               "feature_shapes": shapes, "transcript_words": len(words), "last_word_s": round(float(words[-1]["end"]), 1) if words else None,
               "trimmed_duration_s": round(trimmed, 1), "clips_times_2s": n * CLIP_SECONDS,
               "positive_seconds": int(labels[v].sum() * CLIP_SECONDS)}
        if len(vol[v]) != n or any(s is not None and s[0] != n for s in shapes.values()):
            problems.append(f"video {v}: one extra trailing loudness/feature value beyond the {n} labelled clips; cut to the label length" if max([len(vol[v])] + [s[0] for s in shapes.values() if s]) - n == 1 else f"video {v}: clip counts differ by more than one")
        if abs(trimmed - n * CLIP_SECONDS) > 4:
            problems.append(f"video {v}: trimmed duration {trimmed:.0f}s vs {n * CLIP_SECONDS:.0f}s of clips")
        rows.append(row)
    links = [r["full_link"] for r in rows]
    titles = [r["title"] for r in rows]
    rep = {
        "source": SOURCE, "revision": REVISION, "audited_at": now_utc(), "license": "CC BY-NC 4.0 (dataset card)",
        "videos": len(rows), "leagues": sorted({r["league"] for r in rows}), "seasons_in_titles": sorted({r["season_in_title"] for r in rows if r["season_in_title"]}),
        "duplicate_source_links": len(links) - len(set(links)), "duplicate_titles": len(titles) - len(set(titles)),
        "clips": int(sum(r["clips"] for r in rows)), "positive_clips": int(sum(r["positive_clips"] for r in rows)),
        "positive_share": round(sum(r["positive_clips"] for r in rows) / sum(r["clips"] for r in rows), 4),
        "clip_seconds": CLIP_SECONDS, "feature_dims": {k: rows[0]["feature_shapes"][k][1] for k in MODALITIES if rows[0]["feature_shapes"][k]},
        "time_coordinate": "Clip i covers seconds [2i, 2i+2) of the trimmed broadcast. Original video time = trim_start_s + trimmed time.",
        "label_definition": "1 if the clip was matched to the official highlight video by the dataset authors' frame matching; editorial selection, replays included.",
        "not_downloaded": ["other seven sports", "alignment.tar (label-generation intermediates)", "source or highlight videos"],
        "not_used_as_inputs": ["label.json", "all_filtered_frame_idx.json", "segment_caption.json (model-generated captions)", "txt_clip (one query embedding, not per-clip)"],
        "problems": problems, "games": rows,
    }
    write_json(config.MANIFESTS / "svhighlights_audit.json", rep)
    return rep


def make_split() -> dict:
    """Deterministic split by game, written once before any model is scored. Each game is one group; no game was found twice."""
    path = config.MANIFESTS / "svhighlights_split.json"
    if path.exists():
        return json.loads(path.read_text())
    ids = video_ids()
    order = np.random.default_rng(SPLIT_SEED).permutation(ids).tolist()
    test, val, train = sorted(order[:N_TEST]), sorted(order[N_TEST:N_TEST + N_VAL]), sorted(order[N_TEST + N_VAL:])
    folds = [sorted(train[i::4]) for i in range(4)]            # grouped development folds over the training games
    man = {"version": "svh-football-split-v1", "frozen_at": now_utc(), "seed": SPLIT_SEED, "unit": "game (one source video each; no duplicates found)",
           "train": train, "validation": val, "test": test, "development_folds": folds,
           "ids_sha256": hashlib.sha256(json.dumps([train, val, test]).encode()).hexdigest(),
           "rule": "seeded shuffle of the 40 game ids; first 6 test, next 6 validation, remaining 28 training (70/15/15)"}
    write_json(path, man)
    return man
