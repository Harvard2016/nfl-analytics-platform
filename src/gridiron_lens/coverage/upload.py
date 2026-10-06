"""Coverage inference for an uploaded tracking play (v3 upload contract `coverage-upload-v1`).

The same normalization and tensor layout as the offline pipeline (`normalize._canonical`, `relational.build_tensors`), applied
to one play supplied as rows. The parity test runs a real release play through this path and through the offline cache and
requires the same tensor and the same probabilities.

Scope, stated in every result: the model learned on the released selected-player tracking (route runners and the coverage
defenders the release chose after the play). `release-compatible` mode expects the same kind of input. `broader` mode accepts
other player sets and is experimental: it is a domain shift and carries no benchmark accuracy.

Never read as an input: any label column. A supplied `man_zone` value is returned beside the prediction for comparison only.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field

import numpy as np
import torch

from ..shared import config
from . import neural, relational

SCHEMA = "coverage-upload-v1"
L, W = config.FIELD_LENGTH, config.FIELD_WIDTH
HZ = relational.HORIZONS
REQUIRED = ["player_id", "frame", "side", "x", "y"]
OPTIONAL = ["play_id", "role", "s", "dir", "vx", "vy", "o", "play_direction", "los_x", "man_zone"]
LABEL_COLUMNS = ("man_zone", "coverage", "coverage_type", "team_coverage_man_zone", "team_coverage_type")
MODEL_FILE = config.MODELS / "coverage_v2" / "temporal_gru_seed42.pt"
THRESHOLDS = {"at_snap": 0.55, "post_0_5s": 0.5, "post_1s": 0.5, "post_1_5s": 0.5}      # v2 temporal reliable-prediction cutoffs (weeks 13-14)
TEMPLATE = "play_id,player_id,frame,side,role,x,y,s,dir,o,play_direction,los_x\n" \
           "demo-1,D1,0,defense,,61.2,30.1,0.4,270.0,265.0,right,55\ndemo-1,R1,0,offense,route_runner,54.3,40.2,0.1,90.0,88.0,right,55\ndemo-1,QB,0,offense,passer,50.1,26.7,0.0,0.0,90.0,right,55\n"
DOCS = {
    "schema": SCHEMA, "format": "CSV with a header row, or JSON {\"rows\": [...]} with the same keys. One play per upload.",
    "field": "x runs 0-120 yards along the length (end zones included), y runs 0-53.3 yards across. Angles are degrees clockwise from +y, as in NFL tracking.",
    "required": {"player_id": "stable id for one player across frames", "frame": "integer, 10 frames per second; the smallest frame is treated as the snap unless snap_frame is passed",
                 "side": "offense or defense", "x": "yards", "y": "yards"},
    "motion": "Either s (yards/second) and dir (degrees), or vx and vy (yards/second). If neither is given, velocity is derived from the previous frame only.",
    "orientation": "o (degrees) is the observed body orientation. Direction of travel is not orientation. Without o the upload runs only in broader (experimental) mode with the facing input zeroed.",
    "optional": {"role": "passer | route_runner | other (the passer is excluded from the model, as offline)", "play_direction": "left or right (default right: offense moving toward larger x)",
                 "los_x": "line of scrimmage on the same x axis; required", "play_id": "echoed back", "man_zone": "released or charted label, used only for comparison"},
    "limits": {"defenders": f"1-{relational.D_MAX}", "route_runners": f"1-{relational.R_MAX}", "frames_used": f"snap to +1.5 s ({relational.T_MAX} frames)"},
}


class UploadError(ValueError):
    """Validation failure with messages a person can act on."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass
class Parsed:
    play_id: str
    rows: list[dict]
    label: str | None
    warnings: list[str] = field(default_factory=list)


def parse(data: bytes | str, kind: str) -> Parsed:
    text = data.decode("utf-8-sig") if isinstance(data, bytes) else data
    if kind == "json":
        obj = json.loads(text)
        rows = obj["rows"] if isinstance(obj, dict) else obj
    else:
        rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise UploadError(["The file has no rows."])
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise UploadError([f"Missing required column(s): {', '.join(missing)}. Download the template for the expected header."])
    plays = {str(r.get("play_id") or "") for r in rows}
    if len(plays) > 1:
        raise UploadError([f"The file mixes {len(plays)} plays ({', '.join(sorted(plays)[:4])}...). Upload one play at a time."])
    labels = {str(r[c]) for r in rows for c in LABEL_COLUMNS if r.get(c) not in (None, "")}
    out, problems = [], []
    num = lambda r, k: None if r.get(k) in (None, "", "NA") else float(r[k])
    for i, r in enumerate(rows):
        try:
            out.append({"player_id": str(r["player_id"]), "frame": int(float(r["frame"])), "side": str(r["side"]).strip().lower(), "role": str(r.get("role") or "").strip().lower(),
                        "x": float(r["x"]), "y": float(r["y"]), "s": num(r, "s"), "dir": num(r, "dir"), "vx": num(r, "vx"), "vy": num(r, "vy"), "o": num(r, "o"),
                        "play_direction": str(r.get("play_direction") or "right").strip().lower(), "los_x": num(r, "los_x")})
        except (TypeError, ValueError):
            problems.append(f"Row {i + 2}: a number could not be read.")
            if len(problems) > 5:
                break
    if problems:
        raise UploadError(problems)
    return Parsed(plays.pop() or "upload", out, labels.pop() if len(labels) == 1 else None)


def build(p: Parsed, mode: str = "release-compatible", snap_frame: int | None = None) -> dict:
    """Validated rows -> the model's tensors. Raises UploadError with every problem found."""
    rows, problems, warn = p.rows, [], list(p.warnings)
    if any(r["side"] not in ("offense", "defense") for r in rows):
        problems.append("side must be offense or defense on every row.")
    if any(not (0 <= r["x"] <= L and 0 <= r["y"] <= W) for r in rows):
        problems.append(f"Coordinates must be in yards on a {L:.0f} by {W} field (x 0-{L:.0f}, y 0-{W}).")
    seen = set()
    for r in rows:
        k = (r["player_id"], r["frame"])
        if k in seen:
            problems.append(f"Player {r['player_id']} has more than one row at frame {r['frame']}.")
            break
        seen.add(k)
    dirs, los = {r["play_direction"] for r in rows}, {r["los_x"] for r in rows}
    if len(dirs) != 1 or not dirs <= {"left", "right"}:
        problems.append("play_direction must be the same on every row: left or right.")
    if len(los) != 1 or None in los:
        problems.append("los_x (line of scrimmage, same x axis as the players) is required and must be the same on every row.")
    if problems:
        raise UploadError(problems)
    left, los_x = dirs.pop() == "left", los.pop()
    snap = min(r["frame"] for r in rows) if snap_frame is None else snap_frame
    has_o = all(r["o"] is not None for r in rows)
    if not has_o and mode == "release-compatible":
        raise UploadError(["Observed orientation (o) is missing. The model uses where each defender is facing; direction of travel is not a substitute. Add o, or run broader (experimental) mode."])
    by: dict[str, dict[str, dict[int, dict]]] = {"offense": {}, "defense": {}}
    for r in rows:
        if r["frame"] >= snap and r["role"] != "passer":
            by[r["side"]].setdefault(r["player_id"], {})[r["frame"] - snap] = r
    passer = [r for r in rows if r["role"] == "passer" and r["frame"] == snap]
    if len({r["player_id"] for r in passer}) > 1:
        problems.append("More than one player is marked as the passer.")
    nd, nr = len(by["defense"]), len(by["offense"])
    if not 1 <= nd <= relational.D_MAX:
        problems.append(f"{nd} defenders found; the model accepts 1 to {relational.D_MAX}.")
    if not 1 <= nr <= relational.R_MAX:
        problems.append(f"{nr} offensive players other than the passer found; the model accepts 1 to {relational.R_MAX} route runners. Mark the passer with role=passer and leave linemen out.")
    if problems:
        raise UploadError(problems)
    flip_y = lambda y: W - y if left else y
    snap_off = [r for r in rows if r["side"] == "offense" and r["frame"] == snap]
    ref_y = flip_y(passer[0]["y"]) if passer else float(np.mean([flip_y(r["y"]) for r in snap_off])) if snap_off else None
    if ref_y is None:
        raise UploadError(["No offensive player has a row at the snap frame, so the formation centre cannot be set."])
    if not passer:
        warn.append("No passer marked: the formation centre is the mean lateral position of the offense at the snap.")
    los_c = L - los_x if left else los_x
    T = relational.T_MAX
    n_frames = min(T, 1 + min(max(f) for side in by.values() for f in side.values()))
    arrs, masks, ids, first_gap = [], [], {}, T
    for side, cap in (("defense", relational.D_MAX), ("offense", relational.R_MAX)):
        a, m = np.zeros((T, cap, 6), np.float32), np.zeros(cap, bool)
        order = sorted(by[side], key=lambda s: (0, int(s)) if s.lstrip("-").isdigit() else (1, s))        # numeric ids sort as numbers, as offline
        ids[side] = order
        for i, pid in enumerate(order):
            fr = by[side][pid]
            m[i] = True
            for f in range(T):
                r = fr.get(f)
                if r is None:
                    first_gap = min(first_gap, f)
                    continue
                x, y = (L - r["x"] if left else r["x"]), flip_y(r["y"])
                if r["s"] is not None and r["dir"] is not None:
                    v = relational.unit(np.array((r["dir"] + 180.0) % 360.0 if left else r["dir"])) * r["s"]
                elif r["vx"] is not None and r["vy"] is not None:
                    v = np.array([-r["vx"], -r["vy"]] if left else [r["vx"], r["vy"]])
                else:
                    prev = fr.get(f - 1)                                              # backward difference: the previous frame only, never a later one
                    v = np.zeros(2) if prev is None else np.array([(x - (L - prev["x"] if left else prev["x"])) * 10, (y - flip_y(prev["y"])) * 10])
                o = np.zeros(2) if r["o"] is None else relational.unit(np.array((r["o"] + 180.0) % 360.0 if left else r["o"]))
                a[f, i] = [x - los_c, y - ref_y, v[0], v[1], o[0], o[1]]
        arrs.append(a), masks.append(m)
    usable = min(n_frames, first_gap)
    if first_gap < n_frames:
        warn.append(f"At least one player has no row at frame {first_gap} after the snap. Cutoffs from that frame on are disabled: missing frames are not filled in.")
    if not all((r["s"] is not None and r["dir"] is not None) or (r["vx"] is not None and r["vy"] is not None) for r in rows):
        warn.append("Velocity was derived from positions using the previous frame only; it is noisier than measured speed and direction.")
    if not has_o:
        warn.append("No observed orientation: the facing input is zero. Experimental result outside the trained input distribution.")
    return {"defs": arrs[0], "recs": arrs[1], "dmask": masks[0], "rmask": masks[1], "frames_usable": int(usable), "frames_supplied": int(n_frames), "ids": ids,
            "ref_y": float(ref_y), "los_x": float(los_c), "flipped": left, "warnings": warn, "has_orientation": has_o,
            "mode": mode if has_o else "broader (experimental)", "defenders": nd, "route_runners": nr}


_MODEL: dict = {}


def bundle() -> dict:
    if not _MODEL:
        b = torch.load(MODEL_FILE, weights_only=False)
        m = neural.CoverageNet(b["config"]["temporal"])
        m.load_state_dict(b["state_dict"])
        from ..shared.provenance import sha256_file
        _MODEL.update(model=m.eval(), platt=b["platt"], version=f"cov-v2-temporal-gru-seed{b['seed']}", sha256=sha256_file(MODEL_FILE))
    return _MODEL


def infer(t: dict, label: str | None = None, play_id: str = "upload") -> dict:
    b = bundle()
    with torch.no_grad():
        lg = b["model"](*(torch.as_tensor(t[k])[None] for k in ("defs", "recs", "dmask", "rmask")))[0].numpy()
    hz = {}
    for h, f in HZ.items():
        if t["frames_usable"] <= f:
            hz[h] = {"available": False, "reason": "the play was thrown (or the tracking stops, or a player is missing) before this cutoff; nothing is extrapolated"}
            continue
        p = float(1 / (1 + np.exp(-(b["platt"][h]["a"] * lg[f] + b["platt"][h]["b"]))))
        conf = max(p, 1 - p)
        d, r = t["defs"][f][t["dmask"]], t["recs"][f][t["rmask"]]
        sep = np.linalg.norm(d[:, None, :2] - r[None, :, :2], axis=-1)
        hz[h] = {"available": True, "frame": f, "p_man": p, "p_zone": 1 - p, "lean": "Man" if p >= 0.5 else "Zone", "confidence": conf,
                 "reliable_prediction_cutoff": THRESHOLDS[h], "accepted_under_reliable_policy": bool(conf >= THRESHOLDS[h]),
                 "observed": {"nearest_defender_to_each_route_runner_yards": [round(float(x), 2) for x in sep.min(0)], "mean_defender_depth_yards": round(float(d[:, 0].mean()), 2)}}
    experimental = t["mode"] != "release-compatible"
    return {
        "schema": SCHEMA, "play_id": play_id, "module": "coverage", "kind": "tracking classification",
        "model": {"version": b["version"], "bundle_sha256": b["sha256"], "input_schema": relational.SCHEMA, "calibration": "Platt per cutoff, fitted on weeks 13-14"},
        "mode": t["mode"], "horizons": hz,
        "input_quality": {"defenders": t["defenders"], "route_runners": t["route_runners"], "frames_supplied": t["frames_supplied"], "frames_usable": t["frames_usable"],
                          "observed_orientation": t["has_orientation"], "play_direction_flipped": t["flipped"], "warnings": t["warnings"]},
        "scope": ("Experimental: this input is outside what the model was trained on. No benchmark accuracy applies." if experimental else
                  "Agreement with the released coverage label was measured on the released selected-player tracking. It applies only if this play's players were chosen the same way; it is not validated for arbitrary tracking."),
        "comparison_label": None if label is None else {"value": label, "note": "Supplied with the upload and never used as an input. One play is not an accuracy measurement."},
        "explanation_note": "Probabilities are the model's calibrated outputs at each cutoff. The observed measurements are plain geometry from the upload, not reasons the model gave.",
    }


def run(data: bytes | str, kind: str = "csv", mode: str = "release-compatible") -> dict:
    p = parse(data, kind)
    return infer(build(p, mode), p.label, p.play_id)
