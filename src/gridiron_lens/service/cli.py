"""Command-line use of the same inference code as the API, for files too long for an upload (full games) and for scripted checks.

    PYTHONPATH=src .venv/bin/python -m gridiron_lens.service.cli highlights /path/to/game.mp4 --reel 180
    PYTHONPATH=src .venv/bin/python -m gridiron_lens.service.cli coverage /path/to/play.csv
    PYTHONPATH=src .venv/bin/python -m gridiron_lens.service.cli bundle-score 23

Results stay in the private job store (data/jobs/<id>/), with the same retention as API jobs.
`bundle-score` runs the trained H3 bundle on a benchmark game's released extractor outputs; it cannot score a new video,
because the extractors that produce those inputs are not installed here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from ..coverage import upload as cov_upload
from .store import Store
from .worker import process


def main() -> None:
    ap = argparse.ArgumentParser(description="Local inference without the HTTP service.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("highlights")
    h.add_argument("file", type=Path)
    h.add_argument("--reel", type=float, default=180.0)
    h.add_argument("--lead", type=float, default=2.0)
    h.add_argument("--tail", type=float, default=2.0)
    c = sub.add_parser("coverage")
    c.add_argument("file", type=Path)
    c.add_argument("--mode", default="release-compatible", choices=["release-compatible", "broader"])
    b = sub.add_parser("bundle-score")
    b.add_argument("game", type=int)
    a = ap.parse_args()
    if a.cmd == "coverage":
        print(json.dumps(cov_upload.run(a.file.read_bytes(), "json" if a.file.suffix == ".json" else "csv", a.mode), indent=1))
        return
    if a.cmd == "bundle-score":
        from ..highlights import bundle as B
        from ..highlights import decode as D
        from ..highlights import pipeline as P
        model, tr, man = B.load()
        n = len(P.load_labels()[a.game])
        t0 = time.time()
        s = B.score_raw(model, tr, {k: P.load_modality(a.game, k, n) for k in P.MODALITIES}, P.load_volume()[a.game], chunk=B.CHUNK)
        segs = D.decode(s, 180, block=3, lead_s=2, tail_s=2)
        print(json.dumps({"bundle": man["version"], "game": a.game, "clips": n, "seconds": round(time.time() - t0, 2), "streams": man["streams"],
                          "top_moments_s": [round(x.moment_s, 1) for x in segs[:10]], "reel_output_s": D.output_seconds(segs),
                          "note": "Scores from released extractor outputs for a benchmark game. Ranking scores, not probabilities."}, indent=1))
        return
    st = Store()
    jid, _tok, dest = st.create("highlights", "loudness_baseline", {"reel_seconds": a.reel, "lead_s": a.lead, "tail_s": a.tail, "local_cli": True}, a.file.suffix.lower() or ".bin")
    shutil.copyfile(a.file, dest)
    process(st, st.get(jid))
    row = st.get(jid)
    if row["state"] != "complete":
        raise SystemExit(f"{row['state']}: {row['error']}")
    res = json.loads((st.dir(jid) / "result.json").read_text())
    print(json.dumps({"job": jid, "dir": str(st.dir(jid)), "mode": res["mode"], "duration_s": res["media"]["duration_s"], "reel": res["reel"], "measured": res["measured"],
                      "candidates": [{k: c[k] for k in ("rank_in_reel", "candidate_moment_s", "clip_start_s", "clip_end_s", "ffprobe_duration_s")} for c in res["candidates"][:12]],
                      "clips": int(np.size(res["timeline"]["loudness_db"]))}, indent=1))


if __name__ == "__main__":
    main()
