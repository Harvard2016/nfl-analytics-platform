"""Site export for the highlight timeline. Derived measurements, labels, links and short transcript excerpts only: no video or audio."""
from __future__ import annotations

import json

import joblib
import numpy as np

from ..shared import config
from ..shared.provenance import now_utc, write_json
from . import models as M
from . import pipeline as P

WEB = config.WEB_DEMO / "highlights"
EXCERPT = 100
SHOWN = ["H0 loudness", "H1 commentary", "H3 temporal fusion"]


def pct_rank(a: np.ndarray) -> list[int]:
    """Percentile rank within the game, 0-100. A rank, not a probability."""
    return np.round(100 * np.argsort(np.argsort(a)) / max(1, len(a) - 1)).astype(int).tolist()


def excerpt(words: list[dict], lo: float, hi: float) -> str:
    text = " ".join(w["word"] for w in words if "start" in w and lo <= float(w["start"]) < hi).strip()
    return text if len(text) <= EXCERPT else text[: EXCERPT - 1].rsplit(" ", 1)[0] + "…"


def run() -> dict:
    rep = json.loads((P.OUT / "highlights_ranking.json").read_text())
    split, labels, vol = P.make_split(), P.load_labels(), P.load_volume()
    meta = {m["id"]: m for m in P.metadata()}
    z = np.load(P.OUT / "highlights_scores.npz")
    shipped = rep["shipped"]
    text = joblib.load(config.MODELS / "highlights" / "commentary_tfidf_logistic.joblib")
    names, coef = np.array(text["vectorizer"].get_feature_names_out()), text["model"].coef_[0]
    (WEB / "games").mkdir(parents=True, exist_ok=True)
    index = []
    for split_name in ("test", "validation"):
        for v in split[split_name]:
            y, m = labels[v], meta[v]
            words = json.loads((P.PROC / "whisper" / f"american_football_{v}.json").read_text())
            scores = {n: z[f"{n}|{v}"] for n in dict.fromkeys([shipped, *SHOWN])}
            main = scores[shipped]
            loud = M.loudness_features(vol[v])[:, 0]
            sel = {name: M.select_budget(main, k) for name, k in M.BUDGETS.items()}
            docs = M.clip_documents(v, len(y))
            cands = []
            for a, b in M.runs_of_ones(sel["3 minutes"].astype(int)):
                x = text["vectorizer"].transform([" ".join(docs[a:b])])
                contrib = x.multiply(coef).toarray()[0]
                top = [names[i].replace("n_", "").replace("a_", "(after) ", 1).replace("a_", "") for i in np.argsort(-contrib)[:5] if contrib[i] > 0]
                cands.append({"start_clip": int(a), "end_clip": int(b), "start_s": a * 2, "end_s": b * 2, "source_time_s": round(m["trim_start_s"] + a * 2, 1),
                              "score_rank": int(np.round(100 * (np.argsort(np.argsort(main))[a:b].max()) / (len(main) - 1))),
                              "loudness_above_background": round(float(loud[a:b].max()), 1), "loudness_rank": int(pct_rank(loud)[int(a + np.argmax(loud[a:b]))]),
                              "commentary_terms": top, "excerpt": excerpt(words, a * 2 - 2, b * 2 + 12),
                              "editorial_overlap": round(float(y[a:b].mean()), 2), "in_editorial_highlights": bool(y[a:b].any())})
            cands.sort(key=lambda c: -c["score_rank"])
            missed = [{"start_s": int(a * 2), "end_s": int(b * 2), "source_time_s": round(m["trim_start_s"] + a * 2, 1), "seconds": int((b - a) * 2),
                       "best_score_rank": int(max(pct_rank(main)[a:b])), "excerpt": excerpt(words, a * 2 - 2, b * 2 + 12)}
                      for a, b in M.runs_of_ones(y) if not sel["3 minutes"][a:b].any()]
            missed.sort(key=lambda r: -r["seconds"])
            segs = json.loads((P.PROC / "segments" / f"american_football_{v}.json").read_text())
            gm = rep["results"][shipped][split_name]["per_game"][str(v)]
            game = {
                "id": v, "source_id": m["source_id"], "title": m["title"], "league": m["league"], "split": split_name, "clip_seconds": 2,
                "source": {"full_video": m["full_link"], "official_highlights": m["highlight_link"], "trim_start_s": m["trim_start_s"],
                           "time_note": "Timeline seconds count from the start of the trimmed broadcast. Source video time = trim start + timeline time."},
                "clips": len(y), "label": y.tolist(), "loudness_above_background": np.round(loud, 1).tolist(),
                "scores": {n: pct_rank(s) for n, s in scores.items()}, "shipped_model": shipped,
                "selected": {name: s.astype(int).tolist() for name, s in sel.items()},
                "candidates": cands, "missed_highlights": missed[:25], "missed_highlight_count": len(missed),
                "commentary": [{"start_s": round(float(s["start"]), 1), "end_s": round(float(s["end"]), 1), "excerpt": (s["text"][:EXCERPT - 1] + "…") if len(s["text"]) > EXCERPT else s["text"]} for s in segs],
                "metrics": gm,
            }
            (WEB / "games" / f"{v}.json").write_text(json.dumps(game, separators=(",", ":"), default=lambda o: o.item()))
            index.append({"id": v, "title": m["title"], "split": split_name, "clips": len(y), "average_precision": gm["average_precision"],
                          "chance": gm["chance_average_precision"], "three_minute_precision": gm["3 minutes"]["precision"], "file": f"games/{v}.json"})
    aud = json.loads((config.MANIFESTS / "svhighlights_audit.json").read_text())
    slim = {n: {"info": {k: v for k, v in r["info"].items() if k != "history"}, "validation": r["validation"]["summary"], "test": r["test"]["summary"]} for n, r in rep["results"].items()}
    write_json(WEB / "index.json", {
        "generated_at": now_utc(), "module": "highlights", "task": rep["task"], "scope": rep["scope"], "label": rep["label"], "score_meaning": rep["score_meaning"],
        "calibration": rep["calibration"], "captions": rep["captions"], "budget_note": rep["budget_note"], "selection_rule": rep["selection_rule"], "shipped": shipped,
        "split": rep["split"], "clips": rep["clips"], "positive_share": rep["positive_share"], "run_id": rep["run_id"],
        "attribution": "SVHighlights (Lee, Ki, Kang, Kim; KDD 2026), CC BY-NC 4.0. Features and annotations only; no broadcast video is hosted here.",
        "source": rep["source"], "revision": rep["revision"], "results": slim, "games": index,
        "all_games": [{k: g[k] for k in ("id", "title", "league", "season_in_title", "clips", "positive_share", "full_link")} for g in aud["games"]],
        "event_extension": {"status": "blocked", "needs": "Verified video-time to play alignment (scoreboard clock reading or manual anchors) and authorized footage for visual review. Without them no event label or event metric is reported.",
                            "what_exists": "Game identities are verified from source titles. Ranking, evidence and missed-highlight review work without video."},
    })
    return {"games": len(index), "out": config.rel(WEB)}
