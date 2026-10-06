"""Follow-through on the registered highlights rule: candidates that passed on development folds, rebuilt as a new version.

- H3-R: the H3 architecture with the ranking-aware loss, trained on all 28 training games for the fixed 5 epochs.
- Fusion: learned late fusion of H3 (saved v2 weights) with the v2 commentary word model. Fusion weights come from the
  development out-of-fold scores only.

Each is scored once on the 6 validation games and once on the 6 test games. The validation games steered H3's early
stopping in v2, and the test games were examined in v2: both are comparisons on previously used data, not fresh tests.
Nothing is selected here. H3 stays the shipped ranker for the site and for uploads.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from ..shared import config
from ..shared.provenance import now_utc, sha256_file, write_json
from ..shared.runs import Run
from . import decode as D
from . import experiment_v3 as E
from . import models as M
from . import pipeline as P

OUT = config.REPORTS / "v3"
DIR = config.MODELS / "highlights" / "v3" / "candidates"


def run(log=print) -> dict:
    torch.set_num_threads(4)
    split, labels = P.make_split(), P.load_labels()
    saved = np.load(config.REPORTS / "v2" / "highlights_scores.npz")
    oof = np.load(OUT / "highlights_v3_oof.npz")
    z = lambda a: (a - a.mean()) / (a.std() + 1e-6)
    tr = split["train"]
    fusion = LogisticRegression(C=1.0, max_iter=1000).fit(np.concatenate([np.c_[z(oof[f"H-C|{v}"]), z(oof[f"H1|{v}"])] for v in tr]), np.concatenate([labels[v] for v in tr]))
    feats = {v: M.load_game(v) for v in P.video_ids()}                       # v2 cache: transforms fitted on the 28 training games
    hr = E.train(feats, labels, tr, "H-R")
    DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": hr.state_dict(), "streams": M.STREAMS, "loss": "cross-entropy + 0.5 x within-window pairwise ranking", "epochs": E.EPOCHS, "seed": E.SEED}, DIR / "h3_ranking_aware.pt")
    write_json(DIR / "fusion.json", {"inputs": ["within-game z-score of H3", "within-game z-score of the v2 commentary word model"], "coef": fusion.coef_[0].tolist(), "intercept": float(fusion.intercept_[0]),
                                     "fitted_on": "development out-of-fold scores of the 28 training games"})
    scores = {"H3 (shipped, v2)": {v: saved[f"H3 temporal fusion|{v}"] for v in P.video_ids()},
              "H3-R ranking-aware (v3 candidate)": {v: M.score_game(hr, feats[v], M.STREAMS) for v in P.video_ids()},
              "H3 + commentary, learned fusion (v3 candidate)": {v: fusion.decision_function(np.c_[z(saved[f"H3 temporal fusion|{v}"]), z(saved[f"H1 commentary|{v}"])]) for v in P.video_ids()}}
    rep = {"module": "highlights", "version": "highlights-v3-candidates", "created_at_utc": now_utc(),
           "roles": {"validation": "6 games used for H3 early stopping in v2: development data, optimistic for H3 itself", "test": "6 games examined in v2: a previously examined benchmark, not a fresh test"},
           "fusion_weights": {"h3": float(fusion.coef_[0][0]), "commentary": float(fusion.coef_[0][1])}, "results": {}}
    base = "H3 (shipped, v2)"
    for name, s in scores.items():
        rep["results"][name] = {}
        for part in ("validation", "test"):
            g = split[part]
            ap = np.array([average_precision_score(labels[v], s[v]) for v in g])
            ap0 = np.array([average_precision_score(labels[v], scores[base][v]) for v in g])
            strict = [D.metrics(labels[v], D.decode(s[v], 180, block=3, lead_s=2, tail_s=2), 180) for v in g]
            rep["results"][name][part] = {"mean_average_precision": float(ap.mean()), "per_game": [round(float(x), 4) for x in ap],
                                          "strict_180s_precision": float(np.mean([m["precision"] for m in strict])), "strict_180s_recall": float(np.mean([m["recall_of_labelled_time"] for m in strict])),
                                          **({} if name == base else {"vs_shipped": E.boot_diff(ap, ap0), "games_better": int((ap > ap0).sum())})}
    rep["decision"] = {"shipped": base, "statement": "Candidates are recorded with their weights. H3 stays the shipped ranker: six games per split give wide intervals, both splits have been used before, and the fusion candidate needs a transcript, which uploads do not have yet.",
                       "rollback": "The v2 weights and the v3 bundle are untouched; candidates live in models/highlights/v3/candidates/."}
    rep["files"] = {f.name: sha256_file(f) for f in sorted(DIR.iterdir())}
    write_json(OUT / "highlights_candidates_v3.json", rep)
    r = Run("highlights", "v3-candidates", "v3", seed=E.SEED)
    r.set(target="editorial highlight selection (ranking)", population="6 validation and 6 test games", split=rep["roles"],
          metrics={k: {p: v[p]["mean_average_precision"] for p in v} for k, v in rep["results"].items()})
    r.output("report", OUT / "highlights_candidates_v3.json")
    r.finish()
    for k, v in rep["results"].items():
        log(k, {p: (round(v[p]["mean_average_precision"], 4), v[p].get("vs_shipped", {}).get("interval_95_over_games"), v[p].get("games_better")) for p in v})
    return rep


if __name__ == "__main__":
    run()
