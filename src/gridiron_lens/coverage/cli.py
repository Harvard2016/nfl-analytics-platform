"""Coverage pipeline commands: `bin/coverage-lens <command>`. Add --synthetic to run on the invented fixture."""
from __future__ import annotations

import json

import typer

from ..shared import config
from . import audit as audit_mod
from . import export as export_mod
from . import features as features_mod
from . import fixture as fixture_mod
from . import normalize as normalize_mod
from . import studies as studies_mod
from . import train as train_mod

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__, pretty_exceptions_enable=False)
REAL, SYN = "bdb2026", "synthetic_bdb"


def _src(synthetic: bool) -> str:
    return SYN if synthetic else REAL


def _show(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


@app.command()
def fixture(plays_per_game: int = 40, games_per_week: int = 2):
    """Write the SYNTHETIC fixture (invented data) to data/raw/synthetic_bdb."""
    typer.echo(f"wrote {config.rel(fixture_mod.generate(config.RAW / SYN, plays_per_game, games_per_week))}")


@app.command()
def audit(synthetic: bool = False, no_hash: bool = False):
    """Audit raw files: hashes, columns, labels, keys, frames and players per play."""
    r = audit_mod.run(_src(synthetic), with_hash=not no_hash)
    _show({"synthetic": r["synthetic"], "files": len(r["files"]), "total_bytes": r["total_bytes"],
           "tracked_plays": r.get("tracked_plays", r.get("plays")), "report": f"data/manifests/{_src(synthetic)}_audit.json"})


@app.command()
def normalize(synthetic: bool = False):
    """Direction and snap normalization to Parquet, one file per week, with convention checks."""
    r = normalize_mod.run(_src(synthetic))
    _show({"snap_definition": r["snap_definition"], "weeks": [
        {k: w.get(k) for k in ("week", "plays_written", "offense_behind_los_at_snap", "defense_beyond_los_at_snap",
                               "median_player_speed_at_snap", "direction_vs_displacement_cosine")} for w in r["weeks"]]})


@app.command()
def features(synthetic: bool = False):
    """Cutoff-safe features for each observation window."""
    r = features_mod.run(_src(synthetic))
    _show({w: {k: v[k] for k in ("latest_frame", "plays_with_features", "plays_with_man_zone_label", "tracking_play_status")}
           for w, v in r["windows"].items()})


@app.command()
def train(synthetic: bool = False):
    """Freeze splits, fit baselines, calibrate on development, score the locked test set."""
    r = train_mod.run(_src(synthetic))
    _show({"synthetic": r["synthetic"], "class_counts": r["class_counts"], "test": {
        w: {m: {k: round(v["test"][k], 4) for k in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier")}
            for m, v in res["models"].items()} for w, res in r["windows"].items()}})


@app.command()
def studies(synthetic: bool = False):
    """Play-count funnel, short throws, error analysis, held-out teams, multiclass (dev only), tendencies."""
    r = studies_mod.run(_src(synthetic))
    _show({"funnel": r["funnel"]["steps"], "evaluated": r["funnel"]["evaluated_plays"], "matches_manifest": r["funnel"]["matches_split_manifest"]})


@app.command("export-demo")
def export_demo(synthetic: bool = False, n_plays: int = 120):
    """Write per-play demo files for the website."""
    _show(export_mod.run(_src(synthetic), n_plays))



# ---- v2 experiments (real data only). Each writes a run record under reports/v2/runs/.

@app.command("relational")
def relational_cmd():
    """Build player tensors and relational defender-receiver features for every horizon."""
    from . import relational
    _show({"tensors": relational.build_tensors(), "features": {h: v["plays"] for h, v in relational.build_features()["horizons"].items()}})


@app.command("exp-a")
def exp_a():
    """Experiment A: relational features and Man class weights, chronological folds in weeks 1-12."""
    from . import experiment_a
    _show(experiment_a.cross_validate()["chosen"])


@app.command("exp-b")
def exp_b():
    """Experiment B: temporal interaction model, selection on folds then a fixed-epoch refit with three seeds."""
    from . import experiment_b
    _show(experiment_b.cross_validate()["chosen"])
    _show(experiment_b.final()["seeds"])


@app.command("exp-c")
def exp_c():
    """Experiment C: coverage family, flat boosted trees vs hierarchical temporal head (development weeks only)."""
    from . import experiment_c
    r = experiment_c.run()
    _show({h: {m: {k: round(v[k], 4) for k in ("accuracy", "macro_f1", "log_loss")} for m, v in e["models"].items()} for h, e in r["horizons"].items()})


@app.command("benchmark-v2")
def benchmark_v2_cmd():
    """Freeze v2 choices and compare every model on the previously examined benchmark."""
    from . import benchmark_v2
    r = benchmark_v2.evaluate()
    _show({m: {h: {k: round(e["benchmark"][k], 4) for k in ("accuracy", "macro_f1", "log_loss", "man_recall")} for h, e in v.items()} for m, v in r["models"].items()})


@app.command("export-v3")
def export_v3_cmd():
    """Site export with every selectable model and horizon, evidence, prefix outputs and similar plays."""
    from . import export_v3
    _show(export_v3.run())


if __name__ == "__main__":
    app()
