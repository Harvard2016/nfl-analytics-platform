"""Tracking upload contract. Mechanical checks use invented rows; the parity test uses a real local play and skips without the data."""
import numpy as np
import polars as pl
import pytest

from gridiron_lens.coverage import neural, relational
from gridiron_lens.coverage import upload as U
from gridiron_lens.shared import config

PROC = config.PROCESSED / "bdb2026"
needs_data = pytest.mark.skipif(not (PROC / "tracking_week_18.parquet").exists() or not U.MODEL_FILE.exists(), reason="local tracking and model not present")


def _rows(frames=16, defenders=4, receivers=3, direction="right", with_o=True):
    rng, out = np.random.default_rng(0), []
    for f in range(frames):
        for i in range(defenders):
            out.append({"play_id": "t", "player_id": f"D{i}", "frame": f, "side": "defense", "role": "", "x": 62 + i + 0.05 * f, "y": 5 + 4 * i, "s": 1.0, "dir": 90.0,
                        "o": 270.0 if with_o else "", "play_direction": direction, "los_x": 55})
        for i in range(receivers):
            out.append({"play_id": "t", "player_id": f"R{i}", "frame": f, "side": "offense", "role": "route_runner", "x": 54 + 0.4 * f, "y": 6 + 6 * i + rng.normal(0, 0.01), "s": 4.0, "dir": 90.0,
                        "o": 90.0 if with_o else "", "play_direction": direction, "los_x": 55})
        out.append({"play_id": "t", "player_id": "QB", "frame": f, "side": "offense", "role": "passer", "x": 50, "y": 26, "s": 0.2, "dir": 0.0, "o": 90.0 if with_o else "", "play_direction": direction, "los_x": 55})
    return out


def _csv(rows):
    cols = list(rows[0])
    return ",".join(cols) + "\n" + "\n".join(",".join(str(r[c]) for c in cols) for r in rows)


@needs_data
def test_label_cannot_change_the_prediction_and_is_only_echoed():
    a = U.run(_csv(_rows()))
    b = U.run(_csv([r | {"man_zone": "Man"} for r in _rows()]))
    c = U.run(_csv([r | {"man_zone": "Zone"} for r in _rows()]))
    assert a["horizons"] == b["horizons"] == c["horizons"]
    assert a["comparison_label"] is None and b["comparison_label"]["value"] == "Man"


@needs_data
def test_short_tracks_disable_unsupported_cutoffs():
    r = U.run(_csv(_rows(frames=8)))
    assert r["horizons"]["at_snap"]["available"] and r["horizons"]["post_0_5s"]["available"]
    assert not r["horizons"]["post_1s"]["available"] and not r["horizons"]["post_1_5s"]["available"]
    gap = [x for x in _rows() if not (x["player_id"] == "D1" and x["frame"] == 7)]
    g = U.run(_csv(gap))
    assert g["horizons"]["post_0_5s"]["available"] and not g["horizons"]["post_1s"]["available"]          # a missing frame is not filled in


@needs_data
def test_player_order_and_row_order_do_not_matter():
    rows = _rows()
    a = U.run(_csv(rows))
    b = U.run(_csv(list(reversed(rows))))
    assert a["horizons"] == b["horizons"]


def test_rejections_are_actionable():
    for bad, text in (([r | {"play_id": f"p{i % 2}"} for i, r in enumerate(_rows())], "one play"), (_rows() + [_rows()[0]], "more than one row"),
                      ([r | {"x": 500} for r in _rows()], "yards"), (_rows(defenders=12), "defenders"), (_rows(receivers=8), "route runners"),
                      ([{k: v for k, v in r.items() if k != "los_x"} for r in _rows()], "los_x"), (_rows(with_o=False), "orientation")):
        with pytest.raises(U.UploadError) as e:
            U.build(U.parse(_csv(bad), "csv"))
        assert text in str(e.value)
    with pytest.raises(U.UploadError):
        U.parse("player_id,frame\n1,0\n", "csv")


def test_missing_orientation_runs_only_as_experimental():
    t = U.build(U.parse(_csv(_rows(with_o=False)), "csv"), mode="broader")
    assert t["mode"] == "broader (experimental)" and (t["defs"][..., 4:] == 0).all()


@needs_data
def test_real_play_through_the_upload_path_matches_the_offline_pipeline():
    A = neural.load_arrays()
    saved = np.load(config.REPORTS / "v2" / "coverage_expB_predictions.npz")
    df = pl.read_parquet(PROC / "tracking_week_18.parquet")
    checked = 0
    for (gid, pid), p in list(df.partition_by(["gameId", "playId"], as_dict=True).items())[:40]:
        i = np.where(A["key"] == f"{gid}:{pid}")[0]
        if len(i) == 0:
            continue
        flipped = bool(p["flipped"][0])
        los_raw = config.FIELD_LENGTH - p["los_x"][0] if flipped else p["los_x"][0]
        back = lambda c: (pl.col(c) + 180.0) % 360.0 if flipped else pl.col(c)                           # undo the canonical rotation to get the raw angles
        rows = p.filter(pl.col("side") != "ball").select(
            pl.col("nflId").cast(pl.String).alias("player_id"), pl.col("frameId").alias("frame"), "side",
            pl.when(pl.col("role") == "Passer").then(pl.lit("passer")).otherwise(pl.lit("route_runner")).alias("role"),
            pl.col("x_raw").alias("x"), pl.col("y_raw").alias("y"), "s", back("dir").alias("dir"), back("o").alias("o"),
            pl.lit("left" if flipped else "right").alias("play_direction"), pl.lit(los_raw).alias("los_x")).to_dicts()
        t = U.build(U.parse(_csv(rows), "csv"))
        assert np.abs(t["defs"] - A["defs"][i[0]]).max() < 1e-3 and np.abs(t["recs"] - A["recs"][i[0]]).max() < 1e-3
        out = U.infer(t)
        for j, h in enumerate(relational.HORIZONS):
            if out["horizons"][h]["available"]:
                assert abs(out["horizons"][h]["p_man"] - saved["seed42"][i[0], j]) < 1e-3
            else:
                assert np.isnan(saved["seed42"][i[0], j])
        checked += 1
    assert checked >= 20
