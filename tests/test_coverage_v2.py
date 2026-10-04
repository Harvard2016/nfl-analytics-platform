"""Mechanical checks for the v2 coverage code: transforms, masks, order invariance, future isolation, policies.

These use random or tiny inputs. They verify code behaviour, not model quality on NFL data.
"""
import numpy as np
import pytest

from gridiron_lens.coverage import evalkit, neural, relational


def _players(rng, T=16, D=6, R=4):
    return rng.normal(0, 5, (T, D, 6)), rng.normal(0, 5, (T, R, 6))


def test_temporal_model_is_order_invariant_causal_and_ignores_masked_slots():
    c = neural.structural_checks()
    for kind in ("gru", "tcn"):
        assert c[kind]["player_order_max_abs_diff"] < 1e-5
        assert c[kind]["future_frames_max_abs_diff_on_earlier_outputs"] == 0.0
        assert c[kind]["later_outputs_do_change"] is True
        assert c[kind]["masked_slots_max_abs_diff"] == 0.0
    assert c["reflection_max_abs_diff"] < 1e-9


def test_angle_convention_matches_motion():
    # 0 degrees points to +y, 90 degrees to +x (clockwise from +y), as verified against displacement on the real data
    assert np.allclose(relational.unit(np.array([0.0, 90.0, 180.0, 270.0])), [[0, 1], [1, 0], [0, -1], [-1, 0]], atol=1e-12)


def test_relational_features_refuse_frames_past_the_horizon_and_are_order_invariant():
    rng = np.random.default_rng(0)
    d, r = _players(rng)
    with pytest.raises(AssertionError):
        relational.play_relational(d, r, 10)                       # 16 frames passed for a 10-frame horizon
    a = relational.play_relational(d[:11], r[:11], 10)
    b = relational.play_relational(d[:11][:, rng.permutation(6)], r[:11][:, rng.permutation(4)], 10)
    assert a.keys() == b.keys()
    assert all(np.isclose(a[k], b[k], equal_nan=True) for k in a)


def test_reflection_keeps_separations_and_flips_lateral_terms():
    rng = np.random.default_rng(1)
    d, r = _players(rng)
    P, Q = relational.pair_arrays(d, r), relational.pair_arrays(relational.reflect(d), relational.reflect(r))
    assert np.allclose(P["sep"], Q["sep"]) and np.allclose(P["closing"], Q["closing"]) and np.allclose(P["facing"], Q["facing"])
    assert np.allclose(P["rel"][..., 1], -Q["rel"][..., 1]) and np.allclose(P["rel"][..., 0], Q["rel"][..., 0])


def test_abstention_sweep_denominators_and_separate_policies():
    y = np.array([1] * 40 + [0] * 60)
    p = np.concatenate([np.linspace(0.3, 0.95, 40), np.linspace(0.02, 0.6, 60)])
    sweep = evalkit.abstention_sweep(y, p)
    first = sweep[0]
    assert first["cutoff"] == 0.5 and first["accepted"] == 100 and first["accepted_fraction_of_all"] == 1.0
    for row in sweep:
        assert 0 <= row["man_correct_accepted_over_all_man"] <= row["man_accepted_fraction_of_man"] <= 1
        assert 0 <= row["zone_correct_accepted_over_all_zone"] <= row["zone_accepted_fraction_of_zone"] <= 1
    pol = evalkit.balanced_policy(y, p)
    assert pol["supported"] and pol["calibration_man_precision"] >= 0.80
    hopeless = evalkit.balanced_policy(y, np.random.default_rng(0).random(100))
    assert hopeless["supported"] is False and hopeless["threshold"] == 0.5
