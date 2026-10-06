"""Structural checks for the v3 attention model and the robustness corruption (random inputs: mechanics, not accuracy)."""
import numpy as np
import torch

from gridiron_lens.coverage import experiment_v3 as E


def _inputs(B=3, T=16, D=11, R=6, nd=7, nr=4):
    torch.manual_seed(0)
    d, r = torch.randn(B, T, D, 6), torch.randn(B, T, R, 6)
    dm, rm = torch.zeros(B, D, dtype=torch.bool), torch.zeros(B, R, dtype=torch.bool)
    dm[:, :nd], rm[:, :nr] = True, True
    return d, r, dm, rm


def test_attention_model_is_order_invariant_causal_and_ignores_masked_slots():
    m = E.AttentionNet().eval()
    d, r, dm, rm = _inputs()
    with torch.no_grad():
        base = m(d, r, dm, rm)
        d2, r2 = d.clone(), r.clone()
        d2[:, :, :7], r2[:, :, :4] = d[:, :, torch.randperm(7)], r[:, :, torch.randperm(4)]
        assert (m(d2, r2, dm, rm) - base).abs().max() < 1e-4                         # player reordering
        d3 = d.clone()
        d3[:, 10:] = torch.randn_like(d3[:, 10:]) * 30                               # rewrite later frames
        assert (m(d3, r, dm, rm)[:, :10] - base[:, :10]).abs().max() < 1e-5
        d4, r4 = d.clone(), r.clone()
        d4[:, :, 7:], r4[:, :, 4:] = 1e3, -1e3                                       # garbage in masked slots
        assert (m(d4, r4, dm, rm) - base).abs().max() < 1e-4
        for nd, nr in ((1, 1), (11, 6), (3, 6)):                                     # variable player counts
            assert torch.isfinite(m(*_inputs(nd=nd, nr=nr))).all()


def test_robustness_corruption_never_uses_a_later_frame_and_keeps_minimum_players():
    rng = np.random.default_rng(0)
    d, r = rng.normal(size=(8, 16, 11, 6)).astype(np.float32), rng.normal(size=(8, 16, 6, 6)).astype(np.float32)
    dm, rm = np.zeros((8, 11), bool), np.zeros((8, 6), bool)
    dm[:, :5], rm[:, :3] = True, True
    d1, _r1, dm1, rm1 = E.degrade(d, r, dm, rm, np.random.default_rng(1))
    d_alt = d.copy()
    d_alt[:, 9:] += 100                                                              # change only frames 9+
    d2, *_ = E.degrade(d_alt, r, dm, rm, np.random.default_rng(1))
    assert np.allclose(d1[:, :9], d2[:, :9])                                         # earlier corrupted frames are unaffected
    assert (dm1.sum(1) >= E.NOISE["min_def"]).all() and (rm1.sum(1) >= E.NOISE["min_rec"]).all()
    assert not dm1[~dm].any() and not rm1[~rm].any()                                 # dropout never switches an absent player on
