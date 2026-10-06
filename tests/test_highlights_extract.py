"""The feature fingerprint used to compare a re-created CLIP stream with released features (numbers only; no model is loaded)."""
import numpy as np

from gridiron_lens.highlights import extract_clip as X


def test_fingerprint_separates_same_source_from_a_different_one():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 512) * 3                                           # the shared constant component of one model's features
    game = lambda: base + rng.normal(0, 1, (400, 512))
    same = X.fingerprint(game(), [game(), game(), game()])
    other = X.fingerprint(rng.normal(0, 1, 512) * 3 + rng.normal(0, 1, (400, 512)), [game(), game(), game()])
    assert same["mean_vector_cosine_new_vs_benchmark_games"]["mean"] > 0.95
    assert abs(other["mean_vector_cosine_new_vs_benchmark_games"]["mean"]) < 0.2
    assert abs(same["mean_vector_cosine_shuffled_dimensions_control"]) < 0.2
