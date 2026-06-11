"""Tests for the Step-3 arc_repr size-invariance audit (analyze_arc_delta_transfer.py).

The audit answers: is the (Δs, fine) target distribution size-invariant across
L=3/4/5 under the pow2 constant-cell transfer setting, or did the Hilbert-turn
problem just move into the Δs tail?
"""
import numpy as np

from grid_transformer.data.lj_transferable import _hilbert3d_decode, _hilbert_bits


def test_tail_fraction():
    from analyze_arc_delta_transfer import tail_fraction

    x = np.array([0.5, 1.0, 2.0, 3.0, 10.0])
    assert tail_fraction(x, 2.0) == 2.0 / 5.0  # strictly greater: 3 and 10
    assert tail_fraction(x, 0.0) == 1.0
    assert tail_fraction(x, 100.0) == 0.0


def test_sorted_arc_targets_known_config():
    """Particles at exact cell centers of known Hilbert codes (given shuffled)
    must yield Δs = code-diff / X and fine ≈ 0."""
    from analyze_arc_delta_transfer import sorted_arc_targets

    R, L, N = 8, 2.0, 4
    X = R**3 // N  # 128
    bits = _hilbert_bits(R)
    cell = L / R
    codes = np.array([3, 13, 77, 78], dtype=np.int64)
    ax, ay, az = _hilbert3d_decode(codes, bits)
    pos = (np.stack([ax, ay, az], axis=1).astype(np.float64) + 0.5) * cell

    rng = np.random.default_rng(1)
    shuffled = pos[rng.permutation(N)]

    arc = sorted_arc_targets(shuffled, L, R)  # [N-1, 4]
    assert arc.shape == (N - 1, 4)
    np.testing.assert_allclose(arc[:, 0], np.diff(codes) / X, atol=1e-6)
    np.testing.assert_allclose(arc[:, 1:4], 0.0, atol=1e-6)
