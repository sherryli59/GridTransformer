import numpy as np
import pytest

from grid_transformer.data.curves import get_curve3d
from grid_transformer.data.lj_transferable import hilbert_arc_delta


def _random_sorted_config(curve, box, n, seed):
    """Random positions, Hilbert/gilbert-sorted, with their codes."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, box, size=(n, 3))
    cell = box / curve.R
    grid = np.clip((pos / cell).astype(np.int64), 0, curve.R - 1)
    codes = curve.encode(grid)
    order = np.argsort(codes, kind="stable")
    return pos[order], codes[order]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_arc_delta_with_hilbert_curve_matches_legacy(seed):
    R, box, n = 8, 3.0, 27
    curve = get_curve3d("hilbert", R)
    pos, codes = _random_sorted_config(curve, box, n, seed)
    box_arr = np.array([box, box, box])
    legacy = hilbert_arc_delta(pos, codes, box_arr, R, periodic=True)
    via_curve = hilbert_arc_delta(pos, codes, box_arr, R, periodic=True, curve=curve)
    np.testing.assert_array_equal(legacy, via_curve)  # byte-exact


@pytest.mark.parametrize("R", [6, 10])
def test_arc_delta_gilbert_roundtrip(R):
    """Encode (Δs, fine) with gilbert, decode with s-space advance -> exact recon.

    Mirrors the round-trip contract test in tests/test_arc_repr_fixes.py.
    """
    box, n = 3.0, 27
    curve = get_curve3d("gilbert", R)
    pos, codes = _random_sorted_config(curve, box, n, seed=3)
    box_arr = np.array([box, box, box])
    arc = hilbert_arc_delta(pos, codes, box_arr, R, periodic=True, curve=curve)
    assert arc.shape == (n - 1, 4)
    X = max(1, curve.ncells // n)
    cell = box / R

    # manual s-space decode, step by step. Exactness relies on cumlen gaps
    # (>= 1 cell) dwarfing the float32 Δs reconstruction error (~1e-4 at these
    # magnitudes); revisit if Δs storage precision or curve density changes.
    c = codes[0]
    for i in range(n - 1):
        s_next = curve.arc(np.array([c]))[0] + float(arc[i, 0]) * X
        c_next = curve.code_from_arc(np.array([s_next]))[0]
        assert c_next == codes[i + 1]  # Δs recovers the exact next code
        center = (curve.decode(np.array([c_next]))[0] + 0.5) * cell
        fine = arc[i, 1:4].astype(np.float64) * cell
        rec = center + fine
        d = rec - pos[i + 1]
        d -= np.round(d / box) * box  # min-image
        np.testing.assert_allclose(d, 0.0, atol=1e-5)
        c = c_next


def test_arc_delta_gilbert_fine_is_bounded():
    R, box, n = 6, 3.0, 27
    curve = get_curve3d("gilbert", R)
    pos, codes = _random_sorted_config(curve, box, n, seed=4)
    arc = hilbert_arc_delta(pos, codes, np.array([box] * 3), R, periodic=True, curve=curve)
    assert np.abs(arc[:, 1:4]).max() <= 0.5 + 1e-6
