import math

import numpy as np
import pytest

from grid_transformer.data.curves import GilbertCurve3D, HilbertCurve3D, get_curve3d
from grid_transformer.data.lj_transferable import (
    _hilbert3d_decode,
    _hilbert3d_encode,
    _hilbert_bits,
)


def _all_cells(R):
    g = np.indices((R, R, R)).reshape(3, -1).T
    return np.ascontiguousarray(g, dtype=np.int64)


@pytest.mark.parametrize("R", [4, 8])
def test_hilbert_adapter_matches_legacy_functions(R):
    curve = HilbertCurve3D(R)
    grid = _all_cells(R)
    codes = curve.encode(grid)
    bits = _hilbert_bits(R)
    np.testing.assert_array_equal(
        codes, _hilbert3d_encode(grid[:, 0], grid[:, 1], grid[:, 2], bits=bits)
    )
    dec = curve.decode(codes)
    lx, ly, lz = _hilbert3d_decode(codes, bits=bits)
    np.testing.assert_array_equal(dec, np.stack([lx, ly, lz], axis=1))
    # the defining identity: code index IS the arc length for pow-2 Hilbert
    np.testing.assert_array_equal(curve.arc(codes), codes.astype(np.float64))
    assert curve.ncells == R ** 3


@pytest.mark.parametrize("R", [4, 8])
def test_hilbert_code_from_arc_matches_round_then_clamp(R):
    # must reproduce sample_lj._arc_decode_positions semantics byte-exactly:
    # c_next = clip(round(s), 0, R^3 - 1)
    curve = HilbertCurve3D(R)
    s = np.array([-5.0, -0.4, 0.0, 1.49, 1.51, R**3 - 1 + 0.4, R**3 + 7.0])
    codes, clamped = curve.code_from_arc(s, return_clamped=True)
    np.testing.assert_array_equal(
        codes, np.clip(np.round(s), 0, R**3 - 1).astype(np.int64)
    )
    np.testing.assert_array_equal(
        clamped, np.round(s).astype(np.int64) != codes
    )


@pytest.mark.parametrize("R", [5, 6])
def test_gilbert_encode_decode_roundtrip_all_cells(R):
    curve = GilbertCurve3D(R)
    grid = _all_cells(R)
    codes = curve.encode(grid)
    assert codes.dtype == np.int64
    assert sorted(codes.tolist()) == list(range(R ** 3))  # bijection
    np.testing.assert_array_equal(curve.decode(codes), grid)


@pytest.mark.parametrize("R", [4, 6, 10])
def test_gilbert_cumlen_even_R_is_unit_steps(R):
    # Even cubic grids are fully face-continuous (measured in Task 1: 0 diagonal
    # steps at R=4..10 and at the production sizes 64/86/106), so cumlen == code
    # index exactly. The nearest-even resolution rule makes this the production
    # regime; s-space is the safety net for any other R.
    curve = GilbertCurve3D(R)
    s = curve.arc(np.arange(R ** 3, dtype=np.int64))
    np.testing.assert_array_equal(s, np.arange(R ** 3, dtype=np.float64))


@pytest.mark.parametrize("R", [5, 9])
def test_gilbert_cumlen_odd_R_monotone(R):
    # Odd grids contain jump steps with lengths in {1, sqrt2, 2, 2*sqrt2, 3}
    # (measured from the verified upstream port — NOT just sqrt2/sqrt3).
    # cumlen must still be strictly monotone (invertible) with steps >= 1.
    curve = GilbertCurve3D(R)
    s = curve.arc(np.arange(R ** 3, dtype=np.int64))
    assert s[0] == 0.0
    steps = np.diff(s)
    assert (steps >= 1.0 - 1e-12).all()
    assert steps.max() <= 3.0 + 1e-12


@pytest.mark.parametrize("R", [5, 6])
def test_gilbert_code_from_arc_inverts_arc(R):
    curve = GilbertCurve3D(R)
    codes = np.arange(R ** 3, dtype=np.int64)
    rec, clamped = curve.code_from_arc(curve.arc(codes), return_clamped=True)
    np.testing.assert_array_equal(rec, codes)
    assert not clamped.any()
    # out-of-range s clamps to the ends and reports it
    rec2, cl2 = curve.code_from_arc(np.array([-3.0, curve.s_max + 3.0]), return_clamped=True)
    np.testing.assert_array_equal(rec2, [0, R ** 3 - 1])
    assert cl2.all()
    # midpoint rounds to the NEAREST code along the curve
    mid = 0.5 * (curve.arc(np.array([3])) + curve.arc(np.array([4])))
    rec3 = curve.code_from_arc(mid - 1e-9)
    assert rec3[0] in (3, 4)


def test_get_curve3d_factory_and_memoization():
    a = get_curve3d("hilbert", 8)
    assert isinstance(a, HilbertCurve3D)
    assert get_curve3d("hilbert", 8) is a  # memoized
    b = get_curve3d("gilbert", 6)
    assert isinstance(b, GilbertCurve3D)
    with pytest.raises(ValueError):
        get_curve3d("spectral", 8)
    with pytest.raises(ValueError):
        get_curve3d("hilbert", 6)  # not a power of two


def test_gilbert_disk_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("GILBERT_LUT_DIR", str(tmp_path))
    c1 = GilbertCurve3D(5)
    assert (tmp_path / "gilbert3d_R5.npz").exists()
    c2 = GilbertCurve3D(5)  # loads from disk
    np.testing.assert_array_equal(c1.coords, c2.coords)
    np.testing.assert_array_equal(c1.code_lut, c2.code_lut)
    np.testing.assert_array_equal(c1.cumlen, c2.cumlen)
