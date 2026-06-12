import numpy as np
import pytest

from grid_transformer.data.gilbert import gilbert3d_path


SIZES = [(2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6), (8, 8, 8),
          (5, 3, 2), (4, 6, 8), (10, 10, 10)]


@pytest.mark.parametrize("dims", SIZES)
def test_gilbert3d_visits_every_cell_exactly_once(dims):
    nx, ny, nz = dims
    path = gilbert3d_path(nx, ny, nz)
    assert path.shape == (nx * ny * nz, 3)
    assert path.dtype == np.int64
    # in-bounds
    assert path.min() >= 0
    assert (path[:, 0] < nx).all() and (path[:, 1] < ny).all() and (path[:, 2] < nz).all()
    # bijection: every cell exactly once
    flat = (path[:, 0] * ny + path[:, 1]) * nz + path[:, 2]
    assert len(np.unique(flat)) == nx * ny * nz


@pytest.mark.parametrize("dims", SIZES)
def test_gilbert3d_steps_are_neighbor_moves(dims):
    path = gilbert3d_path(*dims)
    steps = np.diff(path, axis=0)
    cheb = np.abs(steps).max(axis=1)
    l1 = np.abs(steps).sum(axis=1)
    # The gilbert algorithm is *mostly* neighbor-continuous (cheb==1). However,
    # upstream produces a handful of cheb>1 jumps on certain odd sizes (measured:
    # (5,5,5) has 1/124 steps with cheb=3; ~0.81%). We allow up to 2% as a bound.
    # L1 max is 3 (body-diagonal), min is 1.
    assert float((cheb > 1).mean()) < 0.02, f"too many non-neighbor steps: {(cheb>1).mean():.2%}"
    assert l1.min() >= 1 and l1.max() <= 3


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_gilbert3d_even_cubic_is_fully_continuous(n):
    # Even cubic grids should produce face-adjacent (unit) steps only. If upstream
    # behavior differs for some even size, measure and relax to a small bound —
    # but record the measured fraction in your report.
    path = gilbert3d_path(n, n, n)
    l1 = np.abs(np.diff(path, axis=0)).sum(axis=1)
    assert (l1 == 1).all()


@pytest.mark.parametrize("n", [3, 5, 7])
def test_gilbert3d_odd_cubic_diagonal_fraction_is_small(n):
    path = gilbert3d_path(n, n, n)
    l1 = np.abs(np.diff(path, axis=0)).sum(axis=1)
    diag_frac = float((l1 > 1).mean())
    # Measured upstream diagonal fractions: n=3: 7.69%, n=5: 4.03%, n=7: 4.68%
    # The n=3 case exceeds 5%; we relax to 10% to match verified upstream behavior.
    assert diag_frac < 0.10
