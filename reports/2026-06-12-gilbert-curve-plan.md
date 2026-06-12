# Generalized Hilbert (gilbert) Curve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generalized Hilbert space-filling curve (`ordering="gilbert"`, arbitrary non-pow-2 grid resolutions) alongside the existing pow-2 Hilbert curve, so the physical cell size can be held constant (<1% deviation) across box sizes L3/L4/L5 instead of varying up to 1.5× from octave rounding.

**Architecture:** A new curve abstraction (`grid_transformer/data/curves.py`) exposes `encode / decode / arc / code_from_arc` for both curve types. The non-uniform step lengths of gilbert (occasional diagonal steps on odd-size grids) are handled by a precomputed cumulative-arc-length LUT `cumlen[c]`: encode uses `Δs = Δcumlen/X`, decode advances in s-space and inverts via `searchsorted`. For the classical Hilbert curve `cumlen[c] ≡ c`, so the generalized equations reduce byte-exactly to current behavior. Gilbert LUTs (path, inverse, cumlen) are built once per resolution by a Python port of Červený's gilbert3d and cached to disk. Resolution rule for gilbert: nearest **even** integer of L/cell (even sizes minimize diagonal steps).

**Tech Stack:** numpy, torch, pytest. Python: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python`. No GPU needed (do NOT touch the in-flight multi-size training, PID 523122).

**Out of scope (YAGNI):** curve-rail + gilbert (arc baseline trains with `USE_CURVE_RAIL=0`; raise `NotImplementedError`), 2D gilbert, spectral ordering changes, rectangular (non-cubic) boxes, rebuilding existing caches.

**Key existing code (read before starting):**
- `grid_transformer/data/lj_transferable.py:41-160` — `_hilbert3d_encode`, `_hilbert3d_decode`, `_hilbert_bits` (Skilling transpose).
- `grid_transformer/data/lj_transferable.py:333-370` — `hilbert_arc_delta` (the (Δs, fine) target builder).
- `grid_transformer/data/lj_transferable.py:1095-1103` — `_resolution_for_box` (pow-2 constant-cell rule).
- `sample_lj.py:409-470` — `_arc_decode_positions` (s-space decode lives here).
- `sample_lj.py:361-377` — `_rail_resolution_for_box` (sampler's mirror of the resolution rule).
- `tests/test_arc_repr_fixes.py` — has the existing encode/decode round-trip contract test to mirror.

---

## Task 1: gilbert3d path generator

**Files:**
- Create: `grid_transformer/data/gilbert.py`
- Test: `tests/test_gilbert_curve.py`

- [ ] **Step 1: Fetch the reference implementation for verification**

Run:
```bash
curl -sL https://raw.githubusercontent.com/jakubcerveny/gilbert/master/gilbert3d.py -o /tmp/gilbert3d_upstream.py && head -50 /tmp/gilbert3d_upstream.py
```
Keep `/tmp/gilbert3d_upstream.py` open while writing Step 3 — the port below was written from memory and MUST be checked against upstream line-by-line (signs and offsets in the recursion are easy to get wrong; the tests in Step 2 are the hard gate). If the network is unavailable, rely on the tests: full-coverage + Chebyshev-adjacency across many sizes pins the algorithm down.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_gilbert_curve.py`:

```python
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
    # every step moves to one of the 26 neighbors (diagonals allowed, no jumps)
    assert (cheb == 1).all()
    assert l1.min() >= 1 and l1.max() <= 3


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_gilbert3d_even_cubic_is_fully_continuous(n):
    # Even cubic grids should produce face-adjacent (unit) steps only. If upstream
    # behavior differs for some even size, measure and relax to a small bound —
    # but record the measured fraction in the plan/report.
    path = gilbert3d_path(n, n, n)
    l1 = np.abs(np.diff(path, axis=0)).sum(axis=1)
    assert (l1 == 1).all()


@pytest.mark.parametrize("n", [3, 5, 7])
def test_gilbert3d_odd_cubic_diagonal_fraction_is_small(n):
    path = gilbert3d_path(n, n, n)
    l1 = np.abs(np.diff(path, axis=0)).sum(axis=1)
    diag_frac = float((l1 > 1).mean())
    assert diag_frac < 0.05
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/test_gilbert_curve.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'grid_transformer.data.gilbert'`

- [ ] **Step 4: Write the generator**

Create `grid_transformer/data/gilbert.py`. Port of https://github.com/jakubcerveny/gilbert (BSD 2-Clause, Jakub Červený) — verify every branch against `/tmp/gilbert3d_upstream.py` from Step 1:

```python
"""Generalized Hilbert ("gilbert") space-filling curve for arbitrary 3D grids.

Port of https://github.com/jakubcerveny/gilbert (gilbert3d.py, BSD 2-Clause,
Jakub Cerveny). Unlike the classical Hilbert curve (grid_transformer.data.
lj_transferable._hilbert3d_encode), the grid sides need not be powers of two,
which lets the physical cell size be held ~constant across box lengths. The
price: on odd-size grids a small fraction of consecutive steps are diagonal
(length sqrt(2) or sqrt(3) cells) — consumers must use the cumulative arc
length, not the code index, as the along-curve coordinate (see curves.py).
"""
from __future__ import annotations

import numpy as np


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _generate3d(x, y, z, ax, ay, az, bx, by, bz, cx, cy, cz):
    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)
    dax, day, daz = _sgn(ax), _sgn(ay), _sgn(az)
    dbx, dby, dbz = _sgn(bx), _sgn(by), _sgn(bz)
    dcx, dcy, dcz = _sgn(cx), _sgn(cy), _sgn(cz)

    if h == 1 and d == 1:
        for _ in range(w):
            yield (x, y, z)
            x, y, z = x + dax, y + day, z + daz
        return
    if w == 1 and d == 1:
        for _ in range(h):
            yield (x, y, z)
            x, y, z = x + dbx, y + dby, z + dbz
        return
    if w == 1 and h == 1:
        for _ in range(d):
            yield (x, y, z)
            x, y, z = x + dcx, y + dcy, z + dcz
        return

    ax2, ay2, az2 = ax // 2, ay // 2, az // 2
    bx2, by2, bz2 = bx // 2, by // 2, bz // 2
    cx2, cy2, cz2 = cx // 2, cy // 2, cz // 2
    w2 = abs(ax2 + ay2 + az2)
    h2 = abs(bx2 + by2 + bz2)
    d2 = abs(cx2 + cy2 + cz2)
    if (w2 % 2) and (w > 2):
        ax2, ay2, az2 = ax2 + dax, ay2 + day, az2 + daz
    if (h2 % 2) and (h > 2):
        bx2, by2, bz2 = bx2 + dbx, by2 + dby, bz2 + dbz
    if (d2 % 2) and (d > 2):
        cx2, cy2, cz2 = cx2 + dcx, cy2 + dcy, cz2 + dcz

    if (2 * w > 3 * h) and (2 * w > 3 * d):
        yield from _generate3d(x, y, z,
                               ax2, ay2, az2, bx, by, bz, cx, cy, cz)
        yield from _generate3d(x + ax2, y + ay2, z + az2,
                               ax - ax2, ay - ay2, az - az2, bx, by, bz, cx, cy, cz)
    elif 3 * h > 4 * d:
        yield from _generate3d(x, y, z,
                               bx2, by2, bz2, cx, cy, cz, ax2, ay2, az2)
        yield from _generate3d(x + bx2, y + by2, z + bz2,
                               ax, ay, az, bx - bx2, by - by2, bz - bz2, cx, cy, cz)
        yield from _generate3d(x + (ax - dax) + (bx2 - dbx),
                               y + (ay - day) + (by2 - dby),
                               z + (az - daz) + (bz2 - dbz),
                               -bx2, -by2, -bz2, cx, cy, cz,
                               -(ax - ax2), -(ay - ay2), -(az - az2))
    elif 3 * d > 4 * h:
        yield from _generate3d(x, y, z,
                               cx2, cy2, cz2, ax2, ay2, az2, bx, by, bz)
        yield from _generate3d(x + cx2, y + cy2, z + cz2,
                               ax, ay, az, bx, by, bz, cx - cx2, cy - cy2, cz - cz2)
        yield from _generate3d(x + (ax - dax) + (cx2 - dcx),
                               y + (ay - day) + (cy2 - dcy),
                               z + (az - daz) + (cz2 - dcz),
                               -cx2, -cy2, -cz2,
                               -(ax - ax2), -(ay - ay2), -(az - az2), bx, by, bz)
    else:
        yield from _generate3d(x, y, z,
                               bx2, by2, bz2, cx2, cy2, cz2, ax2, ay2, az2)
        yield from _generate3d(x + bx2, y + by2, z + bz2,
                               cx, cy, cz, ax2, ay2, az2, bx - bx2, by - by2, bz - bz2)
        yield from _generate3d(x + (bx2 - dbx) + (cx - dcx),
                               y + (by2 - dby) + (cy - dcy),
                               z + (bz2 - dbz) + (cz - dcz),
                               ax, ay, az, -bx2, -by2, -bz2,
                               -(cx - cx2), -(cy - cy2), -(cz - cz2))
        yield from _generate3d(x + (ax - dax) + bx2 + (cx - dcx),
                               y + (ay - day) + by2 + (cy - dcy),
                               z + (az - daz) + bz2 + (cz - dcz),
                               -cx, -cy, -cz, -(ax - ax2), -(ay - ay2), -(az - az2),
                               bx - bx2, by - by2, bz - bz2)
        yield from _generate3d(x + (ax - dax) + (bx2 - dbx),
                               y + (ay - day) + (by2 - dby),
                               z + (az - daz) + (bz2 - dbz),
                               -bx2, -by2, -bz2, cx, cy, cz,
                               -(ax - ax2), -(ay - ay2), -(az - az2))


def gilbert3d_path(nx: int, ny: int, nz: int) -> np.ndarray:
    """Traversal order of the generalized Hilbert curve on an nx*ny*nz grid.

    Returns [nx*ny*nz, 3] int64: row i is the grid cell with curve code i.
    Pure Python recursion run once per resolution — cache the result (curves.py
    wraps this with a disk cache).
    """
    nx, ny, nz = int(nx), int(ny), int(nz)
    if min(nx, ny, nz) < 1:
        raise ValueError(f"grid sides must be >= 1, got {(nx, ny, nz)}")
    if nx >= ny and nx >= nz:
        gen = _generate3d(0, 0, 0, nx, 0, 0, 0, ny, 0, 0, 0, nz)
    elif ny >= nx and ny >= nz:
        gen = _generate3d(0, 0, 0, 0, ny, 0, nx, 0, 0, 0, 0, nz)
    else:
        gen = _generate3d(0, 0, 0, 0, 0, nz, nx, 0, 0, 0, ny, 0)
    path = np.fromiter(gen, dtype=np.dtype((np.int64, 3)), count=nx * ny * nz)
    return path
```

Note: `np.fromiter` with a structured dtype requires numpy >= 1.23; if it rejects the tuple yield, fall back to `np.array(list(gen), dtype=np.int64)` (same result, slightly more memory, still one-time).

- [ ] **Step 5: Run tests to verify they pass**

Run: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/test_gilbert_curve.py -x -q`
Expected: all PASS. If coverage/adjacency tests fail, the port has a transcription error — diff `_generate3d` against `/tmp/gilbert3d_upstream.py` branch by branch (the three split cases and the offsets `(ax-dax)+(bx2-dbx)` etc. are the usual culprits). Do not "fix" by relaxing tests.

- [ ] **Step 6: Record the measured diagonal fractions**

Run:
```bash
/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -c "
import numpy as np
from grid_transformer.data.gilbert import gilbert3d_path
for n in (64, 86, 106):
    p = gilbert3d_path(n, n, n)
    l1 = np.abs(np.diff(p, axis=0)).sum(axis=1)
    print(f'R={n}: cells={len(p)}, diagonal steps={(l1>1).sum()} ({(l1>1).mean():.4%})')"
```
Expected: runs in well under a minute per size; diagonal fraction 0% for even sizes (or small — record whatever is measured). Paste the output into the commit message body.

- [ ] **Step 7: Commit**

```bash
git add grid_transformer/data/gilbert.py tests/test_gilbert_curve.py
git commit -m "Add gilbert3d generalized Hilbert path generator (arbitrary grid sizes)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Curve abstraction with cumulative-arc-length LUTs

**Files:**
- Create: `grid_transformer/data/curves.py`
- Test: `tests/test_curves_interface.py`

This is the heart of the "non-uniform jump" fix. Both curves expose the same four operations; the along-curve coordinate is **cumulative arc length in cell units** (`arc`), not the raw code. For the classical Hilbert curve `arc(c) == c` exactly, so downstream math is unchanged for hilbert.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_curves_interface.py`:

```python
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


@pytest.mark.parametrize("R", [5, 6])
def test_gilbert_cumlen_properties(R):
    curve = GilbertCurve3D(R)
    s = curve.arc(np.arange(R ** 3, dtype=np.int64))
    assert s[0] == 0.0
    steps = np.diff(s)
    assert (steps > 0).all()  # strictly monotone -> invertible
    allowed = {1.0, math.sqrt(2.0), math.sqrt(3.0)}
    assert all(any(abs(st - a) < 1e-12 for a in allowed) for st in steps)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/test_curves_interface.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'grid_transformer.data.curves'`

- [ ] **Step 3: Implement curves.py**

Create `grid_transformer/data/curves.py`:

```python
"""Space-filling-curve abstraction for the arc representation.

Two curve families share one interface:

- ``HilbertCurve3D`` — classical pow-2 Hilbert (Skilling transpose, wraps the
  existing functions in lj_transferable.py). Every step is face-adjacent, so
  the cumulative arc length equals the code index: ``arc(c) == c``.
- ``GilbertCurve3D`` — generalized Hilbert on an arbitrary cubic grid
  (gilbert.py). Odd grids contain occasional diagonal steps, so code index and
  arc length differ; ``cumlen`` is the precomputed cumulative physical length
  (cell units: face step = 1, diagonals sqrt(2)/sqrt(3)).

All downstream arc math must use ``arc``/``code_from_arc`` (s-space), never
raw code arithmetic — that is what makes the two families interchangeable.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .gilbert import gilbert3d_path
from .lj_transferable import (
    _hilbert3d_decode,
    _hilbert3d_encode,
    _hilbert_bits,
    _is_power_of_two,
)


class HilbertCurve3D:
    def __init__(self, R: int):
        if not _is_power_of_two(int(R)):
            raise ValueError(f"HilbertCurve3D requires power-of-two R, got {R}")
        self.R = int(R)
        self.ncells = self.R ** 3
        self.s_max = float(self.ncells - 1)
        self._bits = _hilbert_bits(self.R)

    def encode(self, grid: np.ndarray) -> np.ndarray:
        g = np.asarray(grid, dtype=np.int64)
        return _hilbert3d_encode(g[..., 0], g[..., 1], g[..., 2], bits=self._bits)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        x, y, z = _hilbert3d_decode(np.asarray(codes, dtype=np.int64), bits=self._bits)
        return np.stack([x, y, z], axis=-1)

    def arc(self, codes: np.ndarray) -> np.ndarray:
        return np.asarray(codes, dtype=np.float64)

    def code_from_arc(self, s: np.ndarray, *, return_clamped: bool = False):
        raw = np.round(np.asarray(s, dtype=np.float64)).astype(np.int64)
        codes = np.clip(raw, 0, self.ncells - 1)
        if return_clamped:
            return codes, raw != codes
        return codes


class GilbertCurve3D:
    def __init__(self, R: int):
        self.R = int(R)
        if self.R < 2:
            raise ValueError(f"GilbertCurve3D requires R >= 2, got {R}")
        self.ncells = self.R ** 3
        self.coords, self.code_lut, self.cumlen = self._build_or_load(self.R)
        self.s_max = float(self.cumlen[-1])

    @staticmethod
    def _build_or_load(R: int):
        cache_dir = Path(
            os.environ.get(
                "GILBERT_LUT_DIR",
                Path.home() / ".cache" / "grid_transformer" / "gilbert",
            )
        )
        cache_file = cache_dir / f"gilbert3d_R{R}.npz"
        if cache_file.exists():
            with np.load(cache_file) as z:
                return (
                    z["coords"].astype(np.int64),
                    z["code_lut"].astype(np.int64),
                    z["cumlen"].astype(np.float64),
                )
        coords = gilbert3d_path(R, R, R)  # [M, 3]
        code_lut = np.empty((R, R, R), dtype=np.int64)
        code_lut[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(
            coords.shape[0], dtype=np.int64
        )
        step_len = np.sqrt(
            (np.diff(coords, axis=0).astype(np.float64) ** 2).sum(axis=1)
        )
        cumlen = np.concatenate([[0.0], np.cumsum(step_len)])
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".npz.tmp")
        np.savez(tmp, coords=coords, code_lut=code_lut, cumlen=cumlen)
        os.replace(tmp, cache_file)  # atomic: safe under multi-worker DataLoader
        return coords, code_lut, cumlen

    def encode(self, grid: np.ndarray) -> np.ndarray:
        g = np.asarray(grid, dtype=np.int64)
        return self.code_lut[g[..., 0], g[..., 1], g[..., 2]]

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return self.coords[np.asarray(codes, dtype=np.int64)]

    def arc(self, codes: np.ndarray) -> np.ndarray:
        return self.cumlen[np.asarray(codes, dtype=np.int64)]

    def code_from_arc(self, s: np.ndarray, *, return_clamped: bool = False):
        s = np.asarray(s, dtype=np.float64)
        clamped = (s < 0.0) | (s > self.s_max)
        sc = np.clip(s, 0.0, self.s_max)
        # nearest code along the curve: cumlen is strictly increasing
        hi = np.searchsorted(self.cumlen, sc, side="left")
        hi = np.clip(hi, 0, self.ncells - 1)
        lo = np.maximum(hi - 1, 0)
        pick_lo = (sc - self.cumlen[lo]) <= (self.cumlen[hi] - sc)
        codes = np.where(pick_lo, lo, hi).astype(np.int64)
        if return_clamped:
            return codes, clamped
        return codes


_CURVE_MEMO: dict[tuple[str, int], object] = {}


def get_curve3d(ordering: str, R: int):
    """Memoized curve factory. ordering in {"hilbert", "gilbert"}."""
    key = (str(ordering).strip().lower(), int(R))
    if key[0] not in ("hilbert", "gilbert"):
        raise ValueError(f"no 3D curve for ordering {ordering!r}")
    if key not in _CURVE_MEMO:
        _CURVE_MEMO[key] = (
            HilbertCurve3D(key[1]) if key[0] == "hilbert" else GilbertCurve3D(key[1])
        )
    return _CURVE_MEMO[key]
```

Note the `pick_lo` tie-break: at the exact midpoint between two codes it picks the lower one; the test allows either. The disk cache write is atomic (`os.replace`) because 8 DataLoader workers may build the same LUT concurrently.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/test_curves_interface.py tests/test_gilbert_curve.py -q`
Expected: all PASS.

- [ ] **Step 5: Check for circular imports**

`curves.py` imports from `lj_transferable.py`; Task 4 makes `lj_transferable.py` import `get_curve3d`. To avoid a cycle, that import in `lj_transferable.py` must be **lazy** (inside the function) — this is noted again in Task 4. Verify now that plain import works:
Run: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -c "from grid_transformer.data.curves import get_curve3d; print(get_curve3d('gilbert', 6).s_max)"`
Expected: prints a float ≈ 215.0 (215 unit steps for R=6 if fully continuous).

- [ ] **Step 6: Commit**

```bash
git add grid_transformer/data/curves.py tests/test_curves_interface.py
git commit -m "Add curve abstraction: arc-length LUTs unify pow-2 Hilbert and gilbert

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Curve-aware arc targets (`hilbert_arc_delta`)

**Files:**
- Modify: `grid_transformer/data/lj_transferable.py:333-370`
- Test: `tests/test_gilbert_integration.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gilbert_integration.py`:

```python
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

    # manual s-space decode, step by step
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/test_gilbert_integration.py -x -q`
Expected: FAIL with `TypeError: hilbert_arc_delta() got an unexpected keyword argument 'curve'`

- [ ] **Step 3: Generalize hilbert_arc_delta**

In `grid_transformer/data/lj_transferable.py`, replace the body of `hilbert_arc_delta` (lines 333-370). The Δs numerator becomes an arc-length difference and the cell centers come from `curve.decode`; everything else is unchanged:

```python
def hilbert_arc_delta(
    sorted_pos: np.ndarray,
    hilbert_codes: np.ndarray,
    box: np.ndarray,
    R: int,
    *,
    periodic: bool = True,
    curve=None,
) -> np.ndarray:
    """Compute (Δs, fine_x, fine_y, fine_z) AR targets from curve-sorted positions.

    Δs    = (s_{i+1} - s_i) / X  where s = cumulative arc length (cell units)
            and X = ncells // N.  For the pow-2 Hilbert curve s == code index,
            so this reduces to the historical (c_{i+1} - c_i) / X byte-exactly.
            For gilbert curves s accounts for diagonal steps (sqrt2/sqrt3).
    fine  = (r_{i+1} - cell_center_{i+1}) / cell_size  (dimensionless, ~[-0.5, 0.5]).

    Normalizing fine by cell_size makes it size-invariant across system sizes at the
    same density — the distribution is identical regardless of (N, L, R) when
    cell_size = L/R is held constant.  Decode: pos = cell_center + fine * cell_size.

    Returns: [N-1, 4] float32.
    """
    N = len(sorted_pos)
    if N < 2:
        return np.zeros((0, 4), dtype=np.float32)
    if curve is None:
        from .curves import get_curve3d  # lazy: curves.py imports this module

        curve = get_curve3d("hilbert", R)
    X = max(1, int(curve.ncells) // N)

    codes = hilbert_codes.astype(np.int64)
    cell_size = np.asarray(box, dtype=np.float64) / float(R)
    cell_centers = (curve.decode(codes).astype(np.float64) + 0.5) * cell_size

    fine = sorted_pos.astype(np.float64) - cell_centers
    if periodic:
        fine = fine - np.round(fine / cell_size) * cell_size

    s = curve.arc(codes)
    delta_s = ((s[1:] - s[:-1]) / float(X)).astype(np.float32)
    fine_dest = (fine[1:] / cell_size).astype(np.float32)  # normalized: ~[-0.5, 0.5]

    return np.concatenate([delta_s[:, None], fine_dest], axis=1)  # [N-1, 4]
```

- [ ] **Step 4: Run new tests AND the existing arc/likelihood suites**

Run: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/test_gilbert_integration.py tests/test_arc_repr_fixes.py tests/test_phase1_likelihood.py tests/test_octahedral_diagnostic.py -q`
Expected: all PASS (the byte-exact hilbert reduction is what protects the existing suites).

- [ ] **Step 5: Commit**

```bash
git add grid_transformer/data/lj_transferable.py tests/test_gilbert_integration.py
git commit -m "hilbert_arc_delta: arc-length form, byte-exact for pow-2 Hilbert

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Dataset integration (`ordering="gilbert"`)

**Files:**
- Modify: `grid_transformer/data/lj_transferable.py` (validation :948-957, `_resolution_for_box` :1095-1103, `_space_filling_codes_3d` :1111-1113, ordering branches :1276 and :1920, cache-build encode ~:1808-1816, rail guard ~:1004)
- Modify: `grid_transformer/training/lightning_module.py:398` (DataModule ordering validation)
- Test: extend `tests/test_gilbert_integration.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gilbert_integration.py`:

```python
def _make_dataset(tmp_path, ordering, cell_size, n=27, box=3.0, n_samples=4):
    """Tiny on-disk h5 + LJTransferableDataset in arc_repr mode."""
    import h5py

    from grid_transformer.data.lj_transferable import LJTransferableDataset

    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, box, size=(n_samples, n, 3)).astype(np.float64)
    h5 = tmp_path / f"toy_{ordering}.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("positions", data=pos)
        f.create_dataset("box", data=np.array([box, box, box]))
    # NOTE: match the h5 schema actually read by LJTransferableDataset — check
    # tests/test_arc_repr_fixes.py for the existing toy-h5 helper and reuse its
    # field names/kwargs verbatim (positions/box naming and constructor args
    # must mirror that helper, including periodic=True, arc_repr=True,
    # continuous targets).
    return LJTransferableDataset(
        str(h5),
        ordering=ordering,
        hilbert_resolution=64,
        cell_size=cell_size,
        periodic=True,
        arc_repr=True,
        use_continuous_head=True,
        continuous_input=True,
    )


def test_resolution_for_box_gilbert_is_nearest_even(tmp_path):
    # cell 0.046875: L=3 -> 64 (even, exact), L=4 -> 86, L=5 -> 106
    ds = _make_dataset(tmp_path, "gilbert", cell_size=0.046875)
    for L, R in {3.0: 64, 4.0: 86, 5.0: 106}.items():
        assert ds._resolution_for_box(np.array([L, L, L])) == R
    ds_h = _make_dataset(tmp_path, "hilbert", cell_size=0.046875)
    for L, R in {3.0: 64, 4.0: 128, 5.0: 128}.items():  # unchanged octave rule
        assert ds_h._resolution_for_box(np.array([L, L, L])) == R


def test_dataset_getitem_gilbert(tmp_path):
    ds = _make_dataset(tmp_path, "gilbert", cell_size=0.046875)
    item = ds[0]
    # adapt key access to the dataset's actual item schema (mirror the assertions
    # in tests/test_arc_repr_fixes.py): arc targets finite, |fine| <= 0.5 + 1e-6
    arc = item["continuous_targets"] if isinstance(item, dict) else item
    assert np.isfinite(np.asarray(arc, dtype=np.float64)).all()


def test_dataset_rejects_gilbert_with_curve_rail(tmp_path):
    with pytest.raises(NotImplementedError):
        _make_dataset(tmp_path, "gilbert", cell_size=0.046875, use_curve_rail=True)
```

**Implementer note:** the toy-h5 helper above is schematic where marked NOTE — `LJTransferableDataset`'s exact constructor kwargs and h5 schema must be copied from the existing helper in `tests/test_arc_repr_fixes.py` (it already builds tiny arc_repr datasets). Reuse that helper via import if possible instead of duplicating it. The three behaviors to pin with tests, whatever the helper looks like:

1. `_resolution_for_box`: with `ordering="gilbert"` and `cell_size=0.046875`, boxes 3.0/4.0/5.0 give R = 64/86/106 (nearest-even rule); with `ordering="hilbert"` unchanged (64/128/128).
2. `__getitem__` with `ordering="gilbert"` returns finite arc targets with `|fine| ≤ 0.5+1e-6` and the per-sample codes strictly increasing after the stable sort (no crash on non-pow-2 R).
3. `ordering="gilbert"` + `use_curve_rail=True` raises `NotImplementedError` at construction; `hilbert_resolution` pow-2 validation does NOT fire for gilbert.

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/test_gilbert_integration.py -x -q`
Expected: FAIL with `ValueError: ordering must be ...` (gilbert rejected at :948).

- [ ] **Step 3: Implement the dataset changes**

All in `grid_transformer/data/lj_transferable.py`:

(a) Validation (:948-957) — accept gilbert, scope the pow-2 check to hilbert only, forbid rail+gilbert:

```python
        if self.ordering not in ("hilbert", "gilbert", "spectral"):
            raise ValueError(
                f"ordering must be 'hilbert', 'gilbert' or 'spectral', got {self.ordering!r}"
            )
        if self.ordering == "hilbert" and not _is_power_of_two(self.hilbert_resolution):
            raise ValueError(
                f"hilbert_resolution must be a power of two, got {self.hilbert_resolution}"
            )
```

and, after `self.use_curve_rail` is set (~:1004):

```python
        if self.use_curve_rail and self.ordering == "gilbert":
            raise NotImplementedError(
                "curve rail + gilbert ordering is not wired (rail waypoint decode and "
                "the sampler's rail mirror assume the pow-2 Hilbert curve); train rails "
                "with ordering='hilbert' or extend _curve_waypoints_3d via get_curve3d."
            )
```

(b) `_resolution_for_box` (:1095-1103) — nearest even for gilbert (diagonal steps come from odd splits; even R keeps the curve fully continuous):

```python
    def _resolution_for_box(self, box: Optional[np.ndarray]) -> int:
        """Grid resolution (cells per axis) for a box. Constant cell size when
        self.cell_size is set: hilbert -> next_pow2(round(max(L)/cell_size))
        (constant only up to an octave); gilbert -> nearest EVEN integer
        (constant to <1%, even sizes avoid diagonal steps). Else the fixed
        global resolution (constant cell count)."""
        if self.cell_size is None or box is None:
            return int(self.hilbert_resolution)
        L = float(np.max(np.asarray(box, dtype=np.float64)))
        n = max(2, int(round(L / self.cell_size)))
        if self.ordering == "gilbert":
            return max(2, int(round(n / 2.0)) * 2)
        return int(1 << int(math.ceil(math.log2(n))))  # next power of two
```

(c) `_space_filling_codes_3d` (:1111-1113) — route through the factory:

```python
    def _space_filling_codes_3d(self, grid: np.ndarray, resolution: int) -> np.ndarray:
        from .curves import get_curve3d  # lazy: curves.py imports this module

        kind = "gilbert" if self.ordering == "gilbert" else "hilbert"
        return get_curve3d(kind, int(resolution)).encode(grid)
```

(d) Ordering branches: at :1276 change `if self.ordering == "hilbert":` to `if self.ordering in ("hilbert", "gilbert"):`, and at :1920 change `dataset.ordering == "hilbert"` to `dataset.ordering in ("hilbert", "gilbert")`. Then run `grep -n '== "hilbert"' grid_transformer/data/lj_transferable.py` and audit every remaining hit the same way (include gilbert wherever the branch means "space-filling-curve ordering", keep hilbert-only where it touches rails or `_hilbert_bits`).

(e) Cache-build path (~:1808-1816) — it calls `_hilbert_bits(R)` + `_hilbert3d_encode` directly (crashes on non-pow-2 R) and `hilbert_arc_delta` without a curve. Replace with the curve object:

```python
            from .curves import get_curve3d

            curve = get_curve3d(
                "gilbert" if dataset.ordering == "gilbert" else "hilbert", R
            )
            codes = curve.encode(grid)
            ...
                hilbert_arc_delta(
                    abs_coords_np, codes, box_np, R, periodic=self.periodic, curve=curve
                )
```
(adapt to the exact local variable names at that site; the key change is: no `_hilbert_bits`, encode via curve, pass `curve=` through).

(f) `grid_transformer/training/lightning_module.py:398` — accept gilbert in the DataModule:

```python
        if self.ordering not in ("hilbert", "gilbert", "spectral"):
            raise ValueError(
                f"ordering must be 'hilbert', 'gilbert' or 'spectral', got {ordering!r}"
            )
```

- [ ] **Step 4: Run the new tests and the full suite**

Run: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/ -q`
Expected: all PASS (suite was 103 green before this plan; now more).

- [ ] **Step 5: Commit**

```bash
git add grid_transformer/data/lj_transferable.py grid_transformer/training/lightning_module.py tests/test_gilbert_integration.py
git commit -m "Dataset: ordering=gilbert with nearest-even constant-cell resolution

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Sampler integration (s-space decode, ordering plumbing)

**Files:**
- Modify: `sample_lj.py` (`_arc_decode_positions` :409-470, `_rail_resolution_for_box` :361-377, p0/encode sites :402 and :538, sampler entry ~:591-700, CLI ~:1280-1320)
- Test: extend `tests/test_gilbert_integration.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gilbert_integration.py`:

```python
import torch


def test_arc_decode_positions_gilbert_roundtrip():
    """Sampler decode with a gilbert curve reconstructs encoder targets exactly."""
    from grid_transformer.data.curves import get_curve3d
    from grid_transformer.data.lj_transferable import hilbert_arc_delta
    from sample_lj import _arc_decode_positions

    R, box, n = 6, 3.0, 27
    curve = get_curve3d("gilbert", R)
    pos, codes = _random_sorted_config(curve, box, n, seed=7)
    box_arr = np.array([box] * 3)
    arc = hilbert_arc_delta(pos, codes, box_arr, R, periodic=True, curve=curve)

    box_t = torch.tensor([[box, box, box]], dtype=torch.float32)
    curr = torch.tensor(pos[:1], dtype=torch.float32)
    for i in range(n - 1):
        delta = torch.tensor(arc[i : i + 1], dtype=torch.float32)
        nxt, diag = _arc_decode_positions(
            curr, delta, box_t, R, n, curve=curve, return_diagnostics=True
        )
        target = torch.tensor(pos[i + 1], dtype=torch.float32)
        d = (nxt[0] - target).double().numpy()
        d -= np.round(d / box) * box
        np.testing.assert_allclose(d, 0.0, atol=1e-4)
        assert int(diag["c_next"][0]) == int(codes[i + 1])
        assert not bool(diag["clamp_hit"][0])
        curr = nxt


def test_arc_decode_positions_hilbert_default_unchanged():
    """curve=None keeps the historical pow-2 behavior (existing tests also cover
    this; this is the explicit equivalence check)."""
    from grid_transformer.data.curves import get_curve3d
    from sample_lj import _arc_decode_positions

    R, box, n = 8, 3.0, 27
    rng = np.random.default_rng(11)
    curr = torch.tensor(rng.uniform(0, box, (5, 3)), dtype=torch.float32)
    delta = torch.tensor(rng.normal(0, 2, (5, 4)), dtype=torch.float32)
    box_t = torch.tensor([[box] * 3], dtype=torch.float32)
    a = _arc_decode_positions(curr, delta, box_t, R, n)
    b = _arc_decode_positions(curr, delta, box_t, R, n, curve=get_curve3d("hilbert", R))
    torch.testing.assert_close(a, b)


def test_rail_resolution_for_box_gilbert_rule():
    from sample_lj import _rail_resolution_for_box

    box = np.array([4.0, 4.0, 4.0])
    assert _rail_resolution_for_box(box, 64, 0.046875) == 128  # hilbert default
    assert _rail_resolution_for_box(box, 64, 0.046875, ordering="gilbert") == 86
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/test_gilbert_integration.py -x -q -k "decode_positions_gilbert or rail_resolution"`
Expected: FAIL with `TypeError: _arc_decode_positions() got an unexpected keyword argument 'curve'`

- [ ] **Step 3: Implement sampler changes**

In `sample_lj.py`:

(a) `_arc_decode_positions` (:409-470) — add `curve=None` kwarg; replace code arithmetic with s-space (existing comments about fine min-imaging stay):

```python
def _arc_decode_positions(
    curr_pos: torch.Tensor,
    delta_arc: torch.Tensor,
    box_size: torch.Tensor,
    R: int,
    n_particles: int,
    *,
    curve=None,
    return_diagnostics: bool = False,
):
    """Decode arc-length prediction back to xyz position.

    s_next = clamp(s(c_curr) + Δs * X, 0, s_max);  c_next = nearest code at s_next
    pos_next = cell_center(c_next) + fine_xyz

    For the pow-2 Hilbert curve s(c) == c, so this is the historical
    c_next = clamp(round(c_curr + Δs·X), 0, R³-1) byte-exactly.
    """
    if curve is None:
        from grid_transformer.data.curves import get_curve3d

        curve = get_curve3d("hilbert", R)
    X = max(1, int(curve.ncells) // n_particles)
    device = curr_pos.device

    box_np = box_size[0].detach().cpu().numpy().astype(np.float64)
    curr_np = curr_pos.detach().cpu().numpy().astype(np.float64)
    curr_wrapped = np.mod(curr_np, box_np[None, :])
    cell_size = box_np / float(R)
    grid = np.clip(np.floor(curr_wrapped / cell_size).astype(np.int64), 0, R - 1)
    c_curr = curve.encode(grid)  # [B]

    delta_s_np = delta_arc[:, 0].detach().cpu().numpy().astype(np.float64)
    s_next = curve.arc(c_curr) + delta_s_np * float(X)
    c_next, clamp_hit = curve.code_from_arc(s_next, return_clamped=True)

    cell_centers = (curve.decode(c_next).astype(np.float64) + 0.5) * cell_size
    cell_centers_t = torch.from_numpy(cell_centers.astype(np.float32)).to(device=device)

    cell_size_t = torch.from_numpy(cell_size.astype(np.float32)).to(device=device)
    fine_raw = delta_arc[:, 1:4]
    wrap_amount = torch.round(fine_raw)
    fine = fine_raw - wrap_amount
    pos = cell_centers_t + fine * cell_size_t
    if not return_diagnostics:
        return pos
    diag = {
        "c_curr": torch.from_numpy(c_curr.astype(np.int64)).to(device=device),
        "c_next": torch.from_numpy(c_next).to(device=device),
        "clamp_hit": torch.from_numpy(clamp_hit).to(device=device),
        "fine_wrap": (wrap_amount != 0).any(dim=-1),
    }
    return pos, diag
```

**Byte-compat check for hilbert:** old `clamp_hit` was `round(c+Δs·X) != clip(...)`; new is `s_next outside [0, s_max]`. For hilbert these differ only when `round(s_next)` clamps but `s_next` itself is within (−0.5, R³−0.5) of the ends — `HilbertCurve3D.code_from_arc` reproduces the OLD semantics exactly (it rounds first, then compares), so use its returned mask and the behavior is unchanged. Verify with `tests/test_phase1_likelihood.py`.

(b) `_rail_resolution_for_box` (:361-377) — add `ordering: str = "hilbert"` kwarg; gilbert branch mirrors the dataset's nearest-even rule:

```python
    n = max(2, int(round(L / float(cell_size))))
    if str(ordering).lower() == "gilbert":
        return max(2, int(round(n / 2.0)) * 2)
    return int(1 << int(math.ceil(math.log2(n))))
```

(c) Encode sites: `empirical_p0_positions` (:538 region) and the arc-code bookkeeping in the sampling loop (the `_arc_R0` / `c_curr` sites around :697 and :907) — thread a `curve` object built once per run from `(ordering, R)` and call `curve.encode` instead of `_hilbert3d_encode(..., bits=...)`. The rail waypoint helper (:396-406) stays hilbert-only: at its top add

```python
    if str(ordering).lower() == "gilbert":
        raise NotImplementedError("curve rail sampling is pow-2 Hilbert only")
```
(thread `ordering` to it from the caller; rails are off for arc models so this is a guard, not a feature).

(d) Sampler entry (~:591) gains `ordering: str = "hilbert"`; build `curve = get_curve3d(ordering, R)` wherever R is resolved (including per-box R via `_rail_resolution_for_box(..., ordering=ordering)`), pass `curve=curve` to every `_arc_decode_positions` call, and use `curve.encode` at the loop encode sites.

(e) CLI (~:1280): add

```python
    parser.add_argument(
        "--ordering",
        choices=("hilbert", "gilbert"),
        default=None,
        help="Space-filling curve family. Default: read from the checkpoint "
        "(model.ordering); falls back to 'hilbert'. Gilbert checkpoints cannot "
        "be sampled with hilbert ordering or vice versa.",
    )
```

and resolve `effective_ordering = args.ordering or getattr(model, "ordering", None) or "hilbert"` next to `_effective_rail_geometry` (:1305). If the checkpoint stores an ordering and the CLI passes a different one, `raise SystemExit` with both values — a silent curve mismatch generates garbage that still "looks like" samples.

- [ ] **Step 4: Run new tests + full suite**

Run: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/ -q`
Expected: all PASS, including the pre-existing `tests/test_phase1_likelihood.py` (hilbert byte-compat) and `tests/test_kv_cache.py`.

- [ ] **Step 5: Commit**

```bash
git add sample_lj.py tests/test_gilbert_integration.py
git commit -m "Sampler: s-space arc decode + --ordering gilbert plumbing

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: train.py + checkpoint ordering + shell script

**Files:**
- Modify: `train.py` (ordering choices :106-110, arc_repr validation ~:369, model hparam threading ~:506)
- Modify: `grid_transformer/training/lightning_module.py` (store `ordering` on the model the same way `cell_size` is stored)
- Modify: `train_sample_lj27_pbc.sh` (pass ORDERING through to sampling too, if not already)
- Test: extend `tests/test_gilbert_integration.py`

- [ ] **Step 1: Locate the cell_size threading to mirror**

Run: `grep -n "cell_size" train.py grid_transformer/training/lightning_module.py | grep -v "^.*#"`
The recent commit "cell_size threading for multi-size caches" (fba70bb) added `cell_size` to the **model's** hparams so `sample_lj._effective_rail_geometry` can read it via `getattr(model, ...)`. `ordering` must follow the identical path: wherever the model LightningModule receives/stores `cell_size`, add `ordering: str = "hilbert"` beside it (constructor arg, `self.ordering = str(ordering)`, included in `save_hyperparameters`), and pass `args.lj_transfer_ordering` at the train.py construction site.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_gilbert_integration.py` (mirror the existing checkpoint-attr test pattern in `tests/test_sample_cli_fixes.py` for constructing a minimal module):

```python
def test_model_checkpoint_stores_ordering(tmp_path):
    """Round-trip: model saved with ordering='gilbert' exposes it after load,
    so sample_lj can resolve the curve family without a CLI flag."""
    # Build the minimal LightningModule exactly as tests/test_sample_cli_fixes.py
    # does (same constructor kwargs), adding ordering="gilbert"; save a checkpoint
    # with trainer.save_checkpoint; reload with the class's load_from_checkpoint;
    # assert getattr(loaded, "ordering") == "gilbert".
```

(The test body must be concretized from `tests/test_sample_cli_fixes.py`'s existing helper — same module class, same minimal kwargs; the new assertion is the `ordering` attribute surviving the save/load round trip.)

- [ ] **Step 3: Run test to verify it fails**

Run: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/test_gilbert_integration.py -x -q -k checkpoint_stores_ordering`
Expected: FAIL (`TypeError` unknown kwarg `ordering`, or attribute missing).

- [ ] **Step 4: Implement**

(a) `train.py:106-110` — add gilbert to the choices of `--lj_transfer_ordering`:

```python
    parser.add_argument(
        "--lj_transfer_ordering",
        choices=("hilbert", "gilbert", "spectral"),
        ...
    )
```
(keep existing default and help text, append: `"'gilbert' = generalized Hilbert, allows non-pow-2 R for constant cell size across boxes."`)

(b) `train.py` ~:369 — the `--arc_repr` help/validation says "Requires ... --ordering hilbert"; update the validation to accept `("hilbert", "gilbert")` and the help text to match.

(c) Thread `ordering` into the model module per Step 1's findings; pass it from train.py beside `cell_size`.

(d) `train_sample_lj27_pbc.sh` — ORDERING env already maps to `--lj_transfer_ordering` for training; check the sampling invocation in the same script and add `--ordering "${ORDERING}"` if the sampler is called there (RUN_SAMPLE branch).

- [ ] **Step 5: Run the full suite**

Run: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add train.py grid_transformer/training/lightning_module.py train_sample_lj27_pbc.sh tests/test_gilbert_integration.py
git commit -m "Thread ordering through train CLI, model hparams, pipeline script

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Validation on real data — Δs audit + diagnostics routing

**Files:**
- Modify: `octahedral_nll_diagnostic.py` (route encode + `hilbert_arc_delta` through `get_curve3d`, add `--ordering`)
- Create: `reports/gilbert_delta_s_audit.py` (one-shot audit script)
- Modify: `reports/action-plan-variants-equivariance-likelihood.md` (record results)

This task is the GO/NO-GO gate before any gilbert training run: redo the Step-3 Δs audit (the one that returned KS ≤ 0.044 for hilbert, see `reports/arc_repr_delta_s/FINDINGS.md`) with gilbert at constant cell.

- [ ] **Step 1: Route octahedral_nll_diagnostic through the curve factory**

In `octahedral_nll_diagnostic.py`, replace the direct `_hilbert3d_encode(..., bits=bits)` (:91) with `get_curve3d(ordering, R).encode(grid)` and pass `curve=` to `hilbert_arc_delta` (:94); add `--ordering {hilbert,gilbert}` defaulting to hilbert. Existing tests (`tests/test_octahedral_diagnostic.py`) must stay green.

- [ ] **Step 2: Write the audit script**

Create `reports/gilbert_delta_s_audit.py`:

```python
"""GO/NO-GO audit: is the (Δs, fine) target size-invariant under gilbert at
constant cell, and how does it compare to hilbert-with-octave-rounding?

For each (L, N): load up to MAX_CONFIGS configs from the existing MCMC h5
(same files used by reports/arc_repr_delta_s), build codes + arc targets with
(a) hilbert at R = next_pow2 and (b) gilbert at R = nearest_even, then report
per-pair KS statistics of Δs and fine marginals vs the L3 reference, plus the
gilbert diagonal-step fraction actually hit by consecutive particles.
"""
import numpy as np
from scipy.stats import ks_2samp

from grid_transformer.data.curves import get_curve3d
from grid_transformer.data.lj_transferable import hilbert_arc_delta

CELL = 3.0 / 64.0
MAX_CONFIGS = 2000
# (h5 path, L, N) — reuse the exact file list from the previous audit run
# (see reports/arc_repr_delta_s/FINDINGS.md header for the source paths):
SOURCES = [
    ("/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L3_rho1.0_N27_T1.0.h5", 3.0, 27),
    ("/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L4_rho1.0_N64_T1.0.h5", 4.0, 64),
    ("/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L5_rho1.0_N125_T1.0.h5", 5.0, 125),
]


def targets_for(path, L, N, ordering):
    import h5py

    n = L / CELL
    if ordering == "gilbert":
        R = max(2, int(round(n / 2.0)) * 2)
    else:
        R = 1 << int(np.ceil(np.log2(max(2, round(n)))))
    curve = get_curve3d(ordering, R)
    cell = L / R
    box = np.array([L, L, L])
    out = []
    with h5py.File(path, "r") as f:
        # adjust the dataset key to the actual h5 layout (same as previous audit)
        pos_all = f["positions"][:MAX_CONFIGS]
    for pos in pos_all:
        w = np.mod(pos, L)
        grid = np.clip((w / cell).astype(np.int64), 0, R - 1)
        codes = curve.encode(grid)
        order = np.argsort(codes, kind="stable")
        arc = hilbert_arc_delta(w[order], codes[order], box, R, periodic=True, curve=curve)
        out.append(arc)
    return np.concatenate(out, axis=0), R, cell


def main():
    ref = {}
    for ordering in ("hilbert", "gilbert"):
        print(f"=== ordering={ordering} ===")
        for path, L, N in SOURCES:
            arc, R, cell = targets_for(path, L, N, ordering)
            key = (ordering,)
            if key not in ref:
                ref[key] = arc  # L3 is the reference
            ks_ds = ks_2samp(ref[key][:, 0], arc[:, 0]).statistic
            ks_fx = ks_2samp(ref[key][:, 1], arc[:, 1]).statistic
            print(
                f"L={L} N={N}: R={R} cell={cell:.5f} ({abs(cell-CELL)/CELL:.2%} off) "
                f"KS(Δs)={ks_ds:.4f} KS(fine_x)={ks_fx:.4f} "
                f"|Δs|>4 frac={float((np.abs(arc[:,0])>4).mean()):.4%}"
            )


if __name__ == "__main__":
    main()
```

(Adjust the h5 key/paths to match what `reports/arc_repr_delta_s` actually used — its FINDINGS.md records the sources.)

- [ ] **Step 3: Run the audit**

Run: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python reports/gilbert_delta_s_audit.py 2>&1 | tee reports/gilbert_delta_s_audit.out`
Expected: gilbert rows show cell within 1% of 0.046875 for all three L (vs 33%/17% off for hilbert) and KS(Δs), KS(fine) ≤ the hilbert baseline (≈0.044). **Gate:** if gilbert KS is materially worse than hilbert's, stop and investigate before any training — the curve's locality might differ more than expected.

- [ ] **Step 4: Run the full test suite one final time**

Run: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Record results + commit**

Append a "Gilbert curve (constant cell)" section to `reports/action-plan-variants-equivariance-likelihood.md` with: the audit table, the measured diagonal-step fractions from Task 1 Step 6, and the decision rule for the next training action (multi-size {L3,L5}→L4 rerun with `ORDERING=gilbert`, comparing held-out L4 OTgap at matched cell vs the current octave-rounded baseline — only after the in-flight run finishes and only on user go-ahead).

```bash
git add octahedral_nll_diagnostic.py reports/gilbert_delta_s_audit.py reports/gilbert_delta_s_audit.out reports/action-plan-variants-equivariance-likelihood.md
git commit -m "Gilbert Δs audit: constant-cell targets across L3/L4/L5

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## What this plan does NOT do (deliberate)

- **No training runs.** The GPU is occupied by the multi-size baseline (PID 523122). The first gilbert training (multi-size rerun at matched cell) is a separate decision after the audit gate and after that run completes.
- **No cache rebuilds** of the 1M-config training caches — the audit reads raw MCMC h5 directly.
- **Rail + gilbert** is guarded with `NotImplementedError`, not implemented (arc baseline has no rail).
- **Hilbert behavior is provably unchanged**: every generalization reduces to the old formula when `arc(c) == c`, and the pre-existing suites (`test_arc_repr_fixes.py`, `test_phase1_likelihood.py`, `test_kv_cache.py`, `test_octahedral_diagnostic.py`) are run at every task as the regression gate.
