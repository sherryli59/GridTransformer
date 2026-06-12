"""Generalized Hilbert ("gilbert") space-filling curve for arbitrary 3D grids.

Port of https://github.com/jakubcerveny/gilbert (gilbert3d.py, BSD 2-Clause,
Copyright (c) 2018 Jakub Červený). Unlike the classical Hilbert curve
(grid_transformer.data.lj_transferable._hilbert3d_encode), the grid sides need
not be powers of two, which lets the physical cell size be held ~constant across
box lengths. The price: on odd-size grids a small fraction of consecutive steps
span more than one cell — measured Euclidean step lengths {1, sqrt2, 2, 2*sqrt2,
3} (e.g. a (3,0,0) jump on 5^3); even cubic grids are fully face-continuous
(all steps length 1, verified up to R=106). Consumers must therefore use the
cumulative arc length, not the code index, as the along-curve coordinate
(wrapped with a disk-cached LUT by the curve abstraction built on top of this).
"""
from __future__ import annotations

import numpy as np


def _sgn(x: int) -> int:
    return -1 if x < 0 else (1 if x > 0 else 0)


def _generate3d(x, y, z, ax, ay, az, bx, by, bz, cx, cy, cz):
    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)

    dax, day, daz = _sgn(ax), _sgn(ay), _sgn(az)  # unit major direction ("right")
    dbx, dby, dbz = _sgn(bx), _sgn(by), _sgn(bz)  # unit ortho direction ("forward")
    dcx, dcy, dcz = _sgn(cx), _sgn(cy), _sgn(cz)  # unit ortho direction ("up")

    # trivial row/column fills
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

    # prefer even steps
    if (w2 % 2) and (w > 2):
        ax2, ay2, az2 = ax2 + dax, ay2 + day, az2 + daz

    if (h2 % 2) and (h > 2):
        bx2, by2, bz2 = bx2 + dbx, by2 + dby, bz2 + dbz

    if (d2 % 2) and (d > 2):
        cx2, cy2, cz2 = cx2 + dcx, cy2 + dcy, cz2 + dcz

    # wide case, split in w only
    if (2 * w > 3 * h) and (2 * w > 3 * d):
        yield from _generate3d(x, y, z,
                               ax2, ay2, az2,
                               bx, by, bz,
                               cx, cy, cz)
        yield from _generate3d(x + ax2, y + ay2, z + az2,
                               ax - ax2, ay - ay2, az - az2,
                               bx, by, bz,
                               cx, cy, cz)

    # do not split in d
    elif 3 * h > 4 * d:
        yield from _generate3d(x, y, z,
                               bx2, by2, bz2,
                               cx, cy, cz,
                               ax2, ay2, az2)
        yield from _generate3d(x + bx2, y + by2, z + bz2,
                               ax, ay, az,
                               bx - bx2, by - by2, bz - bz2,
                               cx, cy, cz)
        yield from _generate3d(x + (ax - dax) + (bx2 - dbx),
                               y + (ay - day) + (by2 - dby),
                               z + (az - daz) + (bz2 - dbz),
                               -bx2, -by2, -bz2,
                               cx, cy, cz,
                               -(ax - ax2), -(ay - ay2), -(az - az2))

    # do not split in h
    elif 3 * d > 4 * h:
        yield from _generate3d(x, y, z,
                               cx2, cy2, cz2,
                               ax2, ay2, az2,
                               bx, by, bz)
        yield from _generate3d(x + cx2, y + cy2, z + cz2,
                               ax, ay, az,
                               bx, by, bz,
                               cx - cx2, cy - cy2, cz - cz2)
        yield from _generate3d(x + (ax - dax) + (cx2 - dcx),
                               y + (ay - day) + (cy2 - dcy),
                               z + (az - daz) + (cz2 - dcz),
                               -cx2, -cy2, -cz2,
                               -(ax - ax2), -(ay - ay2), -(az - az2),
                               bx, by, bz)

    # regular case, split in all w/h/d
    else:
        yield from _generate3d(x, y, z,
                               bx2, by2, bz2,
                               cx2, cy2, cz2,
                               ax2, ay2, az2)
        yield from _generate3d(x + bx2, y + by2, z + bz2,
                               cx, cy, cz,
                               ax2, ay2, az2,
                               bx - bx2, by - by2, bz - bz2)
        yield from _generate3d(x + (bx2 - dbx) + (cx - dcx),
                               y + (by2 - dby) + (cy - dcy),
                               z + (bz2 - dbz) + (cz - dcz),
                               ax, ay, az,
                               -bx2, -by2, -bz2,
                               -(cx - cx2), -(cy - cy2), -(cz - cz2))
        yield from _generate3d(x + (ax - dax) + bx2 + (cx - dcx),
                               y + (ay - day) + by2 + (cy - dcy),
                               z + (az - daz) + bz2 + (cz - dcz),
                               -cx, -cy, -cz,
                               -(ax - ax2), -(ay - ay2), -(az - az2),
                               bx - bx2, by - by2, bz - bz2)
        yield from _generate3d(x + (ax - dax) + (bx2 - dbx),
                               y + (ay - day) + (by2 - dby),
                               z + (az - daz) + (bz2 - dbz),
                               -bx2, -by2, -bz2,
                               cx2, cy2, cz2,
                               -(ax - ax2), -(ay - ay2), -(az - az2))


def gilbert3d_path(nx: int, ny: int, nz: int) -> np.ndarray:
    """Traversal order of the generalized Hilbert curve on an nx*ny*nz grid.

    Returns an array of shape [nx*ny*nz, 3], dtype int64: row i is the grid
    cell (ix, iy, iz) visited at curve position i. The curve visits every cell
    exactly once. On even cubic grids all steps are face-adjacent (L1=1,
    verified up to R=106); on odd grids a small fraction of steps (measured
    1-8%, highest on small grids) span more than one cell, with Euclidean
    lengths up to 3 (e.g. one (3,0,0) jump on 5^3).

    Pure Python recursion; run once per resolution and cache the result (the
    curve abstraction built on top of this wraps it with a disk-cached LUT).

    Args:
        nx: number of grid cells along x (>= 1).
        ny: number of grid cells along y (>= 1).
        nz: number of grid cells along z (>= 1).

    Returns:
        int64 array of shape (nx*ny*nz, 3).
    """
    nx, ny, nz = int(nx), int(ny), int(nz)
    if min(nx, ny, nz) < 1:
        raise ValueError(f"grid sides must be >= 1, got {(nx, ny, nz)}")

    # Choose the longest axis as the primary ("width") direction, matching
    # the upstream gilbert3d entry point exactly.
    if nx >= ny and nx >= nz:
        gen = _generate3d(0, 0, 0,
                          nx, 0, 0,
                          0, ny, 0,
                          0, 0, nz)
    elif ny >= nx and ny >= nz:
        gen = _generate3d(0, 0, 0,
                          0, ny, 0,
                          nx, 0, 0,
                          0, 0, nz)
    else:  # nz is largest
        gen = _generate3d(0, 0, 0,
                          0, 0, nz,
                          nx, 0, 0,
                          0, ny, 0)

    # np.fromiter with object dtype accepts tuple yields; structured dtype
    # requires numpy >= 1.23. Fall back to list if needed.
    try:
        path = np.fromiter(gen, dtype=np.dtype((np.int64, 3)), count=nx * ny * nz)
    except TypeError:
        path = np.array(list(gen), dtype=np.int64)
    return path
