"""Space-filling-curve abstraction for the arc representation.

Two curve families share one interface:

- ``HilbertCurve3D`` — classical pow-2 Hilbert (Skilling transpose, wraps the
  existing functions in lj_transferable.py). Every step is face-adjacent, so
  the cumulative arc length equals the code index: ``arc(c) == c``.
- ``GilbertCurve3D`` — generalized Hilbert on an arbitrary cubic grid
  (gilbert.py). Odd grids contain occasional multi-cell steps (Euclidean
  lengths up to 3), so code index and arc length differ; ``cumlen`` is the
  precomputed cumulative physical length in cell units. Even cubic grids are
  fully face-continuous, so there ``cumlen == code index`` as well — the
  nearest-even resolution rule keeps production in that regime.

All downstream arc math must use ``arc``/``code_from_arc`` (s-space), never
raw code arithmetic — that is what makes the two families interchangeable.
"""
from __future__ import annotations

import os
import zipfile
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
        # _hilbert3d_decode returns uint64; cast so both curve classes return
        # int64 and cell deltas can't silently underflow.
        return np.stack([x, y, z], axis=-1).astype(np.int64)

    def arc(self, codes: np.ndarray) -> np.ndarray:
        return np.asarray(codes, dtype=np.float64)

    def code_from_arc(self, s: np.ndarray, *, return_clamped: bool = False):
        """Nearest code = clip(round(s), 0, R³-1); ``clamped`` is True where
        round(s) itself left [0, R³) — round-first semantics, byte-identical to
        the historical sampler. NOTE: differs from GilbertCurve3D, whose
        ``clamped`` flags s outside [0, s_max] before rounding."""
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
                str(Path.home() / ".cache" / "grid_transformer" / "gilbert"),
            )
        )
        cache_file = cache_dir / f"gilbert3d_R{R}.npz"
        if cache_file.exists():
            try:
                with np.load(cache_file) as z:
                    return (
                        z["coords"].astype(np.int64),
                        z["code_lut"].astype(np.int64),
                        z["cumlen"].astype(np.float64),
                    )
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                # truncated/corrupt cache (full disk, interrupted copy, format
                # drift): fall through and rebuild rather than brick every run
                pass
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
        # Write to a tmp file via an open file handle so numpy doesn't append
        # an extra ".npz" extension, then atomically rename to the target path.
        tmp = cache_file.with_name(cache_file.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, coords=coords, code_lut=code_lut, cumlen=cumlen)
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
        """Nearest code along the curve via searchsorted on the strictly
        monotone ``cumlen``; ``clamped`` is True where s lay outside
        [0, s_max]. NOTE: differs from HilbertCurve3D, whose ``clamped``
        uses historical round-first semantics."""
        s = np.asarray(s, dtype=np.float64)
        clamped = (s < 0.0) | (s > self.s_max)
        sc = np.clip(s, 0.0, self.s_max)
        # nearest code along the curve: cumlen is strictly increasing.
        # searchsorted(..., side="left") on an exact match returns the matching
        # index, so the lo/hi nearest-pick correctly returns that index (since
        # sc - cumlen[lo] == 0 <= cumlen[hi] - sc, pick_lo is True).
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
    """Memoized curve factory. ordering in {"hilbert", "gilbert"}.

    The memo is per-process and not thread-locked; DataLoader fork-workers
    each build/load their own copy (the disk LUT cache makes that cheap).
    """
    key = (str(ordering).strip().lower(), int(R))
    if key[0] not in ("hilbert", "gilbert"):
        raise ValueError(f"no 3D curve for ordering {ordering!r}")
    if key not in _CURVE_MEMO:
        _CURVE_MEMO[key] = (
            HilbertCurve3D(key[1]) if key[0] == "hilbert" else GilbertCurve3D(key[1])
        )
    return _CURVE_MEMO[key]
