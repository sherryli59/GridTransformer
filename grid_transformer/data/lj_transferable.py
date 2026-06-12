from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..physics.energy import DoubleWellPotential, LJ
from ..utils.spatial import spectral_argsort


def _is_power_of_two(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


def resolution_for_box_rule(
    box: Optional[np.ndarray],
    *,
    cell_size: Optional[float],
    hilbert_resolution: int,
    ordering: str = "hilbert",
) -> int:
    """Grid resolution (cells per axis) for a box. Constant cell size when
    cell_size is set: hilbert -> next_pow2(round(max(L)/cell_size)) (constant
    only up to an octave); gilbert -> nearest EVEN integer of L/cell_size
    (constant to <1%; even grids keep the gilbert curve fully face-continuous).
    Else the fixed global resolution (constant cell count)."""
    if cell_size is None or box is None:
        return int(hilbert_resolution)
    L = float(np.max(np.asarray(box, dtype=np.float64)))
    if str(ordering).strip().lower() == "gilbert":
        # Round L/cell directly to the nearest even integer; dividing by 2
        # before any integer rounding avoids banker's-rounding ties on the
        # pre-rounded integer (85 -> 84, 107 -> 108 would be the FARTHER even).
        return max(2, int(round(L / float(cell_size) / 2.0)) * 2)
    n = max(2, int(round(L / float(cell_size))))
    return int(1 << int(math.ceil(math.log2(n))))


def _hilbert_rot(side: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = side - 1 - x
            y = side - 1 - y
        x, y = y, x
    return x, y


def _hilbert_xy2d(side: int, x: int, y: int) -> int:
    d = 0
    s = side // 2
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _hilbert_rot(s, x, y, rx, ry)
        s //= 2
    return d


def _hilbert3d_encode(x: np.ndarray, y: np.ndarray, z: np.ndarray, bits: int) -> np.ndarray:
    """
    Vectorized 3D Hilbert curve encoding based on John Skilling's 2004 algorithm.
    Transforms (x, y, z) coordinates into a 1D Hilbert integer.
    """
    if bits <= 0:
        return np.zeros_like(x, dtype=np.int64)
        
    X0 = x.astype(np.uint64, copy=True)
    X1 = y.astype(np.uint64, copy=True)
    X2 = z.astype(np.uint64, copy=True)

    M = 1 << (bits - 1)
    Q = M
    
    # 1. Inverse Undo (Axes to Transpose)
    while Q > 1:
        P = Q - 1
        
        cond0 = (X0 & Q) != 0
        X0 = np.where(cond0, X0 ^ P, X0)
        
        cond1 = (X1 & Q) != 0
        t1 = (X0 ^ X1) & P
        X0 = np.where(cond1, X0 ^ P, X0 ^ t1)
        X1 = np.where(cond1, X1, X1 ^ t1)
        
        cond2 = (X2 & Q) != 0
        t2 = (X0 ^ X2) & P
        X0 = np.where(cond2, X0 ^ P, X0 ^ t2)
        X2 = np.where(cond2, X2, X2 ^ t2)
        
        Q >>= 1

    # 2. Gray Encode
    X1 ^= X0
    X2 ^= X1

    t = np.zeros_like(X0)
    Q = M
    while Q > 1:
        cond = (X2 & Q) != 0
        t = np.where(cond, t ^ (Q - 1), t)
        Q >>= 1

    X0 ^= t
    X1 ^= t
    X2 ^= t

    # 3. Interleave Transposed Bits
    code = np.zeros_like(X0, dtype=np.int64)
    for b in range(bits):
        xb = (X0 >> b) & 1
        yb = (X1 >> b) & 1
        zb = (X2 >> b) & 1
        code |= (xb.astype(np.int64) << (3 * b + 2))
        code |= (yb.astype(np.int64) << (3 * b + 1))
        code |= (zb.astype(np.int64) << (3 * b))
        
    return code


def _hilbert3d_decode(code: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inverse of :func:`_hilbert3d_encode` (Skilling's TransposeToAxes).

    Maps a 1D Hilbert index back to (x, y, z) grid coordinates. Vectorized over an
    arbitrary-shaped ``code`` array. Round-trips exactly with the encoder for all
    bit depths (verified in tests/test_hilbert3d_roundtrip.py).
    """
    code_arr = np.asarray(code, dtype=np.int64)
    if bits <= 0:
        zero = np.zeros_like(code_arr, dtype=np.uint64)
        return zero.copy(), zero.copy(), zero.copy()

    c = code_arr.astype(np.uint64)
    X0 = np.zeros_like(c)
    X1 = np.zeros_like(c)
    X2 = np.zeros_like(c)

    # De-interleave the transposed bits (inverse of encode step 3).
    for b in range(bits):
        X0 |= ((c >> np.uint64(3 * b + 2)) & np.uint64(1)) << np.uint64(b)
        X1 |= ((c >> np.uint64(3 * b + 1)) & np.uint64(1)) << np.uint64(b)
        X2 |= ((c >> np.uint64(3 * b)) & np.uint64(1)) << np.uint64(b)

    # Gray decode: t = X[n-1] >> 1; for i=n-1..1: X[i]^=X[i-1]; X[0]^=t.
    t = X2 >> np.uint64(1)
    X2 ^= X1
    X1 ^= X0
    X0 ^= t

    # Undo excess work: Q from 2 up to <2^bits, axis index i from n-1 down to 0.
    N = np.uint64(1) << np.uint64(bits)
    Q = np.uint64(2)
    while Q != N:
        P = Q - np.uint64(1)
        for i in (2, 1, 0):
            Xi = X2 if i == 2 else (X1 if i == 1 else X0)
            cond = (Xi & Q) != 0
            if i == 0:
                # Xi is X0; the off-branch is a no-op (t == 0), so only the swap matters.
                X0 = np.where(cond, X0 ^ P, X0)
            else:
                tt = (X0 ^ Xi) & P
                X0 = np.where(cond, X0 ^ P, X0 ^ tt)
                Xi_new = np.where(cond, Xi, Xi ^ tt)
                if i == 2:
                    X2 = Xi_new
                else:
                    X1 = Xi_new
        Q <<= np.uint64(1)

    return X0, X1, X2


def _hilbert_bits(R: int) -> int:
    """Number of bits per axis for a Hilbert grid of resolution ``R``.

    Single source of truth shared by every encode/decode site so the bit count can
    never diverge between training and sampling. ``ceil`` (not ``round``) is correct:
    a grid of side ``R`` needs ``ceil(log2(R))`` bits to address indices ``0..R-1``.
    For power-of-two ``R`` (the usual case) ``ceil == round``; for non-power-of-two
    ``R`` (e.g. a directly-set HILBERT_RESOLUTION during size transfer) only ``ceil``
    addresses the full range.
    """
    return int(math.ceil(math.log2(int(R))))


def min_image_delta(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    return delta - box * np.round(delta / np.maximum(box, 1e-8))


def raw_delta(delta: np.ndarray, box: np.ndarray, *, periodic: bool) -> np.ndarray:
    if periodic:
        return min_image_delta(delta, box)
    return delta


def _rail_relative_from_codes(
    cond_codes: np.ndarray,
    anchor_pos: np.ndarray,
    box: np.ndarray,
    resolution: int,
    offsets: np.ndarray,
    *,
    periodic: bool,
) -> np.ndarray:
    """Decode the Hilbert "curve rail" relative to each conditioning particle.

    Shared core for both the single-sample and batched dataset paths so the two can
    never diverge.

    Args:
        cond_codes: Hilbert index of each conditioning particle, any shape ``[...]``.
        anchor_pos: that particle's position, shape ``[..., 3]`` (same leading dims).
        box:        box lengths ``[3]``.
        resolution: grid resolution R (power of two).
        offsets:    K positive Hilbert index offsets ``[K]``.

    Returns:
        ``[..., K, 3]`` min-image vectors from each particle to the future curve cells.
    """
    R = int(resolution)
    bits = _hilbert_bits(R)  # resolution is a power of two
    offs = np.asarray(offsets, dtype=np.int64)
    base = np.asarray(cond_codes, dtype=np.int64)[..., None]  # [..., 1]
    widx = np.clip(base + offs, 0, R ** 3 - 1)                # [..., K]
    wx, wy, wz = _hilbert3d_decode(widx.reshape(-1), bits=bits)
    wcells = np.stack([wx, wy, wz], axis=-1).astype(np.float64).reshape(*widx.shape, 3)
    cell_size = np.asarray(box, dtype=np.float64) / float(R)
    wcoords = (wcells + 0.5) * cell_size                      # curve cell centers, box frame
    rel = wcoords - np.asarray(anchor_pos, dtype=np.float64)[..., None, :]
    if periodic:
        rel = min_image_delta(rel, np.asarray(box, dtype=np.float64))
    return rel.astype(np.float32)


def fixed_template_waypoints(
    pred_indices: np.ndarray,
    n_particles: int,
    resolution: int,
    box: np.ndarray,
    *,
    k: int,
    window_scale: float = 1.0,
    periodic: bool,
    reference: str = "absolute",
) -> np.ndarray:
    """Sample-INDEPENDENT "fixed template" curve rail.

    The rail for predicting particle ``j`` (1..N-1) depends only on (L->R, N, j), never
    on the actual/generated particle positions, so it cannot drift with the model's own
    samples (kills autoregressive exposure bias). Each particle index has an expected
    Hilbert index ``window_mean[j] = j * X`` with ``X = R**3 // N`` (uniform spacing on
    the curve at the system density). We place ``k`` DETERMINISTIC evenly-spaced indices
    across the window ``[j*X - window_scale*X, j*X + window_scale*X]``, decode them to
    box-frame cell centers, and express them as vectors:

      reference == "absolute": cell center min-imaged to the box origin (~particle 0).
                               A positional anchor; needs fixed-phase data.
      reference == "prev_step": cell center minus decode((j-1)*X) center; the expected
                               curve step (translation-invariant).

    Args:
        pred_indices: particle indices being predicted, shape ``[M]`` (values in 1..N-1).
        n_particles:  N (sets X = R**3 // N).
        resolution:   grid resolution R (power of two).
        box:          box lengths ``[3]``.
        k:            number of waypoints per particle.
        window_scale: half-window in units of X.
        reference:    "absolute" or "prev_step".

    Returns:
        ``[M, k, 3]`` float32 waypoint vectors.
    """
    R = int(resolution)
    bits = _hilbert_bits(R)
    total = R ** 3
    X = max(1, total // int(n_particles))
    j = np.asarray(pred_indices, dtype=np.float64).reshape(-1)          # [M]
    centers = j[:, None] * X                                            # [M, 1]
    # k deterministic evenly-spaced offsets across [-window_scale*X, +window_scale*X].
    # K=1 special case: place the single waypoint one step AHEAD at (j+1)·X so it
    # provides a true forward lookahead rather than the anchor decode(j·X) itself.
    if int(k) == 1:
        frac = np.ones((1,), dtype=np.float64)
    else:
        frac = np.linspace(-1.0, 1.0, int(k), dtype=np.float64)
    idx = centers + frac[None, :] * (float(window_scale) * X)          # [M, k]
    idx = np.clip(np.rint(idx), 0, total - 1).astype(np.int64)
    wx, wy, wz = _hilbert3d_decode(idx.reshape(-1), bits=bits)
    cell_size = np.asarray(box, dtype=np.float64) / float(R)
    wcoords = (np.stack([wx, wy, wz], axis=-1).astype(np.float64).reshape(*idx.shape, 3) + 0.5) * cell_size
    box64 = np.asarray(box, dtype=np.float64)
    if reference == "absolute":
        rel = wcoords  # relative to the box origin (particle 0 frame)
    elif reference == "prev_step":
        prev_idx = np.clip(np.rint((j - 1.0) * X), 0, total - 1).astype(np.int64)
        px, py, pz = _hilbert3d_decode(prev_idx, bits=bits)
        prev = (np.stack([px, py, pz], axis=-1).astype(np.float64) + 0.5) * cell_size  # [M,3]
        rel = wcoords - prev[:, None, :]
    else:
        raise ValueError(f"reference must be 'absolute' or 'prev_step', got {reference!r}")
    if periodic:
        rel = min_image_delta(rel, box64)
    return rel.astype(np.float32)


def fixed_template_anchors(
    pred_indices: np.ndarray,
    n_particles: int,
    resolution: int,
    box: np.ndarray,
) -> np.ndarray:
    """Per-particle absolute anchor positions ``decode(j * X)`` (box frame, [0, L)).

    ``X = R**3 // N``. Used as the reference for the rail-anchored RESIDUAL target:
    ``residual_j = pos_j - anchor_j`` (instead of the drifting ``pos_j - pos_{j-1}``).
    Returns ``[M, 3]`` float64 cell-center positions for each ``pred_indices`` entry.
    """
    R = int(resolution)
    bits = _hilbert_bits(R)
    total = R ** 3
    X = max(1, total // int(n_particles))
    idx = np.clip(np.rint(np.asarray(pred_indices, dtype=np.float64) * X), 0, total - 1).astype(np.int64)
    ax, ay, az = _hilbert3d_decode(idx, bits=bits)
    cell_size = np.asarray(box, dtype=np.float64) / float(R)
    return (np.stack([ax, ay, az], axis=-1).astype(np.float64) + 0.5) * cell_size  # [M, 3]


def _validate_arc_repr_cache(*, ordering: str, periodic: bool, has_absolute_coords: bool) -> None:
    """Preconditions for arc_repr targets on a preprocessed cache.

    Δs targets are space-filling-curve-code differences over the *stored* particle
    order, so the cache must have been built with Hilbert or Gilbert ordering; spectral
    (or any other) ordering yields non-monotone codes and meaningless Δs with no
    runtime error.
    """
    if not has_absolute_coords:
        raise ValueError(
            "arc_repr=True requires absolute_coords in the cache. "
            "Rebuild the cache (preprocess_lj_transferable.py) to include absolute coordinates."
        )
    if not periodic:
        raise ValueError("arc_repr=True is only supported for periodic systems.")
    if str(ordering).strip().lower() not in ("hilbert", "gilbert"):
        raise ValueError(
            f"arc_repr=True requires a Hilbert- or Gilbert-ordered cache, got ordering={ordering!r}. "
            "Δs targets assume space-filling-curve-monotone codes along the stored particle sequence."
        )


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
            For gilbert curves s accounts for multi-cell steps on odd grids.
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


def _cartesian_to_spherical_np(deltas: np.ndarray) -> np.ndarray:
    if deltas.shape[-1] != 3:
        raise ValueError(f"Polar conversion requires trailing dim 3, got {tuple(deltas.shape)}")
    x = deltas[..., 0]
    y = deltas[..., 1]
    z = deltas[..., 2]
    r = np.sqrt(x * x + y * y + z * z).astype(np.float32, copy=False)
    xy = np.sqrt(x * x + y * y).astype(np.float32, copy=False)
    theta = np.arctan2(xy, z).astype(np.float32, copy=False)
    phi = np.arctan2(y, x).astype(np.float32, copy=False)
    return np.stack((r, theta, phi), axis=-1).astype(np.float32, copy=False)


def _load_codebook_artifact(codebook_path: str) -> tuple[torch.Tensor, dict[str, Any]]:
    try:
        codebook = torch.load(codebook_path, map_location="cpu", weights_only=False)
    except TypeError:
        codebook = torch.load(codebook_path, map_location="cpu")
    metadata: dict[str, Any] = {}
    if isinstance(codebook, dict):
        metadata = dict(codebook.get("metadata", {}))
        for key in ("centers", "codebook", "weight", "weights", "embeddings"):
            if key in codebook:
                codebook = codebook[key]
                break
        else:
            raise ValueError(
                f"Codebook dict in {codebook_path} is missing one of "
                "'centers', 'codebook', 'weight', 'weights', or 'embeddings'."
            )
    codebook = torch.as_tensor(codebook, dtype=torch.float32)
    if codebook.ndim != 2:
        raise ValueError(f"Codebook must be rank-2 [K,D], got {tuple(codebook.shape)} from {codebook_path}")
    if codebook.shape[0] <= 0 or codebook.shape[1] <= 0:
        raise ValueError(f"Codebook must be non-empty, got {tuple(codebook.shape)} from {codebook_path}")
    return codebook.contiguous(), metadata


def _load_codebook_tensor(codebook_path: str) -> torch.Tensor:
    codebook, _ = _load_codebook_artifact(codebook_path)
    return codebook


def _quantize_with_codebook(
    deltas: np.ndarray,
    codebook: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    if deltas.ndim < 2:
        raise ValueError(f"deltas must have rank >= 2 for codebook quantization, got {tuple(deltas.shape)}")
    if deltas.shape[-1] != int(codebook.shape[1]):
        raise ValueError(
            f"deltas trailing dim {deltas.shape[-1]} must match codebook dim {int(codebook.shape[1])}"
        )
    original_shape = tuple(deltas.shape[:-1])
    flat_diff = torch.from_numpy(np.asarray(deltas, dtype=np.float32, order="C").reshape(-1, deltas.shape[-1]))
    distances = torch.cdist(flat_diff, codebook, p=2.0)
    token_ids = torch.argmin(distances, dim=-1).to(torch.int64)
    tokens_np = token_ids.view(*original_shape).cpu().numpy().astype(np.int64, copy=False)
    long_jump_mask = np.zeros(original_shape, dtype=np.bool_)
    return tokens_np, long_jump_mask


def _offline_discretize_cache_tensors(
    *,
    deltas: torch.Tensor,
    delta_length: torch.Tensor,
    particle_length: torch.Tensor,
    box_size: torch.Tensor,
    codebook_path: str,
    periodic: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if deltas.ndim != 3:
        raise ValueError(f"deltas must be [B,T,D], got {tuple(deltas.shape)}")
    if delta_length.ndim != 1 or particle_length.ndim != 1:
        raise ValueError("delta_length and particle_length must be rank-1.")
    if box_size.ndim != 2:
        raise ValueError(f"box_size must be [B,D], got {tuple(box_size.shape)}")
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codebook = _load_codebook_tensor(codebook_path).to(device=device, dtype=torch.float32)
    if int(codebook.shape[1]) != int(deltas.shape[-1]):
        raise ValueError(
            f"Codebook dim {int(codebook.shape[1])} does not match delta dim {int(deltas.shape[-1])}."
        )

    n_samples, max_delta_len, coord_dim = deltas.shape
    discrete_ids = torch.zeros((n_samples, max_delta_len), dtype=torch.long)
    reconstructed_coords = torch.zeros((n_samples, max_delta_len + 1, coord_dim), dtype=torch.float32)
    reconstructed_token_coords = torch.zeros((n_samples, max_delta_len, coord_dim), dtype=torch.float32)

    for start in range(0, int(n_samples), int(chunk_size)):
        end = min(start + int(chunk_size), int(n_samples))
        deltas_chunk = deltas[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        delta_length_chunk = delta_length[start:end].to(device=device, dtype=torch.long, non_blocking=True)
        particle_length_chunk = particle_length[start:end].to(device=device, dtype=torch.long, non_blocking=True)
        box_chunk = box_size[start:end].to(device=device, dtype=torch.float32, non_blocking=True)

        batch_size = deltas_chunk.shape[0]
        valid_delta_mask = torch.arange(max_delta_len, device=device).unsqueeze(0) < delta_length_chunk.unsqueeze(1)

        token_ids_chunk = torch.zeros((batch_size, max_delta_len), dtype=torch.long, device=device)
        if bool(valid_delta_mask.any()):
            flat_valid = deltas_chunk[valid_delta_mask]
            distances = torch.cdist(flat_valid, codebook, p=2.0)
            token_ids_chunk[valid_delta_mask] = torch.argmin(distances, dim=-1)

        quantized_deltas_chunk = torch.zeros_like(deltas_chunk)
        if bool(valid_delta_mask.any()):
            quantized_deltas_chunk[valid_delta_mask] = codebook[token_ids_chunk[valid_delta_mask]]

        coords_chunk = torch.zeros((batch_size, max_delta_len + 1, coord_dim), dtype=torch.float32, device=device)
        coords_chunk[:, 1:, :] = torch.cumsum(quantized_deltas_chunk, dim=1)
        if periodic:
            coords_chunk = torch.remainder(coords_chunk, box_chunk.unsqueeze(1))

        valid_particle_mask = torch.arange(max_delta_len + 1, device=device).unsqueeze(0) < particle_length_chunk.unsqueeze(1)
        coords_chunk = coords_chunk * valid_particle_mask.unsqueeze(-1)

        token_coord_mask = torch.arange(max_delta_len, device=device).unsqueeze(0) < delta_length_chunk.unsqueeze(1)
        token_coords_chunk = coords_chunk[:, :-1, :] * token_coord_mask.unsqueeze(-1)

        discrete_ids[start:end] = token_ids_chunk.cpu()
        reconstructed_coords[start:end] = coords_chunk.cpu()
        reconstructed_token_coords[start:end] = token_coords_chunk.cpu()

    return discrete_ids, reconstructed_coords, reconstructed_token_coords


def _load_h5_traj_and_box(path: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5f:
        if "traj" not in h5f:
            raise KeyError(f"'traj' dataset not found in {path}")
        traj = np.asarray(h5f["traj"], dtype=np.float32)
        if "boxlength" not in h5f.attrs:
            raise KeyError(f"'boxlength' attribute not found in {path}")
        box_attr = np.asarray(h5f.attrs["boxlength"], dtype=np.float32).reshape(-1)

    if traj.ndim == 4:
        n_frames, batch_size, n_particles, dim = traj.shape
        traj = traj.reshape(n_frames * batch_size, n_particles, dim)
    elif traj.ndim != 3:
        raise ValueError(f"Expected traj rank 3 or 4, got shape {traj.shape} (file={path})")
    dim = int(traj.shape[-1])
    if dim < 2:
        raise ValueError(
            f"Expected at least 2 coordinate dimensions, got {traj.shape[-1]} (file={path})"
        )

    if box_attr.size == 1:
        box = np.full((dim,), float(box_attr[0]), dtype=np.float32)
    elif box_attr.size >= dim:
        box = np.asarray(box_attr[:dim], dtype=np.float32)
    else:
        pad = np.full((dim - box_attr.size,), float(box_attr[-1]), dtype=np.float32)
        box = np.concatenate([np.asarray(box_attr, dtype=np.float32), pad], axis=0)
    if np.any(box <= 0.0):
        raise ValueError(f"Invalid box lengths in {path}: {box.tolist()}")

    return np.asarray(traj[..., :dim], dtype=np.float32), box


def _center_coords_to_com_zero(coords: np.ndarray) -> np.ndarray:
    return coords - coords.mean(axis=-2, keepdims=True)


def _repeat_factorized_token_coords(
    token_coords: np.ndarray,
    *,
    coord_dim: int,
) -> np.ndarray:
    axis_to_repeat = 1 if token_coords.ndim == 3 else 0
    return np.repeat(token_coords, int(coord_dim), axis=axis_to_repeat)


def _repeat_factorized_curve_waypoints(
    curve_waypoints: np.ndarray,
    *,
    coord_dim: int,
) -> np.ndarray:
    axis_to_repeat = 1 if curve_waypoints.ndim == 4 else 0
    return np.repeat(curve_waypoints, int(coord_dim), axis=axis_to_repeat)


def _compute_lj_energy_batch(
    coords: np.ndarray,
    *,
    box: np.ndarray,
    periodic: bool,
    epsilon: float,
    sigma: float,
    cutoff: Optional[float],
    spring_constant: float,
    chunk_size: int = 2048,
) -> np.ndarray:
    coords_np = np.asarray(coords, dtype=np.float32)
    if coords_np.ndim != 3:
        raise ValueError(f"coords must be [B,N,D], got {tuple(coords_np.shape)}")
    box_np = np.asarray(box, dtype=np.float32).reshape(-1)
    if box_np.size != int(coords_np.shape[-1]):
        raise ValueError(
            f"box must have one entry per coordinate dimension, got {tuple(box_np.shape)} "
            f"for coords dim {coords_np.shape[-1]}"
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    lj = LJ(
        nparticles=None,
        dim=int(coords_np.shape[-1]),
        batch_size=None,
        device="cpu",
        boxlength=torch.as_tensor(box_np, dtype=torch.float32),
        kT=1.0,
        epsilon=float(epsilon),
        sigma=float(sigma),
        cutoff=None if cutoff is None else float(cutoff),
        periodic=bool(periodic),
        spring_constant=float(spring_constant),
    )

    energies: list[np.ndarray] = []
    for start in range(0, int(coords_np.shape[0]), int(chunk_size)):
        end = min(start + int(chunk_size), int(coords_np.shape[0]))
        chunk = torch.from_numpy(coords_np[start:end]).to(dtype=torch.float32)
        with torch.no_grad():
            energy = lj.potential(chunk)
        energies.append(energy.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(energies, axis=0)


def _compute_dw_energy_batch(
    coords: np.ndarray,
    *,
    box: np.ndarray,
    periodic: bool,
    a: float,
    b: float,
    c: float,
    offset: float,
    chunk_size: int = 2048,
) -> np.ndarray:
    coords_np = np.asarray(coords, dtype=np.float32)
    if coords_np.ndim != 3:
        raise ValueError(f"coords must be [B,N,D], got {tuple(coords_np.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    boxlength = None
    if periodic:
        box_np = np.asarray(box, dtype=np.float32).reshape(-1)
        if box_np.size != int(coords_np.shape[-1]):
            raise ValueError(
                f"box must have one entry per coordinate dimension, got {tuple(box_np.shape)} "
                f"for coords dim {coords_np.shape[-1]}"
            )
        boxlength = torch.as_tensor(box_np, dtype=torch.float32)
    dw = DoubleWellPotential(
        a=float(a),
        b=float(b),
        c=float(c),
        offset=float(offset),
        dim=int(coords_np.shape[-2] * coords_np.shape[-1]),
        n_particles=int(coords_np.shape[-2]),
        boxlength=boxlength,
    )

    energies: list[np.ndarray] = []
    for start in range(0, int(coords_np.shape[0]), int(chunk_size)):
        end = min(start + int(chunk_size), int(coords_np.shape[0]))
        chunk = torch.from_numpy(coords_np[start:end]).to(dtype=torch.float32)
        with torch.no_grad():
            energy = dw.potential(chunk)
        energies.append(energy.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(energies, axis=0)


def _compute_target_energy_batch(
    coords: np.ndarray,
    *,
    box: np.ndarray,
    periodic: bool,
    target_system: str,
    lj_epsilon: float,
    lj_sigma: float,
    lj_cutoff: Optional[float],
    lj_spring_constant: float,
    dw_a: float,
    dw_b: float,
    dw_c: float,
    dw_offset: float,
    chunk_size: int = 2048,
) -> np.ndarray:
    system = str(target_system).lower()
    if system == "lj":
        return _compute_lj_energy_batch(
            coords,
            box=box,
            periodic=bool(periodic),
            epsilon=float(lj_epsilon),
            sigma=float(lj_sigma),
            cutoff=None if lj_cutoff is None else float(lj_cutoff),
            spring_constant=float(lj_spring_constant),
            chunk_size=int(chunk_size),
        )
    if system == "dw":
        return _compute_dw_energy_batch(
            coords,
            box=box,
            periodic=bool(periodic),
            a=float(dw_a),
            b=float(dw_b),
            c=float(dw_c),
            offset=float(dw_offset),
            chunk_size=int(chunk_size),
        )
    raise ValueError(f"Unsupported target_system={target_system!r}; expected 'lj' or 'dw'.")


def _rotation_matrix_2d(k: int) -> np.ndarray:
    kk = int(k) % 4
    if kk == 0:
        return np.eye(2, dtype=np.float32)
    if kk == 1:
        return np.array([[0, -1], [1, 0]], dtype=np.float32)
    if kk == 2:
        return np.array([[-1, 0], [0, -1]], dtype=np.float32)
    return np.array([[0, 1], [-1, 0]], dtype=np.float32)


def _rotation_matrix_3d(kx: int, ky: int, kz: int) -> np.ndarray:
    def _pow(mat: np.ndarray, p: int) -> np.ndarray:
        out = np.eye(3, dtype=np.float32)
        for _ in range(int(p) % 4):
            out = mat @ out
        return out

    rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    return _pow(rz, kz) @ _pow(ry, ky) @ _pow(rx, kx)


def _random_right_angle_rotation(
    rng: np.random.Generator,
    coords: np.ndarray,
) -> np.ndarray:
    coords_arr = np.asarray(coords, dtype=np.float32)
    dim = int(coords_arr.shape[-1])
    if dim == 2:
        rot = _rotation_matrix_2d(int(rng.integers(0, 4)))
    elif dim == 3:
        rot = _rotation_matrix_3d(
            int(rng.integers(0, 4)),
            int(rng.integers(0, 4)),
            int(rng.integers(0, 4)),
        )
    else:
        raise ValueError(f"Right-angle data augmentation is only supported in 2D/3D, got dim={dim}")
    return np.asarray(coords_arr @ rot.T, dtype=np.float32)


def _apply_fixed_right_angle_rotation_2d(
    coords: np.ndarray,
    *,
    rotation_k: int,
    box: Optional[np.ndarray] = None,
) -> np.ndarray:
    if int(coords.shape[-1]) != 2:
        raise ValueError(
            f"Four-way 90-degree rotation augmentation is only supported in 2D, got dim={coords.shape[-1]}"
        )
    rot = _rotation_matrix_2d(int(rotation_k))
    if box is not None:
        center = 0.5 * np.asarray(box, dtype=np.float32)[None, :]
        return np.asarray((coords - center) @ rot.T + center, dtype=np.float32)
    return np.asarray(coords @ rot.T, dtype=np.float32)


def _normalize_file_paths(
    file_paths: Optional[Sequence[str]] = None,
    *,
    h5_path: Optional[str] = None,
) -> list[str]:
    if file_paths is None:
        if h5_path is None:
            raise ValueError("Provide `file_paths` (list) or `h5_path` (single file).")
        paths = [str(h5_path)]
    elif isinstance(file_paths, (str, bytes)):
        paths = [str(file_paths)]
    else:
        paths = [str(p) for p in file_paths]

    if not paths:
        raise ValueError("No H5 files provided.")
    return paths


@dataclass(frozen=True)
class RelativeDeltaTokenizer:
    window: float = 3.0
    bins: int = 64
    dim: int = 2
    use_long_jump_token: bool = True
    long_jump_delta: Tuple[float, ...] = ()
    factorized: bool = False

    def __post_init__(self) -> None:
        if int(self.dim) <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if int(self.bins) <= 0:
            raise ValueError(f"bins must be positive, got {self.bins}")
        if float(self.window) <= 0.0:
            raise ValueError(f"window must be > 0, got {self.window}")
        if len(self.long_jump_delta) == 0:
            object.__setattr__(self, "long_jump_delta", tuple(0.0 for _ in range(int(self.dim))))
        elif len(self.long_jump_delta) != int(self.dim):
            raise ValueError(
                f"long_jump_delta must have length dim={self.dim}, got {len(self.long_jump_delta)}"
            )

    @property
    def base_vocab_size(self) -> int:
        if self.factorized:
            return int(self.bins)
        return int(self.bins ** self.dim)

    @property
    def long_jump_id(self) -> Optional[int]:
        if not self.use_long_jump_token:
            return None
        return int(self.base_vocab_size)

    @property
    def vocab_size(self) -> int:
        return int(self.base_vocab_size + (1 if self.use_long_jump_token else 0))

    def encode(self, deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if deltas.ndim < 2 or deltas.shape[-1] != int(self.dim):
            raise ValueError(
                f"deltas must have trailing shape [...,{self.dim}], got {tuple(deltas.shape)}"
            )
        w = float(self.window)
        b = int(self.bins)
        step = (2.0 * w) / float(b)

        clipped = np.clip(deltas, -w, w)
        scaled = (clipped + w) / max(step, 1e-8)
        coords = np.floor(scaled).astype(np.int64)
        coords = np.clip(coords, 0, b - 1)

        if self.factorized:
            tokens = coords.reshape(*coords.shape[:-2], -1)
            outside = (np.abs(deltas) > w).reshape(*deltas.shape[:-2], -1)
            if self.use_long_jump_token:
                tokens[outside] = int(self.long_jump_id)
            return tokens.astype(np.int64, copy=False), outside.astype(np.bool_, copy=False)

        outside = np.any(np.abs(deltas) > w, axis=-1)
        tokens = np.zeros(coords.shape[:-1], dtype=np.int64)
        stride = 1
        for d in range(int(self.dim)):
            tokens += coords[..., d] * stride
            stride *= b
        if self.use_long_jump_token:
            tokens[outside] = int(self.long_jump_id)
        return tokens.astype(np.int64, copy=False), outside.astype(np.bool_, copy=False)

    def decode(self, token_ids: torch.LongTensor) -> torch.Tensor:
        if token_ids.ndim < 1:
            raise ValueError("token_ids must have rank >= 1")
        b = int(self.bins)
        w = float(self.window)
        step = (2.0 * w) / float(b)

        if self.factorized:
            if token_ids.shape[-1] % int(self.dim) != 0:
                raise ValueError(
                    f"Factorized token_ids last dimension must be divisible by dim={self.dim}, "
                    f"got shape {tuple(token_ids.shape)}"
                )
            flat = token_ids.reshape(-1).long()
            long_mask = (
                flat == int(self.long_jump_id)
                if self.use_long_jump_token
                else torch.zeros_like(flat, dtype=torch.bool)
            )
            if self.use_long_jump_token:
                flat = flat.clamp(0, self.base_vocab_size - 1)

            deltas_1d = -w + (flat.float() + 0.5) * step
            if self.use_long_jump_token and long_mask.any():
                deltas_1d[long_mask] = 0.0

            shape = list(token_ids.shape)
            shape[-1] = shape[-1] // int(self.dim)
            shape.append(int(self.dim))
            return deltas_1d.view(*shape)

        flat = token_ids.reshape(-1).long()
        long_mask = torch.zeros_like(flat, dtype=torch.bool)
        if self.use_long_jump_token:
            long_mask = flat == int(self.long_jump_id)
            flat = flat.clamp(0, self.base_vocab_size - 1)

        comps = []
        cur = flat
        for _ in range(int(self.dim)):
            comps.append(torch.remainder(cur, b).to(torch.float32))
            cur = torch.div(cur, b, rounding_mode="floor")
        deltas = torch.stack(
            [(-w + (comp + 0.5) * step) for comp in comps],
            dim=-1,
        )

        if self.use_long_jump_token and long_mask.any():
            lj = torch.tensor(self.long_jump_delta, dtype=deltas.dtype, device=deltas.device)
            deltas[long_mask] = lj

        return deltas.view(*token_ids.shape, int(self.dim))


# Default "curve rail" look-ahead offsets, in Hilbert *index* units (1-to-1 with
# arc length via cell_size). K=6, log-uniform; validated on L=10 ρ=1 data to bracket
# ~84% of consecutive arc-gaps and localize the true next particle to ~0.57σ median.
# Index offsets are ~box-size invariant at fixed cell-size & density (R³/N ≈ const).
DEFAULT_CURVE_RAIL_OFFSETS: Tuple[int, ...] = (64, 215, 724, 2435, 8192, 27554)


class LJTransferableDataset(Dataset):
    """
    Lennard-Jones trajectories tokenized as ordered local relative moves.

    Per-sample pipeline:
      1. Random global torus shift.
      2. Sort shifted particles (Hilbert or spectral/Fiedler ordering).
      3. Minimum-image deltas between consecutive sorted positions (N-1 deltas).
      4. Quantize each delta on a local [-W, +W]^D grid.

    Supports multiple H5 files with potentially different particle counts.
    """

    def __init__(
        self,
        file_paths: Optional[Sequence[str]] = None,
        *,
        h5_path: Optional[str] = None,
        periodic: bool = True,
        hilbert_resolution: int = 128,
        cell_size: Optional[float] = None,
        ordering: str = "hilbert",
        spectral_sigma: float = 1.0,
        local_window: float = 3.0,
        local_bins: int = 64,
        use_long_jump_token: bool = True,
        factorized: bool = False,
        polar: bool = False,
        discrete: bool = False,
        codebook_path: Optional[str] = None,
        random_grid_shift: bool = True,
        use_data_aug: bool = False,
        limit: Optional[int] = None,
        seed: int = 0,
        use_curve_rail: bool = False,
        curve_rail_offsets: Optional[Sequence[int]] = None,
        curve_rail_mode: str = "lookahead",
        curve_rail_window: float = 1.0,
        curve_rail_k: int = 8,
        curve_rail_reference: str = "absolute",
        curve_rail_residual_target: bool = False,
    ) -> None:
        self.file_paths = _normalize_file_paths(file_paths, h5_path=h5_path)
        self.periodic = bool(periodic)
        self.ordering = str(ordering).strip().lower()
        if self.ordering not in ("hilbert", "gilbert", "spectral"):
            raise ValueError(
                f"ordering must be 'hilbert', 'gilbert' or 'spectral', got {self.ordering!r}"
            )
        self.spectral_sigma = float(spectral_sigma)
        if self.spectral_sigma <= 0.0:
            raise ValueError(f"spectral_sigma must be > 0, got {self.spectral_sigma}")
        self.hilbert_resolution = int(hilbert_resolution)
        if self.ordering == "hilbert" and not _is_power_of_two(self.hilbert_resolution):
            raise ValueError(
                f"hilbert_resolution must be a power of two, got {self.hilbert_resolution}"
            )
        # Constant physical cell size: when set, the Hilbert grid resolution is chosen
        # PER BOX as the next power of two of (L / cell_size), so the cell size (and
        # thus the curve's local granularity relative to particles) stays ~constant
        # across box lengths instead of the cell COUNT. Pow-2 rounding makes it exact
        # only up to an octave.
        self.cell_size = None if cell_size is None else float(cell_size)
        if self.cell_size is not None and self.cell_size <= 0.0:
            raise ValueError(f"cell_size must be > 0, got {self.cell_size}")
        self.tokenizer = RelativeDeltaTokenizer(
            window=float(local_window),
            bins=int(local_bins),
            dim=2,
            use_long_jump_token=bool(use_long_jump_token),
            factorized=bool(factorized),
        )
        self.random_grid_shift = bool(random_grid_shift)
        self.use_data_aug = bool(use_data_aug)
        self.factorized = bool(factorized)
        self.polar = bool(polar)
        self.discrete = bool(discrete)
        self.codebook_path = str(codebook_path) if codebook_path is not None else None
        self.codebook: Optional[torch.Tensor] = None
        if self.discrete:
            if self.factorized:
                raise ValueError("discrete=True requires factorized=False.")
            if self.polar:
                raise ValueError("discrete=True is incompatible with polar=True.")
            if self.codebook_path is None:
                raise ValueError("discrete=True requires codebook_path.")
            self.codebook = _load_codebook_tensor(self.codebook_path)
        if self.polar and self.factorized:
            raise ValueError(
                "polar=True with factorized=True is not supported in this checkout because "
                "polar token-sequence conditioning has not been defined."
            )
        if (not self.periodic) and self.random_grid_shift:
            raise ValueError(
                "random_grid_shift=True is only supported for periodic lj_transferable data. "
                "Nonperiodic mode centers each sample to COM=0 and uses a fixed origin-anchored "
                "ordering."
            )
        self.rng = np.random.default_rng(int(seed))

        # Optional "curve rail" (GPS-guide) look-ahead conditioning. Off by default so
        # existing datasets are byte-for-byte unchanged. Only meaningful for 3D periodic
        # Hilbert ordering (the LJ-transferable regime).
        self.use_curve_rail = bool(use_curve_rail)
        self.curve_rail_mode = str(curve_rail_mode).strip().lower()
        if self.curve_rail_mode not in ("lookahead", "fixed_template"):
            raise ValueError(f"curve_rail_mode must be 'lookahead' or 'fixed_template', got {curve_rail_mode!r}")
        self.curve_rail_window = float(curve_rail_window)
        self.curve_rail_reference = str(curve_rail_reference).strip().lower()
        self.curve_rail_residual_target = bool(curve_rail_residual_target)
        if self.curve_rail_residual_target and self.curve_rail_mode != "fixed_template":
            raise ValueError("curve_rail_residual_target=True requires curve_rail_mode='fixed_template'.")
        offsets = DEFAULT_CURVE_RAIL_OFFSETS if curve_rail_offsets is None else curve_rail_offsets
        offsets_arr = np.asarray(sorted(set(int(o) for o in offsets)), dtype=np.int64)
        # K (waypoint count): in lookahead mode it equals the number of offsets; in
        # fixed_template mode it is an independent count of window samples.
        self.curve_rail_k = int(offsets_arr.size) if self.curve_rail_mode == "lookahead" else int(curve_rail_k)
        if self.use_curve_rail:
            if self.curve_rail_mode == "lookahead" and (offsets_arr.size == 0 or np.any(offsets_arr <= 0)):
                raise ValueError(f"curve_rail_offsets must be positive integers, got {offsets}")
            if self.curve_rail_mode == "fixed_template" and self.curve_rail_k <= 0:
                raise ValueError(f"curve_rail_k must be positive, got {self.curve_rail_k}")
            if self.ordering == "gilbert":
                raise NotImplementedError(
                    "curve rail + gilbert ordering is not wired (rail waypoint decode and "
                    "the sampler's rail mirror assume the pow-2 Hilbert curve); train rails "
                    "with ordering='hilbert' or extend _curve_waypoints_3d via get_curve3d."
                )
            if self.ordering != "hilbert":
                raise ValueError("use_curve_rail=True requires ordering='hilbert'.")
            if bool(factorized) and self.curve_rail_mode != "fixed_template":
                raise ValueError(
                    "use_curve_rail=True with factorized=True is only supported for "
                    "curve_rail_mode='fixed_template'."
                )
        self.curve_rail_offsets = offsets_arr

        self._coords_by_file: list[np.ndarray] = []
        self._box_by_file: list[np.ndarray] = []
        self._n_by_file: list[int] = []
        self.coord_dim: Optional[int] = None

        sample_to_file: list[np.ndarray] = []
        sample_to_local: list[np.ndarray] = []
        sample_lengths: list[np.ndarray] = []

        for file_idx, path in enumerate(self.file_paths):
            traj, box_vec = _load_h5_traj_and_box(path)
            _, n_particles, dim = traj.shape

            if self.coord_dim is None:
                self.coord_dim = int(dim)
                self.tokenizer = RelativeDeltaTokenizer(
                    window=float(local_window),
                    bins=int(local_bins),
                    dim=int(dim),
                    use_long_jump_token=bool(use_long_jump_token),
                    factorized=bool(factorized),
                )
            elif int(dim) != int(self.coord_dim):
                raise ValueError(
                    f"All lj_transferable files must have the same coordinate dimension; "
                    f"expected {self.coord_dim}, got {dim} for {path}"
                )
            if self.discrete and self.codebook is not None and int(self.codebook.shape[1]) != int(dim):
                raise ValueError(
                    f"Discrete codebook dim {int(self.codebook.shape[1])} does not match coord dim {dim} for {path}"
                )

            coords = traj
            if not self.periodic:
                coords = _center_coords_to_com_zero(coords)
            n_samples_i = int(coords.shape[0])

            self._coords_by_file.append(coords)
            self._box_by_file.append(box_vec)
            self._n_by_file.append(int(n_particles))

            sample_to_file.append(np.full((n_samples_i,), file_idx, dtype=np.int64))
            sample_to_local.append(np.arange(n_samples_i, dtype=np.int64))
            sample_lengths.append(np.full((n_samples_i,), int(n_particles), dtype=np.int64))

        self.sample_id_to_file_index = np.concatenate(sample_to_file, axis=0)
        self.sample_id_to_local_index = np.concatenate(sample_to_local, axis=0)
        self.sample_lengths = np.concatenate(sample_lengths, axis=0)

        if limit is not None:
            lim = max(1, min(int(limit), int(self.sample_id_to_file_index.shape[0])))
            self.sample_id_to_file_index = self.sample_id_to_file_index[:lim]
            self.sample_id_to_local_index = self.sample_id_to_local_index[:lim]
            self.sample_lengths = self.sample_lengths[:lim]

        self.vocab_size = int(self.codebook.shape[0]) if self.discrete and self.codebook is not None else int(self.tokenizer.vocab_size)
        self.sos_id = int(self.vocab_size)
        if self.coord_dim is None:
            raise RuntimeError("Failed to infer coordinate dimension from lj_transferable inputs.")

    def __len__(self) -> int:
        return int(self.sample_id_to_file_index.shape[0])

    def _resolution_for_box(self, box: Optional[np.ndarray]) -> int:
        """Grid resolution (cells per axis) for a box — delegates to
        ``resolution_for_box_rule``. See that function for the full rule."""
        return resolution_for_box_rule(
            box,
            cell_size=self.cell_size,
            hilbert_resolution=self.hilbert_resolution,
            ordering=self.ordering,
        )

    def _space_filling_codes_2d(self, grid: np.ndarray, resolution: int) -> np.ndarray:
        h = np.empty(grid.shape[0], dtype=np.int64)
        for i in range(grid.shape[0]):
            h[i] = _hilbert_xy2d(int(resolution), int(grid[i, 0]), int(grid[i, 1]))
        return h

    def _space_filling_codes_3d(self, grid: np.ndarray, resolution: int) -> np.ndarray:
        from .curves import get_curve3d  # lazy: curves.py may import this module

        kind = "gilbert" if self.ordering == "gilbert" else "hilbert"
        return get_curve3d(kind, int(resolution)).encode(grid)

    def _space_filling_codes(self, grid: np.ndarray, resolution: Optional[int] = None) -> np.ndarray:
        R = int(self.hilbert_resolution if resolution is None else resolution)
        if int(self.coord_dim) == 2:
            return self._space_filling_codes_2d(grid, R)
        if int(self.coord_dim) == 3:
            return self._space_filling_codes_3d(grid, R)
        raise ValueError(f"Hilbert-like ordering is only supported in 2D/3D, got dim={self.coord_dim}")

    def _encode_deltas(self, deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.discrete:
            assert self.codebook is not None
            return _quantize_with_codebook(deltas, self.codebook)
        return self.tokenizer.encode(deltas)

    def _grid_coords_periodic(self, coords: np.ndarray, box: np.ndarray) -> np.ndarray:
        R = self._resolution_for_box(box)
        wrapped = np.mod(coords, box[None, :])
        scaled = (wrapped / np.maximum(box[None, :], 1e-8)) * float(R)
        grid = np.floor(scaled).astype(np.int64)
        return np.clip(grid, 0, R - 1)

    def _grid_coords_nonperiodic(self, coords: np.ndarray) -> np.ndarray:
        R = int(self.hilbert_resolution)  # nonperiodic uses the fixed global resolution
        extent = np.max(np.abs(coords), axis=-2, keepdims=True)
        extent = np.maximum(extent, 1e-8)
        shifted = (coords / (2.0 * extent)) + 0.5
        grid = np.floor(shifted * float(R)).astype(np.int64)
        return np.clip(grid, 0, R - 1)

    def _hilbert_sort_periodic(
        self, coords: np.ndarray, box: np.ndarray, *, return_codes: bool = False
    ):
        R = self._resolution_for_box(box)
        grid = self._grid_coords_periodic(coords, box)
        codes = self._space_filling_codes(grid, R)
        order = np.argsort(codes, kind="stable")
        if return_codes:
            return order, codes[order]
        return order

    def _hilbert_sort_nonperiodic(self, coords: np.ndarray) -> np.ndarray:
        grid = self._grid_coords_nonperiodic(coords)
        return np.argsort(self._space_filling_codes(grid, int(self.hilbert_resolution)), kind="stable")

    def _spectral_sort_periodic(self, coords: np.ndarray, box: np.ndarray) -> np.ndarray:
        coords_t = torch.from_numpy(coords).to(dtype=torch.float32).unsqueeze(0)
        box_t = torch.from_numpy(box).to(dtype=torch.float32).view(1, -1)
        order_t = spectral_argsort(coords_t, box_size=box_t, sigma=self.spectral_sigma)[0]
        return order_t.detach().cpu().numpy().astype(np.int64, copy=False)

    def _spectral_sort_nonperiodic(self, coords: np.ndarray) -> np.ndarray:
        coords_t = torch.from_numpy(coords).to(dtype=torch.float32)
        n_points = int(coords_t.shape[0])
        if n_points <= 1:
            return np.arange(n_points, dtype=np.int64)

        dx = coords_t[:, None, :] - coords_t[None, :, :]
        dist_sq = torch.sum(dx * dx, dim=-1)
        gamma = 1.0 / (2.0 * float(self.spectral_sigma) * float(self.spectral_sigma))
        W = torch.exp(-dist_sq * gamma)
        W.fill_diagonal_(0.0)

        deg = W.sum(dim=-1).clamp_min(1e-8)
        eye = torch.eye(n_points, dtype=coords_t.dtype)
        L_rw = eye - (W / deg.unsqueeze(-1))
        L = 0.5 * (L_rw + L_rw.transpose(-1, -2))

        _, eigvecs = torch.linalg.eigh(L)
        u2 = eigvecs[:, 1]

        centroid = coords_t.mean(dim=0, keepdim=True)
        radius = torch.linalg.norm(coords_t - centroid, dim=-1)
        sign_score = torch.sum(u2 * (radius - radius.mean()))
        if sign_score < 0.0:
            u2 = -u2
        return torch.argsort(u2).cpu().numpy().astype(np.int64, copy=False)

    def _curve_waypoints_3d(
        self,
        sorted_pos: np.ndarray,
        box: np.ndarray,
        resolution: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Look-ahead "curve rail" (GPS guide) for 3D periodic Hilbert ordering.

        For each conditioning particle j (0..N-2), returns the K future Hilbert-curve
        cell centers at index offsets ``self.curve_rail_offsets`` ahead of particle j's
        own cell, expressed as min-image vectors *relative to particle j*. These depend
        only on deterministic curve geometry (particle j's cell + grid resolution), never
        on where future particles actually sit -> causal-safe and reproducible at
        generation time.

        Returns:
            waypoints: [N-1, K, 3] float32 relative vectors (particle j -> curve cell).
            arclen:    [K] float32 Hilbert arc lengths (offset * cell_size).
        """
        R = int(resolution)
        K = int(self.curve_rail_k)
        n = int(sorted_pos.shape[0])
        cell_size = np.asarray(box, dtype=np.float64) / float(R)

        if self.curve_rail_mode == "fixed_template":
            X = max(1, (R ** 3) // int(n))
            arclen = np.full((K,), float(self.curve_rail_window * X * float(np.mean(cell_size))), dtype=np.float32)
            if n < 2:
                return np.zeros((max(0, n - 1), K, 3), dtype=np.float32), arclen
            # waypoints for predicting particles 1..N-1 (sample-INDEPENDENT template).
            pred = np.arange(1, n, dtype=np.int64)
            rel = fixed_template_waypoints(
                pred, n, R, box, k=K, window_scale=self.curve_rail_window,
                periodic=self.periodic, reference=self.curve_rail_reference,
            )  # [N-1, K, 3]
            return rel, arclen

        offs = self.curve_rail_offsets
        arclen = (offs.astype(np.float64) * float(np.mean(cell_size))).astype(np.float32)
        if n < 2:
            return np.zeros((max(0, n - 1), K, 3), dtype=np.float32), arclen

        grid = self._grid_coords_periodic(sorted_pos, box)
        codes = self._space_filling_codes_3d(grid, R)  # ascending (already Hilbert-sorted)
        rel = _rail_relative_from_codes(
            codes[:-1], sorted_pos[:-1], box, R, offs, periodic=self.periodic
        )  # [N-1, K, 3]
        return rel, arclen

    def _build_item_from_index(
        self,
        idx: int,
        *,
        rotation_k: Optional[int] = None,
    ) -> tuple[dict[str, Any], float]:
        file_idx = int(self.sample_id_to_file_index[idx])
        local_idx = int(self.sample_id_to_local_index[idx])

        coords = self._coords_by_file[file_idx][local_idx]
        box = self._box_by_file[file_idx]
        n_particles = int(self.sample_lengths[idx])

        working = coords.copy()
        if rotation_k is not None:
            working = _apply_fixed_right_angle_rotation_2d(
                working,
                rotation_k=int(rotation_k),
                box=box if self.periodic else None,
            )
        elif self.use_data_aug:
            if self.periodic:
                center = 0.5 * box[None, :]
                working = _random_right_angle_rotation(self.rng, working - center) + center
            else:
                working = _random_right_angle_rotation(self.rng, working)

        if self.periodic and self.random_grid_shift:
            shift = self.rng.uniform(low=0.0, high=1.0, size=(int(self.coord_dim),)).astype(np.float32) * box
            shifted = np.mod(working + shift[None, :], box[None, :])
        elif self.periodic:
            shifted = np.mod(working, box[None, :])
        else:
            shifted = working

        if self.ordering in ("hilbert", "gilbert"):
            if self.periodic:
                order = self._hilbert_sort_periodic(shifted, box=box)
            else:
                order = self._hilbert_sort_nonperiodic(shifted)
        else:
            if self.periodic:
                order = self._spectral_sort_periodic(shifted, box=box)
            else:
                order = self._spectral_sort_nonperiodic(shifted)

        sorted_pos = shifted[order]

        n_predict = n_particles - 1
        if self.use_curve_rail and self.curve_rail_residual_target and int(self.coord_dim) == 3 and n_particles >= 2:
            # Rail-anchored RESIDUAL target: pos_j - decode(j*X), so each particle is
            # absolutely pinned to its template cell (no drifting prev-particle reference).
            R_anchor = self._resolution_for_box(box) if self.periodic else int(self.hilbert_resolution)
            anchors = fixed_template_anchors(np.arange(1, n_particles), n_particles, R_anchor, box)  # [N-1,3]
            deltas = raw_delta(sorted_pos[1:] - anchors, box, periodic=self.periodic).astype(np.float32, copy=False)
        else:
            deltas = np.empty((n_predict, int(self.coord_dim)), dtype=np.float32)
            for i in range(1, n_particles):
                raw = sorted_pos[i] - sorted_pos[i - 1]
                deltas[i - 1] = raw_delta(raw, box, periodic=self.periodic)

        token_ids_np, long_jump_mask_np = self._encode_deltas(deltas)
        token_ids = torch.from_numpy(token_ids_np).long()

        seq_len = len(token_ids_np)
        sequence = torch.empty(seq_len + 1, dtype=torch.long)
        sequence[0] = int(self.sos_id)
        sequence[1:] = token_ids

        input_idx = sequence[:-1].clone()
        target_idx = sequence[1:].clone()

        if self.periodic:
            shifted_coords = np.mod(sorted_pos - sorted_pos[0], box)
        else:
            shifted_coords = sorted_pos - sorted_pos[0]

        token_coords = shifted_coords[:-1]
        if self.tokenizer.factorized:
            token_coords = _repeat_factorized_token_coords(token_coords, coord_dim=self.coord_dim)
        token_coords = token_coords.astype(np.float32, copy=False)
        if int(token_coords.shape[0]) != int(target_idx.shape[0]):
            raise ValueError(
                f"token_coords length {token_coords.shape[0]} does not match target_idx length "
                f"{target_idx.shape[0]} for factorized={self.tokenizer.factorized}."
            )

        target_deltas = deltas
        if self.polar:
            target_deltas = _cartesian_to_spherical_np(deltas)

        density = float(n_particles / max(1e-8, float(np.prod(box, dtype=np.float64))))
        max_abs_relative_displacement = float(np.max(np.abs(deltas))) if deltas.size > 0 else 0.0

        absolute_coords_t = torch.from_numpy(sorted_pos.astype(np.float32, copy=False))
        item: dict[str, Any] = {
            "sequence": sequence,
            "input_idx": input_idx,
            "target_idx": target_idx,
            "seq": target_idx,
            "deltas": torch.from_numpy(target_deltas.astype(np.float32, copy=False)),
            "absolute_coords": absolute_coords_t,
            "abs_coords": absolute_coords_t,
            "box_size": torch.from_numpy(np.asarray(box, dtype=np.float32)),
            "density": torch.tensor(density, dtype=torch.float32),
            "token_coords": torch.from_numpy(token_coords),
            "long_jump_mask": torch.from_numpy(long_jump_mask_np),
            "sample_length": torch.tensor(seq_len, dtype=torch.long),
            "particle_length": torch.tensor(n_particles, dtype=torch.long),
            "sos_id": torch.tensor(self.sos_id, dtype=torch.long),
            "vocab_size": torch.tensor(self.vocab_size, dtype=torch.long),
            "periodic": torch.tensor(self.periodic, dtype=torch.bool),
        }

        if self.use_curve_rail and int(self.coord_dim) == 3:
            R = self._resolution_for_box(box) if self.periodic else int(self.hilbert_resolution)
            waypoints_np, arclen_np = self._curve_waypoints_3d(sorted_pos, box, R)
            # Align to the N-1 conditioning tokens (same axis as token_coords/target_idx).
            if self.tokenizer.factorized:
                waypoints_np = _repeat_factorized_curve_waypoints(waypoints_np, coord_dim=self.coord_dim)
            item["curve_waypoints"] = torch.from_numpy(waypoints_np)          # [N-1, K, D]
            item["curve_arclen"] = torch.from_numpy(arclen_np)                # [K]

        return item, max_abs_relative_displacement

    def _build_batch_3d(
        self,
        coords_chunk: np.ndarray,
        box: np.ndarray,
        shift: Optional[np.ndarray] = None,
        rot_matrix: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        if int(self.coord_dim) != 3:
            raise ValueError(f"_build_batch_3d requires coord_dim=3, got {self.coord_dim}")
        if self.ordering not in ("hilbert", "gilbert"):
            raise ValueError(f"_build_batch_3d requires ordering='hilbert' or 'gilbert', got {self.ordering!r}")
        if coords_chunk.ndim != 3 or coords_chunk.shape[-1] != 3:
            raise ValueError(
                f"coords_chunk must have shape [B, N, 3], got {tuple(coords_chunk.shape)}"
            )

        working = np.asarray(coords_chunk, dtype=np.float32)
        box = np.asarray(box, dtype=np.float32)
        batch_size, n_particles, _ = working.shape
        if rot_matrix is not None:
            rot = np.asarray(rot_matrix, dtype=np.float32)
            if rot.shape != (3, 3):
                raise ValueError(f"rot_matrix must have shape (3, 3), got {tuple(rot.shape)}")
            working = np.asarray(working @ rot.T, dtype=np.float32)

        if shift is not None:
            if not self.periodic:
                raise ValueError("Explicit shift is only supported for periodic _build_batch_3d inputs.")
            shift_arr = np.asarray(shift, dtype=np.float32)
            if shift_arr.shape == (3,):
                shift_arr = shift_arr.reshape(1, 1, 3)
            elif shift_arr.shape == (batch_size, 3):
                shift_arr = shift_arr.reshape(batch_size, 1, 3)
            elif shift_arr.shape != (batch_size, 1, 3):
                raise ValueError(
                    f"shift must have shape (3,), (B,3), or (B,1,3); got {tuple(shift_arr.shape)}"
                )
            shifted = np.mod(working + shift_arr, box[None, None, :])
        elif self.periodic and self.random_grid_shift:
            shift = (
                self.rng.uniform(low=0.0, high=1.0, size=(batch_size, 1, 3)).astype(np.float32)
                * box[None, None, :]
            )
            shifted = np.mod(working + shift, box[None, None, :])
        elif self.periodic:
            shifted = np.mod(working, box[None, None, :])
        else:
            shifted = working

        if self.periodic:
            grid = self._grid_coords_periodic(shifted, box)
            R = self._resolution_for_box(box)
        else:
            grid = self._grid_coords_nonperiodic(shifted)
            R = int(self.hilbert_resolution)
        h_codes = self._space_filling_codes_3d(grid, R)
        order = np.argsort(h_codes, axis=-1, kind="stable")
        sorted_pos = np.take_along_axis(shifted, order[..., None], axis=1)

        if self.use_curve_rail and self.curve_rail_residual_target and n_particles >= 2:
            # Rail-anchored RESIDUAL target (see _build_item_from_index); anchors are
            # sample-independent so broadcast across the batch.
            anchors = fixed_template_anchors(np.arange(1, n_particles), n_particles, R, box)  # [N-1,3]
            deltas = raw_delta(
                sorted_pos[:, 1:, :] - anchors[None], box, periodic=self.periodic
            ).astype(np.float32, copy=False)
        else:
            raw_diffs = sorted_pos[:, 1:, :] - sorted_pos[:, :-1, :]
            deltas = raw_delta(raw_diffs, box, periodic=self.periodic).astype(np.float32, copy=False)

        token_ids, long_jump_mask = self._encode_deltas(deltas)
        token_ids = token_ids.astype(np.int64, copy=False)
        long_jump_mask = long_jump_mask.astype(np.bool_, copy=False)

        sequence = np.empty((batch_size, token_ids.shape[1] + 1), dtype=np.int64)
        sequence[:, 0] = int(self.sos_id)
        sequence[:, 1:] = token_ids
        input_idx = sequence[:, :-1].copy()
        target_idx = sequence[:, 1:].copy()

        if self.periodic:
            shifted_coords = np.mod(sorted_pos - sorted_pos[:, 0:1, :], box)
        else:
            shifted_coords = sorted_pos - sorted_pos[:, 0:1, :]
        token_coords = shifted_coords[:, :-1, :]
        if self.tokenizer.factorized:
            token_coords = _repeat_factorized_token_coords(token_coords, coord_dim=self.coord_dim)
        token_coords = token_coords.astype(np.float32, copy=False)
        if int(token_coords.shape[1]) != int(target_idx.shape[1]):
            raise ValueError(
                f"token_coords length {token_coords.shape[1]} does not match target_idx length "
                f"{target_idx.shape[1]} for factorized={self.tokenizer.factorized}."
            )

        target_deltas = deltas
        if self.polar:
            target_deltas = _cartesian_to_spherical_np(deltas)

        density = np.full(
            (batch_size,),
            float(n_particles / max(1e-8, float(np.prod(box, dtype=np.float64)))),
            dtype=np.float32,
        )
        sample_length = np.full((batch_size,), int(token_ids.shape[1]), dtype=np.int64)
        delta_length = np.full((batch_size,), int(deltas.shape[1]), dtype=np.int64)
        box_size = np.broadcast_to(box[None, :], (batch_size, box.shape[0])).astype(np.float32, copy=True)
        max_abs_relative_displacement = (
            float(np.max(np.abs(deltas))) if deltas.size > 0 else 0.0
        )

        out: dict[str, np.ndarray] = {
            "sequence": sequence,
            "input_idx": input_idx,
            "target_idx": target_idx,
            "deltas": target_deltas,
            "absolute_coords": sorted_pos.astype(np.float32, copy=False),
            "token_coords": token_coords,
            "long_jump_mask": long_jump_mask,
            "box_size": box_size,
            "density": density,
            "sample_length": sample_length,
            "delta_length": delta_length,
            "particle_length": np.full((batch_size,), int(n_particles), dtype=np.int64),
            "max_abs_relative_displacement": np.array(max_abs_relative_displacement, dtype=np.float32),
        }

        if self.use_curve_rail:
            cell_size = np.asarray(box, dtype=np.float64) / float(R)
            if self.curve_rail_mode == "fixed_template":
                # Sample-INDEPENDENT template: identical for every config (depends only on
                # N, R, j), so broadcast one [N-1, K, 3] template across the batch.
                X = max(1, (R ** 3) // int(n_particles))
                pred = np.arange(1, n_particles, dtype=np.int64)
                tmpl = fixed_template_waypoints(
                    pred, n_particles, R, box, k=int(self.curve_rail_k),
                    window_scale=self.curve_rail_window, periodic=self.periodic,
                    reference=self.curve_rail_reference,
                )  # [N-1, K, 3]
                out["curve_waypoints"] = np.broadcast_to(
                    tmpl[None], (batch_size,) + tmpl.shape
                ).copy()
                if self.tokenizer.factorized:
                    out["curve_waypoints"] = _repeat_factorized_curve_waypoints(
                        out["curve_waypoints"], coord_dim=self.coord_dim
                    )
                out["curve_arclen"] = np.full(
                    (int(self.curve_rail_k),),
                    float(self.curve_rail_window * X * float(np.mean(cell_size))),
                    dtype=np.float32,
                )
            else:
                sorted_codes = np.take_along_axis(h_codes, order, axis=1)  # [B, N] ascending
                out["curve_waypoints"] = _rail_relative_from_codes(
                    sorted_codes[:, :-1], sorted_pos[:, :-1, :], box, R, self.curve_rail_offsets,
                    periodic=self.periodic,
                )  # [B, N-1, K, 3]
                if self.tokenizer.factorized:
                    out["curve_waypoints"] = _repeat_factorized_curve_waypoints(
                        out["curve_waypoints"], coord_dim=self.coord_dim
                    )
                out["curve_arclen"] = (
                    self.curve_rail_offsets.astype(np.float64) * float(np.mean(cell_size))
                ).astype(np.float32)

        return out

    def __getitem__(self, idx: int):
        item, _ = self._build_item_from_index(idx)
        return item


class LJTransferableCachedDataset(Dataset):
    """
    Cached/tokenized lj_transferable dataset loaded from a preprocessed .pt file.

    The cache stores padded tensors and per-sample lengths; __getitem__ trims each
    sample to match the output schema of LJTransferableDataset.
    """

    def __init__(
        self,
        cache_path: str,
        *,
        limit: Optional[int] = None,
        discrete: bool = False,
        arc_repr: bool = False,
        codebook_path: Optional[str] = None,
    ) -> None:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid cache format in {cache_path}: expected dict")

        required_keys = (
            "sequence",
            "input_idx",
            "target_idx",
            "deltas",
            "delta_length",
            "token_coords",
            "long_jump_mask",
            "box_size",
            "density",
            "sample_length",
            "vocab_size",
            "sos_id",
        )
        missing = [k for k in required_keys if k not in payload]
        if missing:
            raise KeyError(f"Missing keys in cache {cache_path}: {missing}")

        self.sequence_all = torch.as_tensor(payload["sequence"], dtype=torch.long)
        self.input_idx_all = torch.as_tensor(payload["input_idx"], dtype=torch.long)
        self.target_idx_all = torch.as_tensor(payload["target_idx"], dtype=torch.long)
        is_discrete_cache = bool(dict(payload.get("metadata", {})).get("is_discrete", False))
        self.deltas_all = torch.as_tensor(
            payload["deltas"],
            dtype=(torch.long if is_discrete_cache else torch.float32),
        )
        self.delta_length_all = torch.as_tensor(payload["delta_length"], dtype=torch.long)
        self.token_coords_all = torch.as_tensor(payload["token_coords"], dtype=torch.float32)
        self.long_jump_mask_all = torch.as_tensor(payload["long_jump_mask"], dtype=torch.bool)
        self.box_size_all = torch.as_tensor(payload["box_size"], dtype=torch.float32)
        self.density_all = torch.as_tensor(payload["density"], dtype=torch.float32)
        self.sample_length_all = torch.as_tensor(payload["sample_length"], dtype=torch.long)
        self.particle_length_all = torch.as_tensor(
            payload.get("particle_length", torch.full_like(self.sample_length_all, -1)),
            dtype=torch.long,
        )
        self.absolute_coords_all = None
        if "absolute_coords" in payload:
            self.absolute_coords_all = torch.as_tensor(payload["absolute_coords"], dtype=torch.float32)
        self.target_energy_all = None
        if "target_energy" in payload:
            self.target_energy_all = torch.as_tensor(payload["target_energy"], dtype=torch.float32)
        self.metadata: dict[str, Any] = dict(payload.get("metadata", {}))
        self.curve_rail_mode = str(self.metadata.get("curve_rail_mode", "lookahead") or "lookahead")
        # Optional curve-rail look-ahead scaffold (absent in pre-v13 caches).
        self.curve_waypoints_all = None
        if "curve_waypoints" in payload:
            self.curve_waypoints_all = torch.as_tensor(payload["curve_waypoints"], dtype=torch.float32)

        if self.sample_length_all.ndim != 1:
            raise ValueError(f"sample_length must be rank-1, got {tuple(self.sample_length_all.shape)}")
        n_samples = int(self.sample_length_all.shape[0])
        for name, t in (
            ("sequence", self.sequence_all),
            ("input_idx", self.input_idx_all),
            ("target_idx", self.target_idx_all),
            ("deltas", self.deltas_all),
            ("delta_length", self.delta_length_all),
            ("token_coords", self.token_coords_all),
            ("long_jump_mask", self.long_jump_mask_all),
            ("box_size", self.box_size_all),
            ("density", self.density_all),
            ("particle_length", self.particle_length_all),
        ):
            if t.shape[0] != n_samples:
                raise ValueError(f"{name} has first dim {t.shape[0]} but expected {n_samples}")
        if self.absolute_coords_all is not None and self.absolute_coords_all.shape[0] != n_samples:
            raise ValueError(
                f"absolute_coords has first dim {self.absolute_coords_all.shape[0]} but expected {n_samples}"
            )
        if self.target_energy_all is not None and self.target_energy_all.shape[0] != n_samples:
            raise ValueError(
                f"target_energy has first dim {self.target_energy_all.shape[0]} but expected {n_samples}"
            )
        if self.curve_waypoints_all is not None and self.curve_waypoints_all.shape[0] != n_samples:
            raise ValueError(
                f"curve_waypoints has first dim {self.curve_waypoints_all.shape[0]} but expected {n_samples}"
            )

        self.sample_lengths = self.sample_length_all.numpy().astype(np.int64, copy=False)
        if limit is not None:
            lim = max(1, min(int(limit), n_samples))
            self.sequence_all = self.sequence_all[:lim]
            self.input_idx_all = self.input_idx_all[:lim]
            self.target_idx_all = self.target_idx_all[:lim]
            self.deltas_all = self.deltas_all[:lim]
            self.delta_length_all = self.delta_length_all[:lim]
            self.token_coords_all = self.token_coords_all[:lim]
            self.long_jump_mask_all = self.long_jump_mask_all[:lim]
            self.box_size_all = self.box_size_all[:lim]
            self.density_all = self.density_all[:lim]
            self.sample_length_all = self.sample_length_all[:lim]
            self.particle_length_all = self.particle_length_all[:lim]
            self.sample_lengths = self.sample_lengths[:lim]
            if self.absolute_coords_all is not None:
                self.absolute_coords_all = self.absolute_coords_all[:lim]
            if self.target_energy_all is not None:
                self.target_energy_all = self.target_energy_all[:lim]
            if self.curve_waypoints_all is not None:
                self.curve_waypoints_all = self.curve_waypoints_all[:lim]

        self.periodic = bool(self.metadata.get("periodic", True))
        # Curve-rail geometry (so the sampler can recompute the rail at generation time).
        self.hilbert_resolution = int(self.metadata.get("hilbert_resolution", 128))
        _cs = self.metadata.get("cell_size", None)
        self.cell_size = None if _cs is None else float(_cs)
        _rail_offs = self.metadata.get("curve_rail_offsets", None)
        self.curve_rail_offsets = (
            np.asarray(_rail_offs, dtype=np.int64) if _rail_offs is not None else None
        )
        self.curve_rail_window = float(self.metadata.get("curve_rail_window", 1.0) or 1.0)
        self.curve_rail_reference = str(self.metadata.get("curve_rail_reference", "absolute") or "absolute")
        self.curve_rail_residual_target = bool(self.metadata.get("curve_rail_residual_target", False))
        _ck = self.metadata.get("curve_rail_k", None)
        self.curve_rail_k = (
            int(_ck) if _ck is not None
            else (int(self.curve_waypoints_all.shape[2]) if self.curve_waypoints_all is not None else 0)
        )
        self.coord_dim = int(self.token_coords_all.shape[-1])
        self.factorized = bool(self.metadata.get("factorized", False))
        self.polar = bool(self.metadata.get("polar", False))
        self.discrete = bool(self.metadata.get("is_discrete", self.metadata.get("discrete", False)))
        self.use_curve_rail = bool(self.metadata.get("use_curve_rail", False)) and (
            self.curve_waypoints_all is not None or self.curve_rail_mode == "fixed_template"
        )
        self._fixed_curve_waypoints_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self.codebook_path = self.metadata.get("codebook_path")
        self.vocab_size = int(payload["vocab_size"])
        self.sos_id = int(payload["sos_id"])

        if bool(discrete):
            if not is_discrete_cache:
                raise ValueError(
                    "Dataset initialized with discrete=True, but the cache is continuous. "
                    "Please run preprocess_lj_transferable.py with --discrete to generate an offline discrete cache."
                )

        self.arc_repr = bool(arc_repr)
        if self.arc_repr:
            _validate_arc_repr_cache(
                ordering=str(self.metadata.get("ordering", "hilbert") or "hilbert"),
                periodic=self.periodic,
                has_absolute_coords=self.absolute_coords_all is not None,
            )

    def __len__(self) -> int:
        return int(self.sample_length_all.shape[0])

    def _resolution_for_box(self, box: np.ndarray) -> int:
        """Grid resolution for a box — delegates to ``resolution_for_box_rule``."""
        ordering = str(self.metadata.get("ordering", "hilbert") or "hilbert").strip().lower()
        return resolution_for_box_rule(
            box,
            cell_size=self.cell_size,
            hilbert_resolution=self.hilbert_resolution,
            ordering=ordering,
        )

    def _fixed_template_curve_waypoints_for_sample(
        self,
        *,
        box: torch.Tensor,
        particle_len: int,
        seq_len: int,
    ) -> torch.Tensor:
        box_np = box.detach().cpu().numpy().astype(np.float32, copy=False)
        box_key = tuple(round(float(x), 8) for x in box_np.reshape(-1))
        key = (
            int(particle_len),
            int(seq_len),
            box_key,
            int(self.curve_rail_k),
            float(self.curve_rail_window),
            str(self.curve_rail_reference),
            bool(self.factorized),
            int(self.coord_dim),
            int(self._resolution_for_box(box_np)),
        )
        cached = self._fixed_curve_waypoints_cache.get(key)
        if cached is not None:
            return cached
        n_particles = int(particle_len)
        R = int(key[-1])
        waypoints_np = fixed_template_waypoints(
            np.arange(1, n_particles, dtype=np.int64),
            n_particles,
            R,
            box_np,
            k=int(self.curve_rail_k),
            window_scale=float(self.curve_rail_window),
            periodic=bool(self.periodic),
            reference=str(self.curve_rail_reference),
        )
        if self.factorized:
            waypoints_np = _repeat_factorized_curve_waypoints(waypoints_np, coord_dim=int(self.coord_dim))
        waypoints = torch.from_numpy(waypoints_np).to(dtype=torch.float32)[:seq_len]
        self._fixed_curve_waypoints_cache[key] = waypoints
        return waypoints

    def __getitem__(self, idx: int):
        seq_len = int(self.sample_length_all[idx].item())
        delta_len = int(self.delta_length_all[idx].item())
        deltas_item = (
            self.deltas_all[idx, :delta_len]
            if self.deltas_all.ndim == 2
            else self.deltas_all[idx, :delta_len, :]
        )
        item = {
            "sequence": self.sequence_all[idx, : seq_len + 1],
            "input_idx": self.input_idx_all[idx, :seq_len],
            "target_idx": self.target_idx_all[idx, :seq_len],
            "seq": self.target_idx_all[idx, :seq_len],
            "deltas": deltas_item,
            "box_size": self.box_size_all[idx],
            "density": self.density_all[idx],
            "token_coords": self.token_coords_all[idx, :seq_len, :],
            "long_jump_mask": self.long_jump_mask_all[idx, :seq_len],
            "sample_length": self.sample_length_all[idx],
            "particle_length": self.particle_length_all[idx],
            "sos_id": torch.tensor(self.sos_id, dtype=torch.long),
            "vocab_size": torch.tensor(self.vocab_size, dtype=torch.long),
            "periodic": torch.tensor(self.periodic, dtype=torch.bool),
        }
        if self.absolute_coords_all is not None:
            particle_len = int(self.particle_length_all[idx].item())
            if particle_len <= 0:
                particle_len = int(self.absolute_coords_all.shape[1])
            abs_coords = self.absolute_coords_all[idx, :particle_len, :]
            item["absolute_coords"] = abs_coords
            item["abs_coords"] = abs_coords
        if self.target_energy_all is not None:
            item["target_energy"] = self.target_energy_all[idx]
        if self.curve_waypoints_all is not None:
            item["curve_waypoints"] = self.curve_waypoints_all[idx, :seq_len]
        elif self.use_curve_rail and self.curve_rail_mode == "fixed_template":
            particle_len = int(self.particle_length_all[idx].item())
            if particle_len <= 0:
                particle_len = int(self.absolute_coords_all.shape[1]) if self.absolute_coords_all is not None else seq_len + 1
            item["curve_waypoints"] = self._fixed_template_curve_waypoints_for_sample(
                box=self.box_size_all[idx],
                particle_len=particle_len,
                seq_len=seq_len,
            )
        if self.arc_repr:
            from .curves import get_curve3d  # lazy to avoid circular import

            particle_len = int(self.particle_length_all[idx].item())
            if particle_len <= 0:
                particle_len = int(self.absolute_coords_all.shape[1])
            abs_coords_np = self.absolute_coords_all[idx, :particle_len, :].numpy()
            box_np = self.box_size_all[idx].numpy()
            R = self._resolution_for_box(box_np)
            ordering = str(self.metadata.get("ordering", "hilbert") or "hilbert").strip().lower()
            curve = get_curve3d("gilbert" if ordering == "gilbert" else "hilbert", R)
            cell_size_np = box_np.astype(np.float64) / float(R)
            grid = np.floor(
                np.mod(abs_coords_np, box_np[None, :]).astype(np.float64) / cell_size_np
            ).astype(np.int64)
            grid = np.clip(grid, 0, R - 1)
            codes = curve.encode(grid)
            item["arc_delta"] = torch.from_numpy(
                hilbert_arc_delta(abs_coords_np, codes, box_np, R, periodic=self.periodic, curve=curve)
            )  # [N-1, 4]
        return item


def build_lj_transferable_cache(
    *,
    file_paths: Optional[Sequence[str]] = None,
    h5_path: Optional[str] = None,
    output_path: str,
    periodic: bool = True,
    hilbert_resolution: int = 128,
    cell_size: Optional[float] = None,
    ordering: str = "hilbert",
    spectral_sigma: float = 1.0,
    local_window: float = 3.0,
    local_bins: int = 64,
    use_long_jump_token: bool = True,
    factorized: bool = False,
    polar: bool = False,
    discrete: bool = False,
    codebook_path: Optional[str] = None,
    random_grid_shift: bool = False,
    augment_90deg_rotations: bool = False,
    augment_torus_shift: bool = True,
    num_augmentations: int = 5,
    energy_chunk_size: int = 2048,
    cache_build_chunk_size: int = 16384,
    limit: Optional[int] = None,
    seed: int = 0,
    use_curve_rail: bool = False,
    curve_rail_offsets: Optional[Sequence[int]] = None,
    curve_rail_mode: str = "lookahead",
    curve_rail_window: float = 1.0,
    curve_rail_k: int = 8,
    curve_rail_reference: str = "absolute",
    curve_rail_residual_target: bool = False,
    target_system: str = "lj",
    lj_epsilon: float = 1.0,
    lj_sigma: float = 1.0,
    lj_cutoff: Optional[float] = None,
    lj_spring_constant: float = 0.5,
    dw_a: float = 0.9,
    dw_b: float = -4.0,
    dw_c: float = 0.0,
    dw_offset: float = 4.0,
) -> dict[str, Any]:
    """
    Precompute tokenized lj_transferable samples and store them in a .pt cache.

    Returns a short summary dict.
    """
    dataset = LJTransferableDataset(
        file_paths=file_paths,
        h5_path=h5_path,
        periodic=bool(periodic),
        hilbert_resolution=int(hilbert_resolution),
        cell_size=cell_size,
        ordering=str(ordering),
        spectral_sigma=float(spectral_sigma),
        local_window=float(local_window),
        local_bins=int(local_bins),
        use_long_jump_token=bool(use_long_jump_token),
        factorized=bool(factorized),
        polar=bool(polar),
        discrete=False,
        codebook_path=None,
        random_grid_shift=bool(random_grid_shift),
        limit=limit,
        seed=int(seed),
        use_curve_rail=bool(use_curve_rail),
        curve_rail_offsets=curve_rail_offsets,
        curve_rail_mode=str(curve_rail_mode),
        curve_rail_window=float(curve_rail_window),
        curve_rail_k=int(curve_rail_k),
        curve_rail_reference=str(curve_rail_reference),
        curve_rail_residual_target=bool(curve_rail_residual_target),
    )
    base_n_samples = len(dataset)
    if base_n_samples <= 0:
        raise RuntimeError("No samples found for cache build.")
    if int(num_augmentations) <= 0:
        raise ValueError(f"num_augmentations must be positive, got {num_augmentations}")
    if int(energy_chunk_size) <= 0:
        raise ValueError(f"energy_chunk_size must be positive, got {energy_chunk_size}")
    if int(cache_build_chunk_size) <= 0:
        raise ValueError(f"cache_build_chunk_size must be positive, got {cache_build_chunk_size}")
    if bool(discrete) and bool(factorized):
        raise ValueError("Offline discrete cache generation requires factorized=False.")
    if bool(discrete) and bool(polar):
        raise ValueError("Offline discrete cache generation requires polar=False.")
    if bool(discrete) and codebook_path is None:
        raise ValueError("Offline discrete cache generation requires codebook_path.")
    if augment_90deg_rotations and int(dataset.coord_dim) != 2:
        raise ValueError(
            "augment_90deg_rotations=True is only supported for 2D lj_transferable data."
        )
    rotation_indices = [None]
    if augment_90deg_rotations:
        rotation_indices = [0, 1, 2, 3]

    can_batch_3d = (
        int(dataset.coord_dim) == 3
        and (not augment_90deg_rotations)
        and dataset.ordering in ("hilbert", "gilbert")
    )
    effective_num_augmentations = int(num_augmentations) if (can_batch_3d and bool(dataset.periodic)) else 1
    n_samples = int(base_n_samples * len(rotation_indices) * effective_num_augmentations)

    max_particles = int(dataset.sample_lengths.max())
    max_delta_len = int(max(0, max_particles - 1))
    max_seq_len = int(max_delta_len)
    if dataset.tokenizer.factorized:
        max_seq_len *= int(dataset.coord_dim)

    sequence = torch.empty((n_samples, max_seq_len + 1), dtype=torch.long)
    input_idx = torch.empty((n_samples, max_seq_len), dtype=torch.long)
    target_idx = torch.empty((n_samples, max_seq_len), dtype=torch.long)
    deltas = torch.empty((n_samples, max_delta_len, int(dataset.coord_dim)), dtype=torch.float32)
    absolute_coords = torch.empty((n_samples, max_particles, int(dataset.coord_dim)), dtype=torch.float32)
    token_coords = torch.empty((n_samples, max_seq_len, int(dataset.coord_dim)), dtype=torch.float32)
    long_jump_mask = torch.empty((n_samples, max_seq_len), dtype=torch.bool)
    box_size = torch.empty((n_samples, int(dataset.coord_dim)), dtype=torch.float32)
    density = torch.empty((n_samples,), dtype=torch.float32)
    sample_length = torch.empty((n_samples,), dtype=torch.long)
    delta_length = torch.empty((n_samples,), dtype=torch.long)
    particle_length = torch.empty((n_samples,), dtype=torch.long)
    target_energy = torch.empty((n_samples,), dtype=torch.float32)
    use_rail = bool(use_curve_rail)
    rail_k = int(dataset.curve_rail_k) if use_rail else 0
    compact_fixed_template_rail = bool(use_rail and dataset.curve_rail_mode == "fixed_template")
    curve_waypoints = (
        torch.zeros((n_samples, max_seq_len, rail_k, int(dataset.coord_dim)), dtype=torch.float32)
        if use_rail and not compact_fixed_template_rail
        else None
    )
    max_abs_relative_displacement = 0.0

    out_idx = 0
    if can_batch_3d:
        for file_idx in np.unique(dataset.sample_id_to_file_index):
            sample_ids = np.nonzero(dataset.sample_id_to_file_index == file_idx)[0].astype(np.int64, copy=False)
            if sample_ids.size == 0:
                continue
            local_ids = dataset.sample_id_to_local_index[sample_ids]
            coords_file = dataset._coords_by_file[int(file_idx)]
            box = dataset._box_by_file[int(file_idx)]
            if not np.all(sample_ids == np.arange(sample_ids[0], sample_ids[0] + sample_ids.size)):
                raise RuntimeError("Expected contiguous sample ids per file in batched cache build.")
            for chunk_start in range(0, int(sample_ids.size), int(cache_build_chunk_size)):
                chunk_end = min(chunk_start + int(cache_build_chunk_size), int(sample_ids.size))
                local_ids_chunk = local_ids[chunk_start:chunk_end]
                coords_chunk = coords_file[local_ids_chunk]
                energies_chunk = _compute_target_energy_batch(
                    coords_chunk,
                    box=box,
                    periodic=bool(periodic),
                    target_system=str(target_system),
                    lj_epsilon=float(lj_epsilon),
                    lj_sigma=float(lj_sigma),
                    lj_cutoff=None if lj_cutoff is None else float(lj_cutoff),
                    lj_spring_constant=float(lj_spring_constant),
                    dw_a=float(dw_a),
                    dw_b=float(dw_b),
                    dw_c=float(dw_c),
                    dw_offset=float(dw_offset),
                    chunk_size=int(energy_chunk_size),
                )
                batch_len = int(coords_chunk.shape[0])
                for aug_idx in range(effective_num_augmentations):
                    shift = None
                    rot_matrix = None
                    if periodic and augment_torus_shift:
                        # Both batched augmentations are gated by ``augment_torus_shift``
                        # (default on, preserving prior behavior). The continuous torus
                        # shift and the random 90-degree rotation are legit symmetries for
                        # normal LJ training, but both move particles to *different* Hilbert
                        # cells/indices, which destroys the exact, evenly-spaced cell
                        # alignment of toy data (e.g. the Hilbert-rail toy). Turn the flag
                        # off to keep cell-aligned data verbatim.
                        shift = (
                            dataset.rng.uniform(low=0.0, high=1.0, size=(batch_len, 1, 3)).astype(np.float32)
                            * box[None, None, :]
                        )
                        rot_matrix = _rotation_matrix_3d(
                            int(dataset.rng.integers(0, 4)),
                            int(dataset.rng.integers(0, 4)),
                            int(dataset.rng.integers(0, 4)),
                        )
                    batch = dataset._build_batch_3d(coords_chunk, box, shift=shift, rot_matrix=rot_matrix)
                    max_abs_relative_displacement = max(
                        max_abs_relative_displacement,
                        float(batch["max_abs_relative_displacement"]),
                    )

                    dest_start = out_idx
                    dest_end = dest_start + batch_len
                    n = int(batch["sample_length"][0])
                    n_delta = int(batch["delta_length"][0])
                    sequence[dest_start:dest_end].fill_(int(dataset.sos_id))
                    input_idx[dest_start:dest_end].zero_()
                    target_idx[dest_start:dest_end].zero_()
                    deltas[dest_start:dest_end].zero_()
                    absolute_coords[dest_start:dest_end].zero_()
                    token_coords[dest_start:dest_end].zero_()
                    long_jump_mask[dest_start:dest_end].zero_()

                    sequence[dest_start:dest_end, : n + 1] = torch.from_numpy(batch["sequence"])
                    input_idx[dest_start:dest_end, :n] = torch.from_numpy(batch["input_idx"])
                    target_idx[dest_start:dest_end, :n] = torch.from_numpy(batch["target_idx"])
                    deltas[dest_start:dest_end, :n_delta, :] = torch.from_numpy(batch["deltas"])
                    absolute_coords[dest_start:dest_end, : int(batch["particle_length"][0]), :] = torch.from_numpy(
                        batch["absolute_coords"]
                    )
                    token_coords[dest_start:dest_end, :n] = torch.from_numpy(batch["token_coords"])
                    if curve_waypoints is not None:
                        curve_waypoints[dest_start:dest_end, :n] = torch.from_numpy(batch["curve_waypoints"])
                    long_jump_mask[dest_start:dest_end, :n] = torch.from_numpy(batch["long_jump_mask"])
                    box_size[dest_start:dest_end] = torch.from_numpy(batch["box_size"])
                    density[dest_start:dest_end] = torch.from_numpy(batch["density"])
                    sample_length[dest_start:dest_end] = torch.from_numpy(batch["sample_length"])
                    delta_length[dest_start:dest_end] = torch.from_numpy(batch["delta_length"])
                    particle_length[dest_start:dest_end] = torch.from_numpy(batch["particle_length"])
                    target_energy[dest_start:dest_end] = torch.from_numpy(
                        np.asarray(energies_chunk, dtype=np.float32)
                    )
                    out_idx = dest_end
    else:
        base_target_energy = np.empty((base_n_samples,), dtype=np.float32)
        for file_idx in np.unique(dataset.sample_id_to_file_index):
            sample_ids = np.nonzero(dataset.sample_id_to_file_index == file_idx)[0].astype(np.int64, copy=False)
            if sample_ids.size == 0:
                continue
            local_ids = dataset.sample_id_to_local_index[sample_ids]
            coords_all = dataset._coords_by_file[int(file_idx)][local_ids]
            box = dataset._box_by_file[int(file_idx)]
            base_target_energy[sample_ids] = _compute_target_energy_batch(
                coords_all,
                box=box,
                periodic=bool(periodic),
                target_system=str(target_system),
                lj_epsilon=float(lj_epsilon),
                lj_sigma=float(lj_sigma),
                lj_cutoff=None if lj_cutoff is None else float(lj_cutoff),
                lj_spring_constant=float(lj_spring_constant),
                dw_a=float(dw_a),
                dw_b=float(dw_b),
                dw_c=float(dw_c),
                dw_offset=float(dw_offset),
                chunk_size=int(energy_chunk_size),
            )
        for sample_idx in range(base_n_samples):
            for rotation_k in rotation_indices:
                item, sample_max_abs = dataset._build_item_from_index(sample_idx, rotation_k=rotation_k)
                max_abs_relative_displacement = max(max_abs_relative_displacement, float(sample_max_abs))
                n = int(item["sample_length"].item())
                sequence[out_idx].fill_(int(dataset.sos_id))
                input_idx[out_idx].zero_()
                target_idx[out_idx].zero_()
                deltas[out_idx].zero_()
                absolute_coords[out_idx].zero_()
                token_coords[out_idx].zero_()
                long_jump_mask[out_idx].zero_()

                sequence[out_idx, : n + 1] = item["sequence"]
                input_idx[out_idx, :n] = item["input_idx"]
                target_idx[out_idx, :n] = item["target_idx"]
                n_delta = int(item["deltas"].shape[0])
                deltas[out_idx, :n_delta, :] = item["deltas"]
                particle_len = int(item["particle_length"].item())
                absolute_coords[out_idx, :particle_len, :] = item["absolute_coords"]
                token_coords[out_idx, :n, :] = item["token_coords"]
                if curve_waypoints is not None:
                    curve_waypoints[out_idx, :n] = item["curve_waypoints"]
                long_jump_mask[out_idx, :n] = item["long_jump_mask"]
                box_size[out_idx] = item["box_size"]
                density[out_idx] = item["density"]
                sample_length[out_idx] = item["sample_length"]
                delta_length[out_idx] = n_delta
                particle_length[out_idx] = particle_len
                target_energy[out_idx] = float(base_target_energy[sample_idx])
                out_idx += 1

    vocab_size_value = int(dataset.vocab_size)
    sos_id_value = int(dataset.sos_id)
    deltas_payload: torch.Tensor = deltas
    if bool(discrete):
        codebook_size_value = int(_load_codebook_tensor(str(codebook_path)).shape[0])
        discrete_ids, reconstructed_coords, reconstructed_token_coords = _offline_discretize_cache_tensors(
            deltas=deltas,
            delta_length=delta_length,
            particle_length=particle_length,
            box_size=box_size,
            codebook_path=str(codebook_path),
            periodic=bool(periodic),
            chunk_size=int(cache_build_chunk_size),
        )
        target_idx = discrete_ids.to(dtype=torch.long)
        input_idx = torch.full_like(target_idx, fill_value=codebook_size_value)
        if target_idx.shape[1] > 0:
            input_idx[:, 0] = codebook_size_value
        if target_idx.shape[1] > 1:
            input_idx[:, 1:] = target_idx[:, :-1]
        sequence = torch.full(
            (target_idx.shape[0], target_idx.shape[1] + 1),
            fill_value=codebook_size_value,
            dtype=torch.long,
        )
        sequence[:, 1:] = target_idx
        long_jump_mask.zero_()
        sample_length = delta_length.clone()
        absolute_coords = reconstructed_coords
        token_coords = reconstructed_token_coords
        deltas_payload = discrete_ids.to(dtype=torch.long)
        vocab_size_value = codebook_size_value
        sos_id_value = vocab_size_value

    metadata = {
        "file_paths": list(dataset.file_paths),
        "periodic": bool(periodic),
        "ordering": str(ordering),
        "spectral_sigma": float(spectral_sigma),
        "hilbert_resolution": int(hilbert_resolution),
        "local_window": float(local_window),
        "local_bins": int(local_bins),
        "coord_dim": int(dataset.coord_dim),
        "use_long_jump_token": bool(use_long_jump_token),
        "factorized": bool(factorized),
        "polar": bool(polar),
        "discrete": bool(discrete),
        "is_discrete": bool(discrete),
        "codebook_path": None if codebook_path is None else str(codebook_path),
        "codebook_size": int(vocab_size_value) if bool(discrete) else None,
        "random_grid_shift": bool(random_grid_shift),
        "use_data_aug": False,
        "augment_90deg_rotations": bool(augment_90deg_rotations),
        "num_augmentations": int(effective_num_augmentations),
        "energy_chunk_size": int(energy_chunk_size),
        "cache_build_chunk_size": int(cache_build_chunk_size),
        "rotation_count": int(len(rotation_indices)),
        "base_n_samples": int(base_n_samples),
        "max_abs_relative_displacement": float(max_abs_relative_displacement),
        "seed": int(seed),
        "target_system": str(target_system),
        "lj_epsilon": float(lj_epsilon),
        "lj_sigma": float(lj_sigma),
        "lj_cutoff": None if lj_cutoff is None else float(lj_cutoff),
        "lj_spring_constant": float(lj_spring_constant),
        "dw_a": float(dw_a),
        "dw_b": float(dw_b),
        "dw_c": float(dw_c),
        "dw_offset": float(dw_offset),
        "periodic_box_convention": "zero_to_L",
        "relative_anchor_convention": "first_particle_origin_chain",
        "use_curve_rail": bool(use_rail),
        "curve_rail_offsets": dataset.curve_rail_offsets.tolist() if use_rail else None,
        "curve_rail_mode": str(dataset.curve_rail_mode) if use_rail else None,
        "curve_rail_window": float(dataset.curve_rail_window) if use_rail else None,
        "curve_rail_k": int(dataset.curve_rail_k) if use_rail else None,
        "curve_rail_reference": str(dataset.curve_rail_reference) if use_rail else None,
        "curve_rail_residual_target": bool(dataset.curve_rail_residual_target) if use_rail else False,
        "curve_rail_storage": (
            "fixed_template_generated" if compact_fixed_template_rail
            else ("per_sample" if use_rail else None)
        ),
        "cell_size": None if dataset.cell_size is None else float(dataset.cell_size),
        "version": 14,
    }

    payload = {
        "sequence": sequence,
        "input_idx": input_idx,
        "target_idx": target_idx,
        "deltas": deltas_payload,
        "absolute_coords": absolute_coords,
        "token_coords": token_coords,
        "long_jump_mask": long_jump_mask,
        "box_size": box_size,
        "density": density,
        "sample_length": sample_length,
        "delta_length": delta_length,
        "particle_length": particle_length,
        "target_energy": target_energy,
        "vocab_size": int(vocab_size_value),
        "sos_id": int(sos_id_value),
        "metadata": metadata,
    }
    if curve_waypoints is not None:
        payload["curve_waypoints"] = curve_waypoints
    torch.save(payload, output_path)
    return {
        "output_path": output_path,
        "base_n_samples": int(base_n_samples),
        "n_samples": n_samples,
        "max_particles": max_particles,
        "max_delta_len": max_delta_len,
        "max_seq_len": max_seq_len,
        "vocab_size": int(vocab_size_value),
        "periodic": bool(periodic),
        "ordering": str(ordering),
        "factorized": bool(factorized),
        "polar": bool(polar),
        "discrete": bool(discrete),
        "random_grid_shift": bool(random_grid_shift),
        "num_augmentations": int(effective_num_augmentations),
        "energy_chunk_size": int(energy_chunk_size),
        "cache_build_chunk_size": int(cache_build_chunk_size),
        "rotation_count": int(len(rotation_indices)),
        "max_abs_relative_displacement": float(max_abs_relative_displacement),
    }


def discretize_lj_transferable_cache(
    *,
    source_cache_path: str,
    output_path: str,
    codebook_path: str,
    chunk_size: int = 16384,
) -> dict[str, Any]:
    try:
        payload = torch.load(source_cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source_cache_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid cache format in {source_cache_path}: expected dict")

    metadata = dict(payload.get("metadata", {}))
    if bool(metadata.get("is_discrete", metadata.get("discrete", False))):
        raise ValueError(
            f"Source cache is already discrete: {source_cache_path}. "
            "Expected a continuous cache as the codebook/discretization source."
        )
    if bool(metadata.get("factorized", False)):
        raise ValueError("Offline discrete cache generation requires factorized=False.")
    if bool(metadata.get("polar", False)):
        raise ValueError("Offline discrete cache generation requires polar=False.")

    required_keys = (
        "sequence",
        "input_idx",
        "target_idx",
        "deltas",
        "delta_length",
        "box_size",
        "density",
        "sample_length",
        "particle_length",
        "target_energy",
        "long_jump_mask",
    )
    missing = [k for k in required_keys if k not in payload]
    if missing:
        raise KeyError(f"Missing keys in continuous cache {source_cache_path}: {missing}")

    deltas = torch.as_tensor(payload["deltas"], dtype=torch.float32)
    delta_length = torch.as_tensor(payload["delta_length"], dtype=torch.long)
    particle_length = torch.as_tensor(payload["particle_length"], dtype=torch.long)
    box_size = torch.as_tensor(payload["box_size"], dtype=torch.float32)
    if deltas.ndim != 3:
        raise ValueError(f"Continuous cache deltas must be [B,T,D], got {tuple(deltas.shape)}")

    codebook_size_value = int(_load_codebook_tensor(str(codebook_path)).shape[0])
    periodic = bool(metadata.get("periodic", True))
    discrete_ids, reconstructed_coords, reconstructed_token_coords = _offline_discretize_cache_tensors(
        deltas=deltas,
        delta_length=delta_length,
        particle_length=particle_length,
        box_size=box_size,
        codebook_path=str(codebook_path),
        periodic=periodic,
        chunk_size=int(chunk_size),
    )

    target_idx = discrete_ids.to(dtype=torch.long)
    input_idx = torch.full_like(target_idx, fill_value=codebook_size_value)
    if target_idx.shape[1] > 1:
        input_idx[:, 1:] = target_idx[:, :-1]
    sequence = torch.full(
        (target_idx.shape[0], target_idx.shape[1] + 1),
        fill_value=codebook_size_value,
        dtype=torch.long,
    )
    sequence[:, 1:] = target_idx

    out_metadata = dict(metadata)
    out_metadata["discrete"] = True
    out_metadata["is_discrete"] = True
    out_metadata["codebook_path"] = str(codebook_path)
    out_metadata["codebook_size"] = int(codebook_size_value)
    out_metadata["relative_anchor_convention"] = "first_particle_origin_chain"
    out_metadata["version"] = max(int(out_metadata.get("version", 0)), 12)

    out_payload = dict(payload)
    out_payload["sequence"] = sequence
    out_payload["input_idx"] = input_idx
    out_payload["target_idx"] = target_idx
    out_payload["deltas"] = discrete_ids.to(dtype=torch.long)
    out_payload["absolute_coords"] = reconstructed_coords
    out_payload["token_coords"] = reconstructed_token_coords
    out_payload["long_jump_mask"] = torch.zeros_like(target_idx, dtype=torch.bool)
    out_payload["sample_length"] = delta_length.clone()
    out_payload["vocab_size"] = int(codebook_size_value)
    out_payload["sos_id"] = int(codebook_size_value)
    out_payload["metadata"] = out_metadata

    torch.save(out_payload, output_path)
    sample_length = out_payload["sample_length"]
    n_samples = int(sample_length.shape[0]) if hasattr(sample_length, "shape") else -1
    max_particles = int(reconstructed_coords.shape[1])
    max_delta_len = int(discrete_ids.shape[1])
    return {
        "output_path": output_path,
        "source_cache_path": source_cache_path,
        "n_samples": n_samples,
        "max_particles": max_particles,
        "max_delta_len": max_delta_len,
        "max_seq_len": max_delta_len,
        "vocab_size": int(codebook_size_value),
        "periodic": periodic,
        "ordering": str(out_metadata.get("ordering", "unknown")),
        "factorized": False,
        "polar": False,
        "discrete": True,
        "random_grid_shift": bool(out_metadata.get("random_grid_shift", False)),
        "num_augmentations": int(out_metadata.get("num_augmentations", 1)),
        "energy_chunk_size": int(out_metadata.get("energy_chunk_size", 0)),
        "cache_build_chunk_size": int(chunk_size),
        "rotation_count": int(out_metadata.get("rotation_count", 1)),
        "base_n_samples": int(out_metadata.get("base_n_samples", n_samples)),
        "max_abs_relative_displacement": float(out_metadata.get("max_abs_relative_displacement", 0.0)),
    }


class BucketedBatchSampler(Sampler[list[int]]):
    """Yield batches of indices grouped by equal particle count, then globally shuffled."""

    def __init__(
        self,
        sample_lengths: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.sample_lengths = np.asarray(sample_lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.sample_lengths.ndim != 1:
            raise ValueError("sample_lengths must be a rank-1 sequence")

    def __len__(self) -> int:
        total = 0
        for n in np.unique(self.sample_lengths):
            count = int(np.sum(self.sample_lengths == n))
            if self.drop_last:
                total += count // self.batch_size
            else:
                total += int(math.ceil(count / self.batch_size))
        return total

    def __iter__(self):
        # Draw a fresh NumPy RNG seed from PyTorch's global RNG state so shuffling
        # advances naturally across epochs without relying on a sampler-local counter.
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item()) ^ self.seed
        rng = np.random.default_rng(seed)

        all_batches: list[list[int]] = []
        for n in np.unique(self.sample_lengths):
            idxs = np.nonzero(self.sample_lengths == n)[0].astype(np.int64)
            if self.shuffle:
                rng.shuffle(idxs)

            n_full = len(idxs) // self.batch_size
            for i in range(n_full):
                s = i * self.batch_size
                e = s + self.batch_size
                all_batches.append(idxs[s:e].tolist())

            rem = len(idxs) % self.batch_size
            if (not self.drop_last) and rem > 0:
                all_batches.append(idxs[-rem:].tolist())

        if self.shuffle:
            rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch
