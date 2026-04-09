from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import torch


def wrap_min_image(dr: torch.Tensor, box_size: torch.Tensor) -> torch.Tensor:
    """Apply minimum-image wrapping along each coordinate axis."""
    box_size = torch.as_tensor(box_size, device=dr.device, dtype=dr.dtype)
    if box_size.ndim == 1:
        box_size = box_size.view(*([1] * (dr.ndim - 1)), box_size.shape[0])
    elif box_size.ndim == 2:
        box_size = box_size.view(box_size.shape[0], *([1] * (dr.ndim - 2)), box_size.shape[1])
    else:
        raise ValueError(f"box_size must be rank 1 or 2, got rank {box_size.ndim}")
    return dr - box_size * torch.round(dr / box_size.clamp_min(1e-8))


def spherical_to_cartesian(coords: torch.Tensor) -> torch.Tensor:
    """
    Convert spherical coordinates (..., 3) -> Cartesian (..., 3).

    Convention:
      r: radius
      theta: polar angle from +z in [0, pi]
      phi: azimuth in the xy-plane in [-pi, pi]
    """
    if coords.shape[-1] != 3:
        raise ValueError(f"spherical coords must have trailing dim 3, got {tuple(coords.shape)}")
    r = coords[..., 0]
    theta = coords[..., 1]
    phi = coords[..., 2]
    sin_theta = torch.sin(theta)
    x = r * sin_theta * torch.cos(phi)
    y = r * sin_theta * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack((x, y, z), dim=-1)


def cartesian_to_spherical(coords: torch.Tensor) -> torch.Tensor:
    """
    Convert Cartesian coordinates (..., 3) -> spherical (..., 3).

    Convention:
      r: radius
      theta: polar angle from +z in [0, pi]
      phi: azimuth in the xy-plane in [-pi, pi]
    """
    if coords.shape[-1] != 3:
        raise ValueError(f"cartesian coords must have trailing dim 3, got {tuple(coords.shape)}")
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]
    r = torch.sqrt(x * x + y * y + z * z)
    xy = torch.sqrt(x * x + y * y)
    theta = torch.atan2(xy, z)
    phi = torch.atan2(y, x)
    return torch.stack((r, theta, phi), dim=-1)


def _pad_or_trim_time(coords: torch.Tensor, seq_len: int) -> torch.Tensor:
    if coords.shape[1] == int(seq_len):
        return coords
    if coords.shape[1] > int(seq_len):
        return coords[:, : int(seq_len), :]
    pad = torch.zeros(
        coords.shape[0],
        int(seq_len) - coords.shape[1],
        coords.shape[2],
        device=coords.device,
        dtype=coords.dtype,
    )
    return torch.cat((coords, pad), dim=1)


def _coerce_spatial_dim(coords: torch.Tensor, spatial_dim: int) -> torch.Tensor:
    if coords.shape[-1] == int(spatial_dim):
        return coords
    if coords.shape[-1] > int(spatial_dim):
        return coords[..., : int(spatial_dim)]
    pad = torch.zeros(
        *coords.shape[:-1],
        int(spatial_dim) - coords.shape[-1],
        device=coords.device,
        dtype=coords.dtype,
    )
    return torch.cat((coords, pad), dim=-1)


def build_attention_coords(
    coords: torch.Tensor | None,
    *,
    seq_len: int,
    spatial_dim: int,
    factorized: bool = False,
    polar: bool = False,
) -> torch.Tensor | None:
    """
    Normalize geometry inputs to Cartesian [B, T, spatial_dim] for attention layers.

    Accepted inputs:
      - token-aligned Cartesian coords: [B, T, D]
      - absolute particle coords: [B, N, D] where N == T + 1 (or (T / D) + 1 if factorized)
      - factorized scalar streams: [B, T] or [B, T, 1]
    """
    if coords is None:
        return None
    if int(seq_len) < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if int(spatial_dim) <= 0:
        raise ValueError(f"spatial_dim must be positive, got {spatial_dim}")

    coord_dim = int(spatial_dim)
    if coords.ndim == 3 and coords.shape[-1] != 1:
        coords = _coerce_spatial_dim(coords, coord_dim)
        if factorized:
            if coords.shape[1] == int(seq_len):
                return coords
            if coords.shape[1] == math.ceil(int(seq_len) / coord_dim):
                rep = torch.repeat_interleave(coords, repeats=coord_dim, dim=1)
                return _pad_or_trim_time(rep, int(seq_len))
            if coords.shape[1] == math.ceil(int(seq_len) / coord_dim) + 1:
                rep = torch.repeat_interleave(coords[:, :-1, :], repeats=coord_dim, dim=1)
                return _pad_or_trim_time(rep, int(seq_len))
        else:
            if coords.shape[1] == int(seq_len):
                return coords
            if coords.shape[1] == int(seq_len) + 1:
                return coords[:, :-1, :]
        return _pad_or_trim_time(coords, int(seq_len))

    if coords.ndim == 3 and coords.shape[-1] == 1:
        flat = coords.squeeze(-1)
    elif coords.ndim == 2:
        flat = coords
    else:
        raise ValueError(f"Unsupported coords shape for attention normalization: {tuple(coords.shape)}")

    batch_size, flat_len = flat.shape
    remainder = int(flat_len) % coord_dim
    if remainder != 0:
        pad = torch.zeros(
            batch_size,
            coord_dim - remainder,
            device=flat.device,
            dtype=flat.dtype,
        )
        flat = torch.cat((flat, pad), dim=1)

    deltas = flat.view(batch_size, -1, coord_dim)
    if polar:
        if coord_dim != 3:
            raise ValueError("polar attention reconstruction requires spatial_dim=3")
        deltas = spherical_to_cartesian(deltas)

    anchors = torch.zeros_like(deltas)
    if deltas.shape[1] > 1:
        anchors[:, 1:, :] = torch.cumsum(deltas[:, :-1, :], dim=1)

    if factorized:
        anchors = torch.repeat_interleave(anchors, repeats=coord_dim, dim=1)
    return _pad_or_trim_time(anchors, int(seq_len))


def _is_power_of_two(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


def _normalize_box_size(
    box_size: float | torch.Tensor,
    *,
    batch_size: int,
    coord_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return box size as [B, D], broadcasting scalar/1D inputs as needed."""
    bs = torch.as_tensor(box_size, device=device, dtype=dtype)
    if bs.ndim == 0:
        bs = bs.repeat(coord_dim).view(1, coord_dim)
    elif bs.ndim == 1:
        if bs.numel() == 1:
            bs = bs.repeat(coord_dim)
        if bs.numel() != coord_dim:
            raise ValueError(f"box_size has {bs.numel()} values, expected {coord_dim}")
        bs = bs.view(1, coord_dim)
    elif bs.ndim == 2:
        if bs.shape[1] == 1:
            bs = bs.repeat(1, coord_dim)
        if bs.shape[1] != coord_dim:
            raise ValueError(f"box_size shape {tuple(bs.shape)} incompatible with coord dim {coord_dim}")
    else:
        raise ValueError(f"box_size must be scalar, [D], or [B,D], got rank {bs.ndim}")

    if bs.shape[0] == 1 and batch_size != 1:
        bs = bs.expand(batch_size, -1)
    elif bs.shape[0] != batch_size:
        raise ValueError(f"box_size batch {bs.shape[0]} does not match coords batch {batch_size}")
    return bs


def spectral_argsort(coords: torch.Tensor, box_size: float | torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute per-sample sort indices using the Fiedler vector of a distance kernel graph.

    Args:
        coords: [B, N, D] coordinates.
        box_size: periodic box size (scalar, [D], or [B,D]).
        sigma: RBF kernel width.

    Returns:
        LongTensor [B, N] of ascending spectral order indices.
    """
    if coords.ndim != 3:
        raise ValueError(f"coords must be [B,N,D], got {tuple(coords.shape)}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    batch_size, n_points, coord_dim = coords.shape
    if n_points == 0:
        return torch.empty((batch_size, 0), dtype=torch.long, device=coords.device)
    if n_points == 1:
        return torch.zeros((batch_size, 1), dtype=torch.long, device=coords.device)

    bs = _normalize_box_size(
        box_size,
        batch_size=batch_size,
        coord_dim=coord_dim,
        device=coords.device,
        dtype=coords.dtype,
    )

    dx = coords[:, :, None, :] - coords[:, None, :, :]
    dx = wrap_min_image(dx, bs)
    dist_sq = torch.sum(dx * dx, dim=-1)

    gamma = 1.0 / (2.0 * float(sigma) * float(sigma))
    W = torch.exp(-dist_sq * gamma)
    W.diagonal(dim1=-2, dim2=-1).zero_()

    deg = W.sum(dim=-1).clamp_min(1e-8)
    eye = torch.eye(n_points, device=coords.device, dtype=coords.dtype).unsqueeze(0)
    L_rw = eye - (W / deg.unsqueeze(-1))
    # Use a symmetric matrix for stable batched eigh.
    L = 0.5 * (L_rw + L_rw.transpose(-1, -2))

    _, eigvecs = torch.linalg.eigh(L)
    u2 = eigvecs[:, :, 1]

    two_pi = 2.0 * math.pi
    theta = (coords / bs.unsqueeze(1).clamp_min(1e-8)) * two_pi
    mean_sin = torch.sin(theta).mean(dim=1)
    mean_cos = torch.cos(theta).mean(dim=1)
    centroid_theta = torch.remainder(torch.atan2(mean_sin, mean_cos), two_pi)
    centroid = bs * (centroid_theta / two_pi)

    centered = wrap_min_image(coords - centroid.unsqueeze(1), bs)
    radius = torch.linalg.norm(centered, dim=-1)
    radius_mean = radius.mean(dim=-1, keepdim=True)
    sign_score = torch.sum(u2 * (radius - radius_mean), dim=-1)
    sign = torch.where(sign_score >= 0.0, torch.ones_like(sign_score), -torch.ones_like(sign_score))
    u2 = u2 * sign.unsqueeze(-1)

    return torch.argsort(u2, dim=-1)


def thermodynamic_mst_argsort(coords: torch.Tensor, box_size: float | torch.Tensor) -> torch.Tensor:
    """
    Sort particles by a thermodynamically anchored greedy MST traversal.

    Steps:
      1) Build periodic minimum-image pairwise distances D [B,N,N].
      2) Anchor at particle with lowest local LJ-like energy per sample.
      3) Run batched Prim traversal to flatten the MST into a 1D sequence.
    """
    if coords.ndim != 3:
        raise ValueError(f"coords must be [B,N,D], got {tuple(coords.shape)}")
    batch_size, n_points, coord_dim = coords.shape
    if n_points == 0:
        return torch.empty((batch_size, 0), dtype=torch.long, device=coords.device)
    if n_points == 1:
        return torch.zeros((batch_size, 1), dtype=torch.long, device=coords.device)

    bs = _normalize_box_size(
        box_size,
        batch_size=batch_size,
        coord_dim=coord_dim,
        device=coords.device,
        dtype=coords.dtype,
    )

    dx = coords[:, :, None, :] - coords[:, None, :, :]
    dx = wrap_min_image(dx, bs)
    D = torch.linalg.norm(dx, dim=-1).clamp_min(1e-4)  # [B,N,N]

    # Thermodynamic anchor via simplified LJ local energy:
    # U_i = sum_{j != i} (r_ij^-12 - r_ij^-6)
    eye = torch.eye(n_points, device=coords.device, dtype=torch.bool).unsqueeze(0)
    inv_r6 = D.pow(-6)
    inv_r12 = inv_r6 * inv_r6
    lj_pair = inv_r12 - inv_r6
    lj_pair = lj_pair.masked_fill(eye, 0.0)
    U = lj_pair.sum(dim=-1)  # [B,N]
    current_nodes = torch.argmin(U, dim=1)  # [B]

    seq = torch.zeros((batch_size, n_points), dtype=torch.long, device=coords.device)
    visited = torch.zeros((batch_size, n_points), dtype=torch.bool, device=coords.device)
    min_dist_to_tree = torch.full((batch_size, n_points), float("inf"), device=coords.device, dtype=coords.dtype)
    batch_idx = torch.arange(batch_size, device=coords.device)
    seq[:, 0] = current_nodes

    for k in range(1, n_points):
        visited[batch_idx, current_nodes] = True
        dists_from_new = D[batch_idx, current_nodes, :]  # [B,N]
        min_dist_to_tree = torch.minimum(min_dist_to_tree, dists_from_new)
        min_dist_to_tree = min_dist_to_tree.masked_fill(visited, float("inf"))
        current_nodes = torch.argmin(min_dist_to_tree, dim=1)
        seq[:, k] = current_nodes

    return seq


def _hilbert_d2xy(order_n: int, d: int) -> tuple[int, int]:
    """Distance-to-2D Hilbert coordinate for square side length `order_n`."""
    x = 0
    y = 0
    t = int(d)
    s = 1
    while s < order_n:
        rx = (t // 2) & 1
        ry = (t ^ rx) & 1
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s <<= 1
    return x, y


@lru_cache(maxsize=16)
def _hilbert_id_to_xy_lut(side: int) -> torch.Tensor:
    if not _is_power_of_two(side):
        raise ValueError(f"Hilbert mapping requires power-of-two side length, got {side}")
    lut = torch.empty((side * side, 2), dtype=torch.float32)
    for d in range(side * side):
        x, y = _hilbert_d2xy(side, d)
        lut[d, 0] = float(x)
        lut[d, 1] = float(y)
    return lut


def id_to_center_xyz(
    token_ids: torch.LongTensor,
    *,
    grid_size: int,
    box_size: Optional[torch.Tensor] = None,
    mapping: str = "hilbert",
    sos_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert discrete token IDs to physical 2D cell-center coordinates.

    token_ids: [B, T], values in [0, K-1] (optionally includes SOS token).
    returns:   [B, T, 2] where coords are in the same physical units as box_size.
    """
    if token_ids.ndim != 2:
        raise ValueError(f"token_ids must be [B,T], got {tuple(token_ids.shape)}")
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")

    k_vocab = int(grid_size * grid_size)
    ids = token_ids.long()
    invalid = (ids < 0) | (ids >= k_vocab)
    if sos_id is not None:
        invalid = invalid | (ids == int(sos_id))
    ids_safe = ids.clamp(min=0, max=k_vocab - 1)

    if mapping == "hilbert":
        lut = _hilbert_id_to_xy_lut(int(grid_size)).to(device=ids.device)
        xy = lut[ids_safe]  # [B,T,2], columns=(ix, iy)
    elif mapping == "raster":
        ix = (ids_safe % int(grid_size)).to(dtype=torch.float32)
        iy = torch.div(ids_safe, int(grid_size), rounding_mode="floor").to(dtype=torch.float32)
        xy = torch.stack([ix, iy], dim=-1)
    else:
        raise ValueError(f"Unsupported id->coord mapping '{mapping}'")

    if box_size is None:
        bs = torch.tensor([float(grid_size), float(grid_size)], device=ids.device, dtype=xy.dtype).view(1, 1, 2)
    else:
        bs = torch.as_tensor(box_size, device=ids.device, dtype=xy.dtype)
        if bs.ndim == 1:
            if bs.numel() != 2:
                raise ValueError(f"box_size must have 2 entries, got shape {tuple(bs.shape)}")
            bs = bs.view(1, 1, 2)
        elif bs.ndim == 2:
            if bs.shape[1] != 2:
                raise ValueError(f"box_size must be [B,2], got shape {tuple(bs.shape)}")
            bs = bs[:, None, :]
        else:
            raise ValueError(f"box_size must be rank 1 or 2, got rank {bs.ndim}")

    centers = (xy + 0.5) * (bs / float(grid_size))
    centers = centers.masked_fill(invalid.unsqueeze(-1), 0.0)
    return centers


__all__ = [
    "id_to_center_xyz",
    "spectral_argsort",
    "thermodynamic_mst_argsort",
    "wrap_min_image",
]
