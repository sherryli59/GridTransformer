"""Exact periodic geometry for the hierarchical 8-color mW cell model.

The grid shift is stored in *oriented* coordinates.  For row-vector particle
coordinates ``x`` and signed-permutation matrix ``O`` the grid chart is

    y = (x O^T - shift) mod L.

Cell indices and local cube coordinates are computed only in this chart.  The
inverse uses the same matrix and therefore never relies on cached assignments.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import torch


N_COLORS = 8
LOG_COLOR_ORDERS = math.lgamma(N_COLORS + 1)


def cubic_orientations(device=None, dtype=torch.float32) -> torch.Tensor:
    """Return the 48 signed coordinate-permutation matrices in fixed order."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            mat = torch.zeros(3, 3, dtype=dtype, device=device)
            rows = torch.arange(3, device=device)
            mat[rows, torch.tensor(perm, device=device)] = torch.tensor(
                signs, dtype=dtype, device=device
            )
            mats.append(mat)
    return torch.stack(mats)


def validate_orientation(orientation: torch.Tensor) -> None:
    """Reject matrices outside the full cubic signed-permutation group."""
    if orientation.shape != (3, 3):
        raise ValueError(f"orientation must be [3,3], got {tuple(orientation.shape)}")
    work = orientation.detach().double()
    eye = torch.eye(3, device=work.device, dtype=work.dtype)
    if not torch.allclose(work @ work.T, eye, atol=1e-12, rtol=0):
        raise ValueError("orientation is not orthogonal")
    rounded = work.round()
    if not torch.equal(work, rounded) or not bool((rounded.abs().sum(0) == 1).all()) \
            or not bool((rounded.abs().sum(1) == 1).all()):
        raise ValueError("orientation is not a signed coordinate permutation")


def sample_orientation(*, device=None, dtype=torch.float32, gen=None) -> tuple[torch.Tensor, int]:
    """Sample one of the 48 cubic orientations and return ``(matrix, index)``."""
    index = int(torch.randint(48, (), device=device, generator=gen))
    return cubic_orientations(device=device, dtype=dtype)[index], index


def sample_color_order(*, device=None, gen=None) -> torch.Tensor:
    """Sample a uniform permutation mapping causal stage to physical color."""
    return torch.randperm(N_COLORS, device=device, generator=gen)


def validate_color_order(color_order: torch.Tensor) -> None:
    order = torch.as_tensor(color_order)
    if order.shape != (N_COLORS,) or not torch.equal(
            order.sort().values, torch.arange(N_COLORS, device=order.device)):
        raise ValueError("color_order must be a permutation of 0..7")


def color_stages(color_order: torch.Tensor) -> torch.Tensor:
    """Return the inverse permutation mapping physical color to causal stage."""
    validate_color_order(color_order)
    stages = torch.empty_like(color_order)
    stages[color_order] = torch.arange(N_COLORS, device=color_order.device)
    return stages


def _grid_width(L: float, G: int) -> float:
    if int(G) != G or G < 2 or G % 2:
        raise ValueError(f"G must be an even integer >=2, got {G}")
    if not math.isfinite(float(L)) or L <= 0:
        raise ValueError(f"L must be finite and positive, got {L}")
    return float(L) / int(G)


def _snapped_grid_coordinates(y: torch.Tensor, h: float, G: int) -> torch.Tensor:
    """Return y/h with roundoff-near grid faces pinned to the upper cell."""
    q = y / float(h)
    nearest = torch.round(q)
    tol = 32.0 * torch.finfo(y.dtype).eps * max(1, int(G))
    q = torch.where((q - nearest).abs() <= tol, nearest, q)
    return torch.remainder(q, int(G))


def to_oriented(x: torch.Tensor, L: float, shift: torch.Tensor,
                orientation: torch.Tensor) -> torch.Tensor:
    """Map physical coordinates to the shifted, oriented torus chart."""
    if x.shape[-1] != 3:
        raise ValueError(f"x must end in dimension 3, got {tuple(x.shape)}")
    validate_orientation(orientation)
    shift = torch.as_tensor(shift, device=x.device, dtype=x.dtype)
    if shift.shape != (3,):
        raise ValueError(f"shift must be [3], got {tuple(shift.shape)}")
    O = orientation.to(device=x.device, dtype=x.dtype)
    return torch.remainder(x @ O.T - shift, float(L))


def from_oriented(y: torch.Tensor, L: float, shift: torch.Tensor,
                  orientation: torch.Tensor) -> torch.Tensor:
    """Inverse :func:`to_oriented`, returning wrapped physical coordinates."""
    if y.shape[-1] != 3:
        raise ValueError(f"y must end in dimension 3, got {tuple(y.shape)}")
    validate_orientation(orientation)
    shift = torch.as_tensor(shift, device=y.device, dtype=y.dtype)
    if shift.shape != (3,):
        raise ValueError(f"shift must be [3], got {tuple(shift.shape)}")
    O = orientation.to(device=y.device, dtype=y.dtype)
    return torch.remainder((y + shift) @ O, float(L))


def cell_indices(x: torch.Tensor, L: float, G: int, shift: torch.Tensor,
                 orientation: torch.Tensor) -> torch.Tensor:
    """Return integer ``[...,3]`` half-open cube indices on the periodic grid."""
    h = _grid_width(L, G)
    y = to_oriented(x, L, shift, orientation)
    q = _snapped_grid_coordinates(y, h, G)
    return torch.floor(q).long().clamp_(0, int(G) - 1)


def flatten_cell_indices(indices: torch.Tensor, G: int) -> torch.Tensor:
    if indices.shape[-1] != 3:
        raise ValueError("indices must end in dimension 3")
    if bool(((indices < 0) | (indices >= int(G))).any()):
        raise ValueError("cell index outside [0,G)")
    return (indices[..., 0] * int(G) + indices[..., 1]) * int(G) + indices[..., 2]


def unflatten_cell_indices(cell_ids: torch.Tensor, G: int) -> torch.Tensor:
    ids = torch.as_tensor(cell_ids).long()
    if bool(((ids < 0) | (ids >= int(G) ** 3)).any()):
        raise ValueError("cell id outside [0,G^3)")
    i = torch.div(ids, int(G) ** 2, rounding_mode="floor")
    rem = ids - i * int(G) ** 2
    j = torch.div(rem, int(G), rounding_mode="floor")
    k = rem - j * int(G)
    return torch.stack((i, j, k), -1)


def cell_colors(indices: torch.Tensor) -> torch.Tensor:
    """Map parity-vector cell indices to physical color labels 0..7."""
    if indices.shape[-1] != 3:
        raise ValueError("indices must end in dimension 3")
    parity = torch.remainder(indices, 2)
    return 4 * parity[..., 0] + 2 * parity[..., 1] + parity[..., 2]


def cell_local_coordinates(x: torch.Tensor, L: float, G: int, shift: torch.Tensor,
                           orientation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(indices, local)`` with local coordinates in ``[0,h)^3``."""
    h = _grid_width(L, G)
    y = to_oriented(x, L, shift, orientation)
    q = _snapped_grid_coordinates(y, h, G)
    indices = torch.floor(q).long().clamp_(0, int(G) - 1)
    local = (q - indices.to(q.dtype)) * h
    return indices, local


def physical_from_cell_local(indices: torch.Tensor, local: torch.Tensor, L: float, G: int,
                             shift: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
    """Map cell-local coordinates back to physical coordinates."""
    h = _grid_width(L, G)
    if indices.shape != local.shape or indices.shape[-1] != 3:
        raise ValueError("indices and local must be matching [...,3] tensors")
    if bool((local < 0.0).any()) or bool((local >= h).any()):
        raise ValueError("local coordinate outside the half-open cell")
    y = indices.to(local.dtype) * h + local
    return from_oriented(y, L, shift, orientation)


def morton_codes(indices: torch.Tensor, G: int) -> torch.Tensor:
    """Integer 3-D Morton codes for a power-of-two grid."""
    if int(G) & (int(G) - 1):
        raise ValueError("Morton schedule requires power-of-two G")
    if indices.shape[-1] != 3:
        raise ValueError("indices must end in dimension 3")
    bits = int(math.log2(int(G)))
    code = torch.zeros(indices.shape[:-1], dtype=torch.long, device=indices.device)
    for bit in range(bits):
        code |= ((indices[..., 0] >> bit) & 1) << (3 * bit + 2)
        code |= ((indices[..., 1] >> bit) & 1) << (3 * bit + 1)
        code |= ((indices[..., 2] >> bit) & 1) << (3 * bit)
    return code


def causal_cell_order(G: int, color_order: torch.Tensor) -> torch.Tensor:
    """Cell ids ordered by sampled causal stage, then Morton within a color."""
    validate_color_order(color_order)
    ids = torch.arange(int(G) ** 3, device=color_order.device)
    indices = unflatten_cell_indices(ids, G)
    colors = cell_colors(indices)
    morton = morton_codes(indices, G)
    pieces = []
    for color in color_order.tolist():
        active = ids[colors == color]
        pieces.append(active[torch.argsort(morton[colors == color], stable=True)])
    return torch.cat(pieces)


@dataclass(frozen=True)
class CellGrouping:
    """Recoverable cell/priority grouping for one labeled configuration."""

    indices: torch.Tensor
    cell_ids: torch.Tensor
    colors: torch.Tensor
    counts: torch.Tensor
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor


def group_by_cell_priority(x: torch.Tensor, priorities: torch.Tensor, L: float, G: int,
                           shift: torch.Tensor, orientation: torch.Tensor) -> CellGrouping:
    """Group rows by cell and then by iid priority, with stable row tie breaks."""
    if x.ndim != 2 or x.shape[-1] != 3:
        raise ValueError("x must be [N,3]")
    priorities = torch.as_tensor(priorities, device=x.device)
    if priorities.shape != (x.shape[0],):
        raise ValueError("priorities must have shape [N]")
    indices = cell_indices(x, L, G, shift, orientation)
    ids = flatten_cell_indices(indices, G)
    by_priority = torch.argsort(priorities, stable=True)
    permutation = by_priority[torch.argsort(ids[by_priority], stable=True)]
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(x.shape[0], device=x.device)
    counts = torch.bincount(ids, minlength=int(G) ** 3)
    return CellGrouping(indices, ids, cell_colors(indices), counts, permutation, inverse)


def earlier_stage_mask(particle_colors: torch.Tensor, active_color: int,
                       color_order: torch.Tensor) -> torch.Tensor:
    """Mask particles from strictly earlier causal stages."""
    stages = color_stages(color_order)
    active = torch.as_tensor(active_color, device=stages.device, dtype=torch.long)
    return stages[particle_colors.to(stages.device)] < stages[active]
