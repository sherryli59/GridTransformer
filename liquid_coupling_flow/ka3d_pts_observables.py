"""Cavity point-to-set (PTS) observables for the 3D Kob-Andersen glass.

Self-contained: pure geometry/statistics on configs, no dependency on any
other Task file in this project.

- ``core_overlap`` is a PROXY for the paper's exact overlap (Eq. 5-8): a
  symmetric, same-species nearest-neighbour core average, not the exact
  cell-occupancy overlap.
- ``pts_susceptibility`` implements the paper's Eq. 10 exactly: the disorder
  average (over boundary/cavity-center realizations J) of the per-boundary
  THERMAL variance of q_c (variance over the pairs axis, ddof=0), i.e.
  ``chi_T(R) = < <q_c^2>_J - <q_c>_J^2 >`` -- NOT the variance across centers.
- ``overlap_pdf`` is a plain histogram of overlap values for a bimodality
  check.

Minimum-image convention throughout: ``diff = a - b; diff -= L*round(diff/L)``.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _min_image(diff: Tensor, L: float) -> Tensor:
    return diff - L * torch.round(diff / L)


def _core_direction(ref: Tensor, s_ref: Tensor, search: Tensor, s_search: Tensor,
                     center: Tensor, L: float, r_c: float, b: float) -> Tensor:
    """Core taken on ``ref`` (particles within r_c of ``center``); for each
    core particle, score the nearest SAME-species particle in ``search`` via
    w(dist) = exp(-(dist/b)^2). Returns the per-batch mean over core
    particles, 0 where the core is empty.

    ref, search: [B, N, 3]; s_ref, s_search: [B, N]; center: [3] or [B,3]. -> [B]
    """
    if center.dim() == 2:
        center = center[:, None, :]
    diff_c = _min_image(ref - center, L)
    dist_c = diff_c.norm(dim=-1)                       # [B, N]
    core_mask = dist_c <= r_c                           # [B, N]

    diff = _min_image(ref.unsqueeze(2) - search.unsqueeze(1), L)
    dist = diff.norm(dim=-1)                            # [B, N_ref, N_search]

    same_species = s_ref.unsqueeze(2) == s_search.unsqueeze(1)   # [B, N_ref, N_search]
    dist_masked = torch.where(same_species, dist, torch.full_like(dist, float("inf")))
    nearest, _ = dist_masked.min(dim=2)                  # [B, N_ref]
    w = torch.exp(-(nearest / b) ** 2)
    w = torch.where(torch.isfinite(nearest), w, torch.zeros_like(w))

    core_mask_f = core_mask.to(w.dtype)
    counts = core_mask_f.sum(dim=1)                      # [B]
    sums = (w * core_mask_f).sum(dim=1)                  # [B]
    return torch.where(counts > 0, sums / counts.clamp_min(1), torch.zeros_like(sums))


def core_overlap(X: Tensor, sX: Tensor, Y: Tensor, sY: Tensor, center: Tensor, L: float,
                  r_c: float = 0.5, b: float = 0.2) -> Tensor:
    """Symmetric same-species nearest-neighbour core overlap proxy.

    Core is taken on ``Y`` (particles within r_c of ``center``, min-image);
    for each core particle the nearest same-species particle in ``X`` is
    scored by w(dist) = exp(-(dist/b)^2) and averaged. Symmetrized by also
    taking the core on ``X`` and searching in ``Y``, then averaging both
    directions. Empty-core batches contribute 0 (no division by zero).

    X, Y: [B, N, 3]; sX, sY: [B, N] (or [N], broadcastable);
    center: [3] or one center per row [B,3].
    Returns Tensor[B].
    """
    if sX.dim() == 1:
        sX = sX.unsqueeze(0).expand(X.shape[0], -1)
    if sY.dim() == 1:
        sY = sY.unsqueeze(0).expand(Y.shape[0], -1)
    d_core_on_Y = _core_direction(Y, sY, X, sX, center, L, r_c, b)
    d_core_on_X = _core_direction(X, sX, Y, sY, center, L, r_c, b)
    return 0.5 * (d_core_on_Y + d_core_on_X)


def pts_susceptibility(q_by_center_pairs) -> float:
    """Paper Eq. 10: disorder average (over centers) of the per-center
    thermal variance (over pairs, population variance / ddof=0).

    Input ``q[n_centers, n_pairs]``. NOT the variance across centers.
    """
    q = torch.as_tensor(q_by_center_pairs)
    if q.dim() != 2:
        raise ValueError(f"pts_susceptibility expects [n_centers, n_pairs], got shape {tuple(q.shape)}")
    thermal_var_per_center = q.var(dim=1, unbiased=False)   # [n_centers]
    return float(thermal_var_per_center.mean())


def overlap_pdf(q_flat, bins) -> Tensor:
    """P(q_c): density-normalized histogram of overlap values.

    ``bins`` may be an int bin count or a 1D sequence/Tensor of bin edges.
    Returns the histogram counts/density Tensor (bin_edges are not returned).
    """
    q = torch.as_tensor(q_flat, dtype=torch.get_default_dtype()).flatten()
    if q.numel() == 0:
        raise ValueError("overlap_pdf requires at least one overlap value")
    if isinstance(bins, int):
        hist, _edges = torch.histogram(q, bins=bins, density=True)
    else:
        edges = torch.as_tensor(bins, dtype=q.dtype)
        hist, _edges = torch.histogram(q, bins=edges, density=True)
    return hist
