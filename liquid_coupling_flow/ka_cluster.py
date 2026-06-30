"""Equivariant cluster-move geometry: slot-based reversible cluster selection + the x_R-only equivariant
frame. See docs/superpowers/specs/2026-06-25-ka-equivariant-cluster-move-design.md.

Exactness keystone: cluster membership is a function of (seed slot, fixed scaffold) ONLY (move-invariant ->
reversible), and the frame is a function of x_R ONLY (never x_C -> rigid frame->lab map -> Jacobian 1)."""
from __future__ import annotations
import torch
from liquid_coupling_flow.ka_gridformer import _wrap_pm


def cluster_slots(seed, sc, k, L):
    """Cluster = seed slot + its (k-1) nearest scaffold slots (min-image on the scaffold lattice).
    Deterministic function of (seed, sc, L) ONLY -> move-invariant -> reversible. Returns LongTensor[k]
    with the seed first."""
    d = _wrap_pm(sc - sc[seed][None], L)
    dist2 = (d ** 2).sum(-1)
    idx = dist2.topk(k, largest=False).indices                       # k nearest incl. self (dist 0)
    seed_t = torch.tensor([seed], device=sc.device, dtype=idx.dtype)
    return torch.cat([seed_t, idx[idx != seed]])[:k]                 # ensure seed first, exactly k


def cluster_frame(pos, cluster_idx, sc, L, cutoff=2.4):
    """Equivariant frame from x_R ONLY (never x_C). origin = centroid (min-image) of NON-cluster particles
    within `cutoff` of the cluster scaffold centroid; R[2,2] (rows = frame axes) = principal axis of those
    neighbours' relative positions, with an x_R-only SKEWNESS sign rule (the first moment is 0 after centring,
    so orient by the third moment along the axis -> nonzero generically, rotation-invariant -> equivariant).
    Frame->lab is rigid (rotation+translation) -> Jacobian 1, IFF this is x_C-independent (test enforces it)."""
    dev = pos.device; N = pos.shape[0]
    mask = torch.ones(N, dtype=torch.bool, device=dev); mask[cluster_idx] = False    # x_R = non-cluster
    sc_c = sc[cluster_idx].to(pos.dtype).mean(0)                                      # cluster scaffold centroid (FIXED)
    dR = _wrap_pm(pos[mask] - sc_c[None], L); near = (dR ** 2).sum(-1) <= cutoff ** 2
    rel = dR[near]                                                                    # x_R neighbours rel to sc_c
    if rel.shape[0] < 2:                                                              # degenerate -> axis-aligned frame
        return torch.remainder(sc_c, L), torch.eye(2, device=dev, dtype=pos.dtype)
    origin = torch.remainder(sc_c + rel.mean(0), L)
    c = rel - rel.mean(0); cov = (c.T @ c) / c.shape[0]                               # 2x2 inertia of x_R neighbours
    _, evecs = torch.linalg.eigh(cov); axis = evecs[:, -1]                            # larger-eigenvalue eigenvector
    proj = c @ axis
    if (proj ** 3).sum() < 0: axis = -axis                                            # x_R-only skewness sign rule
    perp = torch.stack([-axis[1], axis[0]]); R = torch.stack([axis, perp], 0)         # rows = frame axes
    return origin, R


def to_frame(x, origin, R, L):
    """Lab -> frame: min-image relative to origin, then rotate into the frame. Rigid."""
    return _wrap_pm(x - origin[None], L) @ R.T


def from_frame(u, origin, R, L):
    """Frame -> lab (inverse of to_frame), wrapped to the torus."""
    return torch.remainder(u @ R + origin[None], L)
