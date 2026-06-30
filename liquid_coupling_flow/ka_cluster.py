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


def cluster_scaffold_center(cluster_idx, sc, L):
    """Min-image-aware centroid of the cluster's scaffold slots (anchored on the seed = cluster_idx[0]).
    A naive mean is WRONG when the cluster straddles a periodic boundary; this handles the wrap. Function of
    (cluster_idx, sc, L) only -> config-independent."""
    seed = sc[cluster_idx[0]].to(sc.dtype)
    return torch.remainder(seed + _wrap_pm(sc[cluster_idx].to(sc.dtype) - seed[None], L).mean(0), L)


def frame_ctx_slots(cluster_idx, sc, L, n_ctx=16):
    """The fixed x_R context-slot set defining the frame: the n_ctx nearest scaffold slots to the cluster
    centre, EXCLUDING the cluster slots. A function of (cluster_idx, sc) ONLY -> config-independent
    -> guarantees x_C-independence of the frame, and is shared across a batch of chains."""
    sc_c = cluster_scaffold_center(cluster_idx, sc, L)
    dsc = (_wrap_pm(sc - sc_c[None], L) ** 2).sum(-1)
    dsc = dsc.clone(); dsc[cluster_idx] = float("inf")                                # exclude cluster slots
    return dsc.topk(n_ctx, largest=False).indices                                    # [n_ctx] fixed x_R slots


def frame_from_positions(ctx_pos, sc_c, L):
    """Equivariant frame from the context neighbours' positions ONLY. ctx_pos [...,M,2] (leading batch dims
    allowed), sc_c [2] the (fixed) cluster scaffold centroid. origin = sc_c + min-image neighbour mean;
    R [...,2,2] (rows = frame axes) = principal axis of the neighbours' relative positions with an x_R-only
    SKEWNESS sign rule (first moment is 0 after centring; orient by the third moment -> rotation-invariant).
    Frame->lab is rigid -> Jacobian 1, IFF ctx_pos is x_R-only (frame_ctx_slots guarantees it)."""
    # eigh is fp32-only and precision-critical (this is the frame keystone) -> force fp32, disable autocast
    with torch.autocast(device_type=ctx_pos.device.type, enabled=False):
        ctx_pos = ctx_pos.float(); sc_c = sc_c.float()
        rel = _wrap_pm(ctx_pos - sc_c, L)                                             # [...,M,2] rel to sc_c
        mu = rel.mean(-2)                                                             # [...,2]
        origin = torch.remainder(sc_c + mu, L)
        c = rel - mu[..., None, :]                                                    # [...,M,2] centred
        cov = (c.transpose(-1, -2) @ c) / c.shape[-2]                                 # [...,2,2]
        _, evecs = torch.linalg.eigh(cov)                                            # ascending evals
        axis = evecs[..., :, -1]                                                      # [...,2] principal axis
        proj = (c * axis[..., None, :]).sum(-1)                                       # [...,M]
        sign = torch.where((proj ** 3).sum(-1, keepdim=True) < 0, -1.0, 1.0)         # x_R-only skewness sign
        axis = axis * sign
        perp = torch.stack([-axis[..., 1], axis[..., 0]], -1)                         # [...,2]
        R = torch.stack([axis, perp], -2)                                            # [...,2,2] rows = axes
    return origin, R


def cluster_frame(pos, cluster_idx, sc, L, n_ctx=16):
    """Convenience: frame from `pos` (shape [N,2] or [B,N,2]) using the fixed scaffold context slots.
    Returns (origin, R) with leading batch dims matching `pos` (origin [...,2], R [...,2,2])."""
    slots = frame_ctx_slots(cluster_idx, sc, L, n_ctx)
    sc_c = cluster_scaffold_center(cluster_idx, sc, L).to(pos.dtype)
    return frame_from_positions(pos[..., slots, :], sc_c, L)


def to_frame(x, origin, R, L):
    """Lab -> frame: min-image relative to origin, then rotate into the frame. Rigid. x [...,k,2]."""
    return torch.einsum("...kj,...ij->...ki", _wrap_pm(x - origin[..., None, :], L), R)


def from_frame(u, origin, R, L):
    """Frame -> lab (inverse of to_frame), wrapped to the torus. u [...,k,2]."""
    return torch.remainder(torch.einsum("...ki,...ij->...kj", u, R) + origin[..., None, :], L)
