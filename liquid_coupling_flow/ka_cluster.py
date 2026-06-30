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
