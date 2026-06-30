"""Proposal-B: ClusterFlow — exact-likelihood, E(2)-equivariant coupling flow over the k cluster particles'
in-frame coordinates. Anchor-centered Gaussian base + per-block geometric distance-attention conditioner.
See docs/superpowers/specs/2026-06-30-ka-cluster-flow-proposal-b-design.md. Interface mirrors ClusterProposal."""
from __future__ import annotations
import os, math, torch, torch.nn as nn, torch.nn.functional as F
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def anchor_base_logp(z, q_scaf, sigma_b):                     # z,q_scaf [B,k,2] -> [B]
    d2 = ((z - q_scaf) ** 2).sum(-1)                          # [B,k]
    return (-d2 / (2 * sigma_b ** 2) - math.log(2 * math.pi * sigma_b ** 2)).sum(-1)


def sample_base(q_scaf, sigma_b, gen=None):                  # [B,k,2]
    return q_scaf + sigma_b * torch.randn(q_scaf.shape, generator=gen, device=q_scaf.device, dtype=q_scaf.dtype)


@torch.no_grad()
def compute_sigma_b(data, s, geo, sc, L, k):
    """Isotropic std of each cluster particle's min-imaged LAB displacement from its own scaffold slot. This is
    the FIXED base width (rotation-invariant, so no frame needed). Expected ~0.747 on ka_reference_N100."""
    from liquid_coupling_flow.ka_gridformer import _wrap_pm
    N = data.shape[1]; pos0, _ = slot_order(data, s, geo, N); disp = []
    for seed in range(0, N, 3):
        cl = KC.cluster_slots(seed, sc, k, L)
        disp.append(_wrap_pm(pos0[:, cl] - sc[cl][None], L).reshape(-1, 2))
    return float(torch.cat(disp, 0).std().item())
