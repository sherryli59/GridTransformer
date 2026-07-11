"""Carve self-supervised 3D KABLJ cavity-condition training pairs from bulk."""
from __future__ import annotations

import torch

from liquid_coupling_flow.ka3d_pt_cavity import assert_cavity_valid


def carve(x, s, center, R, L):
    """Partition one bulk configuration into mobile interior and frozen exterior.

    ``x[N,3]``, ``s[N]``, ``center[3]``. Original indices are retained so
    reassembly is exact and named/species-fixed interior particles remain
    identifiable.
    """
    assert_cavity_valid(R, L)
    if x.ndim != 2 or x.shape[-1] != 3 or s.shape != x.shape[:1]:
        raise ValueError("carve expects x[N,3] and s[N]")
    center = torch.as_tensor(center, device=x.device, dtype=x.dtype)
    if center.shape != (3,):
        raise ValueError("center must have shape [3]")
    delta = x - center; delta = delta - L * torch.round(delta / L)
    mobile = delta.square().sum(-1) < float(R) ** 2
    idx = torch.arange(x.shape[0], device=x.device)
    return {"x_in": x[mobile].clone(), "s_in": s[mobile].clone(),
            "x_out": x[~mobile].clone(), "s_out": s[~mobile].clone(),
            "idx_in": idx[mobile], "idx_out": idx[~mobile], "mobile": mobile,
            "center": center.clone(), "R": float(R), "L": float(L),
            "n_in": int(mobile.sum())}


def reassemble(pair):
    """Reassemble a carved pair in its original particle-index order."""
    N = pair["mobile"].numel()
    x = pair["x_in"].new_empty(N, 3)
    s = pair["s_in"].new_empty(N)
    x[pair["idx_in"]] = pair["x_in"]; x[pair["idx_out"]] = pair["x_out"]
    s[pair["idx_in"]] = pair["s_in"]; s[pair["idx_out"]] = pair["s_out"]
    return x, s


def carve_batch(bulk_x, bulk_s, L, radii, centers_per_config=1, rng=None):
    """Draw uniform centers and radii, returning a list of variable-size pairs."""
    if bulk_x.ndim != 3 or bulk_x.shape[-1] != 3:
        raise ValueError("bulk_x must be [B,N,3]")
    if bulk_s.ndim == 1:
        bulk_s = bulk_s[None].expand(bulk_x.shape[0], -1)
    if bulk_s.shape != bulk_x.shape[:2]:
        raise ValueError("bulk_s must be [N] or [B,N]")
    radii = tuple(float(r) for r in radii)
    if not radii:
        raise ValueError("radii cannot be empty")
    for R in radii:
        assert_cavity_valid(R, L)
    out = []
    for b in range(bulk_x.shape[0]):
        for _ in range(centers_per_config):
            center = torch.rand(3, generator=rng, device=bulk_x.device, dtype=bulk_x.dtype) * L
            ridx = int(torch.randint(len(radii), (), generator=rng, device=bulk_x.device))
            pair = carve(bulk_x[b], bulk_s[b], center, radii[ridx], L)
            pair["bulk_index"] = b
            out.append(pair)
    return out
