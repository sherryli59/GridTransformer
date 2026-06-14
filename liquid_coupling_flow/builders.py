"""Convenience builders for assembling flows."""

from __future__ import annotations

import torch

from .base import DiagGaussian, UniformTorus
from .transforms import AffineElementwise
from .coupling import CouplingLayer, alternating_masks
from .conditioner import MLPConditioner
from .flow import Flow


def build_affine_flow(dim: int = 2, n_layers: int = 12, hidden: int = 128,
                      cond_layers: int = 3, scale_bound: float = 4.0,
                      base: torch.nn.Module | None = None) -> Flow:
    """Affine (RealNVP) coupling flow. Identity-initialised (zero last layer)."""
    if base is None:
        base = DiagGaussian(dim)
    layers = []
    for m in alternating_masks(dim, n_layers):
        cond = MLPConditioner(dim, AffineElementwise.params_per_dim,
                              hidden=hidden, n_layers=cond_layers)
        layers.append(CouplingLayer(m, cond, AffineElementwise(scale_bound=scale_bound)))
    return Flow(base, layers)
