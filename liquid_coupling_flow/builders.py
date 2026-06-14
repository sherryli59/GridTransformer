"""Convenience builders for assembling flows."""

from __future__ import annotations

import torch

from .base import DiagGaussian, UniformTorus
from .transforms import AffineElementwise
from .transforms_spline import RQSplineElementwise, CircularRQSplineElementwise
from .coupling import CouplingLayer, alternating_masks
from .conditioner import MLPConditioner, PeriodicMLPConditioner
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


def build_spline_flow(dim: int = 2, n_layers: int = 8, num_bins: int = 8,
                      tail_bound: float = 5.0, hidden: int = 128, cond_layers: int = 3,
                      base: torch.nn.Module | None = None) -> Flow:
    """Unconstrained RQ-spline flow on R^dim (Gaussian base)."""
    if base is None:
        base = DiagGaussian(dim)
    layers = []
    for m in alternating_masks(dim, n_layers):
        tr = RQSplineElementwise(num_bins=num_bins, tail_bound=tail_bound)
        cond = MLPConditioner(dim, tr.params_per_dim, hidden=hidden, n_layers=cond_layers)
        layers.append(CouplingLayer(m, cond, tr))
    return Flow(base, layers)


def build_circular_spline_flow(dim: int = 2, n_layers: int = 8, L: float = 1.0,
                               num_bins: int = 8, hidden: int = 128, cond_layers: int = 3) -> Flow:
    """Circular RQ-spline flow on the torus [0, L)^dim (uniform base).

    This is the periodic prototype for liquid coordinates: every coupling layer is a
    diffeomorphism of the torus, conditioner is periodic (sin/cos features).
    """
    base = UniformTorus(dim, L)
    layers = []
    for m in alternating_masks(dim, n_layers):
        tr = CircularRQSplineElementwise(num_bins=num_bins, L=L)
        cond = PeriodicMLPConditioner(dim, tr.params_per_dim, L=L,
                                      hidden=hidden, n_layers=cond_layers)
        layers.append(CouplingLayer(m, cond, tr))
    return Flow(base, layers)
