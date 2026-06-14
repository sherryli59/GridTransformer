"""Elementwise invertible transforms used inside coupling layers.

A coupling layer applies one of these to the "transform" dimensions, with the
per-element parameters produced by a conditioner network from the "identity"
dimensions. Each transform exposes:

    params_per_dim : int
    forward(x, params)  -> y,  logdet_elementwise      (same shape as x)
    inverse(y, params)  -> x, -logdet_elementwise

`logdet_elementwise` is the per-coordinate log|dy/dx| (NOT summed); the coupling
layer masks and sums it. Keeping it per-element lets the coupling layer zero out
the identity dimensions cleanly.
"""

from __future__ import annotations

import torch

# Rational-quadratic spline transforms live in transforms_spline.py (Phase B/C).


class AffineElementwise:
    """Affine transform  y = x * exp(s) + t  (RealNVP).

    params[..., 0] -> raw scale (bounded by tanh to keep training stable)
    params[..., 1] -> shift t
    """

    params_per_dim = 2

    def __init__(self, scale_bound: float = 4.0):
        self.scale_bound = float(scale_bound)

    def _split(self, params: torch.Tensor):
        raw_s = params[..., 0]
        t = params[..., 1]
        log_s = torch.tanh(raw_s) * self.scale_bound
        return log_s, t

    def forward(self, x: torch.Tensor, params: torch.Tensor):
        log_s, t = self._split(params)
        y = x * torch.exp(log_s) + t
        return y, log_s

    def inverse(self, y: torch.Tensor, params: torch.Tensor):
        log_s, t = self._split(params)
        x = (y - t) * torch.exp(-log_s)
        return x, -log_s
