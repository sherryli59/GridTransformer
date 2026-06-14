"""Base distributions for the coupling flow.

A normalizing flow needs a base p_base(z) with a tractable log_prob and sampler.
The flow maps z -> x, and  log p(x) = log p_base(f^{-1}(x)) + log|det df^{-1}/dx|.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DiagGaussian(nn.Module):
    """Standard isotropic Gaussian base (for non-periodic / R^d toys)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return (-0.5 * (z ** 2 + math.log(2 * math.pi))).sum(-1)

    def sample(self, n: int, device=None, dtype=None) -> torch.Tensor:
        return torch.randn(n, self.dim, device=device, dtype=dtype)


class UniformTorus(nn.Module):
    """Uniform distribution on the torus [0, L)^dim.

    This is the ideal-gas base for a periodic liquid: density 1/L^dim everywhere,
    so log_prob is a constant. Domain handling is delegated to circular transforms
    (which map [0, L) -> [0, L) bijectively); inputs are wrapped here for safety.
    """

    def __init__(self, dim: int, L: float = 1.0):
        super().__init__()
        self.dim = dim
        self.L = float(L)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        const = -self.dim * math.log(self.L)
        return torch.full(z.shape[:-1], const, device=z.device, dtype=z.dtype)

    def sample(self, n: int, device=None, dtype=None) -> torch.Tensor:
        return torch.rand(n, self.dim, device=device, dtype=dtype) * self.L
