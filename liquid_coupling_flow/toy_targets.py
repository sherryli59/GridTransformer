"""Toy target distributions for proof-of-concept.

Non-periodic (Gaussian base): eight_gaussians, two_moons.
Periodic   (torus base):       torus_mixture  (used in Phase C).

Each returns samples [n, 2]. `torus_mixture` also exposes an exact log-density so
we can compute importance weights / ESS in Phase D.
"""

from __future__ import annotations

import math

import torch


def eight_gaussians(n: int, std: float = 0.15, radius: float = 2.0, device=None) -> torch.Tensor:
    angles = torch.arange(8) * (2 * math.pi / 8)
    centers = torch.stack([angles.cos(), angles.sin()], dim=-1) * radius
    idx = torch.randint(0, 8, (n,))
    return (centers[idx] + std * torch.randn(n, 2)).to(device)


def two_moons(n: int, noise: float = 0.1, device=None) -> torch.Tensor:
    n0 = n // 2
    n1 = n - n0
    t0 = torch.rand(n0) * math.pi
    moon0 = torch.stack([t0.cos(), t0.sin()], dim=-1)
    t1 = torch.rand(n1) * math.pi
    moon1 = torch.stack([1 - t1.cos(), 1 - t1.sin() - 0.5], dim=-1)
    x = torch.cat([moon0, moon1], dim=0)
    x = x + noise * torch.randn_like(x)
    return x.to(device)


class TorusMixture:
    """Mixture of wrapped Gaussians on the torus [0, L)^2 with an exact density.

    Wrapped-Gaussian density per mode is computed by summing image copies over a
    small range of wraps (±3 sigma is plenty). Used to validate circular splines
    and to get a known target for ESS.
    """

    def __init__(self, L: float = 1.0, std: float = 0.06, n_modes: int = 4, seed: int = 0):
        self.L = float(L)
        self.std = float(std)
        g = torch.Generator().manual_seed(seed)
        self.means = torch.rand(n_modes, 2, generator=g) * self.L
        self.weights = torch.full((n_modes,), 1.0 / n_modes)

    def sample(self, n: int, device=None) -> torch.Tensor:
        idx = torch.multinomial(self.weights, n, replacement=True)
        x = self.means[idx] + self.std * torch.randn(n, 2)
        return torch.remainder(x, self.L).to(device)

    def log_prob(self, x: torch.Tensor, n_wrap: int = 3) -> torch.Tensor:
        # x: [..., 2] assumed in [0, L). Sum wrapped images per mode.
        L, s = self.L, self.std
        offsets = torch.arange(-n_wrap, n_wrap + 1, device=x.device, dtype=x.dtype) * L
        # Build all 2D wrap offsets: [(2W+1)^2, 2]
        ox, oy = torch.meshgrid(offsets, offsets, indexing="ij")
        wrap = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)  # [O, 2]
        means = self.means.to(x.device, x.dtype)                      # [M, 2]
        # diff: [..., M, O, 2]
        diff = x[..., None, None, :] - means[:, None, :] - wrap[None, :, :]
        sq = (diff ** 2).sum(-1)                                      # [..., M, O]
        log_norm = -math.log(2 * math.pi * s * s)                    # 2D Gaussian norm
        comp = log_norm - 0.5 * sq / (s * s)                          # [..., M, O]
        # logsumexp over images and modes, with uniform mode weights
        log_w = torch.log(self.weights.to(x.device, x.dtype))         # [M]
        comp = comp + log_w[:, None]
        return torch.logsumexp(comp.flatten(start_dim=-2), dim=-1)
