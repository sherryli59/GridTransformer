"""Conditioner networks: map the identity dims to per-dim transform parameters.

For toys this is an MLP. The eventual liquid model swaps this for a LOCAL
geometric-attention transformer (cutoff over neighbours) — that is the piece that
makes the flow size-transferable. The interface is fixed here:

    conditioner(x: [..., dim]) -> params: [..., dim, params_per_dim]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MLPConditioner(nn.Module):
    def __init__(self, dim: int, params_per_dim: int, hidden: int = 128, n_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.params_per_dim = params_per_dim

        layers: list[nn.Module] = [nn.Linear(dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, dim * params_per_dim)]
        self.net = nn.Sequential(*layers)

        # Zero-init last layer -> identity flow at init (stabilises training).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(*x.shape[:-1], self.dim, self.params_per_dim)


class PeriodicMLPConditioner(nn.Module):
    """MLP conditioner for periodic coordinates: encodes each input coordinate as
    (sin, cos) of 2*pi*x/L so the network is continuous across the box seam (0 == L).
    Masked (transform) dims arrive as 0 -> (0, 1), contributing nothing, as intended.
    """

    def __init__(self, dim: int, params_per_dim: int, L: float = 1.0,
                 hidden: int = 128, n_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.params_per_dim = params_per_dim
        self.L = float(L)

        layers: list[nn.Module] = [nn.Linear(2 * dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, dim * params_per_dim)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = 2 * math.pi / self.L
        feat = torch.cat([torch.sin(k * x), torch.cos(k * x)], dim=-1)
        out = self.net(feat)
        return out.view(*x.shape[:-1], self.dim, self.params_per_dim)
