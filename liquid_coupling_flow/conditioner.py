"""Conditioner networks: map the identity dims to per-dim transform parameters.

For toys this is a plain MLP. The eventual liquid model swaps this for a LOCAL
geometric-attention transformer (cutoff over neighbours) — that is the piece that
makes the flow size-transferable. The interface is fixed here:

    conditioner(x: [..., dim]) -> params: [..., dim, params_per_dim]
"""

from __future__ import annotations

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

        # Zero-init the last layer so the flow starts as the identity map
        # (params = 0 -> affine identity / spline identity). Stabilises training.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(*x.shape[:-1], self.dim, self.params_per_dim)
