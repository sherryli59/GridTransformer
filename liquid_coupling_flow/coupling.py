"""Coupling layer: the workhorse of the flow.

Split coordinates by a binary mask b (1 = identity/condition, 0 = transform):

    params = conditioner(b * x)                # only sees identity dims
    y      = b * x + (1 - b) * T(x; params)    # transform the rest, elementwise
    logdet = sum over the (1-b) dims of log|dT/dx|

Because the identity dims are unchanged, `forward` and `inverse` compute the SAME
params from them -> the layer is trivially invertible and its Jacobian is
triangular (cheap, exact log-det).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def alternating_masks(dim: int, n_layers: int) -> list[torch.Tensor]:
    """Binary masks that alternate which coordinates are transformed.

    For dim=2 this gives [1,0], [0,1], [1,0], ... so that after two layers every
    coordinate has been transformed conditioned on the other.
    """
    masks = []
    for i in range(n_layers):
        m = torch.tensor([float((j + i) % 2 == 0) for j in range(dim)])
        masks.append(m)
    return masks


class CouplingLayer(nn.Module):
    def __init__(self, mask: torch.Tensor, conditioner: nn.Module, transform):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.conditioner = conditioner
        self.transform = transform

    def _params(self, x: torch.Tensor) -> torch.Tensor:
        # Conditioner only sees the identity dims (others zeroed by the mask).
        return self.conditioner(x * self.mask)

    def forward(self, z: torch.Tensor):
        """base -> data direction."""
        params = self._params(z)
        y, ld = self.transform.forward(z, params)
        x = self.mask * z + (1.0 - self.mask) * y
        logdet = ((1.0 - self.mask) * ld).sum(-1)
        return x, logdet

    def inverse(self, x: torch.Tensor):
        """data -> base direction."""
        params = self._params(x)  # identity dims unchanged => same params as forward
        z, ld = self.transform.inverse(x, params)
        z = self.mask * x + (1.0 - self.mask) * z
        logdet = ((1.0 - self.mask) * ld).sum(-1)
        return z, logdet
