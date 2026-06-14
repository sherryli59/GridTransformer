"""The Flow: a stack of coupling layers over a base distribution.

Conventions:
  layer.forward(z) : base -> data,  returns (x, +logdet)
  layer.inverse(x) : data -> base,  returns (z, +logdet of the inverse map)

  log p(x)         = base.log_prob(z) + sum_layers inverse-logdet   (z = f^{-1}(x))
  sample           : z ~ base; x = f(z); log p(x) = base.log_prob(z) - sum forward-logdet
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Flow(nn.Module):
    def __init__(self, base: nn.Module, layers):
        super().__init__()
        self.base = base
        self.layers = nn.ModuleList(layers)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        logdet = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z)
            logdet = logdet + ld
        return self.base.log_prob(z) + logdet

    def sample(self, n: int, device=None, dtype=None):
        """Returns (x, log_prob(x)) with x ~ flow."""
        z = self.base.sample(n, device=device, dtype=dtype)
        logdet = torch.zeros(n, device=z.device, dtype=z.dtype)
        x = z
        for layer in self.layers:
            x, ld = layer.forward(x)
            logdet = logdet + ld
        log_prob = self.base.log_prob(z) - logdet
        return x, log_prob

    def forward_kld(self, x: torch.Tensor) -> torch.Tensor:
        """Max-likelihood loss: E[-log p(x)] over data x."""
        return -self.log_prob(x).mean()
