from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineCoupling(nn.Module):
    """Diagonal affine map on [...,3] with tanh-bounded log-scale (identity when params=0).
    forward:  u_out = u * exp(s) + t,  s = smax*tanh(params[...,:3]),  t = params[...,3:]
    logdet = sum(s). inverse: u = (u_out - t) * exp(-s)."""
    def __init__(self, smax: float = 2.0):
        super().__init__()
        self.smax = float(smax)

    def _st(self, params):
        s = self.smax * torch.tanh(params[..., :3])
        t = params[..., 3:]
        return s, t

    def forward(self, u, params):
        s, t = self._st(params)
        return u * torch.exp(s) + t, s.sum(-1)

    def inverse(self, u_out, params):
        s, t = self._st(params)
        return (u_out - t) * torch.exp(-s)
