"""Autoregressive particle flow on a periodic box (the neighbour-aware fix).

Place particles one at a time in a fixed order; particle i's position is drawn from
a conditional shaped by the ALREADY-PLACED real particles x_<i (excluded volume).
Exact triangular log-det: log q(x) = sum_i log q(x_i | x_<i). Generation conditions on
real placed particles -> no auxiliary, no noise, no base-bootstrap (the failure mode of
the coupling/augmented attempts).

Per particle we autoregress over the two coords too:
  x_i^0 ~ circular-spline density shaped by placed particles (periodic DeepSets context),
  x_i^1 ~ circular-spline density shaped by placed particles weighted by closeness in
          coord-0 to x_i^0 (so it can open a gap in the right column).
Everything conditions only on (x_<i, x_i^{<c}) -> a valid normalised density, exact log q.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .transforms_spline import CircularRQSplineElementwise


def _wrap(d, L):
    return d - L * torch.round(d / L)


class ARParticleFlow(nn.Module):
    def __init__(self, N, L, num_bins=8, hidden=64, n_rbf=16, cutoff=2.4):
        super().__init__()
        self.N = N
        self.L = float(L)
        self.cutoff = float(cutoff)
        self.spline = CircularRQSplineElementwise(num_bins=num_bins, L=L)
        P = self.spline.params_per_dim
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = (n_rbf / cutoff) ** 2
        self.n_rbf = n_rbf
        # context for coord-0: periodic DeepSets over placed particles
        self.node = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head0 = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, P))
        # context for coord-1: message from each placed j weighted by closeness in coord-0
        self.edge1 = nn.Sequential(nn.Linear(n_rbf + 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head1 = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, P))
        self.base_logp = -2 * N * math.log(self.L)

    def _periodic(self, x):  # x [...,2] -> [...,4] sin/cos
        ang = 2 * math.pi * x / self.L
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def _ctx0(self, placed):  # placed [B,k,2] -> [B,hidden]
        B = placed.shape[0]
        if placed.shape[1] == 0:
            return torch.zeros(B, self.node[-2].out_features, device=placed.device, dtype=placed.dtype)
        return self.node(self._periodic(placed)).sum(dim=1)

    def _ctx1(self, placed, x0):  # placed [B,k,2], x0 [B] -> [B,hidden]
        B = placed.shape[0]
        h = self.edge1[-2].out_features
        if placed.shape[1] == 0:
            return torch.zeros(B, h, device=placed.device, dtype=placed.dtype)
        d0 = _wrap(x0[:, None] - placed[:, :, 0], self.L)            # [B,k] closeness in coord-0
        rbf = torch.exp(-self.gamma * (d0[..., None].abs() - self.centers) ** 2)  # [B,k,n_rbf]
        yfeat = self._periodic(placed[:, :, 1:2]).squeeze(-2) if False else torch.cat(
            [torch.sin(2 * math.pi * placed[:, :, 1:2] / self.L),
             torch.cos(2 * math.pi * placed[:, :, 1:2] / self.L)], dim=-1)  # [B,k,2]
        msg = self.edge1(torch.cat([rbf, yfeat], dim=-1))            # [B,k,hidden]
        w = torch.exp(-(d0 ** 2) / (2 * 0.5 ** 2))[..., None]        # localise in coord-0
        return (msg * w).sum(dim=1)

    def log_prob(self, x):  # x [B,N,2] -> [B]
        B = x.shape[0]
        lp = torch.zeros(B, device=x.device, dtype=x.dtype)
        for i in range(self.N):
            placed = x[:, :i, :]
            p0 = self.head0(self._ctx0(placed))                      # [B,P]
            z0, ld0 = self.spline.inverse(x[:, i, 0:1], p0[:, None, :])
            p1 = self.head1(self._ctx1(placed, x[:, i, 0]))
            z1, ld1 = self.spline.inverse(x[:, i, 1:2], p1[:, None, :])
            lp = lp + ld0.squeeze(-1) + ld1.squeeze(-1)
        return lp + self.base_logp

    @torch.no_grad()
    def sample(self, B, device=None):
        x = torch.zeros(B, self.N, 2, device=device)
        logdet = torch.zeros(B, device=device)
        for i in range(self.N):
            placed = x[:, :i, :]
            z0 = torch.rand(B, 1, device=device) * self.L
            p0 = self.head0(self._ctx0(placed))
            x0, ld0 = self.spline.forward(z0, p0[:, None, :])
            z1 = torch.rand(B, 1, device=device) * self.L
            p1 = self.head1(self._ctx1(placed, x0.squeeze(-1)))
            x1, ld1 = self.spline.forward(z1, p1[:, None, :])
            x[:, i, 0] = x0.squeeze(-1)
            x[:, i, 1] = x1.squeeze(-1)
            logdet = logdet + ld0.squeeze() + ld1.squeeze()
        return x, self.base_logp - logdet
