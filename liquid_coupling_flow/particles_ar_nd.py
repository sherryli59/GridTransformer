"""Dimension-general autoregressive particle flow (d=2 for cheap toys, d=3 for real LJ).

Generalizes particles_ar.ARParticleFlow to arbitrary spatial dimension. Places
particles one at a time (fixed order); for particle i, autoregresses over its d
coordinates:
  coord 0   ~ circular-spline shaped by a DeepSets context over the placed particles
  coord c>0 ~ circular-spline shaped by messages from placed particles, weighted by
              proximity to particle i in the already-decided coords (0..c-1)
Everything conditions only on (x_<i, x_i^{<c}) -> valid normalised density, exact
triangular log q. For d=2 this reduces to the original ARParticleFlow.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .transforms_spline import CircularRQSplineElementwise


def _wrap(delta, L):
    return delta - L * torch.round(delta / L)


class ParticleARFlowND(nn.Module):
    def __init__(self, N, L, d=2, num_bins=24, hidden=128, n_rbf=16, cutoff=2.4,
                 local_w=0.3):
        super().__init__()
        self.N, self.L, self.d, self.cutoff = int(N), float(L), int(d), float(cutoff)
        self.local_w = float(local_w)
        self.spline = CircularRQSplineElementwise(num_bins=num_bins, L=L)
        P = self.spline.params_per_dim
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = (n_rbf / cutoff) ** 2
        self.n_rbf = n_rbf
        # coord 0: DeepSets over placed particles (periodic features of the full d-vector)
        self.node = nn.Sequential(nn.Linear(2 * d, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, P))
             for _ in range(d)])
        # coords 1..d-1: message net (rbf of decided-coord distance + periodic of placed coord c)
        self.edges = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_rbf + 2, hidden), nn.SiLU(),
                           nn.Linear(hidden, hidden), nn.SiLU()) for _ in range(d - 1)])
        self.hidden = hidden
        self.base_logp = -self.d * self.N * math.log(self.L)

    def _periodic(self, x):  # x [...,k] -> [...,2k] sin/cos on the torus
        ang = 2 * math.pi * x / self.L
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def _ctx0(self, placed):  # placed [B,k,d] -> [B,hidden]
        B = placed.shape[0]
        if placed.shape[1] == 0:
            return torch.zeros(B, self.hidden, device=placed.device, dtype=placed.dtype)
        return self.node(self._periodic(placed)).sum(dim=1)

    def _ctx_c(self, placed, partial, c):  # placed [B,k,d], partial [B,c] -> [B,hidden]
        B = placed.shape[0]
        if placed.shape[1] == 0:
            return torch.zeros(B, self.hidden, device=placed.device, dtype=placed.dtype)
        dd = _wrap(partial[:, None, :] - placed[:, :, :c], self.L)        # [B,k,c]
        dist = torch.sqrt((dd ** 2).sum(-1) + 1e-12)                      # [B,k]
        rbf = torch.exp(-self.gamma * (dist[..., None] - self.centers) ** 2)  # [B,k,n_rbf]
        pc = self._periodic(placed[:, :, c:c + 1])                       # [B,k,2]
        msg = self.edges[c - 1](torch.cat([rbf, pc], dim=-1))            # [B,k,hidden]
        w = torch.exp(-(dist ** 2) / (2 * self.local_w ** 2))[..., None]
        return (msg * w).sum(dim=1)

    def _params(self, placed, xi_partial, c):
        ctx = self._ctx0(placed) if c == 0 else self._ctx_c(placed, xi_partial, c)
        return self.heads[c](ctx)

    def log_prob(self, x):  # x [B,N,d] -> [B]
        B = x.shape[0]
        lp = torch.zeros(B, device=x.device, dtype=x.dtype)
        for i in range(self.N):
            placed = x[:, :i, :]
            for c in range(self.d):
                p = self._params(placed, x[:, i, :c], c)
                _, ld = self.spline.inverse(x[:, i, c:c + 1], p[:, None, :])
                lp = lp + ld.squeeze(-1)
        return lp + self.base_logp

    @torch.no_grad()
    def sample(self, B, device=None):
        x = torch.zeros(B, self.N, self.d, device=device)
        logdet = torch.zeros(B, device=device)
        for i in range(self.N):
            placed = x[:, :i, :]
            for c in range(self.d):
                p = self._params(placed, x[:, i, :c], c)
                z = torch.rand(B, 1, device=device) * self.L
                xc, ld = self.spline.forward(z, p[:, None, :])
                x[:, i, c] = xc.squeeze(-1)
                logdet = logdet + ld.squeeze()
        return x, self.base_logp - logdet
