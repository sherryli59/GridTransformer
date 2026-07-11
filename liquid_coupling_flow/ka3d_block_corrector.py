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


class CageConditioner(nn.Module):
    """Per-active-particle affine params from a distance-RBF + species-pair message over the conditioning
    set (other block particles + cage). Physical positions; batched [M,A,*] x [M,C,*] -> [M,A,6]. Last
    layer zero-init => identity coupling at init (composition starts == base)."""
    def __init__(self, d_model: int = 96, n_rbf: int = 12, rbf_max: float = 3.0, n_species: int = 2):
        super().__init__()
        self.n_species = n_species
        mu = torch.linspace(0.0, rbf_max, n_rbf)
        self.register_buffer("mu", mu); self.w = float(mu[1] - mu[0])
        self.pair = nn.Embedding(n_species * n_species, 8)
        self.msg = nn.Sequential(nn.Linear(n_rbf + 8, d_model), nn.SiLU(),
                                 nn.Linear(d_model, d_model), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 6))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def params(self, active_x, active_s, cond_x, cond_s, R):
        d = (active_x[:, :, None, :] - cond_x[:, None, :, :]).norm(dim=-1)          # [M,A,C]
        r = torch.exp(-((d[..., None] - self.mu) ** 2) / (2 * self.w ** 2))         # [M,A,C,n_rbf]
        pr = (active_s[:, :, None] * self.n_species + cond_s[:, None, :]).clamp(0, self.n_species ** 2 - 1)
        msg = self.msg(torch.cat([r, self.pair(pr)], -1))                          # [M,A,C,d]
        agg = msg.mean(2)                                                          # permutation-invariant over cond
        return self.head(agg)                                                     # [M,A,6]
