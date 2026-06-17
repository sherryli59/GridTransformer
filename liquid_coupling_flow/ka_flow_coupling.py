"""Species-aware spatial coupling flow for the 2D KA glass (Arm 2 of the benchmark).

Index-partitioned, ONE-COORDINATE-PER-BLOCK coupling -> exactly invertible AND genuinely
local, with NO autoregressive generation (drift-free). A block transforms coordinate c of
one index-set (even/odd) conditioned on (a) the frozen complement set and (b) the active
set's OTHER coordinate (fixed this block, used to localize). Species enter as conditioning
(neighbour species in messages, own species in the head) so pair-type excluded volume is
learnable. Base = uniform positions on the torus.

Drift-free (all active particles transform in parallel) is the property the benchmark pits
against the AR arm's exposure-bias drift.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from .transforms_spline import CircularRQSplineElementwise


def _wrap(delta, L):
    return delta - L * torch.round(delta / L)


class KACouplingFlow(nn.Module):
    def __init__(self, N, L, n_cycles=3, n_species=2, emb=8, num_bins=16, hidden=96,
                 n_rbf=16, cutoff=2.4, local_w=0.4):
        super().__init__()
        self.N, self.L = int(N), float(L)
        self.local_w = float(local_w)
        self.spline = CircularRQSplineElementwise(num_bins=num_bins, L=L)
        P = self.spline.params_per_dim
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = (n_rbf / cutoff) ** 2
        self.sp_emb = nn.Embedding(n_species, emb)
        # block schedule: cycle over (active parity in {0,1}, coord in {0,1})
        self.meta = [(par, c) for _ in range(n_cycles) for par in (0, 1) for c in (0, 1)]
        self.edges = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_rbf + 2 + emb, hidden), nn.SiLU(),
                           nn.Linear(hidden, hidden), nn.SiLU()) for _ in self.meta])
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden + emb, hidden), nn.SiLU(), nn.Linear(hidden, P))
             for _ in self.meta])
        self.base_logp = -2 * self.N * math.log(self.L)

    def _periodic(self, x):                                            # [...,k] -> [...,2k]
        ang = 2 * math.pi * x / self.L
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def _params(self, b, x, s, act, frz, c):
        """spline params for active coord c. x [B,N,2], act/frz long index tensors."""
        xa, xf = x[:, act, :], x[:, frz, :]                            # [B,Na,2],[B,Nf,2]
        loc = 1 - c                                                    # localizing coord (the fixed one)
        dist = _wrap(xa[:, :, loc][:, :, None] - xf[:, :, loc][:, None, :], self.L).abs()  # [B,Na,Nf]
        rbf = torch.exp(-self.gamma * (dist[..., None] - self.centers) ** 2)               # [B,Na,Nf,R]
        Na, Nf = xa.shape[1], xf.shape[1]
        pc = self._periodic(xf[:, :, c:c + 1])[:, None, :, :].expand(-1, Na, -1, -1)        # [B,Na,Nf,2]
        fe = self.sp_emb(s[frz])[None, None].expand(xa.shape[0], Na, Nf, -1)               # [B,Na,Nf,E]
        msg = self.edges[b](torch.cat([rbf, pc, fe], dim=-1))                              # [B,Na,Nf,H]
        w = torch.exp(-(dist ** 2) / (2 * self.local_w ** 2))[..., None]
        ctx = (msg * w).sum(2)                                                             # [B,Na,H]
        ae = self.sp_emb(s[act])[None].expand(xa.shape[0], -1, -1)                         # [B,Na,E]
        return self.heads[b](torch.cat([ctx, ae], dim=-1))                                 # [B,Na,P]

    def _idx(self, par, device):
        i = torch.arange(self.N, device=device)
        act = i[i % 2 == par]; frz = i[i % 2 != par]
        return act, frz

    def log_prob(self, x, s):                                          # data -> base (reverse blocks)
        s = s.long().to(x.device)
        ld = torch.zeros(x.shape[0], device=x.device)
        x = x.clone()
        for b in reversed(range(len(self.meta))):
            par, c = self.meta[b]
            act, frz = self._idx(par, x.device)
            p = self._params(b, x, s, act, frz, c)                     # [B,Na,P]
            z, d = self.spline.inverse(x[:, act, c], p)                # [B,Na] data, [B,Na,P] params
            x[:, act, c] = z
            ld = ld + d.sum(-1)
        return ld + self.base_logp

    @torch.no_grad()
    def sample(self, B, s, device=None):                              # base -> data (forward blocks)
        s = s.long().to(device)
        x = torch.rand(B, self.N, 2, device=device) * self.L
        ld = torch.zeros(B, device=device)
        for b in range(len(self.meta)):
            par, c = self.meta[b]
            act, frz = self._idx(par, device)
            p = self._params(b, x, s, act, frz, c)
            y, d = self.spline.forward(x[:, act, c], p)
            x[:, act, c] = y
            ld = ld + d.sum(-1)
        return x, self.base_logp - ld
