"""Species-aware autoregressive flow for the 2D KA glass (Arm 1 of the benchmark).

Learns q(positions | species): species are FIXED conditioning, not generated. Extends
the monatomic ParticleARFlowND with species embeddings. Each particle's position spline
is conditioned on (its own species) + (messages from placed neighbours carrying the
neighbour's species), so the model can learn PAIR-TYPE excluded volume (A-A / A-B / B-B
have different sigma). Base = uniform positions on the torus; exact triangular log q.

This is the AR contender; locality refinement (curve order + cutoff attention) and the
spatial coupling-flow arm follow. The species-conditioning INTERFACE here is shared by
both arms so the benchmark stays fair.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from .transforms_spline import CircularRQSplineElementwise


def _wrap(delta, L):
    return delta - L * torch.round(delta / L)


class KAARFlow(nn.Module):
    def __init__(self, N, L, d=2, n_species=2, emb=8, num_bins=24, hidden=128,
                 n_rbf=16, cutoff=2.4, local_w=0.4, ctx0_reduce="sum", ctxc_reduce="sum"):
        super().__init__()
        self.N, self.L, self.d, self.cutoff = int(N), float(L), int(d), float(cutoff)
        self.local_w = float(local_w)
        self.ctx0_reduce, self.ctxc_reduce = ctx0_reduce, ctxc_reduce
        self.spline = CircularRQSplineElementwise(num_bins=num_bins, L=L)
        P = self.spline.params_per_dim
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = (n_rbf / cutoff) ** 2
        self.n_rbf = n_rbf
        self.hidden = hidden
        self.sp_emb = nn.Embedding(n_species, emb)                      # species embedding
        # coord-0 DeepSets over placed: periodic position (2d) + neighbour species emb
        self.node = nn.Sequential(nn.Linear(2 * d + emb, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        # per-coord heads take (context + OWN species emb)
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden + emb, hidden), nn.SiLU(), nn.Linear(hidden, P))
             for _ in range(d)])
        # coords 1..d-1: message = rbf(decided-coord dist) + periodic(neighbour coord c) + neighbour species emb
        self.edges = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_rbf + 2 + emb, hidden), nn.SiLU(),
                           nn.Linear(hidden, hidden), nn.SiLU()) for _ in range(d - 1)])
        self.base_logp = -self.d * self.N * math.log(self.L)

    def _periodic(self, x):
        ang = 2 * math.pi * x / self.L
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def _ctx0(self, placed, s_placed):                                 # [B,k,d],[B,k] -> [B,hidden]
        B = placed.shape[0]
        if placed.shape[1] == 0:
            return torch.zeros(B, self.hidden, device=placed.device, dtype=placed.dtype)
        feat = torch.cat([self._periodic(placed), self.sp_emb(s_placed)], dim=-1)
        h = self.node(feat)
        return h.mean(1) if self.ctx0_reduce == "mean" else h.sum(1)

    def _ctx_c(self, placed, s_placed, partial, c):                    # partial [B,c]
        B = placed.shape[0]
        if placed.shape[1] == 0:
            return torch.zeros(B, self.hidden, device=placed.device, dtype=placed.dtype)
        dd = _wrap(partial[:, None, :] - placed[:, :, :c], self.L)
        dist = torch.sqrt((dd ** 2).sum(-1) + 1e-12)                   # [B,k]
        rbf = torch.exp(-self.gamma * (dist[..., None] - self.centers) ** 2)
        pc = self._periodic(placed[:, :, c:c + 1])                     # [B,k,2]
        msg = self.edges[c - 1](torch.cat([rbf, pc, self.sp_emb(s_placed)], dim=-1))
        w = torch.exp(-(dist ** 2) / (2 * self.local_w ** 2))[..., None]
        agg = (msg * w).sum(1)
        if self.ctxc_reduce == "mean":
            agg = agg / (w.sum(1) + 1e-6)
        return agg

    def _params(self, placed, s_placed, xi_partial, s_i, c):
        ctx = (self._ctx0(placed, s_placed) if c == 0
               else self._ctx_c(placed, s_placed, xi_partial, c))
        return self.heads[c](torch.cat([ctx, self.sp_emb(s_i)], dim=-1))

    def log_prob(self, x, s):                                          # x [B,N,d], s [N] or [B,N]
        B = x.shape[0]
        s = (s.expand(B, self.N) if s.dim() == 1 else s).long()
        lp = torch.zeros(B, device=x.device, dtype=x.dtype)
        for i in range(self.N):
            placed, s_pl = x[:, :i, :], s[:, :i]
            for c in range(self.d):
                p = self._params(placed, s_pl, x[:, i, :c], s[:, i], c)
                _, ld = self.spline.inverse(x[:, i, c:c + 1], p[:, None, :])
                lp = lp + ld.squeeze(-1)
        return lp + self.base_logp

    @torch.no_grad()
    def sample(self, B, s, device=None):
        s = s.to(device); s = (s.expand(B, self.N) if s.dim() == 1 else s).long()
        x = torch.zeros(B, self.N, self.d, device=device)
        logdet = torch.zeros(B, device=device)
        for i in range(self.N):
            placed, s_pl = x[:, :i, :], s[:, :i]
            for c in range(self.d):
                p = self._params(placed, s_pl, x[:, i, :c], s[:, i], c)
                z = torch.rand(B, 1, device=device) * self.L
                xc, ld = self.spline.forward(z, p[:, None, :])
                x[:, i, c] = xc.squeeze(-1)
                logdet = logdet + ld.squeeze()
        return x, self.base_logp - logdet
