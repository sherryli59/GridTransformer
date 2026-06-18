"""Clean replication of the GridTransformer recipe that worked on LJ27, for the 2D KA
glass (with species). Key ideas (NOT the failed absolute-spline flow):
  - Hilbert-curve ordering of particles -> consecutive tokens are spatial neighbours.
  - DISPLACEMENT-from-previous representation (min-image) -> excluded volume is a direct,
    LOCAL constraint on the relative coordinate (not an emergent absolute-placement one).
  - FACTORIZED BINNED (categorical) head: each displacement coordinate is a categorical
    over n_bins -> can assign ~0 probability to forbidden (overlapping) bins, which a
    continuous spline could not do.
  - Causal transformer AR over the curve sequence; species as conditioning.
Exact (piecewise-uniform) log q: log p = sum_i [log P(bin_x) + log P(bin_y|bin_x)] - 2N log(L/n_bins).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def hilbert_index(cell_xy, R):
    """Vectorised Hilbert index of integer cells [...,2] on an R x R grid (R a power of 2)."""
    x = cell_xy[..., 0].clone(); y = cell_xy[..., 1].clone()
    d = torch.zeros_like(x)
    s = R // 2
    while s > 0:
        rx = ((x & s) > 0).long()
        ry = ((y & s) > 0).long()
        d += s * s * ((3 * rx) ^ ry)
        swap = ry == 0
        flip = swap & (rx == 1)
        x = torch.where(flip, s - 1 - x, x)
        y = torch.where(flip, s - 1 - y, y)
        xn = torch.where(swap, y, x)
        yn = torch.where(swap, x, y)
        x, y = xn, yn
        s //= 2
    return d


def curve_order(x, L, R=32):
    """Permutation(s) ordering particles along the Hilbert curve. x [...,N,2] -> [...,N]."""
    cell = (x / L * R).long().clamp(0, R - 1)
    return torch.argsort(hilbert_index(cell, R), dim=-1)


def _wrap_pm(d, L):                                   # -> [-L/2, L/2)
    return d - L * torch.round(d / L)


def to_displacements(x, L):
    """x [B,N,2] (already curve-ordered) -> displacements [B,N,2] in [-L/2,L/2).
    delta_0 = x_0 centred; delta_i = min-image(x_i - x_{i-1})."""
    d = torch.empty_like(x)
    d[:, 0] = _wrap_pm(x[:, 0], L)
    d[:, 1:] = _wrap_pm(x[:, 1:] - x[:, :-1], L)
    return d


def from_displacements(d, L):
    """Inverse of to_displacements -> absolute positions in [0,L)."""
    x = torch.cumsum(d, dim=1)
    return torch.remainder(x, L)


class KAGridformer(nn.Module):
    def __init__(self, L, d=2, n_bins=96, n_species=2, d_model=192, n_head=6,
                 n_layer=6, R=32):
        super().__init__()
        self.L, self.d, self.n_bins, self.R = float(L), int(d), int(n_bins), int(R)
        self.bin_w = self.L / self.n_bins
        self.tok = nn.Linear(2 * d, d_model)               # embed continuous displacement (sin/cos)
        self.sp_emb = nn.Embedding(n_species, d_model)
        self.start = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, 4096, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_head, 4 * d_model, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layer)
        self.head_x = nn.Linear(d_model, n_bins)            # P(bin_x | context, species)
        self.bin_x_emb = nn.Embedding(n_bins, d_model)
        self.head_y = nn.Linear(d_model, n_bins)            # P(bin_y | context, species, bin_x)
        self.base_logp = 0.0                                # piecewise-uniform; volume term added in log_prob

    def _periodic(self, dlt):                               # [...,2] disp -> [...,4]
        a = 2 * math.pi * dlt / self.L
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

    def _bin(self, dlt):                                    # disp [-L/2,L/2) -> bin idx [0,n_bins)
        return ((dlt + self.L / 2) / self.bin_w).long().clamp(0, self.n_bins - 1)

    def _bin_center(self, b):
        return (b.float() + 0.5) * self.bin_w - self.L / 2

    def _context(self, dlt, s):
        """Causal hidden states for predicting each particle. dlt,[B,N,2]; s [B,N] long.
        input_j = embed(delta_{j-1}, species_{j-1}); prediction at j conditions on
        particles <j; we add species_j afterwards in the heads."""
        B, N = s.shape
        inp = self.tok(self._periodic(dlt)) + self.sp_emb(s)             # [B,N,d]
        seq = torch.cat([self.start.expand(B, 1, -1), inp[:, :-1]], 1)   # shift right
        seq = seq + self.pos[:, :N]
        mask = torch.triu(torch.ones(N, N, device=s.device, dtype=torch.bool), 1)
        h = self.tr(seq, mask=mask)                                      # [B,N,d], h_j sees <j
        return h + self.sp_emb(s)                                        # add species_j to predict particle j

    def log_prob(self, x, s):
        s = s.long(); B, N = x.shape[0], x.shape[1]
        s = s.expand(B, N).clone() if s.dim() == 1 else s
        order = curve_order(x, self.L, self.R)                           # [B,N] canonicalise
        x = torch.gather(x, 1, order[..., None].expand(-1, -1, self.d))
        s = torch.gather(s, 1, order)
        dlt = to_displacements(x, self.L)
        bx, by = self._bin(dlt[..., 0]), self._bin(dlt[..., 1])
        h = self._context(dlt, s)                                        # [B,N,d]
        lx = F.log_softmax(self.head_x(h), -1)
        ly = F.log_softmax(self.head_y(h + self.bin_x_emb(bx)), -1)
        lp = lx.gather(-1, bx[..., None]).squeeze(-1) + ly.gather(-1, by[..., None]).squeeze(-1)
        return lp.sum(1) - self.d * N * math.log(self.bin_w)            # piecewise-uniform density

    @torch.no_grad()
    def sample(self, B, s, device=None):
        s = s.long().to(device); N = s.shape[-1]
        s = s.expand(B, N) if s.dim() == 1 else s
        dlt = torch.zeros(B, N, 2, device=device)
        for j in range(N):
            inp = self.tok(self._periodic(dlt)) + self.sp_emb(s)
            seq = torch.cat([self.start.expand(B, 1, -1), inp[:, :-1]], 1)[:, :j + 1] + self.pos[:, :j + 1]
            mask = torch.triu(torch.ones(j + 1, j + 1, device=device, dtype=torch.bool), 1)
            h = self.tr(seq, mask=mask)[:, j] + self.sp_emb(s[:, j])     # [B,d]
            bx = torch.multinomial(F.softmax(self.head_x(h), -1), 1).squeeze(-1)
            by = torch.multinomial(F.softmax(self.head_y(h + self.bin_x_emb(bx)), -1), 1).squeeze(-1)
            dx = self._bin_center(bx) + (torch.rand(B, device=device) - 0.5) * self.bin_w
            dy = self._bin_center(by) + (torch.rand(B, device=device) - 0.5) * self.bin_w
            dlt[:, j, 0] = dx; dlt[:, j, 1] = dy
        return from_displacements(dlt, self.L), s
