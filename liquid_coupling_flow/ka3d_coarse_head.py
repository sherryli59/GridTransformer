"""Coarse-joint-then-fine categorical position head for the KA3D cavity AR.

Motivation (measured 2026-07-13): the sequential per-axis head commits the a-axis through a hedged
marginal with a slice-blind tilt (Vax evaluated at (a,0,0)), so clashes are locked in the a/b plane
where the c-only hard mask cannot act (clash 31.6->31.9%). Here the FIRST commitment is a joint 16^3
coarse cell whose tilt is the true 3D pair energy at the cell center, and the exact hard min-sep mask
acts at cell granularity BEFORE any commitment. Fine stage: 8 bins/axis within the cell (slice error
bounded by cw/2=0.156). 16*8 = 128/axis == the old resolution; logq comparable.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class CoarseFineHead(nn.Module):
    def __init__(self, d_model, u_range=2.5, n_coarse=16, n_fine=8):
        super().__init__()
        self.rng, self.nc, self.nf = float(u_range), int(n_coarse), int(n_fine)
        self.n_cells = self.nc ** 3
        self.cw = 2 * self.rng / self.nc
        self.bwf = self.cw / self.nf
        self.half_diag = self.cw * math.sqrt(3.0) / 2
        self._log_bwf3 = 3.0 * math.log(self.bwf)
        self.head_coarse = nn.Linear(d_model, self.n_cells)
        self.cell_emb = nn.Embedding(self.n_cells, d_model)
        self.head_fa = nn.Linear(d_model, self.nf)
        self.head_fb = nn.Linear(d_model, self.nf)
        self.head_fc = nn.Linear(d_model, self.nf)
        self.femb_a = nn.Embedding(self.nf, d_model)
        self.femb_b = nn.Embedding(self.nf, d_model)
        # cached [4096,3] cell centers (buffer -> moves with .to(device))
        ax = (torch.arange(self.nc) + 0.5) * self.cw - self.rng
        gx, gy, gz = torch.meshgrid(ax, ax, ax, indexing="ij")
        self.register_buffer("centers", torch.stack([gx, gy, gz], -1).reshape(-1, 3), persistent=False)

    def _axcell(self, u1):
        return ((u1 + self.rng) / self.cw).long().clamp(0, self.nc - 1)

    def cell_index(self, u):
        a, b, c = self._axcell(u[..., 0]), self._axcell(u[..., 1]), self._axcell(u[..., 2])
        return (a * self.nc + b) * self.nc + c

    def cell_center(self, idx):
        return self.centers[idx]

    def fine_bin(self, u, idx):
        lo = self.cell_center(idx) - self.cw / 2
        return ((u - lo) / self.bwf).long().clamp(0, self.nf - 1)

    def fine_ctr(self, idx, fb):
        lo = self.cell_center(idx) - self.cw / 2
        return lo + (fb.float() + 0.5) * self.bwf
