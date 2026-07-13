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
from liquid_coupling_flow.ka3d_scaffold_ar import ball_squash
from liquid_coupling_flow.ka_energy import SIGMA


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


def coarse_tilt_V(model, anchor_y, cage_x, cage_s, cage_v, sj, R, head, n_chunk=512):
    """True 3D pairwise-potential tilt V[s,i] = sum_j phi(|squash(anchor_y_s+center_i) - cage_x_{s,j}|,
    sj_s, cage_s_{s,j}) over the causal kNN cage, evaluated at every one of `head`'s coarse-cell centers.
    Position construction mirrors KA3DScaffoldEBM._V_axis (ka3d_scaffold_ebm.py:70-84) verbatim: candidate
    u -> y = anchor_y + u -> x = ball_squash(y, R); frameless (the cavity models have use_frame=False so
    _V_axis's `frame` einsum reduces to identity -> no frame argument here). Chunks the coarse cells in
    blocks of `n_chunk` to bound memory (S x n_chunk x Kc pairwise distances live at once).

    anchor_y: [S,3] (pre-squash). cage_x: [S,Kc,3] (already squashed/config-space). cage_s,cage_v: [S,Kc]
    (cage_v False = padding, contributes 0). sj: [S]. Returns V: [S, head.n_cells].
    """
    S = anchor_y.shape[0]
    n_cells = head.centers.shape[0]
    V = anchor_y.new_zeros(S, n_cells)
    for c0 in range(0, n_cells, n_chunk):
        centers = head.centers[c0:c0 + n_chunk]                            # [c,3]
        c = centers.shape[0]
        y = anchor_y[:, None, :] + centers[None, :, :]                     # [S,c,3]
        x, _ = ball_squash(y, R)
        d = (x[:, :, None, :] - cage_x[:, None, :, :]).norm(dim=-1)        # [S,c,Kc]
        phi = model._phi_pair(d, sj[:, None].expand(S, c),
                              cage_s[:, None, :].expand(S, c, cage_s.shape[1]), model.phi, model.pair_emb)
        V[:, c0:c0 + c] = (phi * cage_v[:, None, :]).sum(-1)
    return V


def cells_allowed(head, anchor_y, sj, cage_x, cage_s, cage_v, R, cut, n_chunk=512):
    """Conservative hard min-separation mask at coarse-cell granularity (frameless): forbid cell i iff
    for SOME valid cage neighbour j, dist(squash(anchor_y+center_i), cage_x_j) < cut*sigma(sj,s_j) -
    half_diag_pos, where half_diag_pos is a per-(slot,cell) UPPER BOUND on the post-squash image radius
    of the cell (max over its 8 corners of |squash(anchor_y+corner) - squash(anchor_y+center)|, computed
    once per call alongside the center pass). Because half_diag_pos upper-bounds the squash image, a
    forbidden cell's ENTIRE image lies within the clash zone for that neighbour -> the mask never rejects
    a cell that contains a truly-safe point. Fallback: rows that forbid every cell revert to allow-all
    (keeps the coarse categorical normalised). Memory-heavy (S x n_cells x 8 corners) -> chunked over cells.

    anchor_y: [S,3]. sj: [S]. cage_x: [S,Kc,3]. cage_s,cage_v: [S,Kc]. Returns bool [S, head.n_cells].
    """
    S = anchor_y.shape[0]
    n_cells = head.centers.shape[0]
    dev, dt = anchor_y.device, anchor_y.dtype
    hw = head.cw / 2.0
    corner_offsets = torch.tensor(
        [[sx * hw, sy * hw, sz * hw] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        device=dev, dtype=dt)                                              # [8,3]
    sig_tab = torch.as_tensor(SIGMA, device=dev, dtype=dt)
    sig = sig_tab[sj.long()[:, None], cage_s.long()]                       # [S,Kc]
    valid = cage_v.bool()                                                  # [S,Kc]

    allowed = torch.ones(S, n_cells, dtype=torch.bool, device=dev)
    for c0 in range(0, n_cells, n_chunk):
        centers = head.centers[c0:c0 + n_chunk]                            # [c,3]
        c = centers.shape[0]
        y_center = anchor_y[:, None, :] + centers[None, :, :]              # [S,c,3]
        pos_center, _ = ball_squash(y_center, R)

        corners = centers[:, None, :] + corner_offsets[None, :, :]         # [c,8,3]
        y_corner = anchor_y[:, None, None, :] + corners[None, :, :, :]     # [S,c,8,3]
        pos_corner, _ = ball_squash(y_corner, R)
        half_diag_pos = (pos_corner - pos_center[:, :, None, :]).norm(dim=-1).amax(-1)   # [S,c]

        d = torch.cdist(pos_center, cage_x)                                # [S,c,Kc]
        thresh = cut * sig[:, None, :] - half_diag_pos[:, :, None]         # [S,c,Kc]
        clash = ((d < thresh) & valid[:, None, :]).any(-1)                 # [S,c]
        allowed[:, c0:c0 + c] = ~clash

    allowed = allowed | (~allowed.any(-1, keepdim=True))
    return allowed
