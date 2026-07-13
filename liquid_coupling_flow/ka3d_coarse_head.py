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
import torch.nn.functional as F
from liquid_coupling_flow.ka3d_scaffold_ar import ball_squash
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
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


def fine_clashfree_cube(head, anchor_y, ci, sj, cage_x, cage_s, cage_v, R, cut):
    """The full nf^3 within-cell clash-free tensor for the NESTED min-sep mask (the geometrically-correct
    declasher: mask fine-a by 'any (b,c) survives', fine-b by 'any c survives', fine-c directly -- so a
    clashing (a,b) column is forbidden AT the a/b stage, not discovered too late by a last-axis mask).
    Enumerates all nf^3 fine positions in cell `ci` and flags those >= cut*sigma+margin from every valid
    placed neighbour. Exact (deterministic function of ci+cage, identical at sample & score). Returns bool
    [*L, nf, nf, nf] indexed [.,fa,fb,fc]. Memory nf^3*Kc per row -> keep nf=8 (512 fine cells)."""
    dev, dt = anchor_y.device, anchor_y.dtype
    nf, bwf = head.nf, head.bwf
    lo = head.cell_center(ci) - head.cw / 2.0                              # [*L,3]
    g = (torch.arange(nf, device=dev, dtype=dt) + 0.5) * bwf               # [nf]
    Lsh = ci.shape
    ga, gb, gc = torch.meshgrid(g, g, g, indexing="ij")                    # [nf,nf,nf] each
    u = torch.stack([lo[..., None, None, None, 0] + ga, lo[..., None, None, None, 1] + gb,
                     lo[..., None, None, None, 2] + gc], -1)                # [*L,nf,nf,nf,3]
    pos, _ = ball_squash(anchor_y[..., None, None, None, :] + u, R)        # [*L,nf,nf,nf,3]
    flat = pos.reshape(*Lsh, nf ** 3, 3)
    d = torch.cdist(flat, cage_x)                                          # [*L,nf^3,Kc]
    sig = torch.as_tensor(SIGMA, device=dev, dtype=dt)[sj.long()[..., None], cage_s.long()]   # [*L,Kc]
    margin = bwf * (3.0 ** 0.5) / 2.0
    clash = ((d < (cut * sig + margin)[..., None, :]) & cage_v.bool()[..., None, :]).any(-1)  # [*L,nf^3]
    return (~clash).reshape(*Lsh, nf, nf, nf)


def fine_c_allowed(head, anchor_y, ci, fa, fb, sj, cage_x, cage_s, cage_v, R, cut):
    """Fine-stage hard min-separation mask on the LAST fine axis (c). Given the coarse cell `ci` and the
    already-chosen fine bins (fa, fb), each candidate fine-c bin's 3D position is fully determined, so we
    forbid the fine-c bins that place the particle within cut*sigma of any valid placed neighbour. Unlike
    the conservative coarse `cells_allowed` (which forbids only fully-inside cells and so LEAKS straddling
    cells -- the measured 'coarse-leak', 52-71% of clashes), this is fine-grained (bin width bwf~0.039) and
    AGGRESSIVE: threshold cut*sigma + margin(=bwf*sqrt3/2, covers the sub-bin dither). It stays unbiased
    because cut<=0.9 + margin < the 0.95 data min-gap floor, so no real data placement is forbidden.
    Exactness: depends only on (ci, fa, fb, cage) -- all available at BOTH sample and score time. Returns
    bool [*, head.nf]. Leading dims *L match the caller ([M] sampler / [M*s_chunk] scorer).
    """
    dev, dt = anchor_y.device, anchor_y.dtype
    bwf = head.bwf
    lo = head.cell_center(ci) - head.cw / 2.0                              # [*L,3]
    fc_grid = (torch.arange(head.nf, device=dev, dtype=dt) + 0.5) * bwf    # [nf]
    Lsh = ci.shape
    u = torch.empty(*Lsh, head.nf, 3, device=dev, dtype=dt)
    u[..., 0] = (lo[..., 0] + (fa.to(dt) + 0.5) * bwf)[..., None]
    u[..., 1] = (lo[..., 1] + (fb.to(dt) + 0.5) * bwf)[..., None]
    u[..., 2] = lo[..., 2:3] + fc_grid
    pos, _ = ball_squash(anchor_y[..., None, :] + u, R)                    # [*L,nf,3]
    d = torch.cdist(pos, cage_x)                                           # [*L,nf,Kc]
    sig = torch.as_tensor(SIGMA, device=dev, dtype=dt)[sj.long()[..., None], cage_s.long()]  # [*L,Kc]
    margin = bwf * (3.0 ** 0.5) / 2.0
    thresh = (cut * sig + margin)[..., None, :]                            # [*L,1,Kc]
    clash = ((d < thresh) & cage_v.bool()[..., None, :]).any(-1)          # [*L,nf]
    allowed = ~clash
    return allowed | (~allowed.any(-1, keepdim=True))                      # fallback: all-forbidden -> allow all


def _V_fine_axis(model, anchor_y, cage_x, cage_s, cage_v, sj, R, u_fixed, axis, grid, net, emb):
    """V over the 8 fine bins of `grid` on `axis`: candidate u = u_fixed with `axis` swept to `grid`
    -> y = anchor_y + u (frameless) -> x = ball_squash(y, R) -> pairwise phi to the causal kNN cage.
    Mirrors KA3DScaffoldEBM._V_axis / KA3DScaffoldEBMBatched._V_axis_b but takes the candidate grid
    explicitly (the coarse-cell fine grid is a per-slot ABSOLUTE offset, not `flow`'s global bin grid).

    anchor_y: [S,3]. u_fixed: [S,3] (off-axis coords already at their current partial position; the
    `axis` entry is overwritten by `grid` and is otherwise ignored). grid: [S,nf]. cage_x: [S,Kc,3].
    cage_s,cage_v: [S,Kc]. sj: [S]. Returns V: [S,nf]."""
    S, nb = u_fixed.shape[0], grid.shape[1]
    u = u_fixed[:, None, :].expand(S, nb, 3).clone()
    u[:, :, axis] = grid
    y = anchor_y[:, None, :] + u                                                # frameless
    x, _ = ball_squash(y, R)
    d = (x[:, :, None, :] - cage_x[:, None, :, :]).norm(dim=-1)                 # [S,nb,Kc]
    phi = model._phi_pair(d, sj[:, None].expand(S, nb),
                          cage_s[:, None, :].expand(S, nb, cage_s.shape[1]), net, emb)
    return (phi * cage_v[:, None, :]).sum(-1)                                   # [S,nb]


class KA3DScaffoldEBMCoarse(KA3DScaffoldEBMBatched):
    """Unbatched scorer override that routes the inherited `log_prob_pair` training loss through the
    coarse-cell-then-fine-bin factorization instead of the sequential per-axis Cat3Head. Everything
    else (context transformer, species head, cage kNN, boundary handling) is inherited unchanged from
    KA3DScaffoldEBMBatched -- only `_tilted_lp_u` (called by `log_prob_pair`/`block_log_prob`) is
    overridden. `self.flow` (Cat3Head) is still constructed by the parent but unused by this path;
    it is kept so sample_block/etc. on the parent class remain callable (not exact for this head)."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.coarse = CoarseFineHead(self.d_model, u_range=self.cat_range)

    def _tilted_lp_u(self, h_e, u, frame, anchor_y, cage_x, cage_s, cage_v, sj, R, n_chunk=16):
        """Exact log q(u) = log p(cell) + log p(fa|cell) + log p(fb|cell,fa) + log p(fc|cell,fa,fb)
        - 3 log bwf, teacher-forced (S slots). `frame` is unused: cavity models are frameless."""
        assert not self.use_frame
        co = self.coarse
        S, dev, dt = u.shape[0], u.device, u.dtype

        ci = co.cell_index(u)                                                   # [S]
        centers = co.cell_center(ci)                                            # [S,3]
        cell_lo = centers - co.cw / 2.0
        fa, fb_, fc = co.fine_bin(u, ci).unbind(-1)                             # [S] each

        base_cell = co.head_coarse(h_e)                                         # [S,n_cells]
        V_cell = u.new_zeros(S, co.n_cells)
        grid8 = (torch.arange(co.nf, device=dev, dtype=dt) + 0.5) * co.bwf      # [nf]
        grid_a = cell_lo[:, 0:1] + grid8[None, :]                               # [S,nf]
        grid_b = cell_lo[:, 1:2] + grid8[None, :]
        grid_c = cell_lo[:, 2:3] + grid8[None, :]

        x_at_fa = cell_lo[:, 0] + (fa.to(dt) + 0.5) * co.bwf                    # [S]
        y_at_fb = cell_lo[:, 1] + (fb_.to(dt) + 0.5) * co.bwf                   # [S]
        zeros = torch.zeros(S, device=dev, dtype=dt)
        u_fix_a = torch.stack([zeros, centers[:, 1], centers[:, 2]], -1)        # axis0 overwritten
        u_fix_b = torch.stack([x_at_fa, zeros, centers[:, 2]], -1)              # axis1 overwritten
        u_fix_c = torch.stack([x_at_fa, y_at_fb, zeros], -1)                    # axis2 overwritten

        Vfa = u.new_zeros(S, co.nf)
        Vfb = u.new_zeros(S, co.nf)
        Vfc = u.new_zeros(S, co.nf)
        for c0 in range(0, S, n_chunk):
            sl = slice(c0, c0 + n_chunk)
            args = (anchor_y[sl], cage_x[sl], cage_s[sl], cage_v[sl], sj[sl], R)
            V_cell[sl] = coarse_tilt_V(self, *args, co)
            Vfa[sl] = _V_fine_axis(self, *args, u_fix_a[sl], 0, grid_a[sl], self.phi_a, self.pair_emb_a)
            Vfb[sl] = _V_fine_axis(self, *args, u_fix_b[sl], 1, grid_b[sl], self.phi_b, self.pair_emb_b)
            Vfc[sl] = _V_fine_axis(self, *args, u_fix_c[sl], 2, grid_c[sl], self.phi, self.pair_emb)

        lp_cell = F.log_softmax(base_cell - V_cell, -1).gather(-1, ci[:, None]).squeeze(-1)

        cell_e = co.cell_emb(ci)
        femb_a_v = co.femb_a(fa)
        femb_b_v = co.femb_b(fb_)
        logits_fa = co.head_fa(h_e + cell_e) - Vfa                             # training path: no declash mask
        logits_fb = co.head_fb(h_e + cell_e + femb_a_v) - Vfb
        logits_fc = co.head_fc(h_e + cell_e + femb_a_v + femb_b_v) - Vfc
        lp_fa = F.log_softmax(logits_fa, -1).gather(-1, fa[:, None]).squeeze(-1)
        lp_fb = F.log_softmax(logits_fb, -1).gather(-1, fb_[:, None]).squeeze(-1)
        lp_fc = F.log_softmax(logits_fc, -1).gather(-1, fc[:, None]).squeeze(-1)

        return lp_cell + lp_fa + lp_fb + lp_fc - co._log_bwf3

    # ---- batched-over-M scorer/sampler overrides (mirrors KA3DScaffoldEBMBatched) ----
    def block_log_prob_b(self, xo, so, block_mask, bnd, s_bnd, R, pos_temp=1.0, min_sep=None, top_p=None,
                         nested=False):
        """Coarse-head override: threads `nested` (absent from the inherited signature) into the scorer.
        Reuses the parent's assembly and only stashes `nested` on the instance for _tilted_lp_u_b to read
        (the parent's inherited call site passes only pos_temp/min_sep/top_p)."""
        self._nested_flag = nested
        try:
            return super().block_log_prob_b(xo, so, block_mask, bnd, s_bnd, R,
                                            pos_temp=pos_temp, min_sep=min_sep, top_p=top_p)
        finally:
            self._nested_flag = False

    def _tilted_lp_u_b(self, h_e, u, anchor_y, cage_x, cage_s, cage_v, sj, R, s_chunk=4, pos_temp=1.0,
                       min_sep=None, top_p=None, nested=None):
        """Batched [M,S] exact scorer for the coarse-cell-then-fine-bin factorization. Mirrors the
        unbatched `_tilted_lp_u` override's math, generalized to an extra leading M dim by flattening
        M*s_chunk into the module-level `coarse_tilt_V`/`_V_fine_axis`/`cells_allowed` helpers' native
        S-batch dim (anchor_y is shared across M -- frameless, same scaffold -- so it is tiled, not
        indexed, before flattening). Shapes mirror KA3DScaffoldEBMBatched._tilted_lp_u_b exactly (h_e/u
        [M,S,*], cage_* [M,S,Kc,*], sj [M,S]) so the inherited `block_log_prob_b` routes here unchanged.
        pos_temp divides EVERY logits tensor (coarse + 3 fine) inside its log_softmax. min_sep applies
        `cells_allowed` as a -inf mask on the COARSE logits only (no fine-c mask; brief 2026-07-13)."""
        co = self.coarse
        if nested is None:                                                     # inherited call site can't pass it
            nested = getattr(self, "_nested_flag", False)
        M, S, _ = u.shape
        dev, dt = u.device, u.dtype

        ci = co.cell_index(u)                                                  # [M,S]
        centers = co.cell_center(ci)                                           # [M,S,3]
        cell_lo = centers - co.cw / 2.0
        fa, fb_, fc = co.fine_bin(u, ci).unbind(-1)                            # [M,S] each

        base_cell = co.head_coarse(h_e)                                        # [M,S,n_cells]
        V_cell = u.new_zeros(M, S, co.n_cells)
        grid8 = (torch.arange(co.nf, device=dev, dtype=dt) + 0.5) * co.bwf     # [nf]
        grid_a = cell_lo[..., 0:1] + grid8                                     # [M,S,nf]
        grid_b = cell_lo[..., 1:2] + grid8
        grid_c = cell_lo[..., 2:3] + grid8

        x_at_fa = cell_lo[..., 0] + (fa.to(dt) + 0.5) * co.bwf                 # [M,S]
        y_at_fb = cell_lo[..., 1] + (fb_.to(dt) + 0.5) * co.bwf                # [M,S]
        zeros = torch.zeros(M, S, device=dev, dtype=dt)
        u_fix_a = torch.stack([zeros, centers[..., 1], centers[..., 2]], -1)   # [M,S,3]
        u_fix_b = torch.stack([x_at_fa, zeros, centers[..., 2]], -1)
        u_fix_c = torch.stack([x_at_fa, y_at_fb, zeros], -1)

        Vfa = u.new_zeros(M, S, co.nf)
        Vfb = u.new_zeros(M, S, co.nf)
        Vfc = u.new_zeros(M, S, co.nf)
        allowed_full = torch.ones(M, S, co.n_cells, dtype=torch.bool, device=dev) if min_sep is not None else None
        fc_allow_full = torch.ones(M, S, co.nf, dtype=torch.bool, device=dev) if min_sep is not None else None
        fa_allow_full = torch.ones(M, S, co.nf, dtype=torch.bool, device=dev) if (min_sep is not None and nested) else None
        fb_allow_full = torch.ones(M, S, co.nf, dtype=torch.bool, device=dev) if (min_sep is not None and nested) else None

        for c0 in range(0, S, s_chunk):
            sl = slice(c0, c0 + s_chunk)
            s_ = min(s_chunk, S - c0)
            ay = anchor_y[sl][None].expand(M, s_, 3).reshape(M * s_, 3)        # tile shared anchor over M
            cx = cage_x[:, sl].reshape(M * s_, cage_x.shape[2], 3)
            cs = cage_s[:, sl].reshape(M * s_, cage_s.shape[2])
            cv = cage_v[:, sl].reshape(M * s_, cage_v.shape[2])
            sjf = sj[:, sl].reshape(M * s_)
            V_cell[:, sl] = coarse_tilt_V(self, ay, cx, cs, cv, sjf, R, co).reshape(M, s_, co.n_cells)
            Vfa[:, sl] = _V_fine_axis(self, ay, cx, cs, cv, sjf, R,
                                      u_fix_a[:, sl].reshape(M * s_, 3), 0, grid_a[:, sl].reshape(M * s_, co.nf),
                                      self.phi_a, self.pair_emb_a).reshape(M, s_, co.nf)
            Vfb[:, sl] = _V_fine_axis(self, ay, cx, cs, cv, sjf, R,
                                      u_fix_b[:, sl].reshape(M * s_, 3), 1, grid_b[:, sl].reshape(M * s_, co.nf),
                                      self.phi_b, self.pair_emb_b).reshape(M, s_, co.nf)
            Vfc[:, sl] = _V_fine_axis(self, ay, cx, cs, cv, sjf, R,
                                      u_fix_c[:, sl].reshape(M * s_, 3), 2, grid_c[:, sl].reshape(M * s_, co.nf),
                                      self.phi, self.pair_emb).reshape(M, s_, co.nf)
            if min_sep is not None:
                allow = cells_allowed(co, ay, sjf, cx, cs, cv, R, min_sep)
                allowed_full[:, sl] = allow.reshape(M, s_, co.n_cells)
                cif = ci[:, sl].reshape(M * s_); faf = fa[:, sl].reshape(M * s_); fbf = fb_[:, sl].reshape(M * s_)
                if nested:                                                     # nf^3 cube -> per-axis availability
                    cube = fine_clashfree_cube(co, ay, cif, sjf, cx, cs, cv, R, min_sep)    # [M*s_,nf,nf,nf]
                    arf = torch.arange(M * s_, device=dev)
                    fa_allow_full[:, sl] = cube.any(-1).any(-1).reshape(M, s_, co.nf)
                    fb_allow_full[:, sl] = cube[arf, faf].any(-1).reshape(M, s_, co.nf)
                    fc_allow_full[:, sl] = cube[arf, faf, fbf].reshape(M, s_, co.nf)
                else:
                    fca = fine_c_allowed(co, ay, cif, faf, fbf, sjf, cx, cs, cv, R, min_sep)
                    fc_allow_full[:, sl] = fca.reshape(M, s_, co.nf)

        logits_cell = (base_cell - V_cell) / pos_temp
        if min_sep is not None:
            logits_cell = logits_cell.masked_fill(~allowed_full, float("-inf"))
        lp_cell = F.log_softmax(logits_cell, -1).gather(-1, ci[..., None]).squeeze(-1)

        cell_e = co.cell_emb(ci)
        femb_a_v = co.femb_a(fa)
        femb_b_v = co.femb_b(fb_)
        logits_fa = (co.head_fa(h_e + cell_e) - Vfa) / pos_temp
        logits_fb = (co.head_fb(h_e + cell_e + femb_a_v) - Vfb) / pos_temp
        logits_fc = (co.head_fc(h_e + cell_e + femb_a_v + femb_b_v) - Vfc) / pos_temp
        if nested and min_sep is not None:                                     # same nested masks as the sampler
            logits_fa = logits_fa.masked_fill(~(fa_allow_full | ~fa_allow_full.any(-1, keepdim=True)), float("-inf"))
            logits_fb = logits_fb.masked_fill(~(fb_allow_full | ~fb_allow_full.any(-1, keepdim=True)), float("-inf"))
        if min_sep is not None:
            logits_fc = logits_fc.masked_fill(~(fc_allow_full | ~fc_allow_full.any(-1, keepdim=True)), float("-inf"))
        lp_fa = F.log_softmax(logits_fa, -1).gather(-1, fa[..., None]).squeeze(-1)
        lp_fb = F.log_softmax(logits_fb, -1).gather(-1, fb_[..., None]).squeeze(-1)
        lp_fc = F.log_softmax(logits_fc, -1).gather(-1, fc[..., None]).squeeze(-1)

        return lp_cell + lp_fa + lp_fb + lp_fc - co._log_bwf3

    @torch.no_grad()
    def sample_block_b(self, xo, so, block_mask, bnd, s_bnd, R, gen=None, pos_temp=1.0, min_sep=None,
                       nested=False):
        """Regenerate the block for M configs in parallel through the coarse-cell-then-fine-bin head.
        nested=True (with min_sep): full within-cell nested declash mask (fine-a/b/c by column availability
        over the nf^3 cube) -- the geometrically-correct min separation; block_log_prob_b MUST match.
        Copy of KA3DScaffoldEBMBatched.sample_block_b (ka3d_ebm_batched.py:179-244) with ONLY the
        position section (the old a/b/c Vax/la/lb/lc block) replaced by: coarse-cell categorical (tilted
        by `coarse_tilt_V`, optionally `cells_allowed`-masked at COARSE granularity, /pos_temp) -> sample
        cell -> three 8-way fine axes (tilted via `_V_fine_axis`, same net/off-axis convention as the
        unbatched `_tilted_lp_u` override, conditioned via cell_emb/femb_a/femb_b) -> dither within the
        fine bin -> u = fine center + dither -> y = anchor + u -> ball_squash -> position. Returns
        xo_new [M,n,3], so_new [M,n], logq [M]."""
        co = self.coarse
        M = xo.shape[0]
        order, anchors, anchor_y, rblock, kind, valid, slot_feat, m, n = self._prep(xo, so, block_mask, bnd, s_bnd, R)
        n_ret = int((~block_mask).sum())
        xo, so = xo[:, order].clone(), so[:, order].clone()
        combined = torch.cat([bnd[None].expand(M, m, 3), xo], 1).clone()
        scomb = torch.cat([s_bnd[None].expand(M, m), so], 1).clone()
        blk_s = so[:, n_ret:]                                                        # per-config block species budget
        rem = torch.stack([(blk_s == 0).sum(1).float(), (blk_s == 1).sum(1).float()], -1)   # [M,2]
        rbias = self._r_bias(R, xo.device, xo.dtype)
        ar = torch.arange(M, device=xo.device)
        logq = xo.new_zeros(M)
        for jj in range(n_ret, n):
            vj = (torch.arange(m + n, device=xo.device) < (m + jj))[None]            # [1, m+n]
            h = self._fc_batched(combined, scomb, kind, vj, anchors[jj:jj + 1], slot_feat[jj:jj + 1])[:, 0] + rbias  # [M,d]
            ls = F.log_softmax(self.head_species(h).masked_fill(rem <= 0, float("-inf")), -1)
            sj = torch.multinomial(ls.exp(), 1, generator=gen).squeeze(-1)           # [M]
            he = h + self.sp_out_emb(sj)
            cage_x, cage_s, cage_v = self._cage_knn_b(combined, scomb, vj, anchors[jj:jj + 1])   # [M,1,K,*]
            cx, cs, cv = cage_x[:, 0], cage_s[:, 0], cage_v[:, 0]
            ay = anchor_y[jj:jj + 1].expand(M, 3)

            # ---- coarse cell ----
            V_cell = coarse_tilt_V(self, ay, cx, cs, cv, sj, R, co)                  # [M,n_cells]
            logits_cell = co.head_coarse(he) - V_cell
            if min_sep is not None:
                allowed_cell = cells_allowed(co, ay, sj, cx, cs, cv, R, min_sep)
                logits_cell = logits_cell.masked_fill(~allowed_cell, float("-inf"))
            l_cell = F.log_softmax(logits_cell / pos_temp, -1)
            ci = torch.multinomial(l_cell.exp(), 1, generator=gen).squeeze(-1)       # [M]

            cell_e = co.cell_emb(ci)
            centers = co.cell_center(ci)                                            # [M,3]
            cell_lo = centers - co.cw / 2.0
            grid8 = (torch.arange(co.nf, device=xo.device, dtype=xo.dtype) + 0.5) * co.bwf   # [nf]
            zM = torch.zeros(M, 3, device=xo.device, dtype=xo.dtype)
            # NESTED min-sep: the nf^3 within-cell clash-free cube -> mask fine-a by any(b,c), fine-b by
            # any(c|a), fine-c by (a,b). Forbids a clashing (a,b) column AT the a/b stage (the a,b-plane leak
            # the last-axis masks couldn't reach). fallback: all-clash row -> allow all (keeps normalization).
            cube = fine_clashfree_cube(co, ay, ci, sj, cx, cs, cv, R, min_sep) if (min_sep is not None and nested) else None

            def _msk(logits, allow):
                allow = allow | (~allow.any(-1, keepdim=True))
                return logits.masked_fill(~allow, float("-inf"))

            # ---- fine axis a ----
            u_fix_a = torch.stack([zM[:, 0], centers[:, 1], centers[:, 2]], -1)
            grid_a = cell_lo[:, 0:1] + grid8[None, :]
            Vfa = _V_fine_axis(self, ay, cx, cs, cv, sj, R, u_fix_a, 0, grid_a, self.phi_a, self.pair_emb_a)
            logits_fa = (co.head_fa(he + cell_e) - Vfa) / pos_temp
            if cube is not None:
                logits_fa = _msk(logits_fa, cube.any(-1).any(-1))                   # any (b,c)
            l_fa = F.log_softmax(logits_fa, -1)
            fa = torch.multinomial(l_fa.exp(), 1, generator=gen).squeeze(-1)
            x_at_fa = cell_lo[:, 0] + (fa.to(xo.dtype) + 0.5) * co.bwf
            femb_a_v = co.femb_a(fa)

            # ---- fine axis b ----
            u_fix_b = torch.stack([x_at_fa, zM[:, 1], centers[:, 2]], -1)
            grid_b = cell_lo[:, 1:2] + grid8[None, :]
            Vfb = _V_fine_axis(self, ay, cx, cs, cv, sj, R, u_fix_b, 1, grid_b, self.phi_b, self.pair_emb_b)
            logits_fb = (co.head_fb(he + cell_e + femb_a_v) - Vfb) / pos_temp
            if cube is not None:
                logits_fb = _msk(logits_fb, cube[ar, fa].any(-1))                   # any c given a
            l_fb = F.log_softmax(logits_fb, -1)
            fb_ = torch.multinomial(l_fb.exp(), 1, generator=gen).squeeze(-1)
            y_at_fb = cell_lo[:, 1] + (fb_.to(xo.dtype) + 0.5) * co.bwf
            femb_b_v = co.femb_b(fb_)

            # ---- fine axis c ----
            u_fix_c = torch.stack([x_at_fa, y_at_fb, zM[:, 2]], -1)
            grid_c = cell_lo[:, 2:3] + grid8[None, :]
            Vfc = _V_fine_axis(self, ay, cx, cs, cv, sj, R, u_fix_c, 2, grid_c, self.phi, self.pair_emb)
            logits_fc = (co.head_fc(he + cell_e + femb_a_v + femb_b_v) - Vfc) / pos_temp
            if cube is not None:
                logits_fc = _msk(logits_fc, cube[ar, fa, fb_])                      # (a,b) fixed
            elif min_sep is not None:                                               # last-axis-only fine mask
                fc_allow = fine_c_allowed(co, ay, ci, fa, fb_, sj, cx, cs, cv, R, min_sep)
                logits_fc = logits_fc.masked_fill(~fc_allow, float("-inf"))
            l_fc = F.log_softmax(logits_fc, -1)
            fc = torch.multinomial(l_fc.exp(), 1, generator=gen).squeeze(-1)

            dith = (torch.rand(M, 3, device=xo.device, dtype=xo.dtype, generator=gen) - 0.5) * co.bwf
            fine_ctr = co.fine_ctr(ci, torch.stack([fa, fb_, fc], -1))               # [M,3]
            u = fine_ctr + dith
            y = anchor_y[jj][None] + u                                              # frameless
            pos, logdet_xy = ball_squash(y, R)                                      # [M,3], [M]
            xo[:, jj], so[:, jj] = pos, sj
            combined[:, m + jj], scomb[:, m + jj] = pos, sj
            rem[ar, sj] -= 1
            lp_u = (l_cell.gather(1, ci[:, None]).squeeze(1) + l_fa.gather(1, fa[:, None]).squeeze(1)
                    + l_fb.gather(1, fb_[:, None]).squeeze(1) + l_fc.gather(1, fc[:, None]).squeeze(1)
                    - co._log_bwf3)
            logq = logq + ls.gather(1, sj[:, None]).squeeze(1) + lp_u - logdet_xy
        xo_new, so_new = xo.new_empty(M, n, 3), so.new_empty(M, n)
        xo_new[:, order], so_new[:, order] = xo, so
        return xo_new, so_new, logq
