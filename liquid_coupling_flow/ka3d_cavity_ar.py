"""3D boundary-conditioned local-frame AUTOREGRESSIVE generator for KA cavity interiors.

NOT a flow. Places interior (mobile) particles one at a time in a geometry-invariant
local frame built from ALREADY-PLACED particles (frozen boundary UNION placed interior):
  origin_j = soft centroid of placed particles near a fixed closed-form scaffold slot j;
  local coords (a,b,c) = x_j - origin_j  (a PHYSICAL length ~ a0 = rho^{-1/3}, size-invariant).
origin_j is a function of the PREFIX + boundary + scaffold_j only (NOT of x_j) => the
(a,b,c) <-> x_j map is a pure translation, |det|=1 => exact log-density. The learned
conditional sees ONLY neighbour positions relative to origin_j + species + boundary/interior
kind => translation-, rotation-(via augmentation), and SIZE-invariant (the transfer property
that curve-anchored models lacked; cf. ka_localframe.py, localframe-conditional-transfers).

Why AR (not the vetoed uniform-base flow): the frozen boundary IS most of the cage in a
cavity, so AR's 50%-future-neighbour residual (gap-is-residual-not-drift) is structurally
small; and AR gives EXACT log_prob -> exact IS/SMC proposal for P(q_c).

This module: the GEOMETRY CORE + an exact round-trip test. Network/heads/log_prob/sample
are added only after the geometry is proven bijective (run `python -m ...ka3d_cavity_ar`).
"""
from __future__ import annotations

import math
import torch

RHO = 1.2
A0 = RHO ** (-1.0 / 3.0)                 # ~0.941, physical particle spacing (size-invariant residual scale)
SIGMA_LOC = 1.0                          # locator kernel width (~first shell); affects QUALITY not EXACTNESS
W0_PRIOR = 0.25                          # pseudo-weight anchoring origin_j -> scaffold_j when no near neighbours


def morton_code_3d(x_rel, R, bits=10):
    """3D Morton (Z-order) code of center-relative coords in the ball |r|<R.
    A SPACE-FILLING sequence key: consecutive codes are (mostly) spatial neighbours, which
    gives the AR frame good locality and a size-invariant local residual. x_rel [n,3] -> [n] long.
    """
    q = ((x_rel + R) / (2.0 * R) * (1 << bits)).long().clamp(0, (1 << bits) - 1)   # [n,3] in [0,2^bits)
    code = torch.zeros(x_rel.shape[0], dtype=torch.long, device=x_rel.device)
    for i in range(bits):
        for dim in range(3):
            code |= ((q[:, dim] >> i) & 1) << (3 * i + dim)
    return code


def fibonacci_ball_scaffold(n_slots, R, rho=RHO, bits=10, device=None, dtype=torch.float32):
    """Closed-form scaffold filling the cavity ball, then MORTON-ORDERED so slot j is spatially
    adjacent to slot j+1 (same key as the particle order => scaffold_j lands near particle_j).

    Slot direction: golden-angle spiral; radius = (3 (j+0.5)/(4 pi rho))^(1/3) (radial order
    statistic at density rho). Data-INDEPENDENT (function of n_slots, R only) => identical at
    sample and log_prob time. Returns [n_slots, 3] center-relative, in Morton order.
    """
    j = torch.arange(n_slots, device=device, dtype=dtype)
    r = (3.0 * (j + 0.5) / (4.0 * math.pi * float(rho))).pow(1.0 / 3.0)
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    theta = 2.0 * math.pi * j / phi
    z = (1.0 - (2.0 * j + 1.0) / max(n_slots, 1)).clamp(-1.0, 1.0)               # cos(polar)
    rho_xy = (1.0 - z * z).clamp_min(0.0).sqrt()
    dirs = torch.stack([rho_xy * torch.cos(theta), rho_xy * torch.sin(theta), z], dim=-1)
    sc = dirs * r[:, None]
    return sc[torch.argsort(morton_code_3d(sc, R, bits))]                        # space-filling slot order


def cavity_order(x_rel, R, bits=10):
    """Canonical AR order for interior particles: MORTON (space-filling) code order.
    x_rel [n_in, 3] (center-relative). Returns a LongTensor permutation [n_in]."""
    return torch.argsort(morton_code_3d(x_rel, R, bits))


def _weights(placed_rel, scaffold_j, valid, sigma):
    """Gaussian locator weights of placed particles about scaffold_j, zeroed on invalid rows.
    placed_rel [M,3], scaffold_j [3], valid [M] bool. Returns [M]."""
    d2 = (placed_rel - scaffold_j[None]).square().sum(-1)
    w = torch.exp(-d2 / (2.0 * sigma * sigma))
    return w * valid.to(w.dtype)


def soft_origin(placed_rel, valid, scaffold_j, sigma=SIGMA_LOC, w0=W0_PRIOR):
    """origin_j = pseudo-count-anchored soft centroid of VALID placed particles near scaffold_j.

    A deterministic function of (placed prefix + boundary, scaffold_j) ONLY. The w0 pseudo-count
    at scaffold_j guarantees a well-defined origin with no near neighbours (origin -> scaffold_j)
    and is what makes j=0 well-posed; it does not depend on x_j, so exactness is preserved.
    placed_rel [M,3], valid [M] bool. Returns [3].
    """
    w = _weights(placed_rel, scaffold_j, valid, sigma)
    num = (w[:, None] * placed_rel).sum(0) + w0 * scaffold_j
    den = w.sum() + w0
    return num / den.clamp_min(1e-12)


def to_frame(interior_rel, bnd_rel, scaffold, R, sigma=SIGMA_LOC, w0=W0_PRIOR):
    """Teacher-forced encode: interior positions -> local coords (a,b,c), in CANONICAL order.

    interior_rel [n,3], bnd_rel [m,3] (both center-relative), scaffold [n,3] Morton-ordered.
    Returns (abc [n,3], origins [n,3], order [n]) with interior_rel[order] the Morton-sorted
    interior (aligned to the Morton-sorted scaffold). Boundary is always-placed.
    """
    order = cavity_order(interior_rel, R)
    xo = interior_rel[order]
    n, m = xo.shape[0], bnd_rel.shape[0]
    combined = torch.cat([bnd_rel, xo], dim=0)                       # [m+n,3]; interior at m..m+n-1
    idx = torch.arange(m + n, device=xo.device)
    abc = torch.empty(n, 3, device=xo.device, dtype=xo.dtype)
    origins = torch.empty(n, 3, device=xo.device, dtype=xo.dtype)
    for j in range(n):
        valid = (idx >= m) & (idx < (m + j))                        # interior[0..j-1] ONLY (directional origin)
        o = soft_origin(combined, valid, scaffold[j], sigma, w0)
        origins[j] = o
        abc[j] = xo[j] - o
    return abc, origins, order


def from_frame_seq(abc, bnd_rel, scaffold, sigma=SIGMA_LOC, w0=W0_PRIOR):
    """Sequential decode: local coords (a,b,c) -> interior positions (center-relative), canonical order.
    Inverse of to_frame given the SAME (bnd_rel, scaffold). abc [n,3]. Returns interior_rel [n,3]."""
    n, m = abc.shape[0], bnd_rel.shape[0]
    combined = torch.cat([bnd_rel, torch.zeros(n, 3, device=abc.device, dtype=abc.dtype)], dim=0)
    idx = torch.arange(m + n, device=abc.device)
    xo = torch.empty(n, 3, device=abc.device, dtype=abc.dtype)
    for j in range(n):
        valid = (idx >= m) & (idx < (m + j))                        # interior[0..j-1] ONLY (directional origin)
        o = soft_origin(combined, valid, scaffold[j], sigma, w0)
        xo[j] = o + abc[j]
        combined[m + j] = xo[j]                                     # this placement enters the next prefix
    return xo


import torch.nn as nn
import torch.nn.functional as F

GEOM_PERIODS_3D = torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0])       # fixed PHYSICAL periods (size-invariant)


def build_frames(nbr_rel, n_valid, col_tol=0.99):
    """Vectorized 3D Gram-Schmidt local frames (adapted from mw.mw_generator.build_frames).

    nbr_rel [.,k,3] displacement vectors sorted nearest-first (rows beyond n_valid arbitrary/invalid,
    possibly zero-norm -> treated as invalid). Returns Rf [.,3,3] with rows (e1,e2,e3), R R^T=I,
    det=+1 on every fallback path. Expressing the residual + neighbours in this frame BREAKS rotation
    symmetry, so the factorized head peaks at the true DIRECTIONAL residual instead of collapsing to
    the origin (the shell-center clash). Frame is a function of the PREFIX -> residual rotation |det|=1.
    """
    B, k, _ = nbr_rel.shape
    d1 = nbr_rel[:, 0]
    nz = nbr_rel.norm(dim=-1) > 1e-9
    e1 = F.normalize(torch.where(((n_valid >= 1) & nz[:, 0])[:, None], d1,
         torch.tensor([1., 0., 0.], device=d1.device).expand_as(d1)), dim=-1)
    cos = torch.einsum("bkd,bd->bk", F.normalize(nbr_rel, dim=-1), e1).abs()
    ok = (cos <= col_tol) & nz & (torch.arange(k, device=d1.device)[None] < n_valid[:, None]) \
         & (torch.arange(k, device=d1.device)[None] >= 1)
    idx2 = torch.where(ok.any(1), ok.float().argmax(1), torch.zeros_like(n_valid))
    d2 = torch.gather(nbr_rel, 1, idx2[:, None, None].expand(-1, 1, 3)).squeeze(1)
    axes = torch.eye(3, device=d1.device)
    ax = axes[e1.abs().argmin(-1)]
    use_ax = (~ok.any(1)) | (n_valid < 2)
    d2 = torch.where(use_ax[:, None], ax, d2)
    u2 = d2 - (d2 * e1).sum(-1, keepdim=True) * e1
    e2 = F.normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-2)


class KA3DCavityAR(nn.Module):
    """Boundary-conditioned local-frame AR generator (3D, positions+species, exact log-prob).

    Places interior particles in Morton order; slot j sees its KNN nearest ALREADY-PLACED
    particles (frozen boundary shell UNION placed interior) in the local frame at origin_j, plus
    species and a boundary/interior kind flag. Emits species s_j (canonical count-masked) then the
    three position bins (a, b|a, c|a,b). origin_j is a function of the prefix only => |det|=1 =>
    exact log-density. Fixed physical Fourier periods => size/radius transfer.
    """
    def __init__(self, d_model=128, n_head=4, n_layer=4, n_bins=64, arc_range=4.5, knn=20,
                 n_species=2, sigma=SIGMA_LOC, w0=W0_PRIOR, rho=RHO, knn_bnd=20, bnd_cutoff=3.0):
        super().__init__()
        self.d_model, self.n_bins, self.arc_range, self.knn = d_model, n_bins, arc_range, knn
        self.n_species, self.sigma, self.w0, self.rho = n_species, sigma, w0, rho
        self.knn_bnd, self.bnd_cutoff = knn_bnd, bnd_cutoff        # separate boundary stream (avoid drowning interior)
        self.bin_w = 2.0 * arc_range / n_bins
        self.register_buffer("periods", GEOM_PERIODS_3D)
        enc = 2 * 3 * self.periods.numel()                     # sin/cos x 3 dims x periods
        self.nbr_proj = nn.Linear(enc, d_model)
        self.sp_emb = nn.Embedding(n_species, d_model)
        self.kind_emb = nn.Embedding(2, d_model)               # 0 = placed interior, 1 = frozen boundary
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        # The neighbour cloud alone does not identify which canonical slot is
        # being generated (especially for the empty-prefix/free-cluster
        # steps).  Give the query explicit *scalar* slot geometry without
        # introducing a lab-frame direction: generation fraction, scaffold
        # radius / cavity radius, and physical cavity radius.  The same
        # features are available bit-for-bit in teacher forcing and rollout.
        self.slot_proj = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_head, 4 * d_model, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)   # deterministic across batch size
        self.head_species = nn.Linear(d_model, n_species)
        self.sp_out_emb = nn.Embedding(n_species, d_model)     # own species -> position heads
        self.head_a = nn.Linear(d_model, n_bins)
        self.head_b = nn.Linear(d_model, n_bins)
        self.head_c = nn.Linear(d_model, n_bins)
        self.bin_a_emb = nn.Embedding(n_bins, d_model)
        self.bin_b_emb = nn.Embedding(n_bins, d_model)

    def _periodic(self, rel):
        a = 2 * math.pi * rel.unsqueeze(-1) / self.periods
        return torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(-2)

    def _bin(self, v):
        return ((v + self.arc_range) / self.bin_w).long().clamp(0, self.n_bins - 1)

    def _bin_center(self, b):
        return (b.float() + 0.5) * self.bin_w - self.arc_range

    @staticmethod
    def _slot_features(scaffold_j, slot_index, n_slots, radius):
        """Rotation-invariant canonical-slot features, shape ``[S,3]``."""
        S = scaffold_j.shape[0]
        dev, dtype = scaffold_j.device, scaffold_j.dtype
        idx = torch.as_tensor(slot_index, device=dev, dtype=dtype).reshape(-1)
        if idx.numel() == 1 and S != 1:
            idx = idx.expand(S)
        if idx.numel() != S:
            raise ValueError(f"slot_index has {idx.numel()} entries for {S} query rows")
        frac = idx / max(int(n_slots) - 1, 1)
        rad = torch.as_tensor(radius, device=dev, dtype=dtype).reshape(-1)
        if rad.numel() == 1:
            rad = rad.expand(S)
        return torch.stack([frac, scaffold_j.norm(dim=-1) / rad.clamp_min(1e-6), rad / 2.5], dim=-1)

    def _frame_context(self, combined, scomb, kind, valid_row, origin, scaffold_j, slot_feat=None):
        """KNN transformer context in a LOCAL FRAME, with SEPARATE interior/boundary neighbour streams.

        The frame and the interior-structure tokens are built from INTERIOR neighbours only (identical
        to the boundary-free free cluster, which generates cleanly); nearby boundary atoms (within
        bnd_cutoff of the slot) are a distinct token stream for clash-avoidance. This stops the dense
        frozen shell from drowning the ~12 interior neighbours. combined [P,3], scomb/kind [P] (kind:
        1=boundary,0=interior), valid_row [S,P], origin/scaffold_j [S,3]. Returns (h [S,d], Rf [S,3,3])."""
        S, P = valid_row.shape
        d2 = (combined[None] - scaffold_j[:, None]).square().sum(-1)            # [S,P] proximity to slot
        is_bnd = (kind == 1)[None].expand(S, -1)                               # [S,P]
        valid_int = valid_row & ~is_bnd
        valid_bnd = valid_row & is_bnd & (d2 < self.bnd_cutoff ** 2)           # only RELEVANT nearby boundary

        def gather(mask, k_max):
            idx = d2.masked_fill(~mask, 1e18).topk(min(k_max, P), dim=1, largest=False).indices
            return idx, torch.gather(mask, 1, idx)

        idx_i, v_i = gather(valid_int, self.knn)
        nbr_i = combined[idx_i] - origin[:, None, :]                           # interior neighbours, WORLD [S,ki,3]
        if getattr(self, "use_frame", True):
            Rf = build_frames(nbr_i, v_i.sum(1))                               # frame from INTERIOR neighbours only
        else:
            Rf = torch.eye(3, device=combined.device, dtype=combined.dtype).expand(S, 3, 3)  # global orientation (2D recipe + rot-aug)
        idx_b, v_b = gather(valid_bnd, self.knn_bnd)
        nbr_b = combined[idx_b] - origin[:, None, :]

        def feats(idx, nbr, v):
            loc = torch.einsum('saj,skj->ska', Rf, nbr)                        # neighbours in the interior frame
            f = self.nbr_proj(self._periodic(loc)) + self.sp_emb(scomb[idx]) + self.kind_emb(kind[idx])
            return f * v[..., None]

        if slot_feat is None:
            # Backward-compatible diagnostic path. Production log_prob/sample
            # always supplies the explicit canonical-slot features.
            slot_feat = scaffold_j.new_zeros(S, 3)
        query = self.query.expand(S, 1, -1) + self.slot_proj(slot_feat)[:, None]
        seq = torch.cat([query, feats(idx_i, nbr_i, v_i), feats(idx_b, nbr_b, v_b)], 1)
        pad = torch.cat([torch.zeros(S, 1, dtype=torch.bool, device=seq.device), ~v_i, ~v_b], 1)
        h = self.tr(seq, src_key_padding_mask=pad)[:, 0]
        return h, Rf

    def _origins(self, combined, valid_row, scaffold):
        """Soft-centroid origins for all slots at once (teacher-forced). Returns [S,3]."""
        w = torch.exp(-(combined[None] - scaffold[:, None]).square().sum(-1) / (2 * self.sigma ** 2))
        w = w * valid_row.to(w.dtype)
        num = (w[..., None] * combined[None]).sum(1) + self.w0 * scaffold
        return num / (w.sum(1, keepdim=True) + self.w0)

    def log_prob_pair(self, interior_rel, s_in, bnd_rel, s_bnd, R, preordered=False):
        """Exact log-density of ONE carved cavity interior (center-relative coords). Scalar.
        preordered=True skips cavity_order (interior already in generation slot order) so that
        sample()'s logq matches log_prob exactly — the exactness gate."""
        n, m = interior_rel.shape[0], bnd_rel.shape[0]
        dev = interior_rel.device
        scaffold = fibonacci_ball_scaffold(n, R, self.rho, device=dev, dtype=interior_rel.dtype)
        if preordered:
            xo, so = interior_rel, s_in
        else:
            order = cavity_order(interior_rel, R)
            xo, so = interior_rel[order], s_in[order]
        combined = torch.cat([bnd_rel, xo], 0)
        scomb = torch.cat([s_bnd, so], 0)
        kind = torch.cat([torch.ones(m, dtype=torch.long, device=dev), torch.zeros(n, dtype=torch.long, device=dev)])
        idxp = torch.arange(m + n, device=dev)
        valid_row = idxp[None] < (m + torch.arange(n, device=dev))[:, None]     # context: boundary + interior prefix
        origin_mask = valid_row & (idxp[None] >= m)                            # origin: interior prefix ONLY (directional)
        origin = self._origins(combined, origin_mask, scaffold)
        slot_feat = self._slot_features(scaffold, torch.arange(n, device=dev), n, R)
        h, Rf = self._frame_context(combined, scomb, kind, valid_row, origin, scaffold,
                                    slot_feat=slot_feat)                       # Rf=frame (R is radius)
        # species log-prob with canonical count mask (budget before placing each slot)
        oh = F.one_hot(so, self.n_species).to(h.dtype)
        rem = oh.sum(0, keepdim=True) - (oh.cumsum(0) - oh)                     # [n, n_species] remaining budget
        s_logits = self.head_species(h).masked_fill(rem <= 0, float("-inf"))
        lp_s = F.log_softmax(s_logits, -1).gather(-1, so[:, None]).squeeze(-1)
        e = self.sp_out_emb(so)
        abc = torch.einsum('naj,nj->na', Rf, xo - origin)                      # residual in LOCAL frame
        ba, bb, bc = self._bin(abc[:, 0]), self._bin(abc[:, 1]), self._bin(abc[:, 2])
        la = F.log_softmax(self.head_a(h + e), -1).gather(-1, ba[:, None]).squeeze(-1)
        lb = F.log_softmax(self.head_b(h + e + self.bin_a_emb(ba)), -1).gather(-1, bb[:, None]).squeeze(-1)
        lc = F.log_softmax(self.head_c(h + e + self.bin_a_emb(ba) + self.bin_b_emb(bb)), -1).gather(-1, bc[:, None]).squeeze(-1)
        return (lp_s + la + lb + lc).sum() - 3 * n * math.log(self.bin_w)

    @torch.no_grad()
    def sample_pair(self, bnd_rel, s_bnd, n_A, n_B, R, return_logq=False):
        """Sequentially generate one cavity interior (positions + species) given a frozen boundary.
        Returns interior_rel [n,3], s_in [n] (Morton-slot order); optional exact logq."""
        n, m = n_A + n_B, bnd_rel.shape[0]
        dev = bnd_rel.device
        scaffold = fibonacci_ball_scaffold(n, R, self.rho, device=dev, dtype=bnd_rel.dtype)
        combined = torch.cat([bnd_rel, torch.zeros(n, 3, device=dev, dtype=bnd_rel.dtype)], 0)
        scomb = torch.cat([s_bnd, torch.zeros(n, dtype=torch.long, device=dev)], 0)
        kind = torch.cat([torch.ones(m, dtype=torch.long, device=dev), torch.zeros(n, dtype=torch.long, device=dev)])
        idxp = torch.arange(m + n, device=dev)
        rem = torch.tensor([[float(n_A), float(n_B)]], device=dev)
        xo = torch.zeros(n, 3, device=dev, dtype=bnd_rel.dtype); so = torch.zeros(n, dtype=torch.long, device=dev)
        logq = bnd_rel.new_zeros(())
        for j in range(n):
            valid_row = (idxp < (m + j))[None]                                 # context: boundary + interior prefix
            origin_mask = valid_row & (idxp >= m)[None]                        # origin: interior prefix ONLY (directional)
            origin = self._origins(combined, origin_mask, scaffold[j:j + 1])   # [1,3]
            slot_feat = self._slot_features(scaffold[j:j + 1], j, n, R)
            h, Rf = self._frame_context(combined, scomb, kind, valid_row, origin,
                                        scaffold[j:j + 1], slot_feat=slot_feat)  # Rf=frame (R=radius)
            s_logits = self.head_species(h).masked_fill(rem <= 0, float("-inf"))
            ls = F.log_softmax(s_logits, -1)
            sj = torch.multinomial(ls.exp(), 1).squeeze(-1)
            e = self.sp_out_emb(sj)
            la = F.log_softmax(self.head_a(h + e), -1)
            ba = torch.multinomial(la.exp(), 1).squeeze(-1)
            lb = F.log_softmax(self.head_b(h + e + self.bin_a_emb(ba)), -1)
            bb = torch.multinomial(lb.exp(), 1).squeeze(-1)
            lc = F.log_softmax(self.head_c(h + e + self.bin_a_emb(ba) + self.bin_b_emb(bb)), -1)
            bc = torch.multinomial(lc.exp(), 1).squeeze(-1)
            a = self._bin_center(ba) + (torch.rand(1, device=dev) - 0.5) * self.bin_w
            b = self._bin_center(bb) + (torch.rand(1, device=dev) - 0.5) * self.bin_w
            c = self._bin_center(bc) + (torch.rand(1, device=dev) - 0.5) * self.bin_w
            off_local = torch.stack([a, b, c], -1)                             # frame coords [1,3]
            pos = origin[0] + torch.einsum('naj,na->nj', Rf, off_local)[0]     # frame -> world
            xo[j] = pos; so[j] = sj
            combined[m + j] = pos; scomb[m + j] = sj
            rem[0, sj] -= 1
            logq = (logq + ls.gather(1, sj[:, None]).squeeze() + la.gather(1, ba[:, None]).squeeze()
                    + lb.gather(1, bb[:, None]).squeeze() + lc.gather(1, bc[:, None]).squeeze()
                    - 3.0 * math.log(self.bin_w))
        if return_logq:
            # Keep the generation labels.  Sorting a generated configuration
            # and then rescoring it is not the probability of the sampling
            # path.  The physical KA energy is label-invariant, so SMC can
            # carry these fixed canonical-slot labels while observables ignore
            # them. `log_prob_pair(..., preordered=True)` is the exact reverse
            # evaluation of this returned labeled state.
            return xo, so, logq
        return xo, so


# ---------------------------------------------------------------------------
# Exact round-trip test (the foundation for exact likelihood). Run directly.
# ---------------------------------------------------------------------------
def _mic(x, center, L):
    d = x - center
    return d - L * torch.round(d / L)


def main(dataset="liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt",
         radii=(1.6, 2.0, 2.4), r_ctx=2.5, device="cpu"):
    from liquid_coupling_flow.ka3d_cavity_carve import carve
    d = torch.load(dataset, map_location=device, weights_only=False)
    X, S, L = d["x"].to(device).float(), d["s"].to(device).long(), float(d["L"])
    print(f"3D cavity local-frame GEOMETRY round-trip (exactness foundation), L={L:.3f}", flush=True)
    torch.manual_seed(0)
    ok = True
    for R in radii:
        for t in range(3):
            xb, sb = X[-(t + 1)], S[-(t + 1)]
            center = xb[(97 * t + 13) % xb.shape[0]].clone()
            pair = carve(xb, sb, center, R, L)
            interior_rel = _mic(pair["x_in"], center, L)
            bnd_rel_all = _mic(pair["x_out"], center, L)
            shell = bnd_rel_all.norm(dim=-1) < R + r_ctx                  # only interacting boundary shell
            bnd_rel = bnd_rel_all[shell]
            scaffold = fibonacci_ball_scaffold(pair["n_in"], R, device=device, dtype=interior_rel.dtype)
            abc, origins, order = to_frame(interior_rel, bnd_rel, scaffold, R)
            recon = from_frame_seq(abc, bnd_rel, scaffold)
            err = (recon - interior_rel[order]).abs().max().item()
            rms_abc = abc.norm(dim=-1).mean().item()
            ok &= err < 1e-4
            print(f"  R={R:.1f} c={t}: n_in={pair['n_in']:3d} n_shell={int(shell.sum()):3d}  "
                  f"max round-trip |err|={err:.2e}  mean|abc|={rms_abc:.3f} (~a0={A0:.3f})  "
                  f"-> {'PASS' if err < 1e-4 else 'FAIL'}", flush=True)
    print(f"GEOMETRY {'PASS — proceed to network/heads/log_prob' if ok else 'FAIL — fix before network'}", flush=True)


if __name__ == "__main__":
    main()
