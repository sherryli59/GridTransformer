"""mW generator v7 -- neighbor-centered output charts (radial-first + n1-frame arms).

v6 (global body + per-axis circular spline on the ANCHOR-offset chart) plateaued
at val ~1.0 with a FLAT TF first-shell peak error: a factorized p(u_x)p(u_y)p(u_z)
cannot concentrate mass on the thin first-shell SPHERE |x-neighbor|=1.19 (axis
coupling required).  v7 makes the RADIAL structure around a PHYSICAL particle a
coordinate of the output chart (design:
docs/superpowers/specs/2026-07-10-mw-v7-neighbor-centered-heads-design.md).

ONE module, TWO head arms over a SHARED exact mixture.  For AR step j the locator
n1_j is the nearest ALREADY-PLACED particle to the anchor t_j (deterministic
causal argmin, the ka_localframe origin).  With v = wrap_pm(x_j - n1_j, L):

    p(x_j | h_j) = w(h)*p_local(v | h)*1[v in D] + (1 - w(h))*(1/L^3)

- p_local is a normalized density on the FIXED PHYSICAL domain D; w is a sigmoid.
  Integrates to w + (1-w) = 1 over the box for any D inside the fundamental domain.
- The uniform component gives full support (required for the SMC base) and is the
  ONLY N-dependent term (via L^3) -- p_local and D are physical/local, so the
  learned component is manifestly size-invariant (the transfer thesis).
- Steps with no placed neighbor (j=0): w forced to 0 -> exactly -3 log L.

Arm A ("radial"): p_local(v) = p_r(r) * p_dir(cos th, phi) / r^2 on D = ball(R_CAP);
p_r a bounded RQS on [0,R_CAP], p_dir = bounded RQS(cos th in [-1,1]) x circular
RQS(phi) so the sphere measure dOmega = da dphi is exact (cos th absorbs sin th).
The 1/r^2 Cartesian Jacobian is exact.

Arm B ("frame"): p_local(v) = product of three bounded RQS over u = R_frame v / S_C
on [-1,1]^3, D = the rotated cube.  e1 points from n1 toward the anchor (the
vacancy region); e2 Gram-Schmidt off the 2nd-nearest placed neighbor; e3 = e1 x e2.
|det R| = 1 (asserted in tests); the rotation is exact ONLY because the support is
capped to the inscribed cube (sqrt(3)*S_C <= L/2), never the global/circular chart.

Exactness (log q0 enters SMC weights): sample() DRAWS x_j (Bernoulli(w) then local
forward transforms or a uniform box draw) and then SCORES it through the SAME
mixture log-density used by log_prob (n1, h and the domain indicator are
deterministic functions of the causal prefix), so the two are mirror-exact by
construction (verified by the ungated + perturbed tests, and by a 3D normalization
quadrature over the new mixture/Jacobian/domain surfaces).
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from liquid_coupling_flow.transforms_spline import (
    CircularRQSplineElementwise,
    DEFAULT_MIN_DERIVATIVE,
    RQSplineElementwise,
)
from liquid_coupling_flow.mw.mw_energy import RHO_STAR
from liquid_coupling_flow.mw.mw_generator import (
    ART,
    DEV,
    EV_RMIN,
    R_SHELL1,
    _augment_batch,
    canonical_order,
    load_training_bank,
    mw_scaffold,
    wrap_pm,
)
from liquid_coupling_flow.mw.mw_generator_v4 import (
    CausalGeoBlock,
    CurveRail,
    GeometricBias,
)
from liquid_coupling_flow.mw.mw_generator_v6 import N_GEO_FEAT

R_CAP = 2.5   # radial-arm ball radius (2.5 < L/2 = 2.598 at N=64; fits at every larger N)
S_C = 1.5     # frame-arm cube half-width (sqrt(3)*1.5 = 2.598 <= L/2 at N=64)


# ---------------------------------------------------------------------------
# Exact 1D building blocks: bounded / circular RQ splines with a UNIFORM base.
# ---------------------------------------------------------------------------

def _embed(q, lo, hi):
    """sin/cos embedding of a scalar in [lo,hi] (v5 convention), q[...] -> [...,2]."""
    ang = 2.0 * math.pi * (q - lo) / (hi - lo)
    return torch.stack([torch.sin(ang), torch.cos(ang)], -1)


def _init_spline_head(head: nn.Linear, num_bins: int):
    """Identity-init a spline parameter head (v5/Spline3Head convention).

    Zero weights + zero width/height bias => uniform bins; the derivative section
    (bias[2K:], the interior derivatives for the linear spline or all K for the
    circular one) set to `const` so softplus(const)+min_deriv == 1 => the RQS is
    the identity at init and the density is UNIFORM on its interval.
    """
    const = float(torch.log(torch.expm1(torch.tensor(1.0 - DEFAULT_MIN_DERIVATIVE))))
    nn.init.zeros_(head.weight)
    with torch.no_grad():
        head.bias.zero_()
        head.bias[2 * num_bins:] = const


def _bounded_logp(spline: RQSplineElementwise, t, params, lo, hi):
    """log-density at t of a UNIFORM-base bounded RQS on [lo,hi].

    The RQSplineElementwise is used as a diffeomorphism of [-B,B]; t is mapped
    affinely into [-B,B] (the identity tails never engage for t in [lo,hi]).  The
    affine Jacobians cancel to a clean -log(hi-lo) + spline-inverse-logdet.
    """
    B = spline.tail_bound
    zeta = (2.0 * B) * (t - lo) / (hi - lo) - B
    _, ld = spline.inverse(zeta, params)
    return -math.log(hi - lo) + ld


def _bounded_sample(spline: RQSplineElementwise, params, lo, hi, gen):
    """Draw t ~ uniform-base bounded RQS on [lo,hi]; returns (t, log-density)."""
    B = spline.tail_bound
    xi = torch.rand(params.shape[:-1], device=params.device, dtype=params.dtype, generator=gen)
    zeta_in = (2.0 * B) * xi - B
    zeta_out, ld = spline.forward(zeta_in, params)
    t = lo + (hi - lo) * (zeta_out + B) / (2.0 * B)
    return t, -math.log(hi - lo) - ld


def _circ_logp(spline: CircularRQSplineElementwise, phi, params):
    q = torch.remainder(phi, spline.L)
    _, ld = spline.inverse(q, params)
    return -math.log(spline.L) + ld


def _circ_sample(spline: CircularRQSplineElementwise, params, gen):
    z = torch.rand(params.shape[:-1], device=params.device, dtype=params.dtype, generator=gen) * spline.L
    q, ld = spline.forward(z, params)
    return q, -math.log(spline.L) - ld


# ---------------------------------------------------------------------------
# Arm A -- radial-first spherical head: p_r(r) p_a(cos th) p_phi(phi) / r^2.
# ---------------------------------------------------------------------------

class RadialSpherical3Head(nn.Module):
    """Exact spherical density on ball(r_cap).

    dOmega = da dphi with a = cos th absorbs the sin th Jacobian; the 1/r^2 factor
    converts (r, Omega) density to Cartesian.  AR ctx within the step: r -> a -> phi
    (r-embed feeds the a-head; (r,a)-embeds feed the phi-head).  Identity init =>
    p_r, p_a, p_phi all uniform (uniform in (r, Omega); NOT uniform in v -- the 1/r^2
    is real and is exactly what the normalization-quadrature test checks).
    """

    def __init__(self, d_model, num_bins=32, r_cap=R_CAP):
        super().__init__()
        self.r_cap = float(r_cap)
        self.num_bins = int(num_bins)
        self.r_spline = RQSplineElementwise(num_bins=num_bins, tail_bound=1.0)
        self.a_spline = RQSplineElementwise(num_bins=num_bins, tail_bound=1.0)
        self.phi_spline = CircularRQSplineElementwise(num_bins=num_bins, L=2.0 * math.pi)
        Pr = self.r_spline.params_per_dim
        Pp = self.phi_spline.params_per_dim
        self.r_head = nn.Linear(d_model, Pr)
        self.a_head = nn.Linear(d_model + 2, Pr)
        self.phi_head = nn.Linear(d_model + 4, Pp)
        for hd in (self.r_head, self.a_head, self.phi_head):
            _init_spline_head(hd, num_bins)

    def log_p_r(self, h, r):
        """1D radial marginal log-density (exposed for the analytic-core quadrature)."""
        return _bounded_logp(self.r_spline, r, self.r_head(h), 0.0, self.r_cap)

    def local_logp(self, h, v):
        r = v.norm(dim=-1)
        rc = r.clamp_min(1e-6)
        a = (v[..., 2] / rc).clamp(-1.0, 1.0)
        phi = torch.atan2(v[..., 1], v[..., 0])
        lp_r = _bounded_logp(self.r_spline, r, self.r_head(h), 0.0, self.r_cap)
        er = _embed(r, 0.0, self.r_cap)
        lp_a = _bounded_logp(self.a_spline, a, self.a_head(torch.cat([h, er], -1)), -1.0, 1.0)
        ea = _embed(a, -1.0, 1.0)
        lp_phi = _circ_logp(self.phi_spline, phi, self.phi_head(torch.cat([h, er, ea], -1)))
        lp_local = lp_r + lp_a + lp_phi - 2.0 * torch.log(rc)
        in_D = r <= self.r_cap
        return lp_local, in_D

    def local_sample(self, h, gen=None):
        r, _ = _bounded_sample(self.r_spline, self.r_head(h), 0.0, self.r_cap, gen)
        er = _embed(r, 0.0, self.r_cap)
        a, _ = _bounded_sample(self.a_spline, self.a_head(torch.cat([h, er], -1)), -1.0, 1.0, gen)
        ea = _embed(a, -1.0, 1.0)
        phi, _ = _circ_sample(self.phi_spline, self.phi_head(torch.cat([h, er, ea], -1)), gen)
        sinth = torch.sqrt((1.0 - a * a).clamp_min(0.0))
        nhat = torch.stack([sinth * torch.cos(phi), sinth * torch.sin(phi), a], -1)
        return r[..., None] * nhat


# ---------------------------------------------------------------------------
# Arm B -- n1-frame factored head over the inscribed rotated cube.
# ---------------------------------------------------------------------------

class FrameFactored3Head(nn.Module):
    """Factored bounded RQS over u = R_frame v / s_c on [-1,1]^3 (|det R| = 1)."""

    def __init__(self, d_model, num_bins=32, s_c=S_C):
        super().__init__()
        self.s_c = float(s_c)
        self.num_bins = int(num_bins)
        self.spline = RQSplineElementwise(num_bins=num_bins, tail_bound=1.0)
        P = self.spline.params_per_dim
        self.heads = nn.ModuleList([nn.Linear(d_model + 2 * i, P) for i in range(3)])
        for hd in self.heads:
            _init_spline_head(hd, num_bins)

    def local_logp(self, h, v, R):
        u = torch.einsum("...ij,...j->...i", R, v) / self.s_c
        lp = torch.zeros(h.shape[:-1], device=h.device, dtype=h.dtype)
        ctx = h
        for i in range(3):
            lp = lp + _bounded_logp(self.spline, u[..., i], self.heads[i](ctx), -1.0, 1.0)
            ctx = torch.cat([ctx, _embed(u[..., i], -1.0, 1.0)], -1)
        lp_local = lp - 3.0 * math.log(self.s_c)
        in_D = (u.abs() <= 1.0).all(-1)
        return lp_local, in_D

    def local_sample(self, h, R, gen=None):
        us = []
        ctx = h
        for i in range(3):
            ui, _ = _bounded_sample(self.spline, self.heads[i](ctx), -1.0, 1.0, gen)
            us.append(ui)
            ctx = torch.cat([ctx, _embed(ui, -1.0, 1.0)], -1)
        u = torch.stack(us, -1)
        # v = s_c R^T u (R orthonormal: rows are e1,e2,e3 => R^T maps frame -> world).
        return self.s_c * torch.einsum("...ij,...i->...j", R, u)


# ---------------------------------------------------------------------------
# The generator: v6 body verbatim + neighbor-centered mixture head.
# ---------------------------------------------------------------------------

class MWNeighborChart(nn.Module):
    """v6 global-AR body + a neighbor-centered exact mixture head (radial | frame)."""

    def __init__(self, head="radial", d_model=256, n_layers=4, n_heads=8, rail_k=8,
                 num_bins=32, bound=4.0, knn=12, use_geo_feat=True, r_cap=R_CAP, s_c=S_C):
        super().__init__()
        assert head in ("radial", "frame"), head
        self.head_kind = head
        self.d_model, self.n_layers, self.n_heads = d_model, n_layers, n_heads
        self.rail_k, self.num_bins = rail_k, num_bins
        self.bound = float(bound)
        self.knn = int(knn)
        self.use_geo_feat = bool(use_geo_feat)
        self.r_cap = float(r_cap)
        self.s_c = float(s_c)
        # ---- v6 body (verbatim) ----
        self.prev_proj = nn.Linear(3, d_model)
        self.phase_proj = nn.Sequential(nn.Linear(8, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.rail = CurveRail(d_model, n_heads, rail_k)
        self.geo = GeometricBias(n_heads)
        self.blocks = nn.ModuleList([CausalGeoBlock(d_model, n_heads) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        # ---- head arm + mixture gate ----
        if head == "radial":
            self.head = RadialSpherical3Head(d_model, num_bins=num_bins, r_cap=r_cap)
        else:
            self.head = FrameFactored3Head(d_model, num_bins=num_bins, s_c=s_c)
        self.w_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.w_head.weight)
        nn.init.zeros_(self.w_head.bias)              # w = sigmoid(0) = 0.5 at init (moderate prior)
        # Created LAST so its RNG draw does not shift the shared-body init stream
        # (a fresh use_geo_feat=False model has bit-identical body weights).
        self.geo_feat_proj = nn.Linear(N_GEO_FEAT, d_model)
        nn.init.zeros_(self.geo_feat_proj.weight)
        nn.init.zeros_(self.geo_feat_proj.bias)

    # -- v6 body helpers (verbatim) --

    @staticmethod
    def _phase(T, device):
        z = (torch.arange(T, device=device) + 0.5) / T
        fs = []
        for f in (1.0, 2.0, 4.0, 8.0):
            fs += [torch.sin(2 * math.pi * f * z), torch.cos(2 * math.pi * f * z)]
        return torch.stack(fs, -1)

    def _geo_feat(self, x, anchors, L):
        """Per-anchor three-body summary over the k nearest PLACED-prefix neighbors (v6)."""
        B, T, _ = x.shape
        K = min(self.knn, T)
        jj = torch.arange(T, device=x.device)
        causal = jj[None, None, :] < jj[None, :, None]
        d = wrap_pm(x[:, None, :, :] - anchors[:T][None, :, None, :], L)
        dist2 = d.square().sum(-1)
        idx = dist2.masked_fill(~causal, 1e9).topk(K, dim=2, largest=False).indices
        rel = torch.gather(d, 2, idx[..., None].expand(-1, -1, -1, 3))
        valid = torch.gather(causal.expand(B, T, T), 2, idx)
        rel = rel * valid[..., None].to(rel.dtype)
        vf = valid.to(x.dtype)
        r = rel.norm(dim=-1)
        n_valid = valid.sum(-1)
        nv = n_valid.clamp_min(1).to(x.dtype)
        inv_r2 = 1.0 / r.clamp_min(EV_RMIN).square()
        inv_r2_mean = (inv_r2 * vf).sum(-1) / nv
        occ = n_valid.to(x.dtype) / self.knn
        unit = rel / r.clamp_min(1e-6)[..., None]
        cos = torch.einsum("btkd,btld->btkl", unit, unit)
        tetra = cos + 1.0 / 3.0
        eye = torch.eye(K, device=x.device, dtype=torch.bool)
        pair_valid = (valid[..., :, None] & valid[..., None, :] & (~eye)).to(x.dtype)
        npairs = pair_valid.sum((-1, -2)).clamp_min(1.0)
        tetra_mean = (tetra * pair_valid).sum((-1, -2)) / npairs
        return torch.stack([tetra_mean, inv_r2_mean, occ], dim=-1)

    def _hidden(self, u, x, anchors, L):
        """v6 body: u/x are the anchor-chart offsets / positions of the TF prefix."""
        B, T, _ = u.shape
        prev = torch.cat([torch.zeros(B, 1, 3, device=u.device, dtype=u.dtype), u[:, :-1]], 1)
        h = self.prev_proj(prev) + self.phase_proj(self._phase(anchors.shape[0], u.device)[:T])[None]
        if self.use_geo_feat:
            h = h + self.geo_feat_proj(self._geo_feat(x, anchors, L))
        h[:, :1] = h[:, :1] + self.bos
        h = self.rail(h, anchors, L, L / round(anchors.shape[0] ** (1 / 3)))
        key_pos = torch.cat([anchors[:1][None].expand(B, -1, -1), x[:, :-1]], 1)
        gb = self.geo(anchors[:T], key_pos, L)
        for block in self.blocks:
            h = block(h, gb)
        return self.final_ln(h)

    # -- neighbor-centered geometry --

    def _frame_from_anchor(self, e1_dir, nbr_rel, n_valid, col_tol=0.99):
        """Right-handed orthonormal triad with e1 fixed to (t_j - n1) (adapts build_frames).

        e1_dir points n1 -> anchor.  nbr_rel are placed-neighbor displacements to the
        anchor, sorted nearest-first (index 0 = n1, ~antiparallel to e1 => rejected by
        the collinearity guard, so e2 uses the 2nd-nearest).  Falls back to a coordinate
        axis when < 2 non-collinear valid neighbors.  det(R) = +1 always (e3 = e1 x e2).
        """
        dev = e1_dir.device
        k = nbr_rel.shape[1]
        nz1 = e1_dir.norm(dim=-1) > 1e-9
        e1 = F.normalize(torch.where(
            nz1[:, None], e1_dir,
            torch.tensor([1.0, 0.0, 0.0], device=dev).expand_as(e1_dir)), dim=-1)
        nz = nbr_rel.norm(dim=-1) > 1e-9
        cos = torch.einsum("bkd,bd->bk", F.normalize(nbr_rel, dim=-1), e1).abs()
        valid_k = torch.arange(k, device=dev)[None] < n_valid[:, None]
        ok = (cos <= col_tol) & nz & valid_k
        idx2 = torch.where(ok.any(1), ok.float().argmax(1), torch.zeros_like(n_valid))
        d2 = torch.gather(nbr_rel, 1, idx2[:, None, None].expand(-1, 1, 3)).squeeze(1)
        axes = torch.eye(3, device=dev)
        ax = axes[e1.abs().argmin(-1)]
        use_ax = ~ok.any(1)
        d2 = torch.where(use_ax[:, None], ax, d2)
        u2 = d2 - (d2 * e1).sum(-1, keepdim=True) * e1
        e2 = F.normalize(u2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        return torch.stack([e1, e2, e3], dim=-2)

    def _fields(self, xo, t, L):
        """Vectorized causal fields for every AR step.

        Returns h [B,N,d], v [B,N,3] (= x_j - n1_j), n1_pos [B,N,3], R [B,N,3,3] or
        None, z [B,N] (w logit), has_nbr [B,N].  n1 = the global nearest placed
        particle to the anchor (topk index 0); the incremental sample() mirror uses
        the SAME argmin so the two paths agree exactly.
        """
        B, N, _ = xo.shape
        dev = xo.device
        s = L / (2.0 * self.bound)
        u = wrap_pm(xo - t[None], L) / s                        # v6 anchor chart (body prev content)
        h = self._hidden(u, xo, t, L)
        K = min(self.knn, N)
        jj = torch.arange(N, device=dev)
        causal = jj[None, None, :] < jj[None, :, None]          # [1,N(step),N(cand)]: cand < step
        d = wrap_pm(xo[:, None, :, :] - t[None, :, None, :], L)
        dist2 = d.square().sum(-1)
        idx = dist2.masked_fill(~causal, 1e9).topk(K, dim=2, largest=False).indices
        nbr_pos = torch.gather(xo[:, None, :, :].expand(B, N, N, 3), 2,
                               idx[..., None].expand(-1, -1, -1, 3))       # [B,N,K,3]
        valid = torch.gather(causal.expand(B, N, N), 2, idx)              # [B,N,K]
        n1_pos = nbr_pos[:, :, 0, :]
        v = wrap_pm(xo - n1_pos, L)
        z = self.w_head(h).squeeze(-1)
        has_nbr = (jj >= 1)[None].expand(B, -1)
        R = None
        if self.head_kind == "frame":
            nbr_rel = wrap_pm(nbr_pos - t[None, :, None, :], L) * valid[..., None].to(xo.dtype)
            e1_dir = wrap_pm(t[None, :, :] - n1_pos, L)
            n_valid = valid.sum(-1)
            R = self._frame_from_anchor(e1_dir.reshape(B * N, 3), nbr_rel.reshape(B * N, K, 3),
                                        n_valid.reshape(B * N)).reshape(B, N, 3, 3)
        return h, v, n1_pos, R, z, has_nbr

    def _mixture_logp(self, h, v, R, z, has_nbr, L):
        """log p(x_j) of the two-component mixture, per step (any leading shape)."""
        logw = F.logsigmoid(z)
        log1mw = F.logsigmoid(-z)
        if self.head_kind == "radial":
            lp_local, in_D = self.head.local_logp(h, v)
        else:
            lp_local, in_D = self.head.local_logp(h, v, R)
        branch_local = torch.where(in_D, logw + lp_local,
                                   torch.full_like(logw, float("-inf")))
        log_unif = log1mw - 3.0 * math.log(L)
        logp = torch.logsumexp(torch.stack([branch_local, log_unif], 0), 0)
        return torch.where(has_nbr, logp, torch.full_like(logp, -3.0 * math.log(L)))

    def log_prob(self, x, L, preordered=False):
        B, N, _ = x.shape
        x = torch.remainder(x, L)
        t, rank, R_grid = mw_scaffold(N, L, x.device)
        if not preordered:
            perm = canonical_order(x, L, R_grid, rank)
            x = torch.gather(x, 1, perm[..., None].expand(-1, -1, 3))
        h, v, _n1, R, z, has_nbr = self._fields(x, t, L)
        return self._mixture_logp(h, v, R, z, has_nbr, L).sum(1)

    @torch.no_grad()
    def sample(self, B, N, L, gen=None, return_logq=True):
        device = next(self.parameters()).device
        t, _, _ = mw_scaffold(N, L, device)
        s = L / (2.0 * self.bound)
        x = torch.zeros(B, N, 3, device=device)
        u = torch.zeros(B, N, 3, device=device)
        logq = torch.zeros(B, device=device)
        log_unif_const = -3.0 * math.log(L)
        for j in range(N):
            h = self._hidden(u[:, :j + 1], x[:, :j + 1], t, L)[:, -1]
            if j == 0:
                xj = torch.rand(B, 3, device=device, generator=gen) * L      # no neighbor: pure uniform
                x[:, j] = xj
                u[:, j] = wrap_pm(xj - t[j], L) / s
                logq = logq + log_unif_const
                continue
            placed = x[:, :j, :]
            dd = wrap_pm(placed - t[j][None, None, :], L)
            dist2 = dd.square().sum(-1)
            K = min(self.knn, j)
            idx = dist2.topk(K, dim=1, largest=False).indices
            nbr_pos = torch.gather(placed, 1, idx[..., None].expand(-1, -1, 3))
            n1_pos = nbr_pos[:, 0, :]
            z = self.w_head(h).squeeze(-1)
            w = torch.sigmoid(z)
            R = None
            if self.head_kind == "frame":
                nbr_rel = wrap_pm(nbr_pos - t[j][None, None, :], L)
                e1_dir = wrap_pm(t[j][None, :] - n1_pos, L)
                n_valid = torch.full((B,), K, dtype=torch.long, device=device)
                R = self._frame_from_anchor(e1_dir, nbr_rel, n_valid)
                v_local = self.head.local_sample(h, R, gen=gen)
            else:
                v_local = self.head.local_sample(h, gen=gen)
            x_local = torch.remainder(n1_pos + v_local, L)
            x_unif = torch.rand(B, 3, device=device, generator=gen) * L
            bern = torch.rand(B, device=device, generator=gen) < w
            xj = torch.where(bern[:, None], x_local, x_unif)
            x[:, j] = xj
            u[:, j] = wrap_pm(xj - t[j], L) / s
            v_final = wrap_pm(xj - n1_pos, L)
            has_nbr = torch.ones(B, dtype=torch.bool, device=device)
            logq = logq + self._mixture_logp(h, v_final, R, z, has_nbr, L)
        return (x, logq) if return_logq else x

    @torch.no_grad()
    def _sample_positions_tf(self, h, n1_pos, R, z, has_nbr, L, gen=None):
        """Teacher-forced per-step placement from the mixture (vectorized over [B,N])."""
        w = torch.sigmoid(z)
        if self.head_kind == "radial":
            v_local = self.head.local_sample(h, gen=gen)
        else:
            v_local = self.head.local_sample(h, R, gen=gen)
        x_local = torch.remainder(n1_pos + v_local, L)
        x_unif = torch.rand(h.shape[:-1] + (3,), device=h.device, generator=gen) * L
        bern = torch.rand(h.shape[:-1], device=h.device, generator=gen) < w
        x = torch.where(bern[..., None], x_local, x_unif)
        return torch.where(has_nbr[..., None], x, x_unif)

    @torch.no_grad()
    def teacher_forced_structure(self, x, L, nbins=120, gen=None):
        """Pooled-predecessor TF g(r), peak error, and excluded-core mass (v6 metric)."""
        xo = torch.remainder(x, L)
        t, rank, R_grid = mw_scaffold(xo.shape[1], L, xo.device)
        perm = canonical_order(xo, L, R_grid, rank)
        xo = torch.gather(xo, 1, perm[..., None].expand(-1, -1, 3))
        h, _v, n1_pos, R, z, has_nbr = self._fields(xo, t, L)
        placed = self._sample_positions_tf(h, n1_pos, R, z, has_nbr, L, gen=gen)
        d = wrap_pm(placed[:, :, None] - xo[:, None], L).norm(dim=-1)
        N = xo.shape[1]
        jj = torch.arange(N, device=x.device)
        causal = jj[None, :, None] > jj[None, None, :]
        vals = d[causal.expand(len(xo), -1, -1)]
        edges = torch.linspace(0, L / 2, nbins + 1, device=x.device)
        # torch.histogram is CPU-only; counts rejoin the device for the normalization arithmetic
        counts = torch.histogram(vals.float().cpu(), bins=nbins, range=(0.0, L / 2))[0].to(x.device)
        r = (edges[1:] + edges[:-1]) / 2
        dr = edges[1] - edges[0]
        norm = len(xo) * (N * (N - 1) / 2) * (4 * math.pi * r.square() * dr) / (L ** 3)
        gr = counts / norm.clamp_min(1e-12)
        ipeak = (r - R_SHELL1).abs().argmin()
        peak_err = (gr[ipeak] - 2.18).abs() / 2.18
        core_mass = gr[r < 1.0].mean()
        return {"r": r, "gr": gr, "peak_err": peak_err, "core_mass": core_mass,
                "composite": peak_err + core_mass}


_CKPT_KEYS = ("d_model", "n_layers", "n_heads", "rail_k", "num_bins", "bound",
              "knn", "use_geo_feat", "r_cap", "s_c")


def _checkpoint(model, step, val_nll, struct, L):
    return {
        "state_dict": model.state_dict(), "step": step, "val_nll": val_nll,
        "struct": {k: (float(v) if v.numel() == 1 else v.cpu()) for k, v in struct.items()},
        "train_N": round(RHO_STAR * L ** 3),
        "architecture": f"neighbor_chart_v7_{model.head_kind}",
        "head": model.head_kind,
        **{k: getattr(model, k) for k in _CKPT_KEYS},
    }


def train(steps=30_000, batch=32, lr=3e-4, val_every=500, seed=71,
          out="mw_gen_N64_v7.pt", primary_thin=2, val_frac=0.1,
          extension="mw_ref_N64_ext.pt", device=None, art_path=None,
          extra_banks=None, head="radial", **model_kw):
    """Train v7 and save independent best-NLL and best-structure checkpoints.

    Trainer ported verbatim from v6: multi-bank loader (thin-2 primary + extension),
    on-the-fly augmentation, AdamW lr 3e-4 wd 1e-4, clip_grad_norm 5.0, best_nll +
    best_struct + last checkpoints, and a TF pooled-predecessor g(r) composite
    (peak_err + core_mass) every val cycle.  ``head`` selects the output arm.
    """
    torch.manual_seed(seed)
    device = device or DEV
    if extra_banks is None:
        ext = extension if os.path.exists(extension) else os.path.join(ART, extension)
        extra_banks = [(ext, 32, 1)]
    xtr, xva, L = load_training_bank(art_path=art_path, thin_events=primary_thin,
                                     val_frac=val_frac, extra_banks=extra_banks)
    xtr, xva = xtr.to(device), xva.to(device)
    model = MWNeighborChart(head=head, **model_kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    aug_gen = torch.Generator(device=device).manual_seed(seed + 1)
    eval_gen = torch.Generator(device=device).manual_seed(seed + 2)
    out = out if os.path.isabs(out) else os.path.join(ART, out)
    stem = out[:-3] if out.endswith(".pt") else out
    paths = {"nll": stem + "_best_nll.pt", "struct": stem + "_best_struct.pt",
             "last": stem + "_last.pt"}
    best_nll = best_struct = float("inf")
    N = xtr.shape[1]
    for step in range(steps):
        idx = torch.randint(len(xtr), (batch,), device=device)
        xb = _augment_batch(xtr[idx], L, aug_gen)
        loss = -model.log_prob(xb, L).mean() / N
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite v7 loss at step {step}")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % val_every == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vals = [-model.log_prob(xva[i:i + 64], L).sum()
                        for i in range(0, len(xva), 64)]
                val = float(torch.stack(vals).sum() / (len(xva) * N))
                eval_gen.manual_seed(seed + 2)   # common random numbers across checkpoints
                struct = model.teacher_forced_structure(xva[:64], L, gen=eval_gen)
            model.train()
            ck = _checkpoint(model, step, val, struct, L)
            torch.save(ck, paths["last"])
            if val < best_nll:
                best_nll = val; torch.save(ck, paths["nll"])
            score = float(struct["composite"])
            if score < best_struct:
                best_struct = score; torch.save(ck, paths["struct"])
            print(f"v7-{head} step {step}: train {float(loss):.4f} val {val:.4f} "
                  f"TF peak_err {float(struct['peak_err']):.4f} core {float(struct['core_mass']):.4f} "
                  f"composite {score:.4f}", flush=True)
    return {**paths, "best_nll": best_nll, "best_struct": best_struct}


def load_generator_v7(path, device=None):
    """Construct an MWNeighborChart from a checkpoint's hyperparams and load weights."""
    device = device or DEV
    ck = torch.load(path, map_location=device, weights_only=False)
    kw = {k: ck[k] for k in _CKPT_KEYS}
    model = MWNeighborChart(head=ck["head"], **kw)
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="mW v7 neighbor-centered generator training")
    parser.add_argument("--head", choices=("radial", "frame"), default="radial")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--out", type=str, default="mw_gen_N64_v7.pt")
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    train(steps=args.steps, out=args.out, seed=args.seed, head=args.head)
