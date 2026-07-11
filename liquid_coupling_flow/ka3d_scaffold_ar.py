"""Exact fixed-scaffold autoregressive generator for 3D KA cavity interiors.

The physical state is represented in a *labeled* space: slot labels are fixed
deterministic ball-scaffold anchors.  Equilibrium training configurations are
labeled once by a hard Hungarian permutation; generated and SMC states retain
those labels and are never re-sorted.  KA energies and observables ignore the
labels.

Positions have exact hard-ball support.  An unconstrained pre-position ``y`` is
mapped bijectively to ``|x| < R`` by a radial tanh diffeomorphism, and its analytic
Jacobian is included in both sampling and likelihood evaluation.  A Gaussian-base
autoregressive rational-quadratic spline head gives full support in ``y``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from liquid_coupling_flow.ka3d_cavity_ar import KA3DCavityAR, morton_code_3d
from liquid_coupling_flow.transforms_spline import RQSplineElementwise, DEFAULT_MIN_DERIVATIVE


_LOG2PI = math.log(2.0 * math.pi)


_SCAFFOLD_CACHE: dict = {}


def fixed_ball_scaffold(n_slots: int, R: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """Deterministic low-discrepancy, approximately equal-volume anchors in ``|x|<R``.

    Radial quantiles and angular coordinates use distinct irrational sequences,
    avoiding the radial/polar correlation of a naive Fibonacci construction.
    The final Morton ordering is over the *fixed anchors*, never over particles.

    Data-independent (function of n_slots, R, device, dtype only) -> cached; callers treat the
    result as read-only.
    """
    if n_slots < 1:
        raise ValueError("fixed_ball_scaffold needs at least one slot")
    key = (int(n_slots), float(R), str(device), dtype)
    cached = _SCAFFOLD_CACHE.get(key)
    if cached is not None:
        return cached
    j = torch.arange(n_slots, device=device, dtype=dtype)
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    # One sample per equal-volume radial stratum; deterministically permute which
    # angular point receives each radial rank.
    angular_key = torch.remainder((j + 0.5) / phi, 1.0)
    radial_rank = torch.argsort(torch.argsort(angular_key)).to(dtype)
    radius = float(R) * ((radial_rank + 0.5) / n_slots).pow(1.0 / 3.0)
    z = 1.0 - 2.0 * (j + 0.5) / n_slots
    theta = 2.0 * math.pi * torch.remainder((j + 0.5) / (phi * phi), 1.0)
    xy = (1.0 - z.square()).clamp_min(0.0).sqrt()
    direction = torch.stack([xy * torch.cos(theta), xy * torch.sin(theta), z], dim=-1)
    anchors = radius[:, None] * direction
    anchors = anchors[torch.argsort(morton_code_3d(anchors, float(R)))]
    _SCAFFOLD_CACHE[key] = anchors
    return anchors


def hungarian_order(x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Return ``slot -> particle`` hard assignment minimizing squared distance.

    This is a data-labeling permutation, not a differentiable coordinate map.
    The labeled likelihood never calls it during rollout or SMC mutation.
    """
    if x.ndim != 2 or anchors.ndim != 2 or x.shape != anchors.shape:
        raise ValueError(f"need matching [n,3] arrays, got {tuple(x.shape)} and {tuple(anchors.shape)}")
    from scipy.optimize import linear_sum_assignment
    cost = torch.cdist(x.detach(), anchors.detach()).square().cpu().double().numpy()
    rows, cols = linear_sum_assignment(cost)
    order = torch.empty(x.shape[0], dtype=torch.long)
    order[torch.as_tensor(cols)] = torch.as_tensor(rows)
    return order.to(x.device)


def label_to_scaffold(x: torch.Tensor, s: torch.Tensor, R: float):
    anchors = fixed_ball_scaffold(x.shape[0], R, x.device, x.dtype)
    order = hungarian_order(x, anchors)
    return x[order], s[order], order


def assignment_recovery(x_labeled: torch.Tensor, R: float):
    """Return per-particle and whole-configuration recovery of fixed labels."""
    anchors = fixed_ball_scaffold(x_labeled.shape[0], R, x_labeled.device, x_labeled.dtype)
    order = hungarian_order(x_labeled, anchors)
    identity = torch.arange(x_labeled.shape[0], device=x_labeled.device)
    return float((order == identity).float().mean()), bool(torch.equal(order, identity))


BALL_POW = 8.0   # soft radial bound: near-identity for |x|/R<~0.9, compresses only in the last shell.
# Replaces the atanh map, which diverged logarithmically and inflated the spline's target ~2x near the
# wall where cavity particles pack (|x|/R median 0.80, 90pct 0.96) -> mode blur -> clashes. This map:
#   phi(t) = t (1+t^p)^(-1/p),  x = y*(1+t^p)^(-1/p), t=|y|/R;  phi'(t) = (1+t^p)^(-(p+1)/p)
#   log|dx/dy| = -((p+3)/p) log(1+t^p)   (radial deriv * tangential^2, d=3)
# Inverse: y = x/(1-v^p)^(1/p), v=|x|/R;  log|dy/dx| = -((p+3)/p) log(1-v^p).


def ball_squash(y: torch.Tensor, R: float, p: float = BALL_POW):
    """Biject ``R^3 -> {|x|<R}`` (soft radial map); return ``x, log|dx/dy|``."""
    t = y.norm(dim=-1) / float(R)
    tp = t.clamp_min(0.0).pow(p)
    scale = (1.0 + tp).pow(-1.0 / p)                              # phi(t)/t; smooth at t=0 (-> 1)
    logdet = -((p + 3.0) / p) * torch.log1p(tp)
    return y * scale[..., None], logdet


def ball_unsquash(x: torch.Tensor, R: float, p: float = BALL_POW):
    """Inverse of :func:`ball_squash`; return ``y, log|dy/dx|``."""
    v = x.norm(dim=-1) / float(R)
    if torch.any(v >= 1.0):
        raise ValueError("ball_unsquash received a point outside the open ball")
    v = v.clamp_max(1.0 - 4.0 * torch.finfo(x.dtype).eps)
    vp = v.pow(p)
    inv = (1.0 - vp).pow(-1.0 / p)                                # y/x radial factor
    logdet = -((p + 3.0) / p) * torch.log(1.0 - vp)
    return x * inv[..., None], logdet


class Spline3Head(nn.Module):
    """Full-support ``p(a|h)p(b|h,a)p(c|h,a,b)`` Gaussian-base spline flow."""

    def __init__(self, d_model: int, num_bins: int = 12, tail_bound: float = 5.0):
        super().__init__()
        self.spline = RQSplineElementwise(num_bins=num_bins, tail_bound=tail_bound)
        self.num_bins = int(num_bins)
        P = self.spline.params_per_dim
        self.heads = nn.ModuleList([nn.Linear(d_model + i, P) for i in range(3)])
        const = float(torch.log(torch.expm1(torch.tensor(1.0 - DEFAULT_MIN_DERIVATIVE))))
        for head in self.heads:
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                head.bias.zero_()
                head.bias[2 * self.num_bins:] = const

    @staticmethod
    def _normal_logp(z):
        return -0.5 * z.square() - 0.5 * _LOG2PI

    def log_prob(self, h: torch.Tensor, u: torch.Tensor):
        lp = h.new_zeros(h.shape[:-1])
        ctx = h
        for axis in range(3):
            ui = u[..., axis:axis + 1]
            z, ld = self.spline.inverse(ui, self.heads[axis](ctx))
            lp = lp + (self._normal_logp(z) + ld).squeeze(-1)
            ctx = torch.cat([ctx, ui], dim=-1)
        return lp

    def sample(self, h: torch.Tensor, gen=None):
        lp = h.new_zeros(h.shape[:-1])
        ctx, values = h, []
        for axis in range(3):
            z = torch.randn(h.shape[:-1] + (1,), device=h.device, dtype=h.dtype, generator=gen)
            ui, ld = self.spline.forward(z, self.heads[axis](ctx))
            lp = lp + (self._normal_logp(z) - ld).squeeze(-1)
            values.append(ui)
            ctx = torch.cat([ctx, ui], dim=-1)
        return torch.cat(values, dim=-1), lp

    @torch.no_grad()
    def mode(self, h: torch.Tensor):
        ctx, values = h, []
        for axis in range(3):
            ui, _ = self.spline.forward(h.new_zeros(h.shape[:-1] + (1,)), self.heads[axis](ctx))
            values.append(ui)
            ctx = torch.cat([ctx, ui], dim=-1)
        return torch.cat(values, dim=-1)


class KA3DScaffoldAR(KA3DCavityAR):
    """Fixed-anchor labeled AR with exact hard-ball support and full-support density."""

    def __init__(self, *args, flow_bins=12, flow_tail=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow = Spline3Head(self.d_model, flow_bins, flow_tail)
        self.flow_bins, self.flow_tail = int(flow_bins), float(flow_tail)

    def _labeled(self, interior_rel, s_in, R, preordered):
        if preordered:
            return interior_rel, s_in
        x, s, _ = label_to_scaffold(interior_rel, s_in, R)
        return x, s

    def _contexts(self, xo, so, bnd_rel, s_bnd, R):
        n, m, dev = xo.shape[0], bnd_rel.shape[0], xo.device
        anchors = fixed_ball_scaffold(n, R, dev, xo.dtype)
        combined = torch.cat([bnd_rel, xo], dim=0)
        scomb = torch.cat([s_bnd, so], dim=0)
        kind = torch.cat([torch.ones(m, dtype=torch.long, device=dev),
                          torch.zeros(n, dtype=torch.long, device=dev)])
        idx = torch.arange(m + n, device=dev)
        valid = idx[None] < (m + torch.arange(n, device=dev))[:, None]
        slot_feat = self._slot_features(anchors, torch.arange(n, device=dev), n, R)
        h, frame = self._frame_context(combined, scomb, kind, valid, anchors, anchors,
                                       slot_feat=slot_feat)
        return anchors, h, frame

    def log_prob_pair(self, interior_rel, s_in, bnd_rel, s_bnd, R, preordered=False):
        xo, so = self._labeled(interior_rel, s_in, R, preordered)
        n = xo.shape[0]
        anchors, h, frame = self._contexts(xo, so, bnd_rel, s_bnd, R)
        # Count-masked species likelihood in the fixed slot sequence.
        oh = F.one_hot(so, self.n_species).to(h.dtype)
        rem = oh.sum(0, keepdim=True) - (oh.cumsum(0) - oh)
        s_logits = self.head_species(h).masked_fill(rem <= 0, float("-inf"))
        lp_s = F.log_softmax(s_logits, -1).gather(-1, so[:, None]).squeeze(-1)
        y, logdet_yx = ball_unsquash(xo, R)
        anchor_y, _ = ball_unsquash(anchors, R)
        u = torch.einsum("naj,nj->na", frame, y - anchor_y)
        lp_u = self.flow.log_prob(h + self.sp_out_emb(so), u)
        return (lp_s + lp_u + logdet_yx).sum()

    @torch.no_grad()
    def sample_pair(self, bnd_rel, s_bnd, n_A, n_B, R, return_logq=False, gen=None):
        n, m, dev = n_A + n_B, bnd_rel.shape[0], bnd_rel.device
        anchors = fixed_ball_scaffold(n, R, dev, bnd_rel.dtype)
        anchor_y, _ = ball_unsquash(anchors, R)
        combined = torch.cat([bnd_rel, torch.zeros(n, 3, device=dev, dtype=bnd_rel.dtype)], dim=0)
        scomb = torch.cat([s_bnd, torch.zeros(n, dtype=torch.long, device=dev)], dim=0)
        kind = torch.cat([torch.ones(m, dtype=torch.long, device=dev),
                          torch.zeros(n, dtype=torch.long, device=dev)])
        idx = torch.arange(m + n, device=dev)
        rem = torch.tensor([[float(n_A), float(n_B)]], device=dev)
        xo = torch.zeros(n, 3, device=dev, dtype=bnd_rel.dtype)
        so = torch.zeros(n, dtype=torch.long, device=dev)
        logq = bnd_rel.new_zeros(())
        for j in range(n):
            valid = (idx < (m + j))[None]
            slot_feat = self._slot_features(anchors[j:j + 1], j, n, R)
            h, frame = self._frame_context(combined, scomb, kind, valid,
                                           anchors[j:j + 1], anchors[j:j + 1], slot_feat=slot_feat)
            ls = F.log_softmax(self.head_species(h).masked_fill(rem <= 0, float("-inf")), -1)
            sj = torch.multinomial(ls.exp(), 1, generator=gen).squeeze(-1)
            u, lp_u = self.flow.sample(h + self.sp_out_emb(sj), gen=gen)
            y = anchor_y[j] + torch.einsum("naj,na->nj", frame, u)[0]
            pos, logdet_xy = ball_squash(y, R)
            xo[j], so[j] = pos, sj
            combined[m + j], scomb[m + j] = pos, sj
            rem[0, sj] -= 1
            logq = logq + ls.gather(1, sj[:, None]).squeeze() + lp_u.squeeze() - logdet_xy.squeeze()
        if return_logq:
            return xo, so, logq
        return xo, so


class Cat3Head(nn.Module):
    """Factorized categorical over binned (a,b,c) -- drop-in for Spline3Head (same log_prob(h,u) /
    sample(h) interface). A fine categorical can put ~all mass in one bin => a razor-sharp peak,
    which the Gaussian-base spline cannot; this is exactly what makes the 2D ka_localframe conditional
    sharp. Continuous piecewise-uniform density (bin + uniform dither), exact log-prob."""

    def __init__(self, d_model: int, num_bins: int = 128, u_range: float = 2.5):
        super().__init__()
        self.n = int(num_bins); self.rng = float(u_range); self.bw = 2.0 * u_range / num_bins
        self.head_a = nn.Linear(d_model, self.n)
        self.head_b = nn.Linear(d_model, self.n)
        self.head_c = nn.Linear(d_model, self.n)
        self.bin_a_emb = nn.Embedding(self.n, d_model)
        self.bin_b_emb = nn.Embedding(self.n, d_model)
        self._logbw3 = 3.0 * math.log(self.bw)

    def _bin(self, u):
        return ((u + self.rng) / self.bw).long().clamp(0, self.n - 1)

    def _ctr(self, b):
        return (b.float() + 0.5) * self.bw - self.rng

    def log_prob(self, h, u):
        ba, bb, bc = self._bin(u[..., 0]), self._bin(u[..., 1]), self._bin(u[..., 2])
        la = F.log_softmax(self.head_a(h), -1).gather(-1, ba[..., None]).squeeze(-1)
        lb = F.log_softmax(self.head_b(h + self.bin_a_emb(ba)), -1).gather(-1, bb[..., None]).squeeze(-1)
        lc = F.log_softmax(self.head_c(h + self.bin_a_emb(ba) + self.bin_b_emb(bb)), -1).gather(-1, bc[..., None]).squeeze(-1)
        return la + lb + lc - self._logbw3

    def sample(self, h, gen=None):
        la = F.log_softmax(self.head_a(h), -1); ba = torch.multinomial(la.exp(), 1, generator=gen).squeeze(-1)
        lb = F.log_softmax(self.head_b(h + self.bin_a_emb(ba)), -1); bb = torch.multinomial(lb.exp(), 1, generator=gen).squeeze(-1)
        lc = F.log_softmax(self.head_c(h + self.bin_a_emb(ba) + self.bin_b_emb(bb)), -1); bc = torch.multinomial(lc.exp(), 1, generator=gen).squeeze(-1)
        dith = (torch.rand(h.shape[:-1] + (3,), device=h.device, dtype=h.dtype, generator=gen) - 0.5) * self.bw
        u = torch.stack([self._ctr(ba), self._ctr(bb), self._ctr(bc)], -1) + dith
        lp = (la.gather(-1, ba[..., None]).squeeze(-1) + lb.gather(-1, bb[..., None]).squeeze(-1)
              + lc.gather(-1, bc[..., None]).squeeze(-1) - self._logbw3)
        return u, lp

    @torch.no_grad()
    def mode(self, h):
        ba = self.head_a(h).argmax(-1)
        bb = self.head_b(h + self.bin_a_emb(ba)).argmax(-1)
        bc = self.head_c(h + self.bin_a_emb(ba) + self.bin_b_emb(bb)).argmax(-1)
        return torch.stack([self._ctr(ba), self._ctr(bb), self._ctr(bc)], -1)


class KA3DScaffoldCatAR(KA3DScaffoldAR):
    """Fixed-anchor labeled AR with a CATEGORICAL residual head (2D-proven sharp peak) instead of the
    spline. Everything else (soft ball map, frame, anchors, block machinery) is inherited unchanged."""

    def __init__(self, *args, cat_bins=128, cat_range=2.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow = Cat3Head(self.d_model, cat_bins, cat_range)
        self.cat_bins, self.cat_range = int(cat_bins), float(cat_range)
