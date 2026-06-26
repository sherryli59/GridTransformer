# liquid_coupling_flow/ka_flowhead.py
"""Exact AR rational-quadratic spline-flow placement head for the local-frame generator (Phase 1 / Feature B).
Replaces the binned (a,b) categorical with a continuous flow sharp enough to represent the hard-core contact
peak the categorical smears. Gaussian base on R + RQSplineElementwise (identity linear tails); AR factorization
p(a|h)*p(b|a,h). Identity-initialized for stable training. Returns the NORMALIZED-offset log-density; the caller
adds the arc_scale Jacobian-to-physical. Quetzal structure (transformer context -> small continuous head),
exact instead of diffusion so the SMC corrector's likelihood stays exact."""
from __future__ import annotations
import math, torch, torch.nn as nn
from liquid_coupling_flow.transforms_spline import RQSplineElementwise, DEFAULT_MIN_DERIVATIVE

_LOG2PI = math.log(2 * math.pi)


def _base_logp(z):                                          # standard-normal log-density, summed over last dim
    return (-0.5 * z ** 2 - 0.5 * _LOG2PI)


class SplineFlowHead(nn.Module):
    def __init__(self, d_model, num_bins=8, tail_bound=4.0):
        super().__init__()
        self.spline = RQSplineElementwise(num_bins=num_bins, tail_bound=tail_bound)
        self.num_bins = num_bins
        P = self.spline.params_per_dim                      # 3K - 1
        self.head_a = nn.Linear(d_model, P)
        self.head_b = nn.Linear(d_model + 1, P)             # condition b on the continuous a
        self._identity_init()

    def _identity_init(self):
        # zero weights; widths/heights bias 0 (-> uniform bins); interior-derivative bias = const giving
        # softplus(const)+min_deriv == 1 -> the RQS is exactly the identity at init (flow == Gaussian base).
        K = self.num_bins
        const = float(torch.log(torch.expm1(torch.tensor(1.0 - DEFAULT_MIN_DERIVATIVE))))
        for head in (self.head_a, self.head_b):
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                head.bias.zero_()
                head.bias[2 * K:] = const                   # interior derivatives (P = 3K-1: [2K:] is the K-1 derivs)

    def log_prob(self, h, ab):
        a = ab[..., 0:1]; b = ab[..., 1:2]
        za, lda = self.spline.inverse(a, self.head_a(h))                 # a -> base
        lpa = _base_logp(za) + lda
        zb, ldb = self.spline.inverse(b, self.head_b(torch.cat([h, a], -1)))
        lpb = _base_logp(zb) + ldb
        return (lpa + lpb).squeeze(-1)

    def sample(self, h, gen=None):
        za = torch.randn(h.shape[:-1] + (1,), device=h.device, generator=gen)
        a, lda = self.spline.forward(za, self.head_a(h))                 # base -> a
        lpa = _base_logp(za) - lda
        zb = torch.randn(h.shape[:-1] + (1,), device=h.device, generator=gen)
        b, ldb = self.spline.forward(zb, self.head_b(torch.cat([h, a], -1)))
        lpb = _base_logp(zb) - ldb
        return torch.cat([a, b], -1), (lpa + lpb).squeeze(-1)


# ---------------------------------------------------------------------------
# KAFlowHeadModel — local-frame AR generator with spline-flow placement head
# ---------------------------------------------------------------------------
import torch.nn.functional as F
from liquid_coupling_flow.ka_localframe import KALocalFrameModel, _wrap_pm


class KAFlowHeadModel(KALocalFrameModel):
    """Local-frame AR generator with the categorical (a,b) head replaced by the exact spline flow.
    Species head + frame + KNN context + curve ordering are inherited unchanged; exact likelihood preserved
    (continuous flow density replaces P(bin)/bin_area; the arc_scale Jacobian is the same `jac` term)."""

    def __init__(self, *args, num_bins=8, tail_bound=4.0, **kw):
        super().__init__(*args, **kw)
        self.flow = SplineFlowHead(self.d_model, num_bins=num_bins, tail_bound=tail_bound)

    def _species_logits(self, h, rem):
        lg = self.head_species(h)
        return lg if rem is None else lg.masked_fill(rem <= 0, float("-inf"))

    def log_prob(self, x, s, canonical=None, preordered=False):
        B, N = x.shape[0], x.shape[1]
        s = s.long()
        s = s.expand(B, N).clone() if s.dim() == 1 else s
        L = self._Lof(N)
        if preordered:
            xo, so = x, s
        else:
            order = self.geo._curve_order(x, N)
            xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2))
            so = torch.gather(s, 1, order)
        sc = self.geo._scaffold(N, x.device)
        context, origin = self._local(xo, so, sc, L, N)
        ab = _wrap_pm(xo - origin, L) / self._arc_scale(N)
        lp_ab = self.flow.log_prob(context, ab)                          # [B,N] continuous flow log-density
        s_logits = self.head_species(context)
        if self.canonical if canonical is None else canonical:
            oh = F.one_hot(so, self.n_species).to(s_logits.dtype)
            rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)
            s_logits = s_logits.masked_fill(rem <= 0, float("-inf"))
        lp_s = F.log_softmax(s_logits, -1).gather(-1, so[..., None]).squeeze(-1)
        jac = self.d * N * math.log(self._arc_scale(N))                  # NO bin_w vol term (flow is continuous)
        return (lp_ab + lp_s).sum(1) - jac

    @torch.no_grad()
    def sample(self, B, N, n_B=None, device=None, return_logq=False):
        L = self._Lof(N); sc = self.geo._scaffold(N, device); arc = self._arc_scale(N)
        pos = torch.zeros(B, N, 2, device=device); sp = torch.zeros(B, N, dtype=torch.long, device=device)
        rem = None
        if n_B is not None:
            rem = torch.zeros(B, self.n_species, device=device)
            rem[:, 0] = N - n_B; rem[:, 1] = n_B
        for j in range(N):
            h, origin = self._step(pos, sp, sc[j], j, L)
            sj = torch.multinomial(F.softmax(self._species_logits(h, rem), -1), 1).squeeze(-1)
            ab, _ = self.flow.sample(h)
            if rem is not None:
                rem[torch.arange(B, device=device), sj] -= 1
            pos[:, j] = torch.remainder(origin + ab * arc, L)
            sp[:, j] = sj
        perm = self.geo._curve_order(pos, N)
        pos = torch.gather(pos, 1, perm[..., None].expand(-1, -1, 2))
        sp = torch.gather(sp, 1, perm)
        if return_logq:
            return pos, sp, self.log_prob(pos, sp)                       # exactness trick (matches base model)
        return pos, sp
