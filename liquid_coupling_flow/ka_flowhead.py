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
