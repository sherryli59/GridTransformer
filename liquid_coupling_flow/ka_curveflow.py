# liquid_coupling_flow/ka_curveflow.py
"""Feature A (GPS / QueryAR curve conditioning) for the spline-flow placement head. Adds a per-particle,
deterministic curve-feature (periodic encoding of the scaffold position s_j from geo._scaffold + the
arc-length coord (j+0.5)/N) to the local-frame context before the exact spline flow. Zero-initialized so
it starts identical to B-alone; warm-started from the trained B checkpoint to isolate the curve increment.
Exact likelihood preserved (the curve feature is part of the conditioning, identical in log_prob and
sample). TEST OF THE HYPOTHESIS that 'giving the curve' sharpens the conditional vs B-alone (2.704)."""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from liquid_coupling_flow.ka_flowhead import KAFlowHeadModel
from liquid_coupling_flow.ka_localframe import _wrap_pm


class KACurveFlowModel(KAFlowHeadModel):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        enc = 2 * self.d * self.periods.numel()                          # sin/cos over d coords x K periods
        self.curve_proj = nn.Sequential(nn.Linear(enc + 1, self.d_model), nn.GELU(),
                                        nn.Linear(self.d_model, self.d_model))
        nn.init.zeros_(self.curve_proj[-1].weight); nn.init.zeros_(self.curve_proj[-1].bias)   # zero -> == B at init

    def _curve_feat(self, N, device):
        sc = self.geo._scaffold(N, device)                               # [N,2] curve position s_j (the GPS coord)
        a = 2 * math.pi * sc.unsqueeze(-1) / self.periods.to(device)     # [N,2,K]
        pe = torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(1)      # [N, 2*d*K]
        arclen = ((torch.arange(N, device=device) + 0.5) / N)[:, None]   # [N,1] normalized arc-length
        return self.curve_proj(torch.cat([pe, arclen], -1))              # [N, d_model]

    def log_prob(self, x, s, canonical=None, preordered=False):
        B, N = x.shape[0], x.shape[1]; s = s.long(); s = s.expand(B, N).clone() if s.dim() == 1 else s
        L = self._Lof(N)
        if preordered:
            xo, so = x, s
        else:
            order = self.geo._curve_order(x, N)
            xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s, 1, order)
        sc = self.geo._scaffold(N, x.device)
        context, origin = self._local(xo, so, sc, L, N)
        context = context + self._curve_feat(N, x.device)[None]          # <-- GPS curve conditioning
        ab = _wrap_pm(xo - origin, L) / self._arc_scale(N)
        lp_ab = self.flow.log_prob(context, ab)
        s_logits = self.head_species(context)
        if self.canonical if canonical is None else canonical:
            oh = F.one_hot(so, self.n_species).to(s_logits.dtype)
            rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)
            s_logits = s_logits.masked_fill(rem <= 0, float("-inf"))
        lp_s = F.log_softmax(s_logits, -1).gather(-1, so[..., None]).squeeze(-1)
        return (lp_ab + lp_s).sum(1) - self.d * N * math.log(self._arc_scale(N))

    @torch.no_grad()
    def sample(self, B, N, n_B=None, device=None, return_logq=False):
        L = self._Lof(N); sc = self.geo._scaffold(N, device); arc = self._arc_scale(N)
        cf = self._curve_feat(N, device)                                 # [N, d_model] precomputed once
        pos = torch.zeros(B, N, 2, device=device); sp = torch.zeros(B, N, dtype=torch.long, device=device)
        rem = None
        if n_B is not None:
            rem = torch.zeros(B, self.n_species, device=device); rem[:, 0] = N - n_B; rem[:, 1] = n_B
        for j in range(N):
            h, origin = self._step(pos, sp, sc[j], j, L)
            h = h + cf[j]                                                 # <-- GPS curve conditioning (particle j)
            sj = torch.multinomial(F.softmax(self._species_logits(h, rem), -1), 1).squeeze(-1)
            ab, _ = self.flow.sample(h)
            if rem is not None:
                rem[torch.arange(B, device=device), sj] -= 1
            pos[:, j] = torch.remainder(origin + ab * arc, L); sp[:, j] = sj
        perm = self.geo._curve_order(pos, N)
        pos = torch.gather(pos, 1, perm[..., None].expand(-1, -1, 2)); sp = torch.gather(sp, 1, perm)
        if return_logq:
            return pos, sp, self.log_prob(pos, sp)
        return pos, sp
