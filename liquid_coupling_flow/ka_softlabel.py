# liquid_coupling_flow/ka_softlabel.py
"""Feature B: GEOMETRIC SOFT LABELS + stochastic target sampling for the (a,b) head of the local-frame AR
generator. The head bins a continuous coordinate into n_bins fine bins; a one-hot single-sample target
teaches a spiky conditional and treats physically-near bins as equally wrong as far bins. Soft labels =
kernel-density smoothing of the target (P(bin_i) ~ exp(-(c_i-target)^2/tau)); stochastic target sampling
draws the assigned bin from that kernel. TRAINING-ONLY: base log_prob/sample are untouched, so the
exactness gate is preserved. Species labels are NOT softened (discrete)."""
from __future__ import annotations
import os, time, math, torch, torch.nn.functional as F
from liquid_coupling_flow.ka_localframe import KALocalFrameModel, _wrap_pm, KNN

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def soft_target(centers, target, tau):
    """centers [n_bins], target [...] -> normalized soft categorical [..., n_bins].
    UNITS: `centers`, `target`, and `tau` are all in NORMALIZED (a,b) units (ab = wrap(xo-origin)/arc_scale;
    centers from _bin_center; tau = (sigma_bins*bin_w)^2). This is INTENTIONAL and size-invariant: the head
    bins/learns in the normalized coordinate, so a constant `sigma_bins` smooths consistently across N. The
    PHYSICAL bin width scales as arc_scale=N^(1/6); for the size-transfer follow-on, KEEP sigma_bins in
    normalized-bin units -- do NOT reinterpret tau as a fixed physical width (that would break size-invariance).
    TIE-BREAKING: when target falls exactly on a bin boundary, we shift by eps=bw*1e-5 rightward so that
    argmax matches _bin's floor-division convention (the containing bin is the right one)."""
    bw = centers[1] - centers[0]
    d2 = (centers.view(*([1] * target.dim()), -1) - (target.unsqueeze(-1) + bw * 1e-5)) ** 2
    return F.softmax(-d2 / tau, dim=-1)


def _axis_loss(logp, centers, target, tau, soft, stochastic, gen):
    """logp [B,N,n_bins] log-softmax head; returns (axis_nll[B,N], assigned_bin[B,N]).
    The two flags act on DIFFERENT objects (NOT a clean 2x2) -- `soft` softens the supervised TARGET,
    `stochastic` samples the bin that (a) conditions the b-head and (b) supervises the loss when NOT soft:
        (soft, stochastic) | supervises the loss        | conditions head_b (via `assigned`)
        (F, F)  neither     | hard CE at containing bin   | containing bin            (== base one-hot)
        (T, F)  soft        | soft CE over full P         | containing bin
        (F, T)  stochastic  | hard CE at SAMPLED bin      | sampled bin               (RQ stochastic code)
        (T, T)  both        | soft CE over full P         | sampled bin               (RQ: soft label + stoch code)
    In `both` the sampled bin is the propagated code (conditions head_b) while the soft P supervises -- the
    sampled bin does NOT supervise. This is faithful to RQ; the ablation report must describe the factors
    by what they control, not as a symmetric 2x2."""
    bw = centers[1] - centers[0]                                   # == m.bin_w
    P = soft_target(centers, target, tau) if (soft or stochastic) else None
    if stochastic:
        assigned = torch.multinomial(P.reshape(-1, P.shape[-1]), 1, generator=gen).reshape(target.shape)
    else:                                                          # containing bin == m._bin(target) exactly
        assigned = ((target - (centers[0] - bw / 2)) / bw).long().clamp(0, centers.numel() - 1)
    if soft:
        nll = -(P * logp).sum(-1)
    else:
        nll = -logp.gather(-1, assigned.unsqueeze(-1)).squeeze(-1)
    return nll, assigned


def pos_species_nll(m, context, origin, xo, so, L, N, *, sigma_bins, soft, stochastic, canonical, gen):
    """Per-config position+species NLL given a precomputed context/origin (from TRUE or MIXED prefix).
    Reused by the scheduled-sampling arm. Returns [B]."""
    if soft or stochastic:
        assert sigma_bins > 0, "sigma_bins must be > 0 when soft or stochastic labeling is on"
    tau = (sigma_bins * m.bin_w) ** 2
    centers = m._bin_center(torch.arange(m.n_bins, device=context.device))
    ab = _wrap_pm(xo - origin, L) / m._arc_scale(N)                # normalized target (a,b)
    la = F.log_softmax(m.head_a(context), -1)
    nll_a, asg_a = _axis_loss(la, centers, ab[..., 0], tau, soft, stochastic, gen)
    lb = F.log_softmax(m.head_b(context + m.bin_a_emb(asg_a)), -1)
    nll_b, _ = _axis_loss(lb, centers, ab[..., 1], tau, soft, stochastic, gen)
    s_logits = m.head_species(context)
    use_canon = m.canonical if canonical is None else canonical
    if use_canon:
        oh = F.one_hot(so, m.n_species).to(s_logits.dtype)
        rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)
        s_logits = s_logits.masked_fill(rem <= 0, float("-inf"))
    lp_s = F.log_softmax(s_logits, -1).gather(-1, so[..., None]).squeeze(-1)
    return (nll_a + nll_b - lp_s).sum(1)                           # [B]


class KALocalFrameSoft(KALocalFrameModel):
    def train_loss(self, x, s, *, sigma_bins, soft, stochastic, canonical=None, gen=None):
        B, N = x.shape[0], x.shape[1]; s = s.long()
        s = s.expand(B, N).clone() if s.dim() == 1 else s
        L = self._Lof(N)
        order = self.geo._curve_order(x, N)
        xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s, 1, order)
        sc = self.geo._scaffold(N, x.device)
        context, origin = self._local(xo, so, sc, L, N)
        return pos_species_nll(self, context, origin, xo, so, L, N, sigma_bins=sigma_bins,
                               soft=soft, stochastic=stochastic, canonical=canonical, gen=gen)
