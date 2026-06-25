"""Single-site Metropolis-Hastings kernel with a learned (NonCausalLF) local proposal.
See docs/superpowers/specs/2026-06-25-ka-single-site-mh-kernel-design.md."""
from __future__ import annotations
import math, torch, torch.nn.functional as F
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR
from liquid_coupling_flow.ka_gridformer import _wrap_pm

_SIG = torch.tensor(SIGMA); _EPS = torch.tensor(EPS); _RC = RCUT_FACTOR


def site_energy(xj, sj, pos, s, j, L):
    """Sum of shifted-LJ pair energies of a particle at xj (species int sj) vs all pos_k, k!=j.
    xj [B,2], pos [B,N,2], s [N] long, returns [B]."""
    B, N, _ = pos.shape
    sig = _SIG.to(pos.device, pos.dtype)[sj, s.long()]            # [N]  pair sigma j-vs-k
    eps = _EPS.to(pos.device, pos.dtype)[sj, s.long()]            # [N]
    rc = _RC * sig
    d = xj[:, None, :] - pos                                      # [B,N,2]
    d = d - L * torch.round(d / L)
    r2 = (d ** 2).sum(-1)                                         # [B,N]
    r2[:, j] = 1e12                                              # exclude self
    inv6 = (sig ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (sig / rc) ** 6
    e = torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e))
    return e.sum(-1)                                              # [B]


def site_dE(pos, s, j, xj_new, L):
    sj = int(s[j])
    return site_energy(xj_new, sj, pos, s, j, L) - site_energy(pos[:, j], sj, pos, s, j, L)


def _bin_probs(m, ctx_j):
    """Return (la [B,nbins], lb [B,nbins,nbins]) log-probs: lb[:,a] = log P(bin_b | bin_a)."""
    la = F.log_softmax(m.head_a(ctx_j), -1)                       # [B,nbins]
    nb = m.n_bins
    a_emb = m.bin_a_emb(torch.arange(nb, device=ctx_j.device))    # [nbins,d]
    lb = F.log_softmax(m.head_b(ctx_j[:, None, :] + a_emb[None]), -1)  # [B,nbins,nbins]
    return la, lb

def _offsets_to_bins(off, m):
    """off [B] in arc-units along one dim -> long bin idx in [0,nbins); out-of-range -> -1."""
    AR = m.arc_range
    b = ((off + AR) / m.bin_w).floor().long()
    return torch.where((off >= -AR) & (off < AR), b, torch.full_like(b, -1))

@torch.no_grad()
def site_logq(m, ctx_j, origin_j, xj, arc, L):
    """Folded log q(xj | x_{-j}). xj [B,2] -> [B]."""
    la, lb = _bin_probs(m, ctx_j)
    AR = m.arc_range; period = L / arc                            # torus period in arc-units
    off = _wrap_pm(xj - origin_j, L) / arc                        # [B,2] in [-period/2, period/2]
    B = xj.shape[0]; q = torch.zeros(B, device=xj.device)
    for da in (-1, 0, 1):                                         # fold aliased grid offsets
        for db in (-1, 0, 1):
            oa = off[:, 0] + da * period; ob = off[:, 1] + db * period
            ba = _offsets_to_bins(oa, m); bb = _offsets_to_bins(ob, m)
            ok = (ba >= 0) & (bb >= 0)
            if not ok.any():
                continue
            lpa = la.gather(1, ba.clamp_min(0)[:, None]).squeeze(1)
            lpb = lb.gather(1, ba.clamp_min(0)[:, None, None].expand(-1, 1, m.n_bins)).squeeze(1) \
                    .gather(1, bb.clamp_min(0)[:, None]).squeeze(1)
            contrib = (lpa + lpb).exp() / (m.bin_w * arc) ** 2     # density in PHYSICAL units
            q = q + torch.where(ok, contrib, torch.zeros_like(contrib))
    return q.clamp_min(1e-30).log()

@torch.no_grad()
def site_propose(m, ctx_j, origin_j, arc, L):
    """Sample xj' ~ q(.|x_{-j}); return (xj' [B,2], logq' [B]) with logq' the folded density at xj'."""
    la, lb = _bin_probs(m, ctx_j); B = ctx_j.shape[0]
    ba = torch.multinomial(la.exp(), 1).squeeze(1)                # [B]
    lb_a = lb.gather(1, ba[:, None, None].expand(-1, 1, m.n_bins)).squeeze(1)  # [B,nbins]
    bb = torch.multinomial(lb_a.exp(), 1).squeeze(1)
    a = m._bin_center(ba) + (torch.rand(B, device=ctx_j.device) - 0.5) * m.bin_w
    b = m._bin_center(bb) + (torch.rand(B, device=ctx_j.device) - 0.5) * m.bin_w
    xj = torch.remainder(origin_j + torch.stack([a, b], -1) * arc, L)
    return xj, site_logq(m, ctx_j, origin_j, xj, arc, L)
