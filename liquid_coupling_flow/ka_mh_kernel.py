"""Single-site Metropolis-Hastings kernel with a learned (NonCausalLF) local proposal.
See docs/superpowers/specs/2026-06-25-ka-single-site-mh-kernel-design.md."""
from __future__ import annotations
import math, torch, torch.nn.functional as F
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR

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
