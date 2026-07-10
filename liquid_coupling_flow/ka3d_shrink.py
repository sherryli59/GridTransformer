"""Shrinkage-ensemble KABLJ pair energy (Berthier-Charbonneau-Yaida point-to-set cavity method).

At shrinkage parameter lam in [0,1], each pair's sigma is scaled by lambda_tilde(lam, mobile_i, mobile_j):
lam if both particles are mobile, (1+lam)/2 if exactly one is mobile (one mobile, one pinned), 1.0 if both are
pinned (unshrunk). The cutoff scales with the shrunk sigma: rc = RCUT_FACTOR * lambda_tilde * sigma_ab. eps is
left unscaled (only the core radius shrinks). At lam=1.0, lambda_tilde==1.0 everywhere, so this collapses exactly
onto the standard shifted-LJ/minimum-image convention used by `ka_energy`/`ka_pmc_3d.particle_energies`.

`shrink_pair_row` mirrors `ka_energy.ka_pair_row_scatter`'s O(N) single-particle-row pattern (idx=(rows, i),
candidate position xi) for use in single-site Metropolis moves: delta = shrink_pair_row(new) - shrink_pair_row(old)."""
from __future__ import annotations
import torch
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR


def _lt_values(lam, mi, mj, dtype):
    """mi, mj: broadcastable bool tensors (mobile masks). Returns a `dtype` tensor of the broadcast shape with
    lam where both mobile, (1+lam)/2 where exactly one mobile, 1.0 where both pinned."""
    both = mi & mj
    one = mi ^ mj
    lt = torch.ones_like(both, dtype=dtype)
    lt = torch.where(one, torch.full_like(lt, (1.0 + lam) / 2.0), lt)
    lt = torch.where(both, torch.full_like(lt, float(lam)), lt)
    return lt


def lambda_tilde(lam, mobile_i, mobile_j):
    """mobile_i, mobile_j: [B,N] bool mobile masks. Returns [B,N,N]."""
    B, Ni = mobile_i.shape
    Nj = mobile_j.shape[1]
    mi = mobile_i[:, :, None].expand(B, Ni, Nj)
    mj = mobile_j[:, None, :].expand(B, Ni, Nj)
    return _lt_values(lam, mi, mj, torch.get_default_dtype())


def shrink_particle_energy(x, s, L, lam, mobile):
    """pe_i = sum_{j!=i} e(x_i, x_j) under shrinkage lam, shifted-LJ, minimum image. Returns [B, N].
    At lam=1.0 exactly equals `ka_pmc_3d.particle_energies` (lambda_tilde==1.0 everywhere)."""
    B, N, _ = x.shape
    dtype = x.dtype
    si = s.long()
    t_sig = torch.tensor(SIGMA, device=x.device, dtype=dtype)
    t_eps = torch.tensor(EPS, device=x.device, dtype=dtype)
    a = si[:, :, None].expand(-1, -1, N); b = si[:, None, :].expand(-1, N, -1)
    sig = t_sig[a, b]; eps = t_eps[a, b]
    mi = mobile[:, :, None].expand(-1, -1, N); mj = mobile[:, None, :].expand(-1, N, -1)
    lt = _lt_values(lam, mi, mj, dtype)
    sig_s = lt * sig
    rc = RCUT_FACTOR * sig_s
    diff = x[:, :, None, :] - x[:, None, :, :]; diff = diff - L * torch.round(diff / L)
    eye = torch.eye(N, device=x.device, dtype=torch.bool)[None]
    r2 = (diff ** 2).sum(-1).masked_fill(eye, 1e12); inv6 = (sig_s ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6); src6 = (sig_s / rc) ** 6
    return torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e)).sum(-1)


def shrink_pair_row(x, s, idx, xi, L, lam, mobile):
    """O(N) row energy of the single particle idx=(rows,i) placed at candidate position xi, against all
    other particles, under shrinkage lam. Same shifted/cutoff/min-image convention as `shrink_particle_energy`
    (mirrors `ka_energy.ka_pair_row_scatter`'s signature/pattern). rows,i: [K]; xi: [K,3]. Returns [K]."""
    b_ids, p_ids = idx
    B, N, _ = x.shape
    dtype = x.dtype
    si = s.long()
    if si.dim() == 1:
        si = si[None].expand(B, -1)
    t_sig = torch.tensor(SIGMA, device=x.device, dtype=dtype)
    t_eps = torch.tensor(EPS, device=x.device, dtype=dtype)
    s_row = si[b_ids]                                          # [K,N]
    s_i = si[b_ids, p_ids]                                     # [K]
    sig = t_sig[s_i[:, None].expand(-1, N), s_row]             # [K,N]
    eps = t_eps[s_i[:, None].expand(-1, N), s_row]

    mobile_row = mobile[b_ids]                                 # [K,N]
    mobile_i = mobile[b_ids, p_ids][:, None].expand(-1, N)     # [K,N]
    lt = _lt_values(lam, mobile_i, mobile_row, dtype)          # [K,N]

    sig_s = lt * sig
    rc = RCUT_FACTOR * sig_s
    diff = xi[:, None, :] - x[b_ids]                           # [K,N,3]
    diff = diff - L * torch.round(diff / L)
    r2 = (diff ** 2).sum(-1)                                   # [K,N]
    r2[torch.arange(r2.shape[0], device=x.device), p_ids] = 1e12
    inv6 = (sig_s ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (sig_s / rc) ** 6
    e = torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e))
    return e.sum(-1)                                           # [K]
