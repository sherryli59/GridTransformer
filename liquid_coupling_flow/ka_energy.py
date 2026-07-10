"""Dimension-generic Kob-Andersen binary LJ energy (shifted, minimum image)."""
from __future__ import annotations
import torch

# standard KA interaction matrix (indices 0=A, 1=B)
SIGMA = [[1.0, 0.8], [0.8, 0.88]]
EPS = [[1.0, 1.5], [1.5, 0.5]]
RCUT_FACTOR = 2.5


def _matrix(s, table, device, dtype):
    # s: [N] -> [N,N], or per-config [B,N] -> [B,N,N] pair-parameter matrix
    t = torch.tensor(table, device=device, dtype=dtype)      # [2,2]
    si = s.long()
    if si.dim() == 1:
        return t[si][:, si]                                  # [N,N]
    a = si[:, :, None].expand(-1, -1, si.shape[1])           # [B,N,N]
    b = si[:, None, :].expand(-1, si.shape[1], -1)
    return t[a, b]                                            # [B,N,N]


def ka_energy(x, s, L, per_particle: bool = False):
    """x: [B,N,D] in [0,L)^D; s: [N] or [B,N] in {0,1}. Returns [B] (or [B,N] if per_particle)."""
    B, N, d = x.shape
    dtype = x.dtype
    sig = _matrix(s, SIGMA, x.device, dtype)                  # [N,N]
    eps = _matrix(s, EPS, x.device, dtype)
    rc = RCUT_FACTOR * sig                                    # [N,N]
    diff = x[:, :, None, :] - x[:, None, :, :]                # [B,N,N,2]
    diff = diff - L * torch.round(diff / L)
    r2 = (diff ** 2).sum(-1)                                  # [B,N,N]
    eye = torch.eye(N, device=x.device, dtype=torch.bool)
    r2 = r2.masked_fill(eye, 1e12)
    inv6 = (sig ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (sig / rc) ** 6
    eshift = 4 * eps * (src6 ** 2 - src6)
    within = r2 < rc ** 2
    e = torch.where(within, e - eshift, torch.zeros_like(e))  # [B,N,N], diagonal=0
    if per_particle:
        return e.sum(-1)                                      # [B,N]
    return 0.5 * e.sum(dim=(1, 2))                            # [B]


def ka_pair_row(x, s, i, xi, L):
    """Sum of pair energies between a CANDIDATE position xi [B,2] for particle i and all j != i, with the
    exact same shifted/cutoff/min-image convention as ka_energy. s: [B,N] or [N]. Returns [B].
    Enables O(N)-per-move incremental Metropolis (dU = row_new - row_old)."""
    B, N, _ = x.shape
    dtype = x.dtype
    si = s.long()
    if si.dim() == 1:
        si = si[None].expand(B, -1)
    t_sig = torch.tensor(SIGMA, device=x.device, dtype=dtype)
    t_eps = torch.tensor(EPS, device=x.device, dtype=dtype)
    sig = t_sig[si[:, i:i + 1].expand(-1, N), si]             # [B,N] pair sigma_i,j
    eps = t_eps[si[:, i:i + 1].expand(-1, N), si]
    rc = RCUT_FACTOR * sig
    diff = xi[:, None, :] - x                                  # [B,N,2]
    diff = diff - L * torch.round(diff / L)
    r2 = (diff ** 2).sum(-1)                                   # [B,N]
    r2[:, i] = 1e12                                            # exclude self
    inv6 = (sig ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (sig / rc) ** 6
    e = torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e))
    return e.sum(-1)                                           # [B]


def ka_pair_row_scatter(x, s, idx, xi, L):
    """Row energies for K scattered movers: idx=(b_ids[K], p_ids[K]), xi [K,2] candidate positions.
    Same shifted/cutoff/min-image convention as ka_energy. Returns [K]. One batched kernel for all movers."""
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
    rc = RCUT_FACTOR * sig
    diff = xi[:, None, :] - x[b_ids]                           # [K,N,2]
    diff = diff - L * torch.round(diff / L)
    r2 = (diff ** 2).sum(-1)                                   # [K,N]
    r2[torch.arange(r2.shape[0], device=x.device), p_ids] = 1e12
    inv6 = (sig ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (sig / rc) ** 6
    e = torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e))
    return e.sum(-1)


def ka_forces(x, s, L):
    """Analytic forces for the shifted KA LJ (shift constant -> force unchanged; sharp cutoff standard).
    x [B,N,2], s [N] or [B,N] -> F [B,N,2] with F_i = -dU/dx_i. Fully batched (one kernel chain)."""
    B, N, _ = x.shape
    dtype = x.dtype
    sig = _matrix(s, SIGMA, x.device, dtype)
    eps = _matrix(s, EPS, x.device, dtype)
    if sig.dim() == 2:
        sig = sig[None]; eps = eps[None]
    rc = RCUT_FACTOR * sig
    diff = x[:, :, None, :] - x[:, None, :, :]                # [B,N,N,2] r_i - r_j
    diff = diff - L * torch.round(diff / L)
    r2 = (diff ** 2).sum(-1)
    eye = torch.eye(N, device=x.device, dtype=torch.bool)
    r2 = r2.masked_fill(eye, 1e12)
    s2 = sig ** 2 / r2
    s6 = s2 ** 3
    # dU/dr2 = 4 eps (12 s12 - 6 s6) * (-1/r2) /2 ... use pair force magnitude / r2 form:
    # F_ij = 24 eps (2 s12 - s6) / r2 * diff   (repulsive positive along diff)
    fmag = 24 * eps * (2 * s6 ** 2 - s6) / r2                 # [B,N,N]
    fmag = torch.where(r2 < rc ** 2, fmag, torch.zeros_like(fmag))
    return (fmag[..., None] * diff).sum(2)                    # [B,N,2]
