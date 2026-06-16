"""2D Kob-Andersen binary LJ energy (species-dependent sigma/eps, shifted, min-image)."""
from __future__ import annotations
import torch

# standard KA interaction matrix (indices 0=A, 1=B)
SIGMA = [[1.0, 0.8], [0.8, 0.88]]
EPS = [[1.0, 1.5], [1.5, 0.5]]
RCUT_FACTOR = 2.5


def _matrix(s, table, device, dtype):
    # s: [N] int8 -> [N,N] pair-parameter matrix
    t = torch.tensor(table, device=device, dtype=dtype)      # [2,2]
    si = s.long()
    return t[si][:, si]                                       # [N,N]


def ka_energy(x, s, L, per_particle: bool = False):
    """x: [B,N,2] in [0,L)^2; s: [N] in {0,1}. Returns [B] (or [B,N] if per_particle)."""
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
