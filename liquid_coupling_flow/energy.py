"""Periodic Lennard-Jones energy (reduced units) — the Boltzmann target.

U = sum_{i<j} 4 eps [ (sigma/r_ij)^12 - (sigma/r_ij)^6 ],  r_ij = min-image distance.

Works for any spatial dim d (2D for cheap toys, 3D for the real liquid). Used both
as the training target (energy-based / reweighting) and as the log-density (up to
the partition function) for SMC and ESS:

    log pi(x) = -U(x) / kT   (+ const)

`min_dist` core-clamps r to avoid the r^-12 blow-up on overlaps (the same trick the
GridTransformer refinability benchmark uses); leave None for the exact potential.
"""

from __future__ import annotations

import torch


def lj_energy(x: torch.Tensor, L: float, sigma: float = 1.0, eps: float = 1.0,
              cutoff: float | None = None, shift: bool = True,
              min_dist: float | None = None) -> torch.Tensor:
    """x: [B, N, d] in a periodic box of side L. Returns energy [B]."""
    B, N, d = x.shape
    iu = torch.triu_indices(N, N, offset=1, device=x.device)
    diff = x[:, iu[0], :] - x[:, iu[1], :]          # [B, P, d]
    diff = diff - L * torch.round(diff / L)         # minimum image
    r2 = (diff ** 2).sum(-1)                         # [B, P]
    if min_dist is not None:
        r2 = r2.clamp_min(min_dist ** 2)
    inv2 = (sigma ** 2) / r2
    inv6 = inv2 ** 3
    e_pair = 4.0 * eps * (inv6 ** 2 - inv6)
    if cutoff is not None:
        within = r2 < cutoff ** 2
        if shift:
            ic6 = (sigma / cutoff) ** 6
            e_shift = 4.0 * eps * (ic6 ** 2 - ic6)
            e_pair = torch.where(within, e_pair - e_shift, torch.zeros_like(e_pair))
        else:
            e_pair = torch.where(within, e_pair, torch.zeros_like(e_pair))
    return e_pair.sum(-1)


def lj_log_prob_unnorm(x: torch.Tensor, L: float, kT: float = 1.0, **kwargs) -> torch.Tensor:
    """Unnormalised log Boltzmann density (up to the partition function)."""
    return -lj_energy(x, L, **kwargs) / kT
