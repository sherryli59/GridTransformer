"""Dimension-general periodic LJ MCMC (ground-truth Boltzmann sampler + data source).

Single-particle Metropolis on the torus [0,L)^{N x d}, min-image cutoff+shift LJ.
Used both as the forward-KL training data and as the ground-truth Boltzmann reference
for energy / g(r) comparisons at any N, L, d. Cached under /tmp keyed by (N,L,d).
"""

from __future__ import annotations

import os

import torch


def _pair_energy_one(ri, x, i, L, cutoff):
    """LJ energy of particle i at position ri against all others (d-agnostic)."""
    diff = ri[:, None, :] - x
    diff = diff - L * torch.round(diff / L)
    r2 = (diff ** 2).sum(-1)
    r2[:, i] = 1e9
    within = r2 < cutoff ** 2
    inv6 = (1.0 / r2) ** 3
    ic6 = (1.0 / cutoff) ** 6
    e = 4.0 * (inv6 ** 2 - inv6) - 4.0 * (ic6 ** 2 - ic6)
    return torch.where(within, e, torch.zeros_like(e)).sum(-1)


def mcmc_lj(N, L, kT, cutoff, d=2, device="cpu", n_chains=4096, n_equil=300,
            n_collect=200, every=25, step=0.15, x0=None):
    if x0 is None:
        x = torch.rand(n_chains, N, d, device=device) * L
    else:                                            # start every chain from a given IC
        x = x0.to(device).expand(n_chains, N, d).clone()
    snaps = []
    for sweep in range(n_equil + n_collect):
        for i in range(N):
            xi = x[:, i, :]
            prop = torch.remainder(xi + step * torch.randn_like(xi), L)
            dE = _pair_energy_one(prop, x, i, L, cutoff) - _pair_energy_one(xi, x, i, L, cutoff)
            acc = torch.log(torch.rand(n_chains, device=device)) < (-dE / kT)
            x[:, i, :] = torch.where(acc[:, None], prop, xi)
        if sweep >= n_equil and (sweep - n_equil) % every == 0:
            snaps.append(x.clone())
    return torch.cat(snaps, dim=0)


def get_data(N, L, kT, cutoff, d=2, device="cpu", **kw):
    cache = f"/tmp/lj{d}d_N{N}_L{L:g}.pt"
    if os.path.exists(cache):
        return torch.load(cache, map_location=device)
    data = mcmc_lj(N, L, kT, cutoff, d=d, device=device, **kw).cpu()
    torch.save(data, cache)
    return data.to(device)
