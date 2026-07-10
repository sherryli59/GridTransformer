"""Vectorized parallel-Metropolis KABLJ displacement (dimension-generic; used for 3D).

All mobile particles are proposed at once and scored against the OLD configuration in a single batched pairwise
energy call (~1 GPU launch/sweep vs N sequential launches for exact single-site MC). This is NOT exact detailed
balance -- simultaneous moves make each acceptance a first-order approximation whose bias scales with step^2 --
so it is a fast EQUILIBRATOR for generating bulk/exterior configurations, NOT the exact interior-measurement
dynamics. Its residual bias vs the exact single-site sampler is measured in U/N (see reports .../pmc_validate_3d).

`particle_energies` is the same shifted min-image formula as `ka_energy` (asserted equal in tests), so the move
energetics are guaranteed consistent with the reference energy function used everywhere else."""
from __future__ import annotations
import torch
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR


def _tables(x):
    return (torch.tensor(SIGMA, device=x.device, dtype=x.dtype),
            torch.tensor(EPS, device=x.device, dtype=x.dtype))


def particle_energies(x, s, L):
    """pe_i = sum_{j!=i} e(x_i, x_j), shifted-LJ, minimum image. Returns [B, N].
    Consistent with ka_energy: ka_energy(x,s,L) == 0.5 * particle_energies(x,s,L).sum(1)."""
    B, N, _ = x.shape
    t_sig, t_eps = _tables(x)
    a = s[:, :, None].expand(-1, -1, N); b = s[:, None, :].expand(-1, N, -1)
    sig = t_sig[a, b]; eps = t_eps[a, b]; rc = RCUT_FACTOR * sig
    diff = x[:, :, None, :] - x[:, None, :, :]; diff = diff - L * torch.round(diff / L)
    eye = torch.eye(N, device=x.device, dtype=torch.bool)[None]
    r2 = (diff ** 2).sum(-1).masked_fill(eye, 1e12); inv6 = (sig ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6); src6 = (sig / rc) ** 6
    return torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e)).sum(-1)


def parallel_mc_disp(x, s, L, beta, step, mobile=None):
    """One parallel-Metropolis displacement sweep. beta may be scalar or per-row [B]. If mobile [B,N] is given,
    frozen particles are held fixed (noise + acceptance masked). Returns the updated x [B,N,D]."""
    B, N, _ = x.shape
    t_sig, t_eps = _tables(x)
    a = s[:, :, None].expand(-1, -1, N); b = s[:, None, :].expand(-1, N, -1)
    sig = t_sig[a, b]; eps = t_eps[a, b]; rc = RCUT_FACTOR * sig
    eye = torch.eye(N, device=x.device, dtype=torch.bool)[None]
    noise = step * torch.randn_like(x)
    if mobile is not None:
        noise = noise * mobile[..., None]
    prop = torch.remainder(x + noise, L)

    def cross(xa):                                                      # xa_i scored against OLD x_j
        diff = xa[:, :, None, :] - x[:, None, :, :]; diff = diff - L * torch.round(diff / L)
        r2 = (diff ** 2).sum(-1).masked_fill(eye, 1e12); inv6 = (sig ** 2 / r2) ** 3
        e = 4 * eps * (inv6 ** 2 - inv6); src6 = (sig / rc) ** 6
        return torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e)).sum(-1)

    dE = cross(prop) - cross(x)
    bt = beta.view(-1, 1) if (torch.is_tensor(beta) and beta.ndim == 1) else beta
    acc = torch.log(torch.rand_like(dE)) < (-bt * dE)
    if mobile is not None:
        acc = acc & mobile
    return torch.where(acc[..., None], prop, x)
