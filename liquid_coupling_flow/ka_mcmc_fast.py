"""Scalable swap MC for 2D KA. The per-particle Python loop in ka_mcmc.swap_mcmc is
O(N) kernel launches/sweep -> hours at N=2048. Here the DISPLACEMENT sweep is
PARALLEL-METROPOLIS: all particles propose at once, accept independently using the
others' OLD positions, via one vectorized O(N^2) energy op. That violates detailed
balance by O(step^2) when interacting particles move together, so it MUST be validated
against the exact sequential sampler (ka_mcmc_fast_validate.py) and used with a small
step. Identity swaps stay exact-sequential (fewer, since displacement is now cheap)."""
from __future__ import annotations
import os, torch
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR
from liquid_coupling_flow.ka_mcmc import _pair_energy_one, make_species


def _u_matrix(xa, xb, sa, sb, L, exclude_diag):
    """u(xa_k, xb_j) shifted-LJ -> [B,Na,Nb]. exclude_diag zeroes k==j (self)."""
    sig = torch.tensor(SIGMA, device=xa.device, dtype=xa.dtype)[sa.long()][:, sb.long()]  # [Na,Nb]
    eps = torch.tensor(EPS, device=xa.device, dtype=xa.dtype)[sa.long()][:, sb.long()]
    rc = RCUT_FACTOR * sig
    diff = xa[:, :, None, :] - xb[:, None, :, :]
    diff = diff - L * torch.round(diff / L)
    r2 = (diff ** 2).sum(-1)                                   # [B,Na,Nb]
    if exclude_diag:
        r2 = r2 + torch.eye(xa.shape[1], device=xa.device, dtype=r2.dtype)[None] * 1e12
    inv6 = (sig ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (sig / rc) ** 6
    return torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e))


def swap_mcmc_fast(N, L, kT, s, device="cpu", n_chains=64, n_equil=2000, n_collect=200,
                   every=20, step=0.04, n_swap=None, x0=None):
    """Parallel-Metropolis displacement + exact-sequential identity swaps. Returns (x, s)."""
    B = n_chains
    x = (torch.rand(B, N, 2, device=device) * L) if x0 is None else x0.to(device).expand(B, N, 2).clone()
    sd = s.to(device)
    Aidx = (sd == 0).nonzero().squeeze(-1)
    Bidx = (sd == 1).nonzero().squeeze(-1)
    n_swap = (N // 8 if n_swap is None else n_swap)
    snaps = []
    for sweep in range(n_equil + n_collect):
        # --- parallel displacement (all particles at once) ---
        prop = torch.remainder(x + step * torch.randn_like(x), L)
        Uold = _u_matrix(x, x, sd, sd, L, exclude_diag=True)        # [B,N,N]
        Unew = _u_matrix(prop, x, sd, sd, L, exclude_diag=True)     # prop_k vs x_j (j!=k)
        dE = (Unew - Uold).sum(-1)                                  # [B,N]
        acc = torch.log(torch.rand(B, N, device=device)) < (-dE / kT)
        x = torch.where(acc[:, :, None], prop, x)
        # --- exact-sequential identity swaps (fewer) ---
        if len(Aidx) > 0 and len(Bidx) > 0:
            for _ in range(n_swap):
                i = Aidx[torch.randint(len(Aidx), (1,))].item()
                j = Bidx[torch.randint(len(Bidx), (1,))].item()
                xi, xj = x[:, i, :].clone(), x[:, j, :].clone()
                e_old = (_pair_energy_one(xi, sd[i].expand(B), x, sd, i, L)
                         + _pair_energy_one(xj, sd[j].expand(B), x, sd, j, L))
                xp = x.clone(); xp[:, i, :] = xj; xp[:, j, :] = xi
                e_new = (_pair_energy_one(xj, sd[i].expand(B), xp, sd, i, L)
                         + _pair_energy_one(xi, sd[j].expand(B), xp, sd, j, L))
                a = torch.log(torch.rand(B, device=device)) < (-(e_new - e_old) / kT)
                x[:, i, :] = torch.where(a[:, None], xj, xi)
                x[:, j, :] = torch.where(a[:, None], xi, xj)
        if sweep >= n_equil and (sweep - n_equil) % every == 0:
            snaps.append(x.clone())
    return torch.cat(snaps, 0), sd.cpu()


def get_data_fast(N, L, kT, frac_B=0.35, device="cpu", **kw):
    cache = f"/tmp/kaF2d_N{N}_L{L:g}_T{kT:g}_B{frac_B:g}.pt"
    if os.path.exists(cache):
        d = torch.load(cache, map_location=device); return d["x"].to(device), d["s"].to(device)
    s = make_species(N, frac_B)
    x, sd = swap_mcmc_fast(N, L, kT, s, device=device, **kw)
    torch.save({"x": x.cpu(), "s": sd}, cache)
    return x.to(device), sd.to(device)
