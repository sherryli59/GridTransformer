"""Swap Monte Carlo for 2D Kob-Andersen: displacement moves + A<->B identity swaps.
Swap MC is what makes KA equilibrable in the supercooled regime. Energy uses a local
recompute per move (O(N) per particle) for speed."""
from __future__ import annotations
import os, torch
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR


def _pair_energy_one(ri, ai, x, s, i, L):
    # energy of particle (ri, ai) vs all others; ai: [B] species of particle i
    diff = ri[:, None, :] - x                                # [B,N,2]
    diff = diff - L * torch.round(diff / L)
    r2 = (diff ** 2).sum(-1)                                  # [B,N]
    r2[:, i] = 1e12
    sig = torch.tensor(SIGMA, device=x.device, dtype=x.dtype)[ai.long()][:, s.long()]  # [B,N]
    eps = torch.tensor(EPS, device=x.device, dtype=x.dtype)[ai.long()][:, s.long()]
    rc = RCUT_FACTOR * sig
    inv6 = (sig ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (sig / rc) ** 6
    e = e - 4 * eps * (src6 ** 2 - src6)
    return torch.where(r2 < rc ** 2, e, torch.zeros_like(e)).sum(-1)  # [B]


def swap_mcmc(N, L, kT, s, d=2, device="cpu", n_chains=512, n_equil=2000,
              n_collect=200, every=20, step=0.08, swap_frac=0.2, x0=None):
    """s: [N] fixed species labels. Returns (configs [n,N,2], s)."""
    B = n_chains
    x = (torch.rand(B, N, d, device=device) * L) if x0 is None else x0.to(device).expand(B, N, d).clone()
    sd = s.to(device)
    snaps = []
    for sweep in range(n_equil + n_collect):
        # displacement sweep
        for i in range(N):
            xi = x[:, i, :]; ai = sd[i].expand(B)
            prop = torch.remainder(xi + step * torch.randn_like(xi), L)
            dE = _pair_energy_one(prop, ai, x, sd, i, L) - _pair_energy_one(xi, ai, x, sd, i, L)
            acc = torch.log(torch.rand(B, device=device)) < (-dE / kT)
            x[:, i, :] = torch.where(acc[:, None], prop, xi)
        # swap moves: pick random A and random B index, propose swapping their positions
        # (equivalent to swapping identities) -> accelerates identity relaxation.
        nsw = int(swap_frac * N)
        Aidx = (sd == 0).nonzero().squeeze(-1); Bidx = (sd == 1).nonzero().squeeze(-1)
        for _ in range(nsw):
            i = Aidx[torch.randint(len(Aidx), (1,))].item()
            j = Bidx[torch.randint(len(Bidx), (1,))].item()
            xi, xj = x[:, i, :].clone(), x[:, j, :].clone()
            e_old = (_pair_energy_one(xi, sd[i].expand(B), x, sd, i, L)
                     + _pair_energy_one(xj, sd[j].expand(B), x, sd, j, L))
            xp = x.clone(); xp[:, i, :] = xj; xp[:, j, :] = xi
            e_new = (_pair_energy_one(xj, sd[i].expand(B), xp, sd, i, L)
                     + _pair_energy_one(xi, sd[j].expand(B), xp, sd, j, L))
            acc = torch.log(torch.rand(B, device=device)) < (-(e_new - e_old) / kT)
            x[:, i, :] = torch.where(acc[:, None], xj, xi)
            x[:, j, :] = torch.where(acc[:, None], xi, xj)
        if sweep >= n_equil and (sweep - n_equil) % every == 0:
            snaps.append(x.clone())
    return torch.cat(snaps, 0), sd.cpu()


def make_species(N, frac_B=0.35, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = (torch.rand(N, generator=g) < frac_B).to(torch.int8)
    return s


def get_data(N, L, kT, frac_B=0.35, device="cpu", **kw):
    cache = f"/tmp/ka2d_N{N}_L{L:g}_T{kT:g}_B{frac_B:g}.pt"
    if os.path.exists(cache):
        d = torch.load(cache, map_location=device)
        return d["x"].to(device), d["s"].to(device)
    s = make_species(N, frac_B)
    x, sd = swap_mcmc(N, L, kT, s, device=device, **kw)
    torch.save({"x": x.cpu(), "s": sd}, cache)
    return x.to(device), sd.to(device)


if __name__ == "__main__":
    from liquid_coupling_flow.ka_energy import ka_energy
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N, L, kT = 64, (64 / 0.8) ** 0.5, 1.0
    s = make_species(N, 0.35)
    x, sd = swap_mcmc(N, L, kT, s, device=dev, n_chains=64, n_equil=200, n_collect=20, every=10)
    U = (ka_energy(x, sd, L) / N).mean().item()
    print(f"equilibrated <U>/N = {U:.3f} (should be finite, negative-ish, no infs)")
    assert torch.isfinite(ka_energy(x, sd, L)).all()
    print("swap_mcmc sanity OK")
