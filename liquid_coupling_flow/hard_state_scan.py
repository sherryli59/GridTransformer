"""Phase 1 (GO/NO-GO): find a state point where uniform+SMC CANNOT anneal efficiently.

A learned proposal can only earn its keep where the baseline (annealed SMC from uniform
noise) fails. So before training any flow, establish that such a regime exists. For each
2D LJ state point we compare three numbers for <U>/N:
  - MCMC from an ORDERED grid IC   -> the low-energy (equilibrium-ish) basin
  - MCMC from a DISORDERED uniform IC -> what relaxation-from-noise reaches
  - uniform + SMC at increasing budget -> what annealed importance sampling reaches
HARD <=> MCMC-ordered sits well below MCMC-uniform ~ uniform+SMC (a basin annealing from
disorder cannot reach in the budget) and/or strong seed variance. EASY <=> all agree and
uniform+SMC converges in a few sweeps.

Run:  python -m liquid_coupling_flow.hard_state_scan
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from liquid_coupling_flow.energy import lj_energy
from liquid_coupling_flow.particles import min_image
from liquid_coupling_flow.smc import anneal_smc, SingleParticleMetropolis
from liquid_coupling_flow import mcmc

ART = os.path.join(os.path.dirname(__file__), "artifacts")

N = 16
# (T*, rho): easy -> progressively harder (colder + denser)
STATES = [(1.0, 0.64), (0.7, 0.80), (0.5, 0.90), (0.35, 1.00)]
BUDGETS = [5, 10, 20, 40, 80]
STEP = 0.1


def grid_ic(N, L):
    n = round(N ** 0.5)
    c = (torch.arange(n) + 0.5) * L / n
    return torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).reshape(-1, 2)


def overlap_frac(x, L, thresh=0.8):
    _, r = min_image(x, L)
    r = r + torch.eye(N, device=x.device)[None] * 1e3
    return float((r.min(dim=2).values.reshape(-1) < thresh).float().mean())


def uniform_smc(N, L, kT, cutoff, n_bridge, device, seed):
    torch.manual_seed(seed)
    M = 2000
    x0 = torch.rand(M, N, 2, device=device) * L
    base = -2 * N * math.log(L)
    res = anneal_smc(x0, lambda x: torch.full((x.shape[0],), base, device=x.device),
                     lambda x: -lj_energy(x, L, cutoff=cutoff, shift=True) / kT,
                     SingleParticleMetropolis(step=STEP, n_sweeps=1, L=L),
                     n_bridge=n_bridge, resample_thresh=0.5)
    Us = (lj_energy(res["x"], L, cutoff=cutoff, shift=True) / N).cpu().numpy()
    wn = res["weights"].cpu().numpy()
    return float((wn * Us).sum()), float(res["ess"]), overlap_frac(res["x"], L)


def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    fig, axes = plt.subplots(1, len(STATES), figsize=(5 * len(STATES), 4.4), squeeze=False)
    print(f"{'T*':>5} {'rho':>5} {'cutoff':>6} {'MCMC_ord':>9} {'MCMC_unif':>9} "
          f"{'SMC@80':>9} {'ESS@80':>7} {'ov@80':>6} {'gap(ord)':>8}", flush=True)

    for col, (T, rho) in enumerate(STATES):
        L = math.sqrt(N / rho)
        cutoff = min(2.4, 0.49 * L)
        # references: MCMC from ordered grid IC and from uniform IC (long equilibration)
        x_grid = grid_ic(N, L)
        d_ord = mcmc.mcmc_lj(N, L, T, cutoff, d=2, device=device, n_chains=512,
                             n_equil=2500, n_collect=200, every=20, step=STEP, x0=x_grid)
        d_unif = mcmc.mcmc_lj(N, L, T, cutoff, d=2, device=device, n_chains=512,
                              n_equil=2500, n_collect=200, every=20, step=STEP)
        U_ord = float((lj_energy(d_ord, L, cutoff=cutoff, shift=True) / N).mean())
        U_unif = float((lj_energy(d_unif, L, cutoff=cutoff, shift=True) / N).mean())

        means, stds = [], []
        for nb in BUDGETS:
            vals = [uniform_smc(N, L, T, cutoff, nb, device, s)[0] for s in (0, 1)]
            means.append(np.mean(vals)); stds.append(np.std(vals))
        e80, ess80, ov80 = uniform_smc(N, L, T, cutoff, 80, device, 0)
        gap = means[-1] - U_ord
        print(f"{T:>5.2f} {rho:>5.2f} {cutoff:>6.2f} {U_ord:>9.3f} {U_unif:>9.3f} "
              f"{means[-1]:>9.3f} {100*ess80:>6.1f}% {ov80:>6.3f} {gap:>8.3f}", flush=True)

        ax = axes[0][col]
        ax.errorbar(BUDGETS, means, yerr=stds, fmt="-o", color="C1", capsize=3, label="uniform+SMC")
        ax.axhline(U_ord, color="C2", ls="--", lw=1.5, label=f"MCMC ordered IC {U_ord:.2f}")
        ax.axhline(U_unif, color="C3", ls=":", lw=1.5, label=f"MCMC uniform IC {U_unif:.2f}")
        ax.set_title(f"T*={T}, rho={rho}\ngap(SMC-ordered)={gap:+.3f}")
        ax.set_xlabel("SMC budget (sweeps)"); ax.set_ylabel("<U>/N")
        ax.legend(fontsize=8)

    fig.suptitle("Phase 1 GO/NO-GO: can uniform+SMC anneal the state point? "
                 "(hard = stuck above MCMC-ordered)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(ART, "hard_state_scan.png")
    fig.savefig(out, dpi=120); print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
