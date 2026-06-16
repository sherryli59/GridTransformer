"""Validate swap_mcmc_fast: (1) does parallel-Metropolis reproduce the EXACT sequential
sampler's equilibrium <U>/N and g(r) at N=64? (2) how fast is it at N=256..1024?"""
from __future__ import annotations
import os, time, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
from liquid_coupling_flow.ka_mcmc import swap_mcmc, make_species
from liquid_coupling_flow.ka_mcmc_fast import swap_mcmc_fast
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_observables import partial_gr

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    # (1) correctness vs exact at N=64
    N, rho, kT = 64, 1.2, 0.5
    L = (N / rho) ** 0.5
    s = make_species(N, 0.35)
    xs, sd = swap_mcmc(N, L, kT, s, device=device, n_chains=96, n_equil=2000,
                       n_collect=150, every=15, step=0.08)
    xf, _ = swap_mcmc_fast(N, L, kT, s, device=device, n_chains=96, n_equil=3000,
                           n_collect=150, every=15, step=0.05)
    Us = (ka_energy(xs, sd, L) / N).mean().item()
    Uf = (ka_energy(xf, sd, L) / N).mean().item()
    rc, gs = partial_gr(xs, sd, L, rmax=L / 2, nbins=80, pair=(0, 0))
    _, gf = partial_gr(xf, sd, L, rmax=L / 2, nbins=80, pair=(0, 0))
    dU = abs(Us - Uf)
    print(f"N=64: <U>/N exact={Us:.4f}  fast={Uf:.4f}  |diff|={dU:.4f}", flush=True)
    ok = dU < 0.03
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(rc, gs, label=f"exact seq ({Us:.3f})")
    ax[0].plot(rc, gf, "--", label=f"fast parallel ({Uf:.3f})")
    ax[0].set_title(f"g_AA(r): {'MATCH' if ok else 'MISMATCH'} (|dU|={dU:.4f})")
    ax[0].set_xlabel("r"); ax[0].legend()

    # (2) speed benchmark
    rows = []
    for Nb in [256, 512, 1024]:
        Lb = (Nb / rho) ** 0.5; sb = make_species(Nb, 0.35)
        t = time.time()
        swap_mcmc_fast(Nb, Lb, kT, sb, device=device, n_chains=16, n_equil=20,
                       n_collect=1, every=1, step=0.05, n_swap=Nb // 16)
        dt = (time.time() - t) / 21
        rows.append((Nb, dt))
        print(f"N={Nb}: fast {dt*1000:.0f} ms/sweep -> 4000 sweeps = {dt*4000/60:.1f} min", flush=True)
    ax[1].plot([r[0] for r in rows], [r[1] * 1000 for r in rows], "-o")
    ax[1].set_xlabel("N"); ax[1].set_ylabel("ms/sweep (fast)"); ax[1].set_title("scaling")
    fig.tight_layout(); fig.savefig(os.path.join(ART, "ka_mcmc_fast_validate.png"), dpi=120)
    print(f"VALIDATION: {'PASS' if ok else 'FAIL'} (equilibrium {'matches' if ok else 'DIFFERS'})",
          flush=True)


if __name__ == "__main__":
    main()
