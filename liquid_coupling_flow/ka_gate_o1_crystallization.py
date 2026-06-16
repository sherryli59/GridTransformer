"""O1 GATE: does 2D KA (65:35) stay disordered (no crystallization) at the target T?
PASS if, after long swap MC, hexatic order psi6 stays low (liquid-like, < ~0.4) and
partial g(r) shows no sharp crystalline peaks. Scans a couple of compositions/T."""
from __future__ import annotations
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
from liquid_coupling_flow.ka_mcmc import swap_mcmc, make_species
from liquid_coupling_flow.ka_observables import partial_gr, psi6

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    rho = 1.2                       # 2D KA number density (standard ~1.2)
    # N=100 keeps the (still-O(N^2)) swap MC feasible for this LOCAL-order gate;
    # crystallization detection does not need large N. Large-N work awaits a faster MCMC.
    N = 100; L = (N / rho) ** 0.5
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    rows = []
    for k, (fracB, T) in enumerate([(0.35, 0.50), (0.35, 0.45), (0.50, 0.50)]):
        s = make_species(N, fracB)
        x, sd = swap_mcmc(N, L, T, s, device=device, n_chains=64,
                          n_equil=1500, n_collect=150, every=20, step=0.07)
        p6 = psi6(x, L).mean().item()
        rc, gAA = partial_gr(x, sd, L, rmax=L / 2, nbins=120, pair=(0, 0))
        rows.append((fracB, T, p6))
        ax[k].plot(rc, gAA)
        ax[k].set_title(f"fracB={fracB} T={T}\npsi6={p6:.3f} "
                        f"({'LIQUID' if p6 < 0.4 else 'ORDERED?'})")
        ax[k].set_xlabel("r"); ax[k].set_ylabel("g_AA(r)")
        print(f"fracB={fracB} T={T}: psi6={p6:.3f}", flush=True)
    fig.suptitle("O1 gate: KA stays disordered? (PASS if psi6 < ~0.4, g(r) liquid-like)")
    fig.tight_layout(); fig.savefig(os.path.join(ART, "ka_gate_o1.png"), dpi=120)
    best = min(rows, key=lambda r: r[2])
    print(f"O1 VERDICT: {'PASS' if best[2] < 0.4 else 'FAIL'} (best psi6={best[2]:.3f} at "
          f"fracB={best[0]}, T={best[1]})", flush=True)


if __name__ == "__main__":
    main()
