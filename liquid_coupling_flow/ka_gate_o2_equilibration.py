"""O2 GATE: can swap MC equilibrate the KA glass at N=256..512 in budget?
PASS if <U>/N from an ORDERED-grid IC and a DISORDERED uniform IC converge to the same
value (within ~0.02) at every N -> ergodic -> trustworthy references. Uses the fast
sampler (the small parallel-Metropolis energy bias cancels in the ordered-vs-uniform
GAP, so the IC-agreement test is unaffected). (FRACB,T,RHO) from O1."""
from __future__ import annotations
import os, math, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
from liquid_coupling_flow.ka_mcmc_fast import swap_mcmc_fast
from liquid_coupling_flow.ka_mcmc import make_species
from liquid_coupling_flow.ka_energy import ka_energy

ART = os.path.join(os.path.dirname(__file__), "artifacts")
FRACB, T, RHO = 0.35, 0.50, 1.2     # from O1 (65:35, supercooled, standard 2D KA density)


def grid_ic(N, L):
    n = math.ceil(N ** 0.5); c = (torch.arange(n) + 0.5) * L / n
    g = torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).reshape(-1, 2)
    return g[:N]


def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    sizes = [256, 512]
    rows = []
    for N in sizes:
        L = (N / RHO) ** 0.5
        s = make_species(N, FRACB)
        kw = dict(device=device, n_chains=32, n_equil=4000, n_collect=100, every=20,
                  step=0.05, n_swap=N // 8)
        xo, sd = swap_mcmc_fast(N, L, T, s, x0=grid_ic(N, L), **kw)
        xu, _ = swap_mcmc_fast(N, L, T, s, **kw)
        Uo = (ka_energy(xo, sd, L) / N).mean().item()
        Uu = (ka_energy(xu, sd, L) / N).mean().item()
        rows.append((N, Uo, Uu, abs(Uo - Uu)))
        print(f"N={N}: U_ordered={Uo:.4f} U_uniform={Uu:.4f} gap={abs(Uo-Uu):.4f}", flush=True)
    ok = all(g < 0.02 for *_, g in rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r[0] for r in rows], [r[3] for r in rows], "-o")
    ax.axhline(0.02, ls="--", color="r"); ax.set_xlabel("N"); ax.set_ylabel("|U_ord - U_unif|")
    ax.set_title(f"O2: equilibration gap vs N ({'PASS' if ok else 'FAIL'})")
    fig.tight_layout(); fig.savefig(os.path.join(ART, "ka_gate_o2.png"), dpi=120)
    print(f"O2 VERDICT: {'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
