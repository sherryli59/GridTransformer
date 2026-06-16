"""P1 GATE (glass version): is the KA glass genuinely METASTABLE at the chosen (N,T)?
Annealing-from-disorder WITH swaps reaches the equilibrium basin; WITHOUT swaps
(displacement only) it should get stuck ABOVE it (no fast identity-relaxation pathway).
GO if displacement-only stays clearly above the swap basin at the largest budget ->
the system needs enhanced sampling -> a learned structural proposal has room to help.
NO-GO if displacement-only also reaches the basin (too easy -> wrong T)."""
from __future__ import annotations
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
from liquid_coupling_flow.ka_mcmc_fast import swap_mcmc_fast
from liquid_coupling_flow.ka_mcmc import make_species
from liquid_coupling_flow.ka_energy import ka_energy

ART = os.path.join(os.path.dirname(__file__), "artifacts")
FRACB, T, RHO = 0.35, 0.50, 1.2


def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    N = 512; L = (N / RHO) ** 0.5
    s = make_species(N, FRACB)
    # reference: with-swaps annealing reaches the basin
    xr, sd = swap_mcmc_fast(N, L, T, s, device=device, n_chains=24, n_equil=4000,
                            n_collect=80, every=20, step=0.05, n_swap=N // 8)
    U_ref = (ka_energy(xr, sd, L) / N).mean().item()
    # displacement-only (n_swap=0) at increasing budget
    budgets = [500, 1500, 3000, 5000]
    disp = []
    for nb in budgets:
        xd, _ = swap_mcmc_fast(N, L, T, s, device=device, n_chains=24, n_equil=nb,
                               n_collect=40, every=10, step=0.05, n_swap=0)
        disp.append((ka_energy(xd, sd, L) / N).mean().item())
        print(f"budget={nb}: displacement-only <U>/N={disp[-1]:.4f} (swap basin {U_ref:.4f})", flush=True)
    gap = disp[-1] - U_ref
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(budgets, disp, "-o", label="displacement-only (no swaps)")
    ax.axhline(U_ref, ls="--", color="C2", label=f"swap-MC basin {U_ref:.3f}")
    ax.set_xlabel("annealing budget (sweeps)"); ax.set_ylabel("<U>/N")
    ax.set_title(f"P1 metastability (N={N}): gap={gap:+.3f} "
                 f"({'GO (metastable/hard)' if gap > 0.05 else 'NO-GO (too easy)'})")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(ART, "ka_gate_p1.png"), dpi=120)
    print(f"P1 VERDICT: {'GO' if gap > 0.05 else 'NO-GO'} (gap={gap:+.3f})", flush=True)


if __name__ == "__main__":
    main()
