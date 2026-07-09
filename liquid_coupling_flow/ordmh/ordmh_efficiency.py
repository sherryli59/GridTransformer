"""Task 4 --- efficiency probe (measurement only; run AFTER Gate E passes).

Correctness is already independently established, so this is pure efficiency:

  1. Acceptance rate of the ordered-space MH for the mediocre antipode proposal
     AND a pure-uniform control (w_u=1, permutation-invariant -> zero ordering
     mismatch).  The uniform control's acceptance decay is the irreducible
     "any independence sampler in N particles" cost; the antipode's excess decay
     is the ordering tax.

  2. The ordering-mismatch term, isolated: for Boltzmann configs drawn from the
     reference, evaluate log q~ over K random orderings of each config and report
     the within-config spread (std, in nats).  This is energy-independent and
     finite --- the clean structural penalty with no CNF analog.  Reported total
     and per-particle.

  3. N-scaling of both, N in {3,4,5,6,7,8}.

No gate --- the deliverable is the honest scaling curve.

The learned KA generator is NOT applicable to this single-species WCA system
(it is a 2-species glass model), so the uniform-vs-antipode contrast stands in
for "toy vs better generator".
"""
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ordmh_energy import wca_energy
from ordmh_reference import displacement_mc
from ordmh_toy_proposal import ToyOrderedProposal
from ordmh_sampler import OrderedMH

BETA = 1.0
RHO = 0.5
NS = [3, 4, 5, 6, 7, 8]
LOGDIR = os.path.abspath(os.path.join(HERE, "..", "..", "reports", "logs-2026-07-09"))
os.makedirs(LOGDIR, exist_ok=True)


def box_for(N):
    return float(np.sqrt(N / RHO))


def ordering_spread(proposal, ref_configs, K=64, seed=0):
    """E_config[ std over K random orderings of log q~ ] (nats).  Pure structural
    penalty: the target is flat over orderings, so any spread here is a handicap.
    Returns (mean_within_config_std, total_variance, config_variance_of_mean)."""
    rng = np.random.default_rng(seed)
    M, N, _ = ref_configs.shape
    # build (M, K, N, 2) with independent random permutations per (config, ordering)
    perms = np.stack([[rng.permutation(N) for _ in range(K)] for _ in range(M)])
    variants = np.take_along_axis(
        ref_configs[:, None, :, :], perms[..., None], axis=2)   # (M,K,N,2)
    logq = proposal.log_q_ordered(variants.reshape(M * K, N, 2)).reshape(M, K)
    within_std = logq.std(axis=1)                     # (M,) spread over orderings
    per_config_mean = logq.mean(axis=1)               # (M,)
    return {
        "within_std_mean": float(within_std.mean()),
        "within_var_mean": float((logq.var(axis=1)).mean()),   # E_c[Var_perm]
        "config_var_of_mean": float(per_config_mean.var()),    # Var_c[E_perm]
        "total_var": float(logq.var()),
    }


def acceptance(proposal, N, L, seed=0):
    smp = OrderedMH(N=N, L=L, beta=BETA, proposal=proposal)
    out = smp.run(n_chains=4000, n_steps=700, n_equil=200, thin=5, seed=seed,
                  collect=False)
    return out["acc_rate"]


def main():
    rows = []
    for N in NS:
        L = box_for(N)
        # Boltzmann reference configs (independent equilibrium samples)
        mc = displacement_mc(N=N, L=L, beta=BETA, n_chains=400, n_equil=800,
                             n_collect=1, thin=1, step=0.22, seed=1000 + N)
        ref = mc["configs"][:400]
        U = wca_energy(ref, L)
        betaU_var = float((BETA ** 2) * U.var())      # irreducible energy cost

        anti = ToyOrderedProposal(N=N, L=L, w_u=0.5, seed=N)
        unif = ToyOrderedProposal(N=N, L=L, w_u=1.0, seed=N)

        sp_anti = ordering_spread(anti, ref, K=64, seed=N)
        sp_unif = ordering_spread(unif, ref, K=64, seed=N)  # must be ~0

        acc_anti = acceptance(anti, N, L, seed=N)
        acc_unif = acceptance(unif, N, L, seed=N)

        row = dict(N=N, L=L, betaU_var=betaU_var,
                   ord_std_anti=sp_anti["within_std_mean"],
                   ord_std_anti_pp=sp_anti["within_std_mean"] / N,
                   ord_std_unif=sp_unif["within_std_mean"],
                   ord_var_anti=sp_anti["within_var_mean"],
                   acc_anti=acc_anti, acc_unif=acc_unif)
        rows.append(row)
        print(f"N={N} L={L:.3f} | acc antipode={acc_anti:.4f} uniform={acc_unif:.4f} "
              f"| ord-std antipode={row['ord_std_anti']:.3f} "
              f"(per-particle {row['ord_std_anti_pp']:.3f}) uniform={row['ord_std_unif']:.2e} "
              f"| beta^2 Var(U)={betaU_var:.3f}")

    Ns = np.array([r["N"] for r in rows])
    acc_a = np.array([r["acc_anti"] for r in rows])
    acc_u = np.array([r["acc_unif"] for r in rows])
    ord_a = np.array([r["ord_std_anti"] for r in rows])
    ord_pp = np.array([r["ord_std_anti_pp"] for r in rows])

    # ---- figure 2: acceptance (log) + ordering-mismatch, vs N ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.semilogy(Ns, acc_a, "o-", color="crimson", label="antipode (mediocre)")
    ax.semilogy(Ns, acc_u, "s--", color="tab:blue",
                label="uniform (perm-invariant control)")
    ax.set_xlabel("N"); ax.set_ylabel("acceptance rate")
    ax.set_title("Ordered-space MH acceptance vs N")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(Ns, ord_a, "o-", color="crimson", label="antipode: std_perm(log q~)")
    ax.plot(Ns, [r["ord_std_unif"] for r in rows], "s--", color="tab:blue",
            label="uniform: 0 (by construction)")
    ax.set_xlabel("N"); ax.set_ylabel("ordering-mismatch std (nats)")
    ax.set_title("Ordering-mismatch spread vs N (total)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(Ns, ord_pp, "o-", color="darkgreen")
    ax.set_xlabel("N"); ax.set_ylabel("ordering std / N (nats/particle)")
    ax.set_title("Ordering-mismatch PER PARTICLE\n(flat=fixed penalty; rising=wall)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(ord_pp) * 1.3)

    fig.suptitle("Task 4: efficiency scaling of ordered-space independence-MH "
                 "(2D WCA, rho*=0.5, T*=1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    figpath = os.path.join(LOGDIR, "ordmh_efficiency.png")
    fig.savefig(figpath, dpi=130)
    print(f"\nFIGURE -> {figpath}")

    # quick scaling read-outs
    lr = np.polyfit(Ns, np.log(acc_a), 1)
    print(f"\nacceptance(antipode) ~ exp({lr[0]:.3f} * N)  "
          f"[per added particle x{np.exp(lr[0]):.3f}]")
    lr_u = np.polyfit(Ns, np.log(acc_u), 1)
    print(f"acceptance(uniform)  ~ exp({lr_u[0]:.3f} * N)  "
          f"[per added particle x{np.exp(lr_u[0]):.3f}]")
    slope_ord = np.polyfit(Ns, ord_a, 1)[0]
    print(f"ordering-std slope   = {slope_ord:.3f} nats per particle "
          f"(per-particle std {ord_pp.mean():.3f} +/- {ord_pp.std():.3f})")

    with open(os.path.join(LOGDIR, "ordmh_efficiency_data.pkl"), "wb") as f:
        pickle.dump(rows, f)


if __name__ == "__main__":
    main()
