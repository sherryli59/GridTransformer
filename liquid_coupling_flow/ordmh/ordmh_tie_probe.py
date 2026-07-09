"""Task 5 --- local-canonical-order tie probe (independent of the sampler).

Could a *locally-defined* canonical order reduce the ordering penalty from Task
4?  The idea: if each particle carries an intensive local scalar that is almost
always distinct, we can order particles by it, making the ordered density
transferable and near-tractable (small tie-sums).  If instead a homogeneous
liquid makes many particles indistinguishable in that scalar (large tie-classes),
we are stuck with the full ordered-space penalty.

For Boltzmann reference configs we compute two candidate intensive scalars per
particle --- local potential energy and coordination number within a cutoff ---
order particles by each, and measure the tie-class-size distribution:
  * fraction of particles in non-trivial tie-classes (class size >= 2),
  * max tie-class size,
vs N.  A tie-class = a maximal run of sorted particles whose consecutive scalar
gaps are all below a small tolerance (0 for the integer coordination number).
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

from ordmh_energy import wca_pair_energy, R_C
from ordmh_reference import displacement_mc

BETA = 1.0
RHO = 0.5
NS = [3, 4, 5, 6, 7, 8]
R_COORD = 1.5           # coordination cutoff (~1.3x contact)
LOGDIR = os.path.abspath(os.path.join(HERE, "..", "..", "reports", "logs-2026-07-09"))
os.makedirs(LOGDIR, exist_ok=True)


def box_for(N):
    return float(np.sqrt(N / RHO))


def _pair_dist_matrix(config, L):
    diff = config[:, None, :] - config[None, :, :]
    diff -= L * np.round(diff / L)
    r = np.sqrt(np.einsum("ijd,ijd->ij", diff, diff))
    np.fill_diagonal(r, np.inf)
    return r


def local_scalars(config, L):
    """Per-particle local potential energy u_i and coordination number c_i."""
    r = _pair_dist_matrix(config, L)
    u = np.where(r < R_C, wca_pair_energy(np.minimum(r, R_C - 1e-9)), 0.0).sum(axis=1)
    c = (r < R_COORD).sum(axis=1).astype(float)
    return u, c


def tie_classes(values, tol):
    """Given per-particle scalar values, sort and group into tie-classes where
    consecutive gaps < tol.  Returns list of class sizes."""
    v = np.sort(values)
    sizes = []
    start = 0
    for i in range(1, len(v)):
        if v[i] - v[i - 1] >= tol:
            sizes.append(i - start)
            start = i
    sizes.append(len(v) - start)
    return sizes


def probe_scalar(all_values, tol):
    """Aggregate tie stats over many configs' per-particle scalar arrays."""
    frac_nontrivial = []
    max_class = []
    for vals in all_values:
        sizes = tie_classes(vals, tol)
        n = len(vals)
        in_nontrivial = sum(s for s in sizes if s >= 2)
        frac_nontrivial.append(in_nontrivial / n)
        max_class.append(max(sizes))
    return float(np.mean(frac_nontrivial)), float(np.mean(max_class))


def _self_check():
    # clearly-distinct scalars -> all singletons; identical -> one big class
    assert tie_classes(np.array([0.0, 1.0, 2.0, 3.0]), tol=0.1) == [1, 1, 1, 1]
    assert tie_classes(np.array([1.0, 1.0, 1.0]), tol=0.1) == [3]
    assert tie_classes(np.array([0.0, 0.05, 1.0]), tol=0.1) == [2, 1]


def main():
    _self_check()
    rows = []
    for N in NS:
        L = box_for(N)
        mc = displacement_mc(N=N, L=L, beta=BETA, n_chains=500, n_equil=800,
                             n_collect=2, thin=1, step=0.22, seed=2000 + N)
        configs = mc["configs"][:800]
        u_vals, c_vals = [], []
        for cfg in configs:
            u, c = local_scalars(cfg, L)
            u_vals.append(u); c_vals.append(c)
        # tolerance for the continuous local energy: 5% of its typical spread
        u_std = np.mean([v.std() for v in u_vals if v.std() > 0])
        tol_u = 0.05 * u_std
        fu, mu = probe_scalar(u_vals, tol_u)
        # coordination number is integer -> exact ties (tol just below 1)
        fc, mc_ = probe_scalar(c_vals, tol=0.5)
        row = dict(N=N, L=L, tol_u=tol_u,
                   energy_frac_tied=fu, energy_max_class=mu,
                   coord_frac_tied=fc, coord_max_class=mc_)
        rows.append(row)
        print(f"N={N} | local-energy: frac tied={fu:.3f} max class={mu:.2f} "
              f"(tol={tol_u:.3f}) | coordination#: frac tied={fc:.3f} "
              f"max class={mc_:.2f}")

    Ns = np.array([r["N"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(Ns, [r["energy_frac_tied"] for r in rows], "o-", color="crimson",
            label="local potential energy (5% tol)")
    ax.plot(Ns, [r["coord_frac_tied"] for r in rows], "s-", color="tab:blue",
            label="coordination number (exact)")
    ax.set_xlabel("N"); ax.set_ylabel("fraction of particles in tie-classes")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ambiguous-order fraction vs N")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(Ns, [r["energy_max_class"] / r["N"] for r in rows], "o-",
            color="crimson", label="local energy")
    ax.plot(Ns, [r["coord_max_class"] / r["N"] for r in rows], "s-",
            color="tab:blue", label="coordination number")
    ax.set_xlabel("N"); ax.set_ylabel("max tie-class size / N")
    ax.set_ylim(0, 1.05)
    ax.set_title("Largest ambiguous group (fraction of system)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Task 5: local-canonical-order tie probe (2D WCA, rho*=0.5, T*=1)\n"
                 "high tie fraction => homogeneous liquid => no cheap local order "
                 "=> stuck with full ordered-space penalty", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    figpath = os.path.join(LOGDIR, "ordmh_tie_probe.png")
    fig.savefig(figpath, dpi=130)
    print(f"\nFIGURE -> {figpath}")

    with open(os.path.join(LOGDIR, "ordmh_tie_probe_data.pkl"), "wb") as f:
        pickle.dump(rows, f)


if __name__ == "__main__":
    main()
