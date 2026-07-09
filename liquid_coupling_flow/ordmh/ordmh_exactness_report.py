"""Gate E report + figure (deliverable 1).

Runs the ordered-space independence-MH sampler with the mediocre toy proposal at
N=3 and N=5, and overlays its physical observables (P(U), g(r)) on the
independent reference (grid quadrature for N=3; displacement MC for both).  Also
prints the decisive "MH is load-bearing" numbers: raw proposal E_q[U] vs the
MH-corrected <U> vs the ground truth.
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
from ordmh_reference import (grid_quadrature, displacement_mc, radial_distribution)
from ordmh_toy_proposal import ToyOrderedProposal
from ordmh_sampler import OrderedMH

BETA = 1.0
LOGDIR = os.path.abspath(os.path.join(HERE, "..", "..", "reports", "logs-2026-07-09"))
os.makedirs(LOGDIR, exist_ok=True)


def shared_hist(a, b, nbins=45):
    lo = min(a.min(), b.min())
    hi = max(np.quantile(a, 0.999), np.quantile(b, 0.999))
    edges = np.linspace(lo, hi, nbins + 1)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    cent = 0.5 * (edges[:-1] + edges[1:])
    # normalised counts for a total-variation distance
    ca, _ = np.histogram(a, bins=edges); cb, _ = np.histogram(b, bins=edges)
    ca = ca / ca.sum(); cb = cb / cb.sum()
    l1 = 0.5 * np.abs(ca - cb).sum()
    ks = np.abs(np.cumsum(ca) - np.cumsum(cb)).max()
    return cent, pa, pb, l1, ks


def run_size(N, L, seed):
    print(f"\n=== N={N}  L={L:.4f}  rho*={N/(L*L):.3f}  T*=1 ===")
    # ---- independent reference ----
    mc = displacement_mc(N=N, L=L, beta=BETA, n_chains=768, n_equil=1000,
                         n_collect=1500, thin=3, step=0.22, seed=seed)
    U_ref = wca_energy(mc["configs"], L)
    grid = grid_quadrature(N=N, L=L, beta=BETA, n_grid=64) if N in (2, 3) else None

    # ---- raw proposal (NO MH): E_q[U] ----
    prop0 = ToyOrderedProposal(N=N, L=L, seed=seed)
    xr, _ = prop0.sample_ordered(300000)
    Ur = wca_energy(xr, L)
    meanU_raw = float(Ur[np.isfinite(Ur)].mean())

    # ---- ordered-space independence-MH ----
    prop = ToyOrderedProposal(N=N, L=L, seed=seed)
    smp = OrderedMH(N=N, L=L, beta=BETA, proposal=prop)
    out = smp.run(n_chains=3000, n_steps=3000, n_equil=800, thin=2, seed=seed)
    U_smp = wca_energy(out["configs"], L)

    gt = grid["mean_U"] if grid else mc["mean_U"]
    gt_err = grid["mean_U_err"] if grid else mc["mean_U_err"]
    gt_name = "grid quad (bedrock)" if grid else "displacement MC"

    print(f"  ground truth <U>       = {gt:.5f} +/- {gt_err:.5f}   [{gt_name}]")
    print(f"  displacement MC <U>    = {mc['mean_U']:.5f} +/- {mc['mean_U_err']:.5f}")
    print(f"  RAW proposal E_q[U]    = {meanU_raw:.5f}   (no MH -> WRONG)")
    print(f"  ordered-MH <U>         = {out['mean_U']:.5f} +/- {out['mean_U_err']:.5f}"
          f"   (acc {out['acc_rate']:.3f})")
    diff = abs(out["mean_U"] - gt)
    tol = 5 * out["mean_U_err"] + gt_err
    print(f"  |ordered-MH - GT|      = {diff:.5f}   tol {tol:.5f}   -> "
          f"{'PASS' if diff < tol else 'FAIL'}")
    print(f"  raw-vs-GT gap          = {abs(meanU_raw - gt):.5f}  "
          f"({abs(meanU_raw - gt)/gt_err:.0f} sigma) -> MH is load-bearing")

    cent_u, pu_ref, pu_smp, l1, ks = shared_hist(U_ref, U_smp)
    print(f"  P(U): total-variation  = {l1:.4f}   KS = {ks:.4f}")
    cr, gref = radial_distribution(mc["configs"], L, nbins=45)
    cs, gsmp = radial_distribution(out["configs"], L, nbins=45)
    mask = cr > 0.85
    grmax = float(np.max(np.abs(gref[mask] - gsmp[mask])))
    print(f"  g(r): max |diff|       = {grmax:.4f}")

    return {
        "N": N, "L": L, "gt": gt, "gt_err": gt_err, "gt_name": gt_name,
        "mc_meanU": mc["mean_U"], "raw_meanU": meanU_raw,
        "mh_meanU": out["mean_U"], "mh_err": out["mean_U_err"],
        "acc": out["acc_rate"], "l1": l1, "ks": ks, "grmax": grmax,
        "pu": (cent_u, pu_ref, pu_smp),
        "gr": (cr, gref, cs, gsmp),
        "grid_pu": grid["PU"] if grid else None,
    }


def make_figure(res, figpath):
    """P(U) (top) and g(r) (bottom), reference vs ordered-MH.  P(U) x-axis is
    clipped to the sampled-mass window so the (near-perfect) overlay is visible;
    the grid-quad bedrock curve's overlap tail would otherwise stretch it."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, r in enumerate(res):
        cu, pu_ref, pu_smp = r["pu"]
        ax = axes[0, col]
        ax.plot(cu, pu_ref, color="k", lw=2.2, label="reference MC", alpha=0.8)
        ax.plot(cu, pu_smp, color="crimson", lw=1.4, ls="--",
                marker="o", ms=3, label="ordered-space MH")
        if r["grid_pu"] is not None:
            gc, gp = r["grid_pu"]
            gp_dens = gp / (gp.sum() * (gc[1] - gc[0]))
            ax.plot(gc, gp_dens, color="tab:blue", lw=1.0, alpha=0.7,
                    label="grid quad (bedrock)")
        ax.set_xlim(cu.min(), cu.max())        # bound to MC/MH mass window
        ax.set_title(f"N={r['N']}: P(U)   [TV={r['l1']:.3f}, KS={r['ks']:.3f}]")
        ax.set_xlabel("U"); ax.set_ylabel("P(U)"); ax.legend(fontsize=8)

        cr, gref, cs, gsmp = r["gr"]
        ax = axes[1, col]
        ax.plot(cr, gref, color="k", lw=2.2, label="reference MC", alpha=0.8)
        ax.plot(cs, gsmp, color="crimson", lw=1.4, ls="--", marker="o", ms=3,
                label="ordered-space MH")
        ax.axhline(1.0, color="gray", lw=0.6, ls=":")
        ax.set_title(f"N={r['N']}: g(r)   [max|diff|={r['grmax']:.3f}]")
        ax.set_xlabel("r"); ax.set_ylabel("g(r)"); ax.legend(fontsize=8)
    fig.suptitle("Gate E: ordered-space independence-MH (mediocre proposal) vs "
                 "independent reference\n2D WCA liquid, rho*=0.5, T*=1",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figpath, dpi=130)
    print(f"\nFIGURE -> {figpath}")


def main():
    figpath = os.path.join(LOGDIR, "ordmh_exactness.png")
    datapath = os.path.join(LOGDIR, "ordmh_exactness_data.pkl")
    if "--replot" in sys.argv and os.path.exists(datapath):
        with open(datapath, "rb") as f:
            res = pickle.load(f)
        make_figure(res, figpath)
        return
    L3 = float(np.sqrt(6.0))    # N=3
    L5 = float(np.sqrt(10.0))   # N=5, rho*=0.5
    res = [run_size(3, L3, seed=101), run_size(5, L5, seed=202)]
    make_figure(res, figpath)
    with open(datapath, "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main()
