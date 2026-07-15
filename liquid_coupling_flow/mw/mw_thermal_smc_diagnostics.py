"""Reference-calibrated energy and g(r) diagnostics for exact thermal SMC."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

from liquid_coupling_flow.mw.mw_reference import g_r


def _reference_subset(ref, n, seed):
    cfgs = torch.as_tensor(ref["cfgs"]).float()
    U = torch.as_tensor(ref["U"]).float()
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(cfgs), generator=gen)[:min(n, len(cfgs))]
    return cfgs[idx], U[idx]


def diagnose(smc_path, ref_path, plot_path, metrics_path, *, ref_configs=4096,
             nbins_energy=50, nbins_gr=120, seed=0):
    smc = torch.load(smc_path, map_location="cpu", weights_only=False)
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    if smc.get("endpoint_is_weighted", True):
        raise ValueError("diagnostics require the SMC endpoint resampled to uniform weights")

    x = torch.as_tensor(smc["x"]).float()
    U = torch.as_tensor(smc["U"]).float()
    N = int(smc["protocol"]["N"])
    L = float(smc["protocol"]["L"])
    ref_x, ref_U = _reference_subset(ref, ref_configs, seed)
    if x.shape[1:] != ref_x.shape[1:]:
        raise ValueError(f"endpoint/ref shape mismatch: {tuple(x.shape)} vs {tuple(ref_x.shape)}")

    u = (U / N).numpy()
    ref_u = (ref_U / N).numpy()
    # All histogram scaling comes from the reference distribution.  Generated
    # tails are counted and reported, but never allowed to stretch the axes.
    lo, hi = np.quantile(ref_u, [0.001, 0.999])
    edges = np.linspace(lo, hi, nbins_energy + 1)
    outside = float(np.mean((u < lo) | (u > hi)))

    r_ref, gr_ref = g_r(ref_x, L, nbins=nbins_gr)
    r_smc, gr_smc = g_r(x, L, nbins=nbins_gr)
    if not torch.allclose(r_ref, r_smc):
        raise RuntimeError("g(r) grids do not match")
    dg = gr_smc - gr_ref

    hist = smc["history"]
    relax = [h for h in hist if h["stage"] not in {"endpoint_resample", "endpoint"}]
    cost = np.asarray([h["cost_units"] for h in relax], dtype=float)
    means = np.asarray([h["U_per_N_mean"] for h in relax])
    q10 = np.asarray([h["U_per_N_q10"] for h in relax])
    q90 = np.asarray([h["U_per_N_q90"] for h in relax])
    ref_mean = float(ref_u.mean())
    ref_q10, ref_q90 = np.quantile(ref_u, [0.1, 0.9])
    ks_result = ks_2samp(u, ref_u)

    metrics = {
        "smc_path": str(smc_path), "reference_path": str(ref_path),
        "M": len(x), "N": N, "L": L, "reference_configs": len(ref_x),
        "energy_U_per_N": {
            "smc_mean": float(u.mean()), "smc_std": float(u.std()),
            "ref_mean": ref_mean, "ref_std": float(ref_u.std()),
            "mean_error": float(u.mean() - ref_mean),
            "ks": float(ks_result.statistic), "ks_pvalue": float(ks_result.pvalue),
            "wasserstein": float(wasserstein_distance(u, ref_u)),
            "reference_axis_q001_q999": (float(lo), float(hi)),
            "smc_fraction_outside_reference_axis": outside,
        },
        "g_r": {
            "r": r_ref, "smc": gr_smc, "reference": gr_ref,
            "max_abs_error": float(dg.abs().max()),
            "rmse": float(torch.sqrt(torch.mean(dg.square()))),
        },
        "history": hist,
        "energy_counter": smc["energy_counter"],
        "flow_accounting": {
            "initial_flow_density_stages": smc["initial_flow_density_stages"],
            "initial_flow_density_batches": smc["initial_flow_density_batches"],
            "per_rung_flow_density_evaluations": smc["per_rung_flow_density_evaluations"],
            "reverse_ode_calls": smc["reverse_ode_calls"],
        },
    }

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax[0].fill_between(cost, q10, q90, color="C0", alpha=.18, label="SMC 10–90%")
    ax[0].plot(cost, means, "o-", color="C0", lw=1.8, ms=4, label="SMC mean")
    ax[0].axhspan(ref_q10, ref_q90, color="0.5", alpha=.14, label="data 10–90%")
    ax[0].axhline(ref_mean, color="k", ls="--", lw=1.4, label="data mean")
    ax[0].set(xlabel="cost-weighted mW energy evaluations", ylabel=r"$U/N$",
              title="Energy relaxation")
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].hist(ref_u, bins=edges, density=True, histtype="stepfilled", alpha=.28,
               color="0.35", label=f"data (n={len(ref_u)})")
    ax[1].hist(u, bins=edges, density=True, histtype="step", lw=2.0,
               color="C0", label=f"SMC endpoint (M={len(u)})")
    ax[1].set_xlim(lo, hi)
    ax[1].set(xlabel=r"$U/N$", ylabel="density", title="Endpoint energy distribution")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].text(.03, .97, f"KS={metrics['energy_U_per_N']['ks']:.3f}\n"
                f"W1={metrics['energy_U_per_N']['wasserstein']:.3f}\n"
                f"outside={outside:.1%}", transform=ax[1].transAxes, va="top", fontsize=8)

    ax[2].plot(r_ref, gr_ref, color="k", lw=1.8, label="data")
    ax[2].plot(r_smc, gr_smc, color="C0", lw=1.6, label="SMC endpoint")
    ax[2].set(xlabel=r"$r/\sigma$", ylabel=r"$g(r)$", title="Radial distribution")
    ax[2].legend(frameon=False, fontsize=8)
    ax[2].text(.03, .05, f"RMSE={metrics['g_r']['rmse']:.3f}\n"
                f"max|Δ|={metrics['g_r']['max_abs_error']:.3f}",
                transform=ax[2].transAxes, ha="left", va="bottom", fontsize=8)

    fig.suptitle("N=64 exact eRSI→thermal SMC vs equilibrium mW data", y=1.01)
    fig.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    torch.save(metrics, metrics_path)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smc", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--plot", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--ref-configs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    m = diagnose(a.smc, a.reference, a.plot, a.metrics,
                 ref_configs=a.ref_configs, seed=a.seed)
    print({"energy_U_per_N": m["energy_U_per_N"],
           "g_r": {k: v for k, v in m["g_r"].items() if k not in {"r", "smc", "reference"}},
           "flow_accounting": m["flow_accounting"]}, flush=True)


if __name__ == "__main__":
    main()
