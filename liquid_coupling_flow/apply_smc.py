"""Apply the annealed-SMC corrector to the AR flow and re-plot energy / g(r) / ESS.

The raw AR flow has correct medium-range structure but a soft hard core (residual
sub-1.0 sigma overlaps) that wrecks energy and ESS. SMC bridges the flow proposal
q -> Boltzmann target along (1-beta) log q + beta (-U/kT), resampling + relaxing the
walkers with a periodic random-walk Metropolis kernel, using the flow's EXACT log q
evaluated on the moved configs.

Run:  python -m liquid_coupling_flow.apply_smc
"""

from __future__ import annotations

import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from liquid_coupling_flow.particles_ar import ARParticleFlow
from liquid_coupling_flow.energy import lj_energy
from liquid_coupling_flow.particles import min_image
from liquid_coupling_flow.smc import (anneal_smc, RWMetropolis,
                                      SingleParticleMetropolis, systematic_resample)
from liquid_coupling_flow.demo_particles_mle import get_data
from liquid_coupling_flow.plot_ar_diagnostics import compute_gr, energy_pdf

ART = os.path.join(os.path.dirname(__file__), "artifacts")
CKPT = os.path.join(ART, "ar_flow_ckpt.pt")


def overlap_frac(x, L, N, thresh=0.8):
    _, r = min_image(x, L)
    r = r + torch.eye(N, device=x.device)[None] * 1e3
    return float((r.min(dim=2).values.reshape(-1) < thresh).float().mean())


def main(device="cuda" if torch.cuda.is_available() else "cpu",
         M=4000, n_bridge=40, n_steps=1, step=0.15, kernel_kind="single"):
    ck = torch.load(CKPT, map_location=device)
    cfg = ck["config"]
    N, L, kT, cutoff = cfg["N"], cfg["L"], cfg["kT"], cfg["cutoff"]

    flow = ARParticleFlow(N=N, L=L, num_bins=cfg["num_bins"], hidden=cfg["hidden"],
                          cutoff=cutoff).to(device)
    flow.load_state_dict(ck["state_dict"])
    flow.eval()

    x0 = ck["samples"][:M].to(device)
    data = get_data(N, L, kT, cutoff, device).to(device)

    def logq_fn(x):
        return flow.log_prob(x)

    def log_target_fn(x):
        return -lj_energy(x, L, cutoff=cutoff, shift=True) / kT

    t = time.time()
    with torch.no_grad():
        _ = logq_fn(x0)
    per_beta = (N * n_steps + 1) if kernel_kind == "single" else (n_steps + 1)
    print(f"one logq eval on {M}: {time.time()-t:.2f}s  kernel={kernel_kind}  "
          f"(~{n_bridge*per_beta} evals total)", flush=True)

    if kernel_kind == "single":
        kernel = SingleParticleMetropolis(step=step, n_sweeps=n_steps, L=L)
    else:
        kernel = RWMetropolis(step=step, n_steps=n_steps, L=L)
    t = time.time()
    res = anneal_smc(x0, logq_fn, log_target_fn, kernel, n_bridge=n_bridge,
                     resample_thresh=0.5, schedule="linear", verbose=True)
    print(f"SMC done in {time.time()-t:.0f}s", flush=True)

    # equally-weighted ensemble for structure plots
    idx = systematic_resample(res["weights"])
    x_smc = res["x"][idx]

    with torch.no_grad():
        U_data = (lj_energy(data, L, cutoff=cutoff, shift=True) / N)
        U_flow = (lj_energy(x0, L, cutoff=cutoff, shift=True) / N)
        U_smc = (lj_energy(x_smc, L, cutoff=cutoff, shift=True) / N)
    ov_flow, ov_smc = overlap_frac(x0, L, N), overlap_frac(x_smc, L, N)
    print(f"plain-IS ESS (flow alone) {100*res['plain_is_ess']:.2f}%  ->  "
          f"SMC ESS {100*res['ess']:.2f}%", flush=True)
    print(f"overlaps  flow {ov_flow:.3f} -> SMC {ov_smc:.3f}", flush=True)
    print(f"<U>/N  data {U_data.mean():.3f} | flow(median) {U_flow.median():.3f} | "
          f"SMC(median) {U_smc.median():.3f}", flush=True)
    print(f"mean acceptance {np.mean(res['acc_history']):.2f}", flush=True)

    # ---- figure: energy | g(r) | ESS-vs-beta ----
    Um, Uf, Us = U_data.cpu().numpy(), U_flow.cpu().numpy(), U_smc.cpu().numpy()
    lo, hi, nb = math.floor(Um.min() - 0.5), math.ceil(Um.max() + 2.0), 70
    cm, pm, _ = energy_pdf(Um, lo, hi, nb)
    cf, pf, of_ = energy_pdf(Uf, lo, hi, nb)
    cs, ps, os_ = energy_pdf(Us, lo, hi, nb)

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))

    ax[0].fill_between(cm, pm, step="mid", alpha=0.45, color="C0", label="MCMC (target)")
    ax[0].step(cf, pf, where="mid", color="C3", lw=1.6, label=f"flow raw (ovf {100*of_:.0f}%)")
    ax[0].step(cs, ps, where="mid", color="C2", lw=2.0, label=f"flow+SMC (ovf {100*os_:.0f}%)")
    ax[0].axvline(Um.mean(), color="C0", ls=":", lw=1)
    ax[0].set_xlabel("U / N"); ax[0].set_ylabel("pdf"); ax[0].set_xlim(lo, hi)
    ax[0].set_title("Energy distribution (capped)")
    ax[0].legend(fontsize=9)

    rmax = L / 2.0
    rc_m, g_m = compute_gr(data[:4000], L, rmax, 90, d=2)
    rc_f, g_f = compute_gr(x0, L, rmax, 90, d=2)
    rc_s, g_s = compute_gr(x_smc, L, rmax, 90, d=2)
    ax[1].plot(rc_m, g_m, color="C0", lw=2, label="MCMC")
    ax[1].plot(rc_f, g_f, color="C3", lw=1.6, label="flow raw")
    ax[1].plot(rc_s, g_s, color="C2", lw=2, label="flow+SMC")
    ax[1].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[1].axvline(2 ** (1 / 6), color="grey", ls="--", lw=1)
    ax[1].set_xlabel(r"$r/\sigma$"); ax[1].set_ylabel("g(r)"); ax[1].set_xlim(0, rmax)
    ax[1].set_title("Radial distribution g(r)")
    ax[1].legend(fontsize=9)

    betas = np.linspace(0, 1, n_bridge + 1)[1:]
    ax[2].plot(betas, 100 * np.array(res["ess_history"]), color="C2", lw=2, label="SMC ESS(beta)")
    ax[2].axhline(100 * res["plain_is_ess"], color="C3", ls="--", lw=1.5,
                  label=f"flow plain-IS {100*res['plain_is_ess']:.2f}%")
    ax[2].axhline(100 * res["ess"], color="C0", ls=":", lw=1.5,
                  label=f"final SMC {100*res['ess']:.1f}%")
    ax[2].set_xlabel(r"$\beta$ (anneal q$\to\pi$)"); ax[2].set_ylabel("ESS %")
    ax[2].set_title(f"ESS along anneal (acc {np.mean(res['acc_history']):.2f})")
    ax[2].legend(fontsize=9)

    fig.suptitle(f"AR flow + SMC ({kernel_kind} moves) — 2D LJ N={N} L={L} kT={kT}  |  "
                 f"ESS {100*res['plain_is_ess']:.2f}% -> {100*res['ess']:.1f}%, "
                 f"overlaps {ov_flow:.3f} -> {ov_smc:.3f}, acc {np.mean(res['acc_history']):.2f}",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(ART, f"ar_smc_{kernel_kind}.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}", flush=True)
    torch.save({"x_smc": x_smc.cpu(), "weights": res["weights"].cpu(),
                "ess": res["ess"], "plain_is_ess": res["plain_is_ess"],
                "acc": float(np.mean(res["acc_history"]))},
               os.path.join(ART, f"smc_result_{kernel_kind}.pt"))


if __name__ == "__main__":
    main()
