"""Size-transfer test (2D, CONSTANT density rho = N/L^2 = 0.64).

Take the AR flow trained ONLY at N=16 (L=5) and, WITHOUT retraining, rebuild it at
larger N with the matching box (same density) -- the conditioner has no N-dependent
shapes, so the trained weights load directly. Then flow + single-particle SMC, and
compare reweighted energy / g(r) / ESS to fresh MCMC at each size. This is the
headline claim: locality -> size-transfer, exactness (SMC) -> correct at scale.

Run:  python -m liquid_coupling_flow.size_transfer
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from liquid_coupling_flow.particles_ar import ARParticleFlow
from liquid_coupling_flow.energy import lj_energy
from liquid_coupling_flow.particles import min_image
from liquid_coupling_flow.smc import anneal_smc, SingleParticleMetropolis, ess_fraction
from liquid_coupling_flow import mcmc
from liquid_coupling_flow.plot_ar_diagnostics import compute_gr
from liquid_coupling_flow.plot_smc_reweighted import gr_weighted

ART = os.path.join(os.path.dirname(__file__), "artifacts")
CKPT = os.path.join(ART, "ar_flow_ckpt.pt")

# constant density 0.64: (N, L) with N/L^2 = 0.64
SIZES = [(16, 5.0), (36, 7.5), (64, 10.0)]
KT, CUTOFF = 1.0, 2.4


def overlap_frac(x, L, N, thresh=0.8):
    _, r = min_image(x, L)
    r = r + torch.eye(N, device=x.device)[None] * 1e3
    return float((r.min(dim=2).values.reshape(-1) < thresh).float().mean())


def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    torch.manual_seed(0)
    ck = torch.load(CKPT, map_location=device)
    cfg = ck["config"]
    rows = []
    fig, axes = plt.subplots(1, len(SIZES), figsize=(6 * len(SIZES), 4.6), squeeze=False)

    for col, (N, L) in enumerate(SIZES):
        rho = N / L ** 2
        print(f"\n=== N={N} L={L} (rho={rho:.3f}) ===", flush=True)
        data = mcmc.get_data(N, L, KT, CUTOFF, d=2, device=device,
                             n_chains=2048, n_equil=200, n_collect=120, every=30)
        U_mcmc = (lj_energy(data, L, cutoff=CUTOFF, shift=True) / N)
        print(f"MCMC {data.shape[0]} configs  <U>/N {U_mcmc.mean():.3f}", flush=True)

        flow = ARParticleFlow(N=N, L=L, num_bins=cfg["num_bins"], hidden=cfg["hidden"],
                              cutoff=CUTOFF).to(device)
        flow.load_state_dict(ck["state_dict"])  # trained at N=16, loads at any N
        flow.eval()

        M = min(2500, data.shape[0])
        with torch.no_grad():
            x0, _ = flow.sample(M, device=device)
        raw_ov = overlap_frac(x0, L, N)

        # uniform baseline plain-IS ESS (proposal = uniform on torus)
        xu = torch.rand(M, N, 2, device=device) * L
        with torch.no_grad():
            logw_u = -lj_energy(xu, L, cutoff=CUTOFF, shift=True) / KT - (-2 * N * math.log(L))
        ess_unif = ess_fraction(logw_u)

        def logq_fn(x):
            return flow.log_prob(x)

        def log_target_fn(x):
            return -lj_energy(x, L, cutoff=CUTOFF, shift=True) / KT

        kernel = SingleParticleMetropolis(step=0.15, n_sweeps=1, L=L)
        res = anneal_smc(x0, logq_fn, log_target_fn, kernel, n_bridge=24,
                         resample_thresh=0.5, schedule="linear", verbose=False)
        x, w = res["x"], res["weights"]
        smc_ov = overlap_frac(x, L, N)
        Us = (lj_energy(x, L, cutoff=CUTOFF, shift=True) / N).cpu().numpy()
        wn = w.cpu().numpy()
        mu_rw = float((wn * Us).sum())
        se_rw = float(np.sqrt((wn ** 2 * (Us - mu_rw) ** 2).sum()))
        print(f"raw-flow overlaps {raw_ov:.3f} | plain-IS ESS flow {100*res['plain_is_ess']:.2f}% "
              f"vs uniform {100*ess_unif:.2f}% | SMC ESS {100*res['ess']:.1f}% | "
              f"SMC overlaps {smc_ov:.3f} | <U>/N rw {mu_rw:.3f}+/-{se_rw:.3f} (MCMC {U_mcmc.mean():.3f})",
              flush=True)
        rows.append((N, rho, raw_ov, res["plain_is_ess"], ess_unif, res["ess"], smc_ov,
                     mu_rw, se_rw, float(U_mcmc.mean())))

        rmax = L / 2.0
        rc_b, g_b = compute_gr(data[:4000], L, rmax, 90, d=2)
        rc_w, g_w = gr_weighted(x, w, L, rmax, 90, d=2)
        ax = axes[0][col]
        ax.plot(rc_b, g_b, color="C0", lw=2.4, label="Boltzmann (MCMC)")
        ax.plot(rc_w, g_w, color="C2", lw=1.6, label="flow+SMC (reweighted)")
        ax.axvline(2 ** (1 / 6), color="grey", ls="--", lw=1)
        ax.axhline(1.0, color="k", lw=0.5, ls=":")
        ax.set_xlim(0, rmax); ax.set_xlabel(r"$r/\sigma$"); ax.set_ylabel("g(r)")
        tag = " (TRAIN SIZE)" if N == 16 else ""
        ax.set_title(f"N={N}, L={L}{tag}\nSMC ESS {100*res['ess']:.0f}%, "
                     f"<U>/N {mu_rw:.3f} (MCMC {U_mcmc.mean():.3f})")
        ax.legend(fontsize=8)

    fig.suptitle("Size transfer at constant density (rho=0.64): AR flow trained at N=16 only, "
                 "+ single-particle SMC", fontsize=12)
    fig.tight_layout()
    out = os.path.join(ART, "size_transfer_gr.png")
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}", flush=True)

    print("\n N   rho   raw_ov  IS_flow  IS_unif  SMC_ESS  SMC_ov  <U>/N_rw      MCMC", flush=True)
    for (N, rho, rov, isf, isu, se, sov, mu, semu, umc) in rows:
        print(f"{N:3d} {rho:.3f}  {rov:.3f}  {100*isf:6.2f}%  {100*isu:6.2f}%  "
              f"{100*se:5.1f}%  {sov:.3f}  {mu:6.3f}+/-{semu:.3f}  {umc:6.3f}", flush=True)
    torch.save(rows, os.path.join(ART, "size_transfer_rows.pt"))


if __name__ == "__main__":
    main()
