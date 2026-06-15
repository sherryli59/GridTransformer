"""Workstream B (light): firm up the 3D ESS by giving SMC a finer anneal (more bridge
steps) on the ALREADY-TRAINED 3D checkpoint -- no retraining. Reports energy/g(r)/ESS
and writes a fresh figure for comparison against the n_bridge=30 baseline (ESS 53%).

Run:  python -m liquid_coupling_flow.firm_3d
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from liquid_coupling_flow.particles_ar_nd import ParticleARFlowND
from liquid_coupling_flow.energy import lj_energy
from liquid_coupling_flow.particles import min_image
from liquid_coupling_flow.smc import anneal_smc, SingleParticleMetropolis
from liquid_coupling_flow import mcmc
from liquid_coupling_flow.plot_ar_diagnostics import compute_gr
from liquid_coupling_flow.plot_smc_reweighted import gr_weighted

ART = os.path.join(os.path.dirname(__file__), "artifacts")
CKPT = os.path.join(ART, "ar_flow_3d_ckpt.pt")


def overlap_frac(x, L, N, thresh=0.8):
    _, r = min_image(x, L)
    r = r + torch.eye(N, device=x.device)[None] * 1e3
    return float((r.min(dim=2).values.reshape(-1) < thresh).float().mean())


def main(device="cuda" if torch.cuda.is_available() else "cpu",
         M=2500, n_bridge=50, n_sweeps=1, step=0.12):
    torch.manual_seed(0)
    ck = torch.load(CKPT, map_location=device)
    cfg = ck["config"]
    N, L, kT, cutoff, D = cfg["N"], cfg["L"], cfg["kT"], cfg["cutoff"], cfg["d"]
    flow = ParticleARFlowND(N=N, L=L, d=D, num_bins=cfg["num_bins"], hidden=cfg["hidden"],
                            cutoff=cutoff, ctx0_reduce=cfg.get("ctx0_reduce", "sum")).to(device)
    flow.load_state_dict(ck["state_dict"]); flow.eval()
    data = mcmc.get_data(N, L, kT, cutoff, d=D, device=device)
    print(f"3D firm-up: N={N} L={L:.3f} cutoff={cutoff} | n_bridge={n_bridge} n_sweeps={n_sweeps}",
          flush=True)

    with torch.no_grad():
        x0, _ = flow.sample(M, device=device)

    def logq_fn(x):
        return flow.log_prob(x)

    def log_target_fn(x):
        return -lj_energy(x, L, cutoff=cutoff, shift=True) / kT

    kernel = SingleParticleMetropolis(step=step, n_sweeps=n_sweeps, L=L)
    res = anneal_smc(x0, logq_fn, log_target_fn, kernel, n_bridge=n_bridge,
                     resample_thresh=0.5, schedule="linear", verbose=True)
    x, w = res["x"], res["weights"]
    Us = (lj_energy(x, L, cutoff=cutoff, shift=True) / N).cpu().numpy()
    Um = (lj_energy(data, L, cutoff=cutoff, shift=True) / N).cpu().numpy()
    wn = w.cpu().numpy()
    mu_rw = float((wn * Us).sum()); se_rw = float(np.sqrt((wn ** 2 * (Us - mu_rw) ** 2).sum()))
    print(f"SMC ESS {100*res['ess']:.1f}% (was 53% at n_bridge=30) | overlaps {overlap_frac(x,L,N):.3f} | "
          f"<U>/N rw {mu_rw:.3f}+/-{se_rw:.3f} (MCMC {Um.mean():.3f})", flush=True)

    lo, hi, nb = math.floor(Um.min() - 0.5), math.ceil(Um.max() + 1.0), 60
    cw, ew = np.histogram(Us, bins=nb, range=(lo, hi), weights=wn)
    pdf_rw = cw / (ew[1] - ew[0]); centers = 0.5 * (ew[:-1] + ew[1:])
    cm, em = np.histogram(Um, bins=nb, range=(lo, hi))
    pdf_bz = cm / (len(Um) * (em[1] - em[0]))
    rmax = L / 2.0
    rc_b, g_b = compute_gr(data[:4000], L, rmax, 70, d=D)
    rc_w, g_w = gr_weighted(x, w, L, rmax, 70, d=D)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].fill_between(0.5 * (em[:-1] + em[1:]), pdf_bz, step="mid", alpha=0.45, color="C0",
                       label="Boltzmann (MCMC)")
    ax[0].step(centers, pdf_rw, where="mid", color="C2", lw=2, label="flow+SMC reweighted")
    ax[0].axvline(Um.mean(), color="C0", ls=":", lw=1)
    ax[0].set_xlabel("U / N"); ax[0].set_ylabel("pdf"); ax[0].set_xlim(lo, hi)
    ax[0].set_title(f"Energy: <U>/N rw {mu_rw:.3f}+/-{se_rw:.3f} (MCMC {Um.mean():.3f})")
    ax[0].legend(fontsize=9)
    ax[1].plot(rc_b, g_b, color="C0", lw=2.4, label="Boltzmann (MCMC)")
    ax[1].plot(rc_w, g_w, color="C2", lw=1.6, label="flow+SMC reweighted")
    ax[1].axvline(2 ** (1 / 6), color="grey", ls="--", lw=1)
    ax[1].axhline(1.0, color="k", lw=0.5, ls=":")
    ax[1].set_xlabel(r"$r/\sigma$"); ax[1].set_ylabel("g(r)"); ax[1].set_xlim(0, rmax)
    ax[1].set_title(f"g(r) 3D firmed  (SMC ESS {100*res['ess']:.0f}%, n_bridge={n_bridge})")
    ax[1].legend(fontsize=9)
    fig.suptitle(f"3D LJ N={N} firmed SMC (n_bridge={n_bridge}, n_sweeps={n_sweeps})", fontsize=12)
    fig.tight_layout()
    out = os.path.join(ART, "lj3d_firmed.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
