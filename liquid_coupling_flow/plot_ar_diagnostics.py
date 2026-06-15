"""Energy distribution (capped) + g(r): AR particle flow vs MCMC ground truth.

Loads the checkpoint written by train_ar_ckpt.py (flow samples + config), loads the
same cached 2D LJ MCMC data the flow was trained on, and makes two physics panels:

  (left)  per-particle energy U/N histogram, MCMC vs flow. The flow has a heavy
          r^-12 tail from residual overlaps, so we CAP the x-axis and annotate the
          overflow mass instead of letting a few configs blow the plot up. We also
          show the importance-reweighted flow energy (exact log q -> reweight to the
          Boltzmann target) -- the payoff of an exact-likelihood model.
  (right) radial distribution g(r) with correct 2D shell normalization. The overlap
          defect shows up as spurious density below the MCMC onset (~0.9 sigma).

Run:  python -m liquid_coupling_flow.plot_ar_diagnostics
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from liquid_coupling_flow.particles import min_image
from liquid_coupling_flow.energy import lj_energy
from liquid_coupling_flow.demo_particles_mle import get_data

ART = os.path.join(os.path.dirname(__file__), "artifacts")
CKPT = os.path.join(ART, "ar_flow_ckpt.pt")


def compute_gr(x, L, rmax, nbins, d=2):
    """Radial distribution g(r). x [B,N,d]. Plateau normalized to 1 (uses rho=(N-1)/V)."""
    B, N = x.shape[0], x.shape[1]
    _, r = min_image(x, L)                                   # [B,N,N]
    iu = torch.triu_indices(N, N, offset=1, device=x.device)
    dists = r[:, iu[0], iu[1]].reshape(-1).cpu()             # unique pairs
    dists = dists[dists < rmax]
    counts = torch.histc(dists, bins=nbins, min=0.0, max=rmax)
    edges = torch.linspace(0.0, rmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if d == 2:
        shell = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    else:
        shell = (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho = (N - 1) / (L ** d)
    ideal_per_frame = 0.5 * N * rho * shell                  # unique pairs, ideal gas
    g = (counts / B) / ideal_per_frame
    return centers.numpy(), g.numpy()


def energy_pdf(vals, lo, hi, nbins):
    """PDF normalized over ALL samples -> in-range area = fraction in range (honest tail)."""
    vals = np.asarray(vals)
    counts, edges = np.histogram(vals, bins=nbins, range=(lo, hi))
    pdf = counts / (len(vals) * (edges[1] - edges[0]))
    centers = 0.5 * (edges[:-1] + edges[1:])
    overflow = float((vals > hi).mean())
    return centers, pdf, overflow


def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    ck = torch.load(CKPT, map_location=device)
    cfg = ck["config"]
    N, L, kT, cutoff = cfg["N"], cfg["L"], cfg["kT"], cfg["cutoff"]
    x = ck["samples"].to(device)
    logq = ck["logq"].to(device)
    data = get_data(N, L, kT, cutoff, device).to(device)
    print(f"loaded {x.shape[0]} flow samples, {data.shape[0]} MCMC configs "
          f"(overlaps {ck['overlap']:.3f}, ESS {100*ck['ess']:.2f}%)")

    # --- energies (per particle) ---
    U_mcmc = (lj_energy(data, L, cutoff=cutoff, shift=True) / N).cpu().numpy()
    U_flow_t = lj_energy(x, L, cutoff=cutoff, shift=True)
    U_flow = (U_flow_t / N).cpu().numpy()

    # importance weights (exact likelihood -> reweight flow to Boltzmann)
    logw = (-U_flow_t / kT - logq)
    logw = logw - torch.logsumexp(logw, 0)
    w = logw.exp().cpu().numpy()

    lo = math.floor(U_mcmc.min() - 0.5)
    hi = math.ceil(U_mcmc.max() + 2.0)           # cap just above the physical band
    nb = 70
    c_m, p_m, _ = energy_pdf(U_mcmc, lo, hi, nb)
    c_f, p_f, ovf = energy_pdf(U_flow, lo, hi, nb)
    # reweighted flow PDF (same bins, weighted)
    counts_w, edges = np.histogram(U_flow, bins=nb, range=(lo, hi), weights=w)
    p_rw = counts_w / (edges[1] - edges[0])      # weights already sum to 1

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

    ax[0].fill_between(c_m, p_m, step="mid", alpha=0.45, color="C0", label="MCMC (target)")
    ax[0].step(c_f, p_f, where="mid", color="C3", lw=1.8, label=f"AR flow (raw)")
    ax[0].step(c_m, p_rw, where="mid", color="C2", lw=1.8, ls="--",
               label="AR flow, reweighted")
    ax[0].axvline(U_mcmc.mean(), color="C0", ls=":", lw=1, alpha=0.8)
    ax[0].set_xlabel("U / N  (per-particle energy)")
    ax[0].set_ylabel("pdf")
    ax[0].set_xlim(lo, hi)
    ax[0].set_title(f"Energy distribution (capped at {hi})\n"
                    f"flow overflow > {hi}: {100*ovf:.1f}%  |  "
                    f"MCMC mean {U_mcmc.mean():.2f}, flow median {np.median(U_flow):.2f}")
    ax[0].legend(fontsize=9)

    # --- g(r) ---
    rmax = L / 2.0
    rc_m, g_m = compute_gr(data[:4000], L, rmax, 90, d=2)
    rc_f, g_f = compute_gr(x, L, rmax, 90, d=2)
    ax[1].plot(rc_m, g_m, color="C0", lw=2, label="MCMC (target)")
    ax[1].plot(rc_f, g_f, color="C3", lw=2, label="AR flow")
    ax[1].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[1].axvline(2 ** (1 / 6), color="grey", ls="--", lw=1, label=r"$2^{1/6}\sigma$ (LJ min)")
    ax[1].axvline(0.8, color="C3", ls=":", lw=1, alpha=0.7, label="overlap thresh 0.8")
    ax[1].set_xlabel(r"$r/\sigma$")
    ax[1].set_ylabel("g(r)")
    ax[1].set_xlim(0, rmax)
    ax[1].set_title("Radial distribution g(r)  (2D, periodic)")
    ax[1].legend(fontsize=9)

    fig.suptitle(f"AR particle flow vs MCMC — 2D LJ, N={N}, L={L}, kT={kT}  "
                 f"(overlaps {ck['overlap']:.3f}, x-ESS {100*ck['ess']:.2f}%)",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(ART, "ar_energy_gr.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    print(f"g(r) below 0.9 sigma: MCMC {g_m[rc_m < 0.9].mean():.3f}  "
          f"flow {g_f[rc_f < 0.9].mean():.3f}  (overlap signature)")


if __name__ == "__main__":
    main()
