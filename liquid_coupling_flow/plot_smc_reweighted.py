"""SMC *reweighted* estimator vs the Boltzmann ground truth (MCMC).

Unlike plot/apply scripts that resample to an equal-weight ensemble, this uses the
SMC importance weights directly: every Boltzmann average is <A> = sum_b w_b A(x_b).
That is the estimator you actually use for importance reweighting, and it keeps all
the information in the weights (no resampling noise). We re-run the single-particle
SMC so x and weights stay a matched pair, then compare weighted energy + g(r) to the
long MCMC run (the ground-truth Boltzmann ensemble for a many-body LJ liquid).

Run:  python -m liquid_coupling_flow.plot_smc_reweighted
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
from liquid_coupling_flow.smc import anneal_smc, SingleParticleMetropolis
from liquid_coupling_flow.demo_particles_mle import get_data
from liquid_coupling_flow.plot_ar_diagnostics import compute_gr

ART = os.path.join(os.path.dirname(__file__), "artifacts")
CKPT = os.path.join(ART, "ar_flow_ckpt.pt")


def gr_weighted(x, w, L, rmax, nbins, d=2):
    """g(r) weighted by per-config weights w (normalized, sum=1). Same shell
    normalization as the unweighted compute_gr -> plateau 1."""
    B, N = x.shape[0], x.shape[1]
    _, r = min_image(x, L)
    iu = torch.triu_indices(N, N, offset=1, device=x.device)
    dists = r[:, iu[0], iu[1]]                       # [B, P]
    P = dists.shape[1]
    wp = w[:, None].expand(-1, P).reshape(-1).cpu().numpy()
    dd = dists.reshape(-1).cpu().numpy()
    counts, edges = np.histogram(dd, bins=nbins, range=(0.0, rmax), weights=wp)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell = (np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) if d == 2
             else (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3))
    ideal = 0.5 * N * ((N - 1) / (L ** d)) * shell   # per (weighted) frame
    return centers, counts / ideal


def main(device="cuda" if torch.cuda.is_available() else "cpu",
         M=4000, n_bridge=40, n_steps=1, step=0.15):
    torch.manual_seed(0)
    ck = torch.load(CKPT, map_location=device)
    cfg = ck["config"]
    N, L, kT, cutoff = cfg["N"], cfg["L"], cfg["kT"], cfg["cutoff"]

    flow = ARParticleFlow(N=N, L=L, num_bins=cfg["num_bins"], hidden=cfg["hidden"],
                          cutoff=cutoff).to(device)
    flow.load_state_dict(ck["state_dict"]); flow.eval()

    x0 = ck["samples"][:M].to(device)
    data = get_data(N, L, kT, cutoff, device).to(device)

    def logq_fn(x):
        return flow.log_prob(x)

    def log_target_fn(x):
        return -lj_energy(x, L, cutoff=cutoff, shift=True) / kT

    kernel = SingleParticleMetropolis(step=step, n_sweeps=n_steps, L=L)
    res = anneal_smc(x0, logq_fn, log_target_fn, kernel, n_bridge=n_bridge,
                     resample_thresh=0.5, schedule="linear", verbose=True)
    x, w = res["x"], res["weights"]                  # MATCHED weighted ensemble
    print(f"SMC ESS {100*res['ess']:.1f}%  (plain-IS {100*res['plain_is_ess']:.2f}%)", flush=True)

    # ---- reweighted observables vs Boltzmann (MCMC) ----
    with torch.no_grad():
        U_smc = (lj_energy(x, L, cutoff=cutoff, shift=True) / N)
        U_mcmc = (lj_energy(data, L, cutoff=cutoff, shift=True) / N)
    Us, wn = U_smc.cpu().numpy(), w.cpu().numpy()
    Um = U_mcmc.cpu().numpy()

    # self-normalized IS estimate + standard error of <U>/N
    mu_rw = float((wn * Us).sum())
    se_rw = float(np.sqrt((wn ** 2 * (Us - mu_rw) ** 2).sum()))
    mu_bz = float(Um.mean())
    se_bz = float(Um.std() / math.sqrt(len(Um)))
    print(f"<U>/N  reweighted {mu_rw:.4f} +/- {se_rw:.4f} | Boltzmann(MCMC) {mu_bz:.4f} +/- {se_bz:.4f}",
          flush=True)

    lo, hi, nb = math.floor(Um.min() - 0.5), math.ceil(Um.max() + 1.0), 70
    # weighted energy pdf (weights sum to 1 -> integrate to fraction-in-range)
    cw, ew = np.histogram(Us, bins=nb, range=(lo, hi), weights=wn)
    pdf_rw = cw / (ew[1] - ew[0])
    cm, em = np.histogram(Um, bins=nb, range=(lo, hi))
    pdf_bz = cm / (len(Um) * (em[1] - em[0]))
    centers = 0.5 * (ew[:-1] + ew[1:])
    ovf_rw = float(wn[Us > hi].sum())

    rmax = L / 2.0
    rc_b, g_b = compute_gr(data[:4000], L, rmax, 90, d=2)
    rc_w, g_w = gr_weighted(x, w, L, rmax, 90, d=2)
    g_below = float(g_w[rc_w < 0.9].mean())

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].fill_between(0.5 * (em[:-1] + em[1:]), pdf_bz, step="mid", alpha=0.45,
                       color="C0", label="Boltzmann (MCMC)")
    ax[0].step(centers, pdf_rw, where="mid", color="C2", lw=2.0,
               label="flow+SMC, reweighted")
    ax[0].axvline(mu_bz, color="C0", ls=":", lw=1)
    ax[0].set_xlabel("U / N"); ax[0].set_ylabel("pdf"); ax[0].set_xlim(lo, hi)
    ax[0].set_title(f"Energy: reweighted vs Boltzmann\n"
                    f"<U>/N  reweighted {mu_rw:.3f}+/-{se_rw:.3f}  |  "
                    f"Boltzmann {mu_bz:.3f}  (overflow {100*ovf_rw:.2f}%)")
    ax[0].legend(fontsize=9)

    ax[1].plot(rc_b, g_b, color="C0", lw=2.4, label="Boltzmann (MCMC)")
    ax[1].plot(rc_w, g_w, color="C2", lw=1.6, label="flow+SMC, reweighted")
    ax[1].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[1].axvline(2 ** (1 / 6), color="grey", ls="--", lw=1, label=r"$2^{1/6}\sigma$")
    ax[1].set_xlabel(r"$r/\sigma$"); ax[1].set_ylabel("g(r)"); ax[1].set_xlim(0, rmax)
    ax[1].set_title(f"g(r): reweighted vs Boltzmann  (reweighted g(<0.9$\\sigma$)={g_below:.3f})")
    ax[1].legend(fontsize=9)

    fig.suptitle(f"AR flow + single-particle SMC, IMPORTANCE-REWEIGHTED vs Boltzmann — "
                 f"2D LJ N={N} L={L} kT={kT}  (ESS {100*res['ess']:.1f}%)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(ART, "ar_smc_reweighted.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
