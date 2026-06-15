"""3D Lennard-Jones liquid: train the dimension-general AR flow, then SMC-correct.

Fresh 3D run (the 2D pipeline was a cheap proof-of-concept). N=32 at reduced density
rho*=0.65, T*=1.0 -- a dense disordered LJ liquid (below T_c~1.31, above rho_c). Box
is small (L/2 limits the cutoff), so g(r) is shown to the first coordination shell.

Pipeline: 3D MCMC ground truth -> forward-KL train ParticleARFlowND(d=3) -> save
checkpoint -> single-particle SMC -> reweighted energy + g(r) vs Boltzmann.

Run:  python -m liquid_coupling_flow.train_3d
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

from liquid_coupling_flow.particles_ar_nd import ParticleARFlowND
from liquid_coupling_flow.energy import lj_energy
from liquid_coupling_flow.particles import min_image
from liquid_coupling_flow.smc import anneal_smc, SingleParticleMetropolis
from liquid_coupling_flow import mcmc
from liquid_coupling_flow.plot_ar_diagnostics import compute_gr
from liquid_coupling_flow.plot_smc_reweighted import gr_weighted

ART = os.path.join(os.path.dirname(__file__), "artifacts")
CKPT = os.path.join(ART, "ar_flow_3d_ckpt.pt")

N, RHO, KT, CUTOFF, D = 32, 0.65, 1.0, 1.8, 3
L = (N / RHO) ** (1.0 / 3.0)


def overlap_frac(x, L, N, thresh=0.8):
    _, r = min_image(x, L)
    r = r + torch.eye(N, device=x.device)[None] * 1e3
    return float((r.min(dim=2).values.reshape(-1) < thresh).float().mean())


def main(device="cuda" if torch.cuda.is_available() else "cpu",
         steps=3500, lr=1e-3, num_bins=24, hidden=128, M=2500):
    os.makedirs(ART, exist_ok=True)
    torch.manual_seed(0)
    print(f"3D LJ: N={N} rho*={RHO} L={L:.3f} (L/2={L/2:.3f}) cutoff={CUTOFF} kT={KT}", flush=True)

    data = mcmc.get_data(N, L, KT, CUTOFF, d=D, device=device,
                         n_chains=2048, n_equil=250, n_collect=150, every=25)
    U_data = (lj_energy(data, L, cutoff=CUTOFF, shift=True) / N).mean().item()
    print(f"MCMC {data.shape[0]} configs  <U>/N {U_data:.3f}", flush=True)

    flow = ParticleARFlowND(N=N, L=L, d=D, num_bins=num_bins, hidden=hidden,
                            cutoff=CUTOFF).to(device)
    xs, lp = flow.sample(32, device=device)
    print(f"invertibility {((lp - flow.log_prob(xs)).abs().max()):.2e}", flush=True)

    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    B, t0 = 512, time.time()
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        loss = -flow.log_prob(data[idx]).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            print(f"  step {step:4d}  -logq {loss.item():.3f}  (base {-flow.base_logp:.1f})  "
                  f"{time.time()-t0:.0f}s", flush=True)
    flow.eval()
    torch.save({"config": dict(N=N, L=L, kT=KT, cutoff=CUTOFF, d=D,
                               num_bins=num_bins, hidden=hidden),
                "state_dict": flow.state_dict(), "U_data": U_data}, CKPT)
    print(f"saved {CKPT}", flush=True)

    with torch.no_grad():
        x0, _ = flow.sample(M, device=device)
    raw_ov = overlap_frac(x0, L, N)

    def logq_fn(x):
        return flow.log_prob(x)

    def log_target_fn(x):
        return -lj_energy(x, L, cutoff=CUTOFF, shift=True) / KT

    kernel = SingleParticleMetropolis(step=0.12, n_sweeps=1, L=L)
    res = anneal_smc(x0, logq_fn, log_target_fn, kernel, n_bridge=30,
                     resample_thresh=0.5, schedule="linear", verbose=True)
    x, w = res["x"], res["weights"]
    smc_ov = overlap_frac(x, L, N)
    Us = (lj_energy(x, L, cutoff=CUTOFF, shift=True) / N).cpu().numpy()
    Um = (lj_energy(data, L, cutoff=CUTOFF, shift=True) / N).cpu().numpy()
    wn = w.cpu().numpy()
    mu_rw = float((wn * Us).sum()); se_rw = float(np.sqrt((wn ** 2 * (Us - mu_rw) ** 2).sum()))
    print(f"raw-flow overlaps {raw_ov:.3f} | plain-IS ESS {100*res['plain_is_ess']:.2f}% -> "
          f"SMC ESS {100*res['ess']:.1f}% | SMC overlaps {smc_ov:.3f} | "
          f"<U>/N rw {mu_rw:.3f}+/-{se_rw:.3f} (MCMC {Um.mean():.3f})", flush=True)

    # ---- reweighted energy + g(r) vs Boltzmann ----
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
    ax[1].axvline(2 ** (1 / 6), color="grey", ls="--", lw=1, label=r"$2^{1/6}\sigma$")
    ax[1].axhline(1.0, color="k", lw=0.5, ls=":")
    ax[1].set_xlabel(r"$r/\sigma$"); ax[1].set_ylabel("g(r)"); ax[1].set_xlim(0, rmax)
    ax[1].set_title(f"g(r) 3D  (SMC ESS {100*res['ess']:.0f}%, overlaps {smc_ov:.3f})")
    ax[1].legend(fontsize=9)
    fig.suptitle(f"3D LJ liquid N={N} rho*={RHO} T*={KT}: AR flow + single-particle SMC, "
                 f"reweighted vs Boltzmann", fontsize=12)
    fig.tight_layout()
    out = os.path.join(ART, "lj3d_smc_reweighted.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
