"""CONTROL: does the flow earn its keep, or is SMC just doing (annealed) MCMC?

Run SMC from the FLOW proposal and from a UNIFORM proposal under the IDENTICAL bridge
and single-particle kernel, and record the convergence trajectory (overlap fraction
and median energy vs bridge step). If the flow's better starting point does not buy
faster convergence than starting from uniform noise, then the flow is dead weight and
SMC (= annealed MCMC) is doing all the work -- which is the honest worry.

Reference: plain MCMC equilibrates in ~200 single-particle sweeps; the SMC here uses
n_bridge sweeps total, so both should also be compared to that cost.

Run:  python -m liquid_coupling_flow.control_flow_vs_uniform
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
from liquid_coupling_flow import mcmc

ART = os.path.join(os.path.dirname(__file__), "artifacts")
CKPT = os.path.join(ART, "ar_flow_ckpt.pt")          # original sum/sum (best base)
SIZES = [(16, 5.0), (64, 10.0)]
KT, CUTOFF = 1.0, 2.4


def overlap_frac(x, L, N, thresh=0.8):
    _, r = min_image(x, L)
    r = r + torch.eye(N, device=x.device)[None] * 1e3
    return float((r.min(dim=2).values.reshape(-1) < thresh).float().mean())


def run(N, L, x0, logq_fn, M, n_bridge, device):
    def obs(x):
        U = (lj_energy(x, L, cutoff=CUTOFF, shift=True) / N)
        return (overlap_frac(x, L, N), float(U.median()))
    kernel = SingleParticleMetropolis(step=0.15, n_sweeps=1, L=L)
    res = anneal_smc(x0, logq_fn, lambda x: -lj_energy(x, L, cutoff=CUTOFF, shift=True) / KT,
                     kernel, n_bridge=n_bridge, resample_thresh=0.5, observable_fn=obs)
    ov = np.array([o[0] for o in res["obs_history"]])
    me = np.array([o[1] for o in res["obs_history"]])
    Us = (lj_energy(res["x"], L, cutoff=CUTOFF, shift=True) / N).cpu().numpy()
    wn = res["weights"].cpu().numpy()
    mu = float((wn * Us).sum())
    return ov, me, mu, res["ess"]


def main(device="cuda" if torch.cuda.is_available() else "cpu", M=1500, n_bridge=24):
    torch.manual_seed(0)
    ck = torch.load(CKPT, map_location=device); cfg = ck["config"]
    fig, axes = plt.subplots(2, len(SIZES), figsize=(6 * len(SIZES), 8), squeeze=False)
    print(f"{'N':>3} {'init':>8} {'sweeps':>6} {'final_ov':>8} {'<U>/N_rw':>10} {'MCMC':>8}", flush=True)

    for col, (N, L) in enumerate(SIZES):
        data = mcmc.get_data(N, L, KT, CUTOFF, d=2, device=device,
                             n_chains=2048, n_equil=200, n_collect=120, every=30)
        U_mcmc = float((lj_energy(data, L, cutoff=CUTOFF, shift=True) / N).mean())

        flow = ARParticleFlow(N=N, L=L, num_bins=cfg["num_bins"], hidden=cfg["hidden"],
                              cutoff=CUTOFF, ctx0_reduce=cfg.get("ctx0_reduce", "sum"),
                              ctxc_reduce=cfg.get("ctxc_reduce", "sum")).to(device)
        flow.load_state_dict(ck["state_dict"]); flow.eval()

        base_unif = -2 * N * math.log(L)
        with torch.no_grad():
            x_flow, _ = flow.sample(M, device=device)
        x_unif = torch.rand(M, N, 2, device=device) * L

        ovF, meF, muF, essF = run(N, L, x_flow, flow.log_prob, M, n_bridge, device)
        ovU, meU, muU, essU = run(N, L, x_unif,
                                  lambda x: torch.full((x.shape[0],), base_unif, device=x.device),
                                  M, n_bridge, device)
        for name, ov, mu in [("flow", ovF, muF), ("uniform", ovU, muU)]:
            print(f"{N:>3} {name:>8} {n_bridge:>6} {ov[-1]:>8.3f} {mu:>10.3f} {U_mcmc:>8.3f}", flush=True)

        steps = np.arange(len(ovF))
        ax0, ax1 = axes[0][col], axes[1][col]
        ax0.plot(steps, ovF, "-o", ms=3, color="C2", label="flow-init")
        ax0.plot(steps, ovU, "-o", ms=3, color="C1", label="uniform-init")
        ax0.axhline(0.0, color="k", lw=0.5, ls=":")
        ax0.set_title(f"N={N}: overlap fraction vs SMC sweep"); ax0.set_ylabel("overlap frac")
        ax0.legend(fontsize=9)
        ax1.plot(steps, meF, "-o", ms=3, color="C2", label="flow-init")
        ax1.plot(steps, meU, "-o", ms=3, color="C1", label="uniform-init")
        ax1.axhline(U_mcmc, color="C0", ls="--", lw=1.2, label=f"MCMC <U>/N {U_mcmc:.2f}")
        ax1.set_ylim(U_mcmc - 0.5, max(meU.max(), meF.max()) * 0.5 + 1)
        ax1.set_title(f"N={N}: median U/N vs SMC sweep"); ax1.set_xlabel("SMC sweep (bridge step)")
        ax1.set_ylabel("median U/N"); ax1.legend(fontsize=9)

    fig.suptitle(f"Control: flow-init vs uniform-init, identical SMC ({n_bridge} sweeps; "
                 f"MCMC equilibration ~200 sweeps)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(ART, "control_flow_vs_uniform.png")
    fig.savefig(out, dpi=120); print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
