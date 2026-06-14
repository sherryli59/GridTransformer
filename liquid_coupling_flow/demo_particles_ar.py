"""Train the autoregressive particle flow on 2D LJ data; check generation + ESS.

The AR flow gives log q(x) directly (no auxiliary), so reweighting is clean:
    log w = -U(x)/kT - log q(x)   (+ const)   -> ESS in x-space.
The metric that the coupling/augmented attempts failed: SAMPLE overlaps + ESS.

Run:  python -m liquid_coupling_flow.demo_particles_ar
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from liquid_coupling_flow.particles_ar import ARParticleFlow
from liquid_coupling_flow.particles import min_image
from liquid_coupling_flow.energy import lj_energy
from liquid_coupling_flow.demo_particles_mle import get_data

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def main(device="cpu", steps=4000, lr=1e-3):
    os.makedirs(ART, exist_ok=True)
    torch.manual_seed(0)
    N, L, kT, cutoff = 16, 5.0, 1.0, 2.4
    data = get_data(N, L, kT, cutoff, device).to(device)
    U_data = (lj_energy(data, L, cutoff=cutoff, shift=True) / N).mean().item()
    print(f"data <U>/N {U_data:.3f}")

    flow = ARParticleFlow(N=N, L=L, num_bins=8, hidden=64, cutoff=cutoff).to(device)
    # correctness: invertibility
    xs, lp_s = flow.sample(64, device=device)
    inv = (lp_s - flow.log_prob(xs)).abs().max().item()
    print(f"invertibility logp(sample)==logp(eval): {inv:.2e}")

    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    B = 512
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        loss = -flow.log_prob(data[idx]).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        opt.step()
        if step % 400 == 0 or step == steps - 1:
            print(f"  step {step:4d}  -logq {loss.item():.3f}  (base {-flow.base_logp:.1f})")

    with torch.no_grad():
        x, logq = flow.sample(8000, device=device)
        U = lj_energy(x, L, cutoff=cutoff, shift=True)
        logw = -U / kT - logq
        logw = logw - torch.logsumexp(logw, 0)
        w = logw.exp()
        ess = float(1.0 / (w ** 2).sum() / w.shape[0])
        good = torch.isfinite(U)
        Uf = (U[good] / N).mean().item()
        _, r = min_image(x, L)
        r = r + torch.eye(N, device=x.device)[None] * 1e3
        nn = r.min(dim=2).values.reshape(-1)
        overlap = float((nn < 0.8).float().mean())
    print(f"<U>/N data {U_data:.3f} | flow(finite) {Uf:.3f} | overlaps {overlap:.3f} | finite {good.float().mean():.3f}")
    print(f"x-space ESS {100*ess:.2f}%")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].scatter(data[0, :, 0].cpu(), data[0, :, 1].cpu(), s=60); ax[0].set_title("MCMC")
    ax[1].scatter(x[0, :, 0].cpu(), x[0, :, 1].cpu(), s=60); ax[1].set_title("AR flow")
    for k in (0, 1):
        ax[k].set_aspect("equal"); ax[k].set_xlim(0, L); ax[k].set_ylim(0, L)
    _, rd = min_image(data[:2000], L); rd = rd + torch.eye(N, device=data.device)[None] * 1e3
    ax[2].hist(rd.min(dim=2).values.reshape(-1).cpu().numpy(), bins=60, range=(0, 3), density=True, alpha=0.5, label="MCMC")
    ax[2].hist(nn.cpu().numpy(), bins=60, range=(0, 3), density=True, alpha=0.5, label="AR flow")
    ax[2].axvline(2 ** (1 / 6), color="r", ls="--"); ax[2].legend(); ax[2].set_xlabel("nn dist")
    fig.suptitle(f"AR particle flow — ESS {100*ess:.2f}%, <U>/N {Uf:.2f} (data {U_data:.2f}), overlaps {overlap:.2f}")
    fig.tight_layout()
    out = os.path.join(ART, "phaseE_ar.png"); fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main(device="cuda" if torch.cuda.is_available() else "cpu")
