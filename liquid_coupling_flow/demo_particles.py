"""Phase E: train the augmented particle flow on a small 2D LJ liquid by reverse-KL.

This is the Boltzmann-generator objective — NO data needed, train directly against
the (soft-core) LJ energy:

    L(theta) = E_{(x,a)~q}[ log q(x,a) + U(x)/kT ]      (min at q = Boltzmann joint)

The auxiliary a has a uniform target (constant), so it drops out of the gradient up
to a constant. Temperature is annealed high->1 for stability (uniform init overlaps).
We then measure the joint-space importance-sampling ESS against the same soft target.

Soft core: U uses min_dist clamp + cutoff (a disordered liquid with excluded volume);
exact-LJ reweighting would just need the flow run in float64. This is a PROOF that the
flow trains and yields structured, non-overlapping configs with usable ESS.

Run:  python -m liquid_coupling_flow.demo_particles
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from liquid_coupling_flow.particles import build_particle_flow, min_image
from liquid_coupling_flow.energy import lj_energy

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def main(device="cpu"):
    os.makedirs(ART, exist_ok=True)
    torch.manual_seed(0)
    N, d, L = 16, 2, 6.0          # rho = 16/36 = 0.44
    kT_final = 1.0
    cutoff, min_dist = 2.5, 0.8

    flow = build_particle_flow(N=N, d=d, L=L, n_layers=10, num_bins=8,
                               cutoff=cutoff, hidden=64).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"particle flow: N={N} d={d} L={L}  {n_params} params")
    opt = torch.optim.Adam(flow.parameters(), lr=2e-4)

    steps, B = 4000, 256
    for step in range(steps):
        frac = step / steps
        kT = max(kT_final, 5.0 - 4.0 * min(1.0, frac / 0.8))  # 5 -> 1, slow anneal
        x, a, logq = flow.sample(B, device=device)
        U = lj_energy(x, L, cutoff=cutoff, shift=True, min_dist=min_dist)
        loss = (logq + U / kT).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        opt.step()
        if step % 300 == 0 or step == steps - 1:
            with torch.no_grad():
                Um = lj_energy(x, L, cutoff=cutoff, shift=True, min_dist=min_dist).mean().item()
            print(f"  step {step:4d}  kT {kT:.2f}  loss {loss.item():.3f}  <U>/N {Um/N:.3f}")

    # ----- evaluate: joint-space ESS against the soft target at kT=1 -----
    with torch.no_grad():
        x, a, logq = flow.sample(4000, device=device)
        U = lj_energy(x, L, cutoff=cutoff, shift=True, min_dist=min_dist)
        log_pi = -U / kT_final - N * d * math.log(L)         # + log r(a)=const(uniform)
        logw = log_pi - logq
        logw = logw - torch.logsumexp(logw, 0)
        w = logw.exp()
        ess = float(1.0 / (w ** 2).sum() / w.shape[0])
        meanU = (U / N).mean().item()
        # nearest-neighbour distance distribution (structure / no-overlap check)
        _, r = min_image(x, L)
        r = r + torch.eye(N, device=x.device)[None] * 1e3  # ignore self
        nn = r.min(dim=2).values.reshape(-1)
        frac_overlap = float((nn < 0.8).float().mean())

    print(f"<U>/N {meanU:.3f}   joint-ESS {100*ess:.1f}%   nn<0.8 frac {frac_overlap:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    cfg = x[0].cpu().numpy()
    ax[0].scatter(cfg[:, 0], cfg[:, 1], s=60)
    ax[0].set_xlim(0, L); ax[0].set_ylim(0, L); ax[0].set_aspect("equal")
    ax[0].set_title("one flow config")
    ax[1].hist(nn.cpu().numpy(), bins=60, range=(0, 3), density=True)
    ax[1].axvline(2 ** (1 / 6), color="r", ls="--", label="LJ min 2^(1/6)")
    ax[1].set_xlabel("nearest-neighbour distance"); ax[1].legend()
    fig.suptitle(f"Phase E — 2D LJ liquid flow (N={N}, <U>/N {meanU:.2f}, ESS {100*ess:.1f}%)")
    fig.tight_layout()
    out = os.path.join(ART, "phaseE_lj2d_particles.png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    main(device=dev)
