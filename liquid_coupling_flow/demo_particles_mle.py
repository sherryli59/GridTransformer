"""Phase E (forward-KL): train the particle flow on 2D LJ MCMC data by maximum
likelihood, then reweight to the exact Boltzmann distribution and report ESS.

Reverse-KL got stuck (demo_particles.py); forward-KL on real equilibrium configs is
the reliable route (it worked beautifully on the torus toy). Pipeline:
  1. cheap vectorised 2D LJ Metropolis -> equilibrium training configs.
  2. train the augmented particle flow: minimise -E_{x_data, a~uniform}[log q(x,a)].
  3. sample, reweight against the (exact) LJ energy -> joint-space ESS + structure.

Run:  python -m liquid_coupling_flow.demo_particles_mle
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


def _pair_energy_one(ri, x, i, L, cutoff):
    """Energy of particle i at position ri [C,2] vs all others x [C,N,2]."""
    diff = ri[:, None, :] - x
    diff = diff - L * torch.round(diff / L)
    r2 = (diff ** 2).sum(-1)
    r2[:, i] = 1e9
    within = r2 < cutoff ** 2
    inv6 = (1.0 / r2) ** 3
    ic6 = (1.0 / cutoff) ** 6
    e = 4.0 * (inv6 ** 2 - inv6) - 4.0 * (ic6 ** 2 - ic6)
    return torch.where(within, e, torch.zeros_like(e)).sum(-1)


def mcmc_lj(N, L, kT, cutoff, device, n_chains=4096, n_equil=300, n_collect=200,
            every=25, step=0.15):
    x = torch.rand(n_chains, N, 2, device=device) * L
    snaps, n_acc, n_tot = [], 0, 0
    for sweep in range(n_equil + n_collect):
        for i in range(N):
            xi = x[:, i, :]
            prop = torch.remainder(xi + step * torch.randn_like(xi), L)
            dE = _pair_energy_one(prop, x, i, L, cutoff) - _pair_energy_one(xi, x, i, L, cutoff)
            acc = torch.log(torch.rand(n_chains, device=device)) < (-dE / kT)
            x[:, i, :] = torch.where(acc[:, None], prop, xi)
            n_acc += int(acc.sum()); n_tot += acc.numel()
        if sweep >= n_equil and (sweep - n_equil) % every == 0:
            snaps.append(x.clone())
    data = torch.cat(snaps, dim=0)
    return data, n_acc / n_tot


def main(device="cpu"):
    os.makedirs(ART, exist_ok=True)
    torch.manual_seed(0)
    N, L, kT, cutoff = 16, 5.0, 1.0, 2.4          # rho = 16/25 = 0.64, supercritical fluid

    print("generating 2D LJ MCMC data ...")
    data, acc = mcmc_lj(N, L, kT, cutoff, device)
    with torch.no_grad():
        U_data = (lj_energy(data, L, cutoff=cutoff, shift=True) / N).mean().item()
    print(f"  data: {data.shape[0]} configs, MCMC acc {acc:.2f}, <U>/N(data) {U_data:.3f}")

    flow = build_particle_flow(N=N, d=2, L=L, n_layers=12, num_bins=8,
                               cutoff=cutoff, hidden=64).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=5e-4)
    steps, B = 4000, 512
    for step_i in range(steps):
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        x = data[idx]
        a = torch.rand(B, N, 2, device=device) * L
        loss = -flow.log_prob(x, a).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        opt.step()
        if step_i % 400 == 0 or step_i == steps - 1:
            print(f"  step {step_i:4d}  -logq {loss.item():.3f}")

    with torch.no_grad():
        x, a, logq = flow.sample(8000, device=device)
        U = lj_energy(x, L, cutoff=cutoff, shift=True)            # EXACT (no clamp)
        log_pi = -U / kT - N * 2 * math.log(L)
        logw = log_pi - logq
        logw = logw - torch.logsumexp(logw, 0)
        w = logw.exp()
        ess = float(1.0 / (w ** 2).sum() / w.shape[0])
        U_flow = (U / N).mean().item()
        U_rw = float((w * (U / N)).sum())                        # reweighted <U>/N
        _, r = min_image(x, L)
        r = r + torch.eye(N, device=x.device)[None] * 1e3
        nn = r.min(dim=2).values.reshape(-1)
        overlap = float((nn < 0.8).float().mean())
    print(f"<U>/N  data {U_data:.3f} | flow {U_flow:.3f} | reweighted {U_rw:.3f}")
    print(f"joint-ESS {100*ess:.1f}%   nn<0.8 frac {overlap:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].scatter(data[0, :, 0].cpu(), data[0, :, 1].cpu(), s=60)
    ax[0].set_title("MCMC config"); ax[0].set_aspect("equal"); ax[0].set_xlim(0, L); ax[0].set_ylim(0, L)
    ax[1].scatter(x[0, :, 0].cpu(), x[0, :, 1].cpu(), s=60)
    ax[1].set_title("flow config"); ax[1].set_aspect("equal"); ax[1].set_xlim(0, L); ax[1].set_ylim(0, L)
    _, rd = min_image(data[:2000], L)
    rd = rd + torch.eye(N, device=data.device)[None] * 1e3
    ax[2].hist(rd.min(dim=2).values.reshape(-1).cpu().numpy(), bins=60, range=(0, 3),
               density=True, alpha=0.5, label="MCMC")
    ax[2].hist(nn.cpu().numpy(), bins=60, range=(0, 3), density=True, alpha=0.5, label="flow")
    ax[2].axvline(2 ** (1 / 6), color="r", ls="--"); ax[2].legend(); ax[2].set_xlabel("nn distance")
    fig.suptitle(f"Phase E (MLE) — 2D LJ N={N}: <U>/N data {U_data:.2f} / flow {U_flow:.2f}, ESS {100*ess:.1f}%")
    fig.tight_layout()
    out = os.path.join(ART, "phaseE_lj2d_mle.png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main(device="cuda" if torch.cuda.is_available() else "cpu")
