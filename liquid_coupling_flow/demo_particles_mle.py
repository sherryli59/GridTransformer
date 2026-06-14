"""Phase E (forward-KL, informative auxiliary): train the particle flow on 2D LJ
MCMC data by maximum likelihood, then reweight to the Boltzmann distribution.

KEY FIX over the first attempt: the augmented auxiliary `a` is trained as a NOISY COPY
of x (r(a|x) = wrapped-Gaussian(x, sigma)), not uniform noise. Conditioning the
x-updates on a uniform-noise auxiliary cannot inject x-x correlations (excluded
volume) -> the flow stayed at the uniform-base constant. With a ~ N(x, sigma) the
conditioner sees REAL neighbour structure, while the joint likelihood stays exact:

    log pi(x, a) = -U(x)/kT + log r(a|x)          r(a|x) = wrapped-Gaussian(x, sigma)
    log w        = log pi(x, a) - log q(x, a)      (exact; partition fn cancels)

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
DATA_CACHE = "/tmp/lj2d_N16_L5.pt"


def _pair_energy_one(ri, x, i, L, cutoff):
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
    snaps = []
    for sweep in range(n_equil + n_collect):
        for i in range(N):
            xi = x[:, i, :]
            prop = torch.remainder(xi + step * torch.randn_like(xi), L)
            dE = _pair_energy_one(prop, x, i, L, cutoff) - _pair_energy_one(xi, x, i, L, cutoff)
            acc = torch.log(torch.rand(n_chains, device=device)) < (-dE / kT)
            x[:, i, :] = torch.where(acc[:, None], prop, xi)
        if sweep >= n_equil and (sweep - n_equil) % every == 0:
            snaps.append(x.clone())
    return torch.cat(snaps, dim=0)


def aux_logprob(a, x, L, sigma):
    """log r(a|x) = wrapped(min-image) Gaussian, summed over particles+coords -> [B]."""
    d = a - x
    d = d - L * torch.round(d / L)
    return (-0.5 * (d / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))).sum(dim=(1, 2))


def get_data(N, L, kT, cutoff, device):
    if os.path.exists(DATA_CACHE):
        return torch.load(DATA_CACHE, map_location=device)
    data = mcmc_lj(N, L, kT, cutoff, device).cpu()
    torch.save(data, DATA_CACHE)
    return data.to(device)


def main(device="cpu", sigma_aux=0.3, n_layers=12, lr=1e-3, steps=6000, tag=""):
    os.makedirs(ART, exist_ok=True)
    torch.manual_seed(0)
    N, L, kT, cutoff = 16, 5.0, 1.0, 2.4

    data = get_data(N, L, kT, cutoff, device).to(device)
    with torch.no_grad():
        U_data = (lj_energy(data, L, cutoff=cutoff, shift=True) / N).mean().item()
    print(f"data: {data.shape[0]} configs, <U>/N(data) {U_data:.3f}  [sigma_aux={sigma_aux}]")

    flow = build_particle_flow(N=N, d=2, L=L, n_layers=n_layers, num_bins=8,
                               cutoff=cutoff, hidden=64).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    B = 512
    for step_i in range(steps):
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        x = data[idx]
        a = torch.remainder(x + sigma_aux * torch.randn_like(x), L)   # informative auxiliary
        loss = -flow.log_prob(x, a).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        opt.step()
        if step_i % 500 == 0 or step_i == steps - 1:
            print(f"  step {step_i:4d}  -logq {loss.item():.3f}  (base {-flow.base_logp:.1f})")

    with torch.no_grad():
        x, a, logq = flow.sample(8000, device=device)
        U = lj_energy(x, L, cutoff=cutoff, shift=True)
        log_r = aux_logprob(a, x, L, sigma_aux)
        logw = -U / kT + log_r - logq
        logw = logw - torch.logsumexp(logw, 0)
        w = logw.exp()
        ess = float(1.0 / (w ** 2).sum() / w.shape[0])
        U_flow = (U / N).mean().item()
        good = torch.isfinite(U)
        U_flow_finite = (U[good] / N).mean().item()
        _, r = min_image(x, L)
        r = r + torch.eye(N, device=x.device)[None] * 1e3
        nn = r.min(dim=2).values.reshape(-1)
        overlap = float((nn < 0.8).float().mean())
    print(f"<U>/N  data {U_data:.3f} | flow(finite) {U_flow_finite:.3f} | overlaps {overlap:.3f}")
    print(f"joint-ESS {100*ess:.2f}%   finite-energy frac {good.float().mean():.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].scatter(data[0, :, 0].cpu(), data[0, :, 1].cpu(), s=60); ax[0].set_title("MCMC")
    ax[1].scatter(x[0, :, 0].cpu(), x[0, :, 1].cpu(), s=60); ax[1].set_title("flow")
    for k in (0, 1):
        ax[k].set_aspect("equal"); ax[k].set_xlim(0, L); ax[k].set_ylim(0, L)
    _, rd = min_image(data[:2000], L)
    rd = rd + torch.eye(N, device=data.device)[None] * 1e3
    ax[2].hist(rd.min(dim=2).values.reshape(-1).cpu().numpy(), bins=60, range=(0, 3), density=True, alpha=0.5, label="MCMC")
    ax[2].hist(nn.cpu().numpy(), bins=60, range=(0, 3), density=True, alpha=0.5, label="flow")
    ax[2].axvline(2 ** (1 / 6), color="r", ls="--"); ax[2].legend(); ax[2].set_xlabel("nn distance")
    fig.suptitle(f"Phase E aux{sigma_aux} L{n_layers} — ESS {100*ess:.2f}%, <U>/N flow {U_flow_finite:.2f} (data {U_data:.2f})")
    fig.tight_layout()
    out = os.path.join(ART, f"phaseE_lj2d_mle{tag}.png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main(device="cuda" if torch.cuda.is_available() else "cpu")
