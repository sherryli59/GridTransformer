"""Annealed Sequential Monte Carlo / Annealed Importance Sampling corrector.

Turns a proposal q (e.g. the coupling flow, with exact log q) into an
asymptotically-unbiased sampler for a target pi ∝ exp(-E) known only up to a
constant (the LJ Boltzmann distribution). Bridges q -> pi along

    log pi_beta(x) = (1-beta) * log q(x) + beta * log_target(x),   beta: 0 -> 1

carrying importance weights and resampling when ESS drops, with an MCMC kernel
leaving each pi_beta invariant. This is the inference-time route to good ESS when
the flow alone is imperfect (see Sequential Boltzmann Generators, arXiv:2502.18462).

`log_target` may be unnormalised (E up to a constant): the constant cancels in
the per-step weight increment and in the Metropolis ratio.
"""

from __future__ import annotations

import torch


def ess_fraction(log_weights: torch.Tensor) -> float:
    w = torch.softmax(log_weights, dim=0)
    return float(1.0 / (w ** 2).sum() / w.shape[0])


def systematic_resample(weights: torch.Tensor) -> torch.Tensor:
    """Low-variance systematic resampling. weights: normalised [M] -> indices [M]."""
    M = weights.shape[0]
    positions = (torch.arange(M, device=weights.device) + torch.rand(1, device=weights.device)) / M
    cumsum = torch.cumsum(weights, dim=0)
    cumsum[-1] = 1.0
    return torch.searchsorted(cumsum, positions).clamp(max=M - 1)


class RWMetropolis:
    """Random-walk Metropolis kernel (gradient-free), optionally periodic on [0,L)."""

    def __init__(self, step: float = 0.1, n_steps: int = 10, L: float | None = None):
        self.step = float(step)
        self.n_steps = int(n_steps)
        self.L = L

    def __call__(self, x: torch.Tensor, log_pi_fn) -> tuple[torch.Tensor, float]:
        logp_x = log_pi_fn(x)
        acc_rates = []
        for _ in range(self.n_steps):
            prop = x + self.step * torch.randn_like(x)
            if self.L is not None:
                prop = torch.remainder(prop, self.L)
            logp_p = log_pi_fn(prop)
            accept = torch.log(torch.rand(x.shape[0], device=x.device)) < (logp_p - logp_x)
            # broadcast the accept mask over all trailing dims (works for x of any rank:
            # [M, D] scalar toys AND [M, N, d] particle configs)
            acc_mask = accept.view(-1, *([1] * (x.dim() - 1)))
            x = torch.where(acc_mask, prop, x)
            logp_x = torch.where(accept, logp_p, logp_x)
            acc_rates.append(accept.float().mean())
        return x, float(torch.stack(acc_rates).mean())


class SingleParticleMetropolis:
    """Single-particle random-walk Metropolis for [M, N, d] particle configs.

    Proposes ONE particle at a time (sweeping over all N), so acceptance does not
    collapse as (single-particle-acc)^N the way a whole-config move does in a dense
    liquid. Calls `log_pi_fn` on the full config after each proposal, so the exact
    bridge density `(1-beta) log q + beta log_target` is preserved (no bias). For an
    autoregressive flow that means a full log q recompute per particle index -- the
    cost of an exact, genuinely-rejuvenating kernel.
    """

    def __init__(self, step: float = 0.15, n_sweeps: int = 1, L: float | None = None):
        self.step = float(step)
        self.n_sweeps = int(n_sweeps)
        self.L = L

    def __call__(self, x: torch.Tensor, log_pi_fn) -> tuple[torch.Tensor, float]:
        M, N, d = x.shape
        logp_x = log_pi_fn(x)
        acc_rates = []
        for _ in range(self.n_sweeps):
            for j in range(N):
                prop = x.clone()
                pj = prop[:, j, :] + self.step * torch.randn(M, d, device=x.device)
                if self.L is not None:
                    pj = torch.remainder(pj, self.L)
                prop[:, j, :] = pj
                logp_p = log_pi_fn(prop)
                accept = torch.log(torch.rand(M, device=x.device)) < (logp_p - logp_x)
                x = torch.where(accept.view(-1, 1, 1), prop, x)
                logp_x = torch.where(accept, logp_p, logp_x)
                acc_rates.append(accept.float().mean())
        return x, float(torch.stack(acc_rates).mean())


def anneal_smc(x0, logq_fn, log_target_fn, kernel, n_bridge: int = 50,
               resample_thresh: float = 0.5, schedule: str = "linear", verbose: bool = False,
               observable_fn=None):
    """Run annealed SMC from samples x0 ~ q to the target.

    Returns dict with final samples, normalised weights, ESS history, and the
    plain-importance-sampling ESS (betas {0,1}, no moves) for comparison.
    """
    device = x0.device
    M = x0.shape[0]
    if schedule == "linear":
        betas = torch.linspace(0.0, 1.0, n_bridge + 1, device=device)
    elif schedule == "quadratic":
        betas = torch.linspace(0.0, 1.0, n_bridge + 1, device=device) ** 2
    else:
        raise ValueError(schedule)

    # Plain IS baseline (no annealing, no moves): w = target/q
    with torch.no_grad():
        logw_plain = log_target_fn(x0) - logq_fn(x0)
    plain_ess = ess_fraction(logw_plain)

    x = x0.clone()
    logw = torch.zeros(M, device=device)
    ess_hist, acc_hist, obs_hist = [], [], []
    with torch.no_grad():
        if observable_fn is not None:
            obs_hist.append(observable_fn(x))          # beta=0 (the bare proposal)
        logq_x = logq_fn(x)
        logt_x = log_target_fn(x)
        for k in range(1, len(betas)):
            db = (betas[k] - betas[k - 1])
            logw = logw + db * (logt_x - logq_x)
            ess = ess_fraction(logw)
            ess_hist.append(ess)
            if ess < resample_thresh:
                w = torch.softmax(logw, dim=0)
                idx = systematic_resample(w)
                x = x[idx]
                logw = torch.zeros(M, device=device)

            beta = betas[k]

            def log_pi(xx, beta=beta):
                return (1 - beta) * logq_fn(xx) + beta * log_target_fn(xx)

            x, acc = kernel(x, log_pi)
            acc_hist.append(acc)
            if observable_fn is not None:
                obs_hist.append(observable_fn(x))
            logq_x = logq_fn(x)
            logt_x = log_target_fn(x)
            if verbose and (k % max(1, len(betas) // 10) == 0):
                print(f"  beta {beta:.2f}  ESS {100*ess:.1f}%  acc {acc:.2f}")

    w = torch.softmax(logw, dim=0)
    final_ess = ess_fraction(logw)
    return {
        "x": x, "log_weights": logw, "weights": w,
        "ess": final_ess, "ess_history": ess_hist, "acc_history": acc_hist,
        "obs_history": obs_hist, "plain_is_ess": plain_ess,
    }
