"""Phase C+D: circular-spline flow on a periodic torus + ESS via importance weights.

This is the keystone proof-of-concept for the project thesis:
  * a flow on a PERIODIC domain (circular splines) can fit a multimodal target, AND
  * because its log p is EXACT, importance weights w = p_target/p_flow give a high
    effective sample size (ESS) -> the reweighting/exactness route works.

TorusMixture has an exact normalised density, so the measured ESS is meaningful:
ESS -> 100% iff the flow matches the target.

Run:  python -m liquid_coupling_flow.demo_phaseC
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from liquid_coupling_flow.builders import build_circular_spline_flow
from liquid_coupling_flow.toy_targets import TorusMixture
from liquid_coupling_flow.training import train_flow

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def importance_ess(flow, target, n=20000, L=1.0):
    with torch.no_grad():
        x, logp_flow = flow.sample(n)
        x = torch.remainder(x, L)
        logp_tgt = target.log_prob(x)
        logw = logp_tgt - logp_flow
        logw = logw - torch.logsumexp(logw, dim=0)  # normalise
        w = logw.exp()
        ess = 1.0 / (w ** 2).sum()
        return float(ess / n), float(logp_flow.mean()), x


def main():
    os.makedirs(ART, exist_ok=True)
    torch.manual_seed(0)
    L = 1.0
    target = TorusMixture(L=L, std=0.06, n_modes=5, seed=3)

    flow = build_circular_spline_flow(dim=2, n_layers=10, L=L, num_bins=12, hidden=128)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"circular-spline flow: 10 layers, {n_params} params")

    train_flow(flow, lambda b: target.sample(b), steps=4000, batch=512, lr=5e-4)

    ess_frac, mean_logp, xs = importance_ess(flow, target, n=20000, L=L)
    # Cross-entropy of target samples under flow (lower = better fit)
    with torch.no_grad():
        nll = -flow.log_prob(target.sample(5000)).mean().item()
    print(f"final NLL (target samples under flow): {nll:.4f}")
    print(f"importance-sampling ESS: {100*ess_frac:.1f}%   (100% = perfect match)")

    xt = target.sample(20000)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for a, data, title in [(ax[0], xt, "target (torus)"), (ax[1], xs, "flow samples")]:
        a.scatter(data[:, 0], data[:, 1], s=2, alpha=0.25)
        a.set_title(title)
        a.set_xlim(0, L)
        a.set_ylim(0, L)
        a.set_aspect("equal")
    fig.suptitle(f"Phase C/D — circular-spline torus flow (ESS {100*ess_frac:.1f}%, NLL {nll:.3f})")
    fig.tight_layout()
    out = os.path.join(ART, "phaseC_torus_flow.png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
