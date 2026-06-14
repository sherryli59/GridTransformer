"""Phase A proof-of-concept: an affine coupling flow learns a 2D toy by MLE.

Trains on `eight_gaussians` (the classic multimodal sanity target) and saves a
scatter of flow samples vs. target. Affine RealNVP is a *weak* transform for
sharp multimodal targets — this establishes the training loop works; Phase B
(splines) is expected to fit it more crisply.

Run:  python -m liquid_coupling_flow.demo_phaseA
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from liquid_coupling_flow.builders import build_affine_flow
from liquid_coupling_flow.toy_targets import eight_gaussians
from liquid_coupling_flow.training import train_flow

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def main():
    os.makedirs(ART, exist_ok=True)
    torch.manual_seed(0)

    flow = build_affine_flow(dim=2, n_layers=16, hidden=128)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"affine flow: 16 layers, {n_params} params")

    hist = train_flow(flow, lambda b: eight_gaussians(b), steps=4000, batch=512, lr=1e-3)

    with torch.no_grad():
        xs, _ = flow.sample(3000)
        xt = eight_gaussians(3000)
        # held-out NLL estimate
        nll = flow.forward_kld(eight_gaussians(5000)).item()
    print(f"final held-out NLL: {nll:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for a, data, title in [(ax[0], xt, "target"), (ax[1], xs, "flow samples")]:
        a.scatter(data[:, 0], data[:, 1], s=3, alpha=0.3)
        a.set_title(title)
        a.set_xlim(-3, 3)
        a.set_ylim(-3, 3)
        a.set_aspect("equal")
    fig.suptitle(f"Phase A — affine flow on eight_gaussians (NLL {nll:.3f})")
    fig.tight_layout()
    out = os.path.join(ART, "phaseA_eight_gaussians.png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
