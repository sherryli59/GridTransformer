"""Sanity test for annealed SMC on a problem with a known answer.

Proposal q = N(0,1); target = N(2, 0.5). A good sampler must (a) move samples to
the target and (b) report a reweighted mean ~ 2.0 with ESS far above plain IS.

Run:  python -m liquid_coupling_flow.tests.test_smc
"""

from __future__ import annotations

import math

import torch

from liquid_coupling_flow.smc import anneal_smc, RWMetropolis


def test_smc_shifted_gaussian():
    torch.manual_seed(0)
    M = 4000
    mu, sig = 2.0, 0.5

    def logq(x):
        return (-0.5 * x ** 2 - 0.5 * math.log(2 * math.pi)).sum(-1)

    def logt(x):
        return (-0.5 * ((x - mu) / sig) ** 2 - math.log(sig) - 0.5 * math.log(2 * math.pi)).sum(-1)

    x0 = torch.randn(M, 1)
    kernel = RWMetropolis(step=0.3, n_steps=10, L=None)
    out = anneal_smc(x0, logq, logt, kernel, n_bridge=40, resample_thresh=0.5)

    mean_est = float((out["weights"] * out["x"][:, 0]).sum())
    print(f"plain-IS ESS {100*out['plain_is_ess']:.1f}%  ->  SMC ESS {100*out['ess']:.1f}%")
    print(f"reweighted mean {mean_est:.3f} (true 2.000)")

    assert abs(mean_est - mu) < 0.15, f"mean off: {mean_est}"
    assert out["ess"] > out["plain_is_ess"], "SMC should improve ESS over plain IS"
    assert out["ess"] > 0.2, f"SMC ESS too low: {out['ess']}"
    print("[PASS] SMC recovers the shifted-Gaussian mean with improved ESS")


if __name__ == "__main__":
    test_smc_shifted_gaussian()
    print("SMC TEST PASSED")
