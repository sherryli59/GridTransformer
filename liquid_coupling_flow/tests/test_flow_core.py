"""Correctness tests for the coupling-flow core (Phase A, affine transform).

Two properties that, if wrong, silently break everything downstream:
  1. invertibility:  f^{-1}(f(z)) == z, and log_prob(sample) == log_prob(eval)
  2. exact log-det:  the analytic triangular log-det == the true Jacobian log-det
     (checked against torch.autograd's Jacobian).

Runnable directly (no pytest dependency):  python -m liquid_coupling_flow.tests.test_flow_core
"""

from __future__ import annotations

import torch

from liquid_coupling_flow.base import DiagGaussian
from liquid_coupling_flow.transforms import AffineElementwise
from liquid_coupling_flow.coupling import CouplingLayer, alternating_masks
from liquid_coupling_flow.conditioner import MLPConditioner
from liquid_coupling_flow.flow import Flow


def build_affine_flow(dim: int = 2, n_layers: int = 6, seed: int = 0) -> Flow:
    torch.manual_seed(seed)
    base = DiagGaussian(dim)
    layers = []
    for m in alternating_masks(dim, n_layers):
        cond = MLPConditioner(dim, AffineElementwise.params_per_dim, hidden=64, n_layers=2)
        # De-identity-ise so the test exercises a non-trivial map (zero-init -> identity).
        torch.nn.init.normal_(cond.net[-1].weight, std=0.1)
        torch.nn.init.normal_(cond.net[-1].bias, std=0.1)
        layers.append(CouplingLayer(m, cond, AffineElementwise()))
    return Flow(base, layers)


def test_invertibility_and_logp():
    flow = build_affine_flow()
    x, logp_sample = flow.sample(128)
    logp_eval = flow.log_prob(x)
    err = (logp_sample - logp_eval).abs().max().item()
    assert err < 1e-4, f"log_prob(sample) != log_prob(eval): {err}"

    z = torch.randn(128, 2)
    xx = z
    for layer in flow.layers:
        xx, _ = layer.forward(xx)
    zz = xx
    for layer in reversed(flow.layers):
        zz, _ = layer.inverse(zz)
    rt = (z - zz).abs().max().item()
    assert rt < 1e-4, f"roundtrip error {rt}"
    print(f"[PASS] invertibility: roundtrip {rt:.2e}; logp(sample)==logp(eval) {err:.2e}")


def test_logdet_vs_autograd():
    flow = build_affine_flow()

    def f(z_single: torch.Tensor) -> torch.Tensor:
        x = z_single.unsqueeze(0)
        for layer in flow.layers:
            x, _ = layer.forward(x)
        return x.squeeze(0)

    z = torch.randn(6, 2)
    max_err = 0.0
    for i in range(z.shape[0]):
        J = torch.autograd.functional.jacobian(f, z[i])
        logdet_auto = torch.slogdet(J)[1]
        x = z[i].unsqueeze(0)
        ld = torch.zeros(1)
        for layer in flow.layers:
            x, d = layer.forward(x)
            ld = ld + d
        max_err = max(max_err, (ld.squeeze(0) - logdet_auto).abs().item())
    assert max_err < 1e-4, f"analytic log-det != autograd: {max_err}"
    print(f"[PASS] analytic log-det == autograd Jacobian (max err {max_err:.2e})")


if __name__ == "__main__":
    test_invertibility_and_logp()
    test_logdet_vs_autograd()
    print("ALL PHASE-A TESTS PASSED")
