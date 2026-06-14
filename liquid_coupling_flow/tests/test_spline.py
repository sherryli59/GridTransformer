"""Correctness tests for the rational-quadratic spline transforms.

Same two properties as the core test, for both spline variants:
  1. invertibility:  inverse(forward(x)) == x   (on R for RQS, on the circle for circular)
  2. exact log-det:  analytic forward log-det == autograd d(forward)/dx

Run in float64 so we test the MATH, not float32 rounding. (In float32 the spline
inverse is accurate to ~3e-4, which is fine for training/sampling; reweighting that
needs tight likelihoods can run the flow in double.)

Run:  python -m liquid_coupling_flow.tests.test_spline
"""

from __future__ import annotations

import torch

from liquid_coupling_flow.transforms_spline import (
    RQSplineElementwise,
    CircularRQSplineElementwise,
)

DT = torch.float64


def _random_params(M, P, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(M, 1, P, generator=g, dtype=DT)  # [M, D=1, P]


def test_rqs_unconstrained():
    K = 8
    tr = RQSplineElementwise(num_bins=K, tail_bound=5.0)
    params = _random_params(256, tr.params_per_dim, seed=1)
    x = (torch.rand(256, 1, dtype=DT) * 8 - 4)  # inside [-4,4]: exercise the spline region

    y, ld_f = tr.forward(x, params)
    x2, ld_i = tr.inverse(y, params)
    rt = (x - x2).abs().max().item()
    assert rt < 1e-8, f"RQS roundtrip error {rt}"
    assert (ld_f + ld_i).abs().max().item() < 1e-8, "fwd/inv logdet not negatives"

    max_err = 0.0
    for i in range(8):
        xi = x[i, 0].clone().requires_grad_(True)
        y_i = tr.forward(xi.view(1, 1), params[i:i + 1])[0].sum()
        dydx = torch.autograd.grad(y_i, xi)[0]
        ld_analytic = tr.forward(xi.view(1, 1), params[i:i + 1])[1].view(())
        max_err = max(max_err, (ld_analytic - torch.log(dydx.abs())).abs().item())
    assert max_err < 1e-8, f"RQS log-det vs autograd {max_err}"
    print(f"[PASS] unconstrained RQS (float64): roundtrip {rt:.2e}, log-det err {max_err:.2e}")


def test_circular_rqs():
    K = 8
    L = 1.7
    tr = CircularRQSplineElementwise(num_bins=K, L=L)
    params = _random_params(256, tr.params_per_dim, seed=2)
    x = torch.rand(256, 1, dtype=DT) * L

    y, ld_f = tr.forward(x, params)
    assert (y >= -1e-8).all() and (y <= L + 1e-8).all(), "circular output left [0,L]"
    x2, ld_i = tr.inverse(y, params)
    rt = (torch.remainder(x - x2 + L / 2, L) - L / 2).abs().max().item()
    assert rt < 1e-8, f"circular roundtrip error {rt}"

    max_err = 0.0
    for i in range(8):
        xi = x[i, 0].clone().requires_grad_(True)
        y_i = tr.forward(xi.view(1, 1), params[i:i + 1])[0].sum()
        dydx = torch.autograd.grad(y_i, xi)[0]
        ld_analytic = tr.forward(xi.view(1, 1), params[i:i + 1])[1].view(())
        max_err = max(max_err, (ld_analytic - torch.log(dydx.abs())).abs().item())
    assert max_err < 1e-8, f"circular RQS log-det vs autograd {max_err}"
    print(f"[PASS] circular RQS (float64): roundtrip {rt:.2e}, log-det err {max_err:.2e}")


if __name__ == "__main__":
    test_rqs_unconstrained()
    test_circular_rqs()
    print("ALL SPLINE TESTS PASSED")
