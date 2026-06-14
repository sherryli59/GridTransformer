"""Correctness tests for the augmented particle coupling flow.

  1. joint invertibility:  log_prob(sample) == log_prob recomputed
  2. exact joint log-det:  analytic sum-of-layer log-det == autograd Jacobian of the
     full (x_z, a_z) -> (x, a) map (the decisive check that locality+augmentation
     didn't break the triangular-Jacobian likelihood).

Float64 so we test the math. Run: python -m liquid_coupling_flow.tests.test_particles
"""

from __future__ import annotations

import torch

from liquid_coupling_flow.particles import build_particle_flow

DT = torch.float64


def _perturb(flow, scale=0.05):
    # Move conditioners off the zero-init so params actually depend on positions
    # (a realistic, well-conditioned regime — like an early-training flow). Larger
    # perturbations create pathologically steep splines whose float64 inverse is
    # ill-conditioned, which is a numerics artefact, not a flow-correctness issue.
    for p in flow.parameters():
        if p.dim() > 0:
            p.data = p.data + scale * torch.randn_like(p)


def test_joint_invertibility():
    torch.manual_seed(0)
    N, d, L = 6, 2, 6.0
    flow = build_particle_flow(N=N, d=d, L=L, n_layers=8, num_bins=8, cutoff=2.5).double()
    _perturb(flow)
    x, a, logp = flow.sample(16, dtype=DT)
    logp2 = flow.log_prob(x, a)
    err = (logp - logp2).abs().max().item()
    assert err < 1e-7, f"logp(sample) != logp(eval): {err}"
    print(f"[PASS] joint invertibility: logp(sample)==logp(eval) ({err:.2e})")


def test_joint_logdet_vs_autograd():
    torch.manual_seed(1)
    N, d, L = 3, 2, 6.0
    flow = build_particle_flow(N=N, d=d, L=L, n_layers=6, num_bins=8, cutoff=4.0).double()
    _perturb(flow)
    nd = N * d
    z = torch.rand(2 * nd, dtype=DT) * L

    def f(zf):
        x = zf[:nd].view(1, N, d)
        a = zf[nd:].view(1, N, d)
        for layer in flow.layers:
            x, a, _ = layer.forward(x, a)
        return torch.cat([x.reshape(-1), a.reshape(-1)])

    J = torch.autograd.functional.jacobian(f, z)
    logdet_auto = torch.slogdet(J)[1]

    x = z[:nd].view(1, N, d)
    a = z[nd:].view(1, N, d)
    ld = torch.zeros(1, dtype=DT)
    for layer in flow.layers:
        x, a, d_ = layer.forward(x, a)
        ld = ld + d_
    err = (ld.squeeze(0) - logdet_auto).abs().item()
    assert err < 1e-7, f"joint log-det != autograd: {err}"
    print(f"[PASS] joint log-det == autograd Jacobian ({err:.2e})")


if __name__ == "__main__":
    test_joint_invertibility()
    test_joint_logdet_vs_autograd()
    print("ALL PARTICLE-FLOW CORRECTNESS TESTS PASSED")
