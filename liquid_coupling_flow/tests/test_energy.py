"""Tests for the periodic LJ energy: analytic two-body value, translation
invariance, and the minimum at r = 2^(1/6).

Run:  python -m liquid_coupling_flow.tests.test_energy
"""

from __future__ import annotations

import torch

from liquid_coupling_flow.energy import lj_energy

DT = torch.float64


def test_two_body_value():
    L = 10.0
    r = 1.3
    x = torch.tensor([[[0.0, 0.0, 0.0], [r, 0.0, 0.0]]], dtype=DT)
    e = lj_energy(x, L).item()
    expected = 4.0 * (r ** -12 - r ** -6)
    assert abs(e - expected) < 1e-9, f"{e} vs {expected}"
    print(f"[PASS] two-body LJ energy matches analytic ({e:.6f})")


def test_translation_invariance():
    # Use a jittered lattice (no near-overlaps) so the stiff r^-12 term doesn't
    # amplify float round-off; then translation invariance must hold ~machine eps.
    L = 6.0
    torch.manual_seed(0)
    g = torch.arange(3, dtype=DT)
    grid = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), dim=-1).reshape(-1, 3)
    x = ((grid * (L / 3) + 0.1 * torch.randn(grid.shape[0], 3)) % L).unsqueeze(0)
    e0 = lj_energy(x, L)
    shift = torch.rand(1, 1, 3, dtype=DT) * L
    e1 = lj_energy(torch.remainder(x + shift, L), L)
    rel = ((e0 - e1).abs() / e0.abs().clamp_min(1e-8)).max().item()
    assert rel < 1e-10, f"translation invariance broken: rel {rel}"
    print(f"[PASS] translation invariance on jittered lattice (rel err {rel:.2e})")


def test_minimum_location():
    L = 10.0
    rmin = 2.0 ** (1.0 / 6.0)
    x = torch.tensor([[[0.0, 0, 0], [rmin, 0, 0]]], dtype=DT)
    e = lj_energy(x, L).item()
    assert abs(e - (-1.0)) < 1e-9, f"LJ min should be -1, got {e}"
    # numerically a true minimum: neighbours are higher
    for dr in (-0.02, 0.02):
        xx = torch.tensor([[[0.0, 0, 0], [rmin + dr, 0, 0]]], dtype=DT)
        assert lj_energy(xx, L).item() > e
    print(f"[PASS] LJ minimum -1.0 at r=2^(1/6) ({rmin:.4f})")


if __name__ == "__main__":
    test_two_body_value()
    test_translation_invariance()
    test_minimum_location()
    print("ALL ENERGY TESTS PASSED")
