import os
import pytest
import torch

from liquid_coupling_flow.ka3d_pt_cavity import (
    assert_cavity_valid, build_ladder, equilibrate_cavity, two_arm_converged,
)


def test_geometry_guard_and_ladder():
    L = (512 / 1.2) ** (1 / 3)
    assert_cavity_valid(2.4, L)
    with pytest.raises(AssertionError):
        assert_cavity_valid(3.2, L)
    betas, lams = build_ladder(0.5, 6)
    assert betas.shape == lams.shape == (6,)
    assert abs(float(betas[0]) - 2.0) < 1e-6 and abs(float(lams[0]) - 1.0) < 1e-6
    assert torch.all(betas[1:] < betas[:-1]) and torch.all(lams[1:] < lams[:-1])


def test_two_arm_gate_uses_tail_means():
    assert two_arm_converged([1, .8, .52, .50, .49, .51], [0, .2, .47, .50, .52, .48], .05)
    assert not two_arm_converged([1, .9, .8, .75, .76], [0, .1, .2, .25, .24], .1)
    assert not two_arm_converged([1, .5], [0, .5], .1)


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("RUN_SLOW_GPU") != "1",
                    reason="set RUN_SLOW_GPU=1 for the manual GPU convergence smoke")
def test_valid_small_cavity_two_arms_meet_from_equilibrated_ref():
    d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt",
                   map_location="cuda", weights_only=False)
    x = d["x"][-1:].float().cuda(); s = d["s"][-1:].long().cuda(); L = float(d["L"])
    center = x[0, 0].clone(); dd = x - center; dd = dd - L * torch.round(dd / L)
    mob = dd.square().sum(-1) < 1.6 ** 2
    ref = equilibrate_cavity(x, s, mob, center, 1.6, L, T=1.0, n_rep=6,
                             n_sweep=4000, exch_every=50, init="ref", seed=1)
    rnd = equilibrate_cavity(x, s, mob, center, 1.6, L, T=1.0, n_rep=6,
                             n_sweep=4000, exch_every=50, init="random", seed=2)
    assert two_arm_converged(ref["q_c_traj"], rnd["q_c_traj"], q_tol=0.1)
