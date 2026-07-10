import math, torch
from liquid_coupling_flow.mw.mw_base import UniformBase
from liquid_coupling_flow.mw.mw_smc import ess, next_lambda, smc_run, mutation_sweeps
from liquid_coupling_flow.mw.mw_energy import mw_energy

def test_ess_bounds():
    assert abs(ess(torch.zeros(100)) - 100.0) < 1e-6
    w = torch.full((100,), -1e9); w[0] = 0.0
    assert ess(w) < 1.01

def test_next_lambda_monotone_and_floor():
    g = torch.Generator().manual_seed(0)
    phi = -torch.rand(256, generator=g) * 50            # heterogeneous -> partial step
    lam = next_lambda(torch.zeros(256), phi, 0.0, 0.6, 256)
    assert 1e-4 <= lam <= 1.0
    assert ess(lam * phi) >= 0.6 * 256 * 0.99           # lands at/above target

def test_uniform_base_reduction():
    # with a LOGQ_CONST base, log q0 must NOT be evaluated per move — at most once (the final bookkeeping
    # call). A per-move call count of N*B would betray the reduction failing.
    class Spy(UniformBase):
        calls = 0
        def log_q(self, x):
            Spy.calls += 1
            return super().log_q(x)
    b = Spy(8, 2.6)
    g = torch.Generator().manual_seed(0)
    x = b.sample(16, g)
    x2, U, lq, info = mutation_sweeps(x, b, lam=0.5, beta=2.0, L=2.6, n_sweeps=2, step=0.1, gen=g)
    assert torch.isfinite(U).all() and Spy.calls <= 1

def test_mutation_stationarity_tiny():
    # N=2, L=4, beta=2, lam=1: long mutation-only chain must reproduce the bedrock <U> (G1 value).
    from liquid_coupling_flow.mw.mw_bedrock import quad_N2
    Uq, err = quad_N2(4.0, 2.0, 64)
    b = UniformBase(2, 4.0); g = torch.Generator().manual_seed(1)
    x = b.sample(64, g)
    us = []
    for it in range(600):
        x, U, _, _ = mutation_sweeps(x, b, 1.0, 2.0, 4.0, n_sweeps=5, step=0.35, gen=g)
        if it > 100: us.append(U.mean().item())
    u = sum(us) / len(us)
    assert abs(u - Uq) < max(0.02, 5 * err), f"mutation kernel off: {u} vs bedrock {Uq}"

def test_smc_runs_and_saves(tmp_path):
    b = UniformBase(8, 2.6)
    out = smc_run(b, 8, 2.6, beta=2.0, B=64, n_sweeps=2, step=0.2, seed=0, save_tag="_smoke")
    # plan-bug fix: the brief's line was torch.isfinite(out["logZ"] * 1.0), but logZ is a python
    # float per the contract and torch.isfinite requires a Tensor — math.isfinite is the right check.
    assert out["history"][-1]["lam"] == 1.0 and math.isfinite(out["logZ"])
    assert out["evals"] > 0

def test_final_sweeps_bookkeeping():
    # the lam=1 finisher must (a) be bookkept as a final history entry, (b) cost extra evals, and
    # (c) leave logZ IDENTICAL to the finisher-less run — the exactness property: a pi_1-invariant
    # kernel appended after the ladder must never touch the weight path.
    out5 = smc_run(UniformBase(8, 2.6), 8, 2.6, beta=2.0, B=32, n_sweeps=2, step=0.2, seed=0,
                   save_tag="_fsbk5", final_sweeps=5)
    out0 = smc_run(UniformBase(8, 2.6), 8, 2.6, beta=2.0, B=32, n_sweeps=2, step=0.2, seed=0,
                   save_tag="_fsbk0", final_sweeps=0)
    last = out5["history"][-1]
    assert last["lam"] == 1.0 and last.get("final") is True
    assert not out0["history"][-1].get("final", False)
    assert out5["evals"] > out0["evals"]
    assert out5["logZ"] == out0["logZ"]

def test_g2_smoke():
    # tiny budgets -> not a PASS/FAIL exactness check (both arms are far too short to converge),
    # just proves g2() runs end-to-end and returns the three non-vacuous metric entries.
    # n_sweeps=3/final_sweeps=0/null_M=20 keep it fast (real-gate defaults are the controller's).
    from liquid_coupling_flow.mw.mw_gates import g2
    out = g2(8, B=64, n_ref_equil=300, n_ref_collect=200, save_tag="_g2smoke",
             n_sweeps=3, final_sweeps=0, null_M=20)
    for key in ("mean_U_per_N", "tv_U_per_N", "g_r"):
        assert key in out
    assert math.isfinite(out["mean_U_per_N"]["diff"])
    assert math.isfinite(out["mean_U_per_N"]["tol"])
    assert math.isfinite(out["tv_U_per_N"]["tv"])
    assert math.isfinite(out["g_r"]["max_dg"])
    # null calibration present: all M values of both distributional metrics (full-data rule)
    assert out["null"]["tv"].shape == (20,) and torch.isfinite(out["null"]["tv"]).all()
    assert out["null"]["max_dg"].shape == (20,) and torch.isfinite(out["null"]["max_dg"]).all()
    assert math.isfinite(out["tv_U_per_N"]["null95"]) and math.isfinite(out["g_r"]["null95"])
