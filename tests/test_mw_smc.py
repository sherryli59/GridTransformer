import math, os, torch
from liquid_coupling_flow.mw.mw_base import UniformBase, GeneratorBase
from liquid_coupling_flow.mw.mw_smc import ess, next_lambda, smc_run, mutation_sweeps
from liquid_coupling_flow.mw.mw_energy import mw_energy, RHO_STAR

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

def test_g3_smoke(tmp_path):
    # tiny N=8 unit, redirected save/plot paths so this can never touch the real mw_g3.pt or
    # collide with the N=64 G2 gate running concurrently on the GPU. ess_target=0.8 (not the
    # certified 0.95) keeps the rung count -- and wall-clock -- small for a smoke test.
    from liquid_coupling_flow.mw.mw_gates import g3
    art_path = str(tmp_path / "mw_g3_smoke.pt")
    plot_path = str(tmp_path / "mw_g3_scaling_smoke.png")
    out = g3(Ns=(8,), seeds=(0,), B=64, ess_target=0.8, step=0.0666,
             art_path=art_path, plot_path=plot_path)
    assert len(out["units"]) == 1
    u = next(iter(out["units"].values()))
    assert u["T"] > 0 and math.isfinite(u["evals"]) and math.isfinite(u["logZ"])
    assert os.path.exists(art_path)
    assert os.path.exists(plot_path)

def test_generator_base_interface():
    # GeneratorBase wraps a v10 model for mw_smc. smc_run has no max_rungs bound and a random-init
    # model at the hard ambient beta could stall or run long, so this smokes ONLY the base<->smc
    # contract (the explicitly sanctioned fallback): sample shapes, finite preordered log_q, exactness
    # on a freshly sampled (canonical) config, and ONE non-const-base mutation sweep (the LOGQ_CONST
    # =False path -- base.log_q is called per site-move, the whole point of the generator base).
    from liquid_coupling_flow.mw.mw_generator_v10 import MWV4ToroidalResidual
    torch.manual_seed(0)
    N = 64
    L = (N / RHO_STAR) ** (1.0 / 3.0)
    model = MWV4ToroidalResidual(d_model=24, n_layers=1, n_heads=2, n_mix=6, rail_k=4,
                                 num_bins=7).eval()
    base = GeneratorBase(model, N, L)
    assert base.LOGQ_CONST is False
    x = base.sample(16, torch.Generator().manual_seed(1))
    assert x.shape == (16, N, 3)
    lq = base.log_q(x)
    assert lq.shape == (16,) and torch.isfinite(lq).all()
    # exactness: sample()'s own log q equals log_q(preordered=True) on the freshly (canonically) drawn x
    xs, logq = model.sample(16, N, L, gen=torch.Generator().manual_seed(2))
    assert torch.allclose(base.log_q(xs), model.log_prob(xs, L, preordered=True), atol=3e-4)
    # one mutation sweep through the non-const base must stay finite and report a valid acceptance rate
    x2, U, lq2, info = mutation_sweeps(x, base, lam=0.5, beta=1.0, L=L, n_sweeps=1, step=0.08,
                                       gen=torch.Generator().manual_seed(3))
    assert torch.isfinite(U).all() and torch.isfinite(lq2).all()
    assert 0.0 <= info["acc"] <= 1.0 and info["evals"] == N * 16


def test_g3_protocol_mismatch(tmp_path):
    # protocol purity: a banked unit may only be reused under the IDENTICAL protocol -- a second
    # call against the same art_path with a different ess_target must raise loudly (ValueError),
    # never silently splice heterogeneous-protocol T values into the baseline fit. B=32 keeps the
    # single real unit cheap; the second call must fail BEFORE running any SMC.
    import pytest
    from liquid_coupling_flow.mw.mw_gates import g3
    art_path = str(tmp_path / "mw_g3_pp.pt")
    plot_path = str(tmp_path / "mw_g3_pp.png")
    g3(Ns=(8,), seeds=(0,), B=32, ess_target=0.8, step=0.0666,
       art_path=art_path, plot_path=plot_path)
    with pytest.raises(ValueError, match="DIFFERENT protocol"):
        g3(Ns=(8,), seeds=(0,), B=32, ess_target=0.9, step=0.0666,
           art_path=art_path, plot_path=plot_path)
