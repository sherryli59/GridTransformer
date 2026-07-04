import torch
from liquid_coupling_flow.ka_local_smc import ess, next_beta, _resample, _canonicalize

def test_ess_bounds_and_uniform():
    lw = torch.zeros(64)
    assert abs(ess(lw) - 64) < 1e-4                       # uniform weights -> ESS = B
    lw2 = torch.full((64,), -1e9); lw2[0] = 0.0
    assert ess(lw2) < 1.01                                # degenerate -> ESS ~ 1

def test_next_beta_monotone_and_bounded():
    torch.manual_seed(0)
    logw = torch.zeros(128); U = torch.randn(128) * 10 - 300
    b = next_beta(logw, U, 0.8, 2.0, ess_target=0.6)
    assert 0.8 < b <= 2.0
    b2 = next_beta(logw, U, 0.8, 2.0, ess_target=0.9)     # stricter target -> smaller step
    assert b2 <= b + 1e-9

def test_resample_preserves_counts_and_shapes():
    torch.manual_seed(0)
    pos = torch.rand(16, 100, 2)
    # fixed count (35) per row via per-row permutation, THEN sort -> canonical rows all identical
    # (matches the real invariant: smc_run asserts (s == s[0:1]).all() after _canonicalize).
    s = torch.stack([torch.randperm(100) for _ in range(16)]) < 35
    s = torch.sort(s.long(), dim=1).values                  # canonical
    logw = torch.randn(16)
    p2, s2, lw2 = _resample(pos, s, logw)
    assert p2.shape == pos.shape and torch.equal(lw2, torch.zeros(16))
    assert (s2.sum(1) == s[0].sum()).all()                  # counts uniform, preserved by resampling

def test_smoke_two_rungs_gpu():
    """2-rung dry run at tiny B: weights finite, ESS sane, species counts invariant."""
    if not torch.cuda.is_available():
        return
    from liquid_coupling_flow.ka_local_smc import smc_run
    out = smc_run("arm0", N=100, B=16, ess_target=0.5, max_rungs=2, n_mut=1, n_disp=5, seed=0)
    assert all(torch.isfinite(torch.tensor(h["ess"])) for h in out["history"])
    assert torch.isfinite(out["logw"]).all()
    assert (out["s"].sum(1) == out["s"][0].sum()).all()
