import torch
from liquid_coupling_flow.mw.mw_reference import mc_run, g_r

def test_smoke_and_proofs():
    out = mc_run(N=8, L=2.6, beta=2.0, n_equil=400, n_collect=200, every=4, seed=0)
    assert out["cfgs"].shape[1:] == (8, 3) and out["cfgs"].shape[0] >= 100
    assert 0.2 < out["acc"] < 0.6                       # adaptive step landed
    assert "flat_budget" in out and "coll_drift" in out

def test_gr_normalization():
    g = torch.Generator().manual_seed(0)
    cfgs = torch.rand(64, 32, 3, generator=g) * 3.0     # ideal gas -> g(r) ~ 1
    r, gr = g_r(cfgs, 3.0)
    assert abs(float(gr[(r > 0.8) & (r < 1.4)].mean()) - 1.0) < 0.1
