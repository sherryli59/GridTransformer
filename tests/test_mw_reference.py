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

def test_mc_warm_init():
    # init_cfgs warm-start: chains begin AT the given configs, so the first tracked U/N must sit
    # near the init configs' own energy (a low-energy jittered lattice), nowhere near the
    # random-uniform init level (core overlaps, U/N >> 0).
    from liquid_coupling_flow.mw.mw_energy import mw_energy_chunked
    N, B, L = 8, 4, 2.6
    g = torch.Generator().manual_seed(0)
    # jittered 2x2x2 lattice: one particle per cell, tiny jitter -> no core overlaps
    grid = torch.stack(torch.meshgrid(*[torch.arange(2)] * 3, indexing="ij"), -1).reshape(8, 3).float()
    init = ((grid + 0.5) * (L / 2))[None].expand(B, -1, -1).clone()
    init = torch.remainder(init + 0.05 * torch.randn(B, N, 3, generator=g), L)
    u_init = float((mw_energy_chunked(init, L) / N).mean())

    out = mc_run(N=N, L=L, beta=2.0, n_equil=4, n_collect=4, every=2, seed=0, B=B,
                 track_every=1, init_cfgs=init)
    first_tracked = out["traj"][0][1]
    assert abs(first_tracked - u_init) < 1.0, (first_tracked, u_init)   # started FROM init (~-1.5/N)
    u_rand = float((mw_energy_chunked(torch.rand(B, N, 3, generator=g) * L, L) / N).mean())
    assert abs(first_tracked - u_init) < 0.1 * abs(u_rand - u_init)     # nowhere near random-init level
