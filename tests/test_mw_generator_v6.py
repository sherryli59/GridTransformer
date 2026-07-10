import torch

from liquid_coupling_flow.mw.mw_generator import mw_scaffold, wrap_pm
from liquid_coupling_flow.mw.mw_generator_v6 import MWGlobalSpline, train
from liquid_coupling_flow.mw.mw_energy import RHO_STAR


def _model(use_geo_feat=True, seed=10):
    torch.manual_seed(seed)
    return MWGlobalSpline(d_model=32, n_layers=1, n_heads=2, rail_k=4,
                          num_bins=8, bound=4.0, knn=6, use_geo_feat=use_geo_feat)


def test_sample_logprob_exact_every_draw():
    m = _model(); L = 3.7
    x, lq = m.sample(12, 8, L, gen=torch.Generator().manual_seed(5))
    scored = m.log_prob(x, L, preordered=True)
    assert torch.allclose(lq, scored, atol=2e-4), float((lq - scored).abs().max())


def test_sample_logprob_exact_perturbed():
    m = _model(); L = 3.7
    torch.manual_seed(6)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.03 * torch.randn_like(p))
    x, lq = m.sample(12, 8, L, gen=torch.Generator().manual_seed(7))
    scored = m.log_prob(x, L, preordered=True)
    assert torch.allclose(lq, scored, atol=3e-4), float((lq - scored).abs().max())


def test_transfer_chart_fixed_bound():
    m = _model(); L = 3.7
    # s = L/(2*bound), and |u| < bound must hold for ANY N (N-independent normalization).
    for N in (8, 27):
        x, _ = m.sample(4, N, L, gen=torch.Generator().manual_seed(N))
        t, _, _ = mw_scaffold(N, L, "cpu")
        s = L / (2 * m.bound)
        u = wrap_pm(x - t[None], L) / s
        assert bool((u.abs() < m.bound).all()), float(u.abs().max())
        assert abs(s - L / (2 * m.bound)) < 1e-12


def test_geo_feat_zero_init_matches_no_feat():
    m = _model(use_geo_feat=True); L = 3.7
    x = torch.rand(5, 8, 3, generator=torch.Generator().manual_seed(11)) * L
    lp_on = m.log_prob(x, L)
    m.use_geo_feat = False
    lp_off = m.log_prob(x, L)
    # Zero-initialized geo_feat_proj => the feature adds exactly nothing at init.
    assert torch.allclose(lp_on, lp_off, atol=1e-6), float((lp_on - lp_off).abs().max())
    # geo_feat_proj is created LAST (after the body), so a fresh no-feat model built
    # with the same seed has bit-identical body weights and must produce the same log_prob.
    ref = _model(use_geo_feat=False)
    assert torch.allclose(ref.log_prob(x, L), lp_off, atol=1e-6), \
        float((ref.log_prob(x, L) - lp_off).abs().max())


def test_train_writes_checkpoints(tmp_path):
    N, L = 8, (8 / RHO_STAR) ** (1 / 3)
    cfgs = torch.rand(32, N, 3, generator=torch.Generator().manual_seed(9)) * L
    bank = tmp_path / "bank.pt"
    torch.save({"cfgs": cfgs}, bank)
    result = train(steps=1, batch=2, val_every=1, out=str(tmp_path / "v6.pt"),
                   primary_thin=1, val_frac=0.5, art_path=str(bank), extra_banks=[],
                   device="cpu", d_model=16, n_layers=1, n_heads=2, rail_k=4,
                   num_bins=4, knn=4)
    for key in ("nll", "struct", "last"):
        ck = torch.load(result[key], map_location="cpu", weights_only=False)
        assert ck["architecture"] == "global_spline_v6"
        assert {"peak_err", "core_mass", "composite"} <= ck["struct"].keys()
        assert ck["use_geo_feat"] is True
