import math
import torch

from liquid_coupling_flow.mw.mw_generator import mw_scaffold, wrap_pm
from liquid_coupling_flow.mw.mw_generator_v5 import CircularSpline3Head, MWLocalFrameAR, train
from liquid_coupling_flow.mw.mw_energy import RHO_STAR


def _model():
    torch.manual_seed(10)
    return MWLocalFrameAR(knn=4, d_model=32, n_layers=1, n_heads=2, num_bins=8)


def test_circular_head_is_normalized_and_bounded():
    head = CircularSpline3Head(8, num_bins=8, bound=4.0)
    torch.manual_seed(3)
    with torch.no_grad():
        for p in head.parameters():
            p.add_(0.04 * torch.randn_like(p))
    h = torch.randn(1, 8).expand(8001, -1)
    a = torch.linspace(-4, 4, 8002)[:-1, None]
    lp = head.logp_a(h, a).squeeze()
    assert abs(float(lp.exp().mean() * 8.0) - 1.0) < 2e-3
    u, _ = head.sample(torch.randn(1024, 8), gen=torch.Generator().manual_seed(4))
    assert bool((u >= -4).all() and (u < 4).all())


def test_prefix_origins_vectorized_match_sequential_definition():
    m = _model(); B, N, L = 3, 8, 3.7
    x = torch.rand(B, N, 3, generator=torch.Generator().manual_seed(2)) * L
    anchors, _, _ = mw_scaffold(N, L, "cpu")
    got = m._origins(x, anchors, L)
    want = []
    for j in range(N):
        if j == 0:
            o = anchors[j].expand(B, 3)
        else:
            d = wrap_pm(x[:, :j] - anchors[j], L)
            w = torch.exp(-d.square().sum(-1) / (2 * m.sigma_origin ** 2))
            o = torch.remainder(anchors[j] + (w[..., None] * d).sum(1) / w.sum(1, keepdim=True), L)
        want.append(o)
    assert torch.allclose(got, torch.stack(want, 1), atol=1e-6)


def test_sample_logprob_exact_every_draw():
    m = _model(); L = 3.7
    x, lq = m.sample(12, 8, L, gen=torch.Generator().manual_seed(5))
    scored = m.log_prob(x, L, preordered=True)
    assert torch.allclose(lq, scored, atol=2e-4), float((lq - scored).abs().max())


def test_sample_logprob_exact_with_perturbed_weights():
    m = _model(); L = 3.7
    torch.manual_seed(6)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.03 * torch.randn_like(p))
    x, lq = m.sample(12, 8, L, gen=torch.Generator().manual_seed(7))
    scored = m.log_prob(x, L, preordered=True)
    assert torch.allclose(lq, scored, atol=3e-4), float((lq - scored).abs().max())


def test_geometry_frames_and_features_are_finite():
    m = _model(); B, N, L = 2, 8, 3.7
    x = torch.rand(B, N, 3, generator=torch.Generator().manual_seed(8)) * L
    anchors, _, _ = mw_scaffold(N, L, "cpu")
    origins, frames, rel, valid = m._geometry(x, anchors, L)
    eye = torch.eye(3).expand(B, N, 3, 3)
    assert torch.allclose(frames @ frames.transpose(-1, -2), eye, atol=1e-5)
    feat = m._token_features(frames, rel, valid)
    assert origins.shape == (B, N, 3) and torch.isfinite(feat).all()
    assert feat.shape[-1] == 16


def test_output_chart_covers_torus_without_rotated_cube_aliasing():
    m = _model(); L = 3.7
    s = L / (2 * m.tail_bound)
    # Every global-chart point is a unique representative except the measure-zero seam.
    u = torch.tensor([[-3.9, 3.8, 2.7], [3.5, -2.1, -3.7]])
    origin = torch.tensor([[0.2, 3.5, 1.0], [3.6, 0.1, 2.0]])
    x = torch.remainder(origin + s * u, L)
    recovered = wrap_pm(x - origin, L) / s
    assert torch.allclose(u, recovered, atol=1e-6)


def test_train_writes_independent_nll_and_structure_checkpoints(tmp_path):
    N, L = 8, (8 / RHO_STAR) ** (1 / 3)
    cfgs = torch.rand(32, N, 3, generator=torch.Generator().manual_seed(9)) * L
    bank = tmp_path / "bank.pt"
    torch.save({"cfgs": cfgs}, bank)
    result = train(steps=1, batch=2, val_every=1, out=str(tmp_path / "v5.pt"),
                   primary_thin=1, val_frac=0.5, art_path=str(bank), extra_banks=[],
                   device="cpu", knn=4, d_model=16, n_layers=1, n_heads=2, num_bins=4)
    for key in ("nll", "struct", "last"):
        ck = torch.load(result[key], map_location="cpu", weights_only=False)
        assert ck["architecture"] == "local_origin_circular_spline_v5a_exact"
        assert {"peak_err", "core_mass", "composite"} <= ck["struct"].keys()
