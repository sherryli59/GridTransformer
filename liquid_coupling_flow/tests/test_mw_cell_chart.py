import torch

from liquid_coupling_flow.mw.mw_cell_chart import (
    AnchoredCubeCat3Head,
    anchored_cube_forward,
    anchored_cube_inverse,
    clip_anchor,
)


def test_cube_chart_forward_inverse_at_branches_and_faces():
    dtype = torch.float64
    anchor = torch.tensor([0.2, 0.55, 0.83], dtype=dtype)
    u = torch.tensor(
        [[-1.0, -0.7, -1e-12], [0.0, 1e-12, 0.4], [0.999999, -0.1, 0.0]],
        dtype=dtype,
    )
    local, forward_ld = anchored_cube_forward(u, anchor, width=2.6)
    recovered, inverse_ld = anchored_cube_inverse(local, anchor, width=2.6)
    assert torch.allclose(recovered, u, atol=2e-12, rtol=0)
    assert torch.allclose(forward_ld + inverse_ld, torch.zeros_like(forward_ld), atol=2e-12)
    assert bool((local >= 0).all()) and bool((local < 2.6).all())


def test_cube_chart_logdet_matches_autograd():
    anchor = torch.tensor([0.21, 0.47, 0.78], dtype=torch.float64)
    for u in (
        torch.tensor([-0.4, -0.2, 0.7], dtype=torch.float64),
        torch.tensor([0.4, 0.2, -0.7], dtype=torch.float64),
    ):
        u.requires_grad_(True)
        local, analytic = anchored_cube_forward(u, anchor, width=2.3)
        jac = torch.autograd.functional.jacobian(
            lambda z: anchored_cube_forward(z, anchor, width=2.3)[0], u
        )
        measured = torch.logdet(jac)
        assert torch.allclose(measured, analytic, atol=2e-12, rtol=0)


def test_anchor_clipping_is_pinned_and_finite():
    anchor = torch.tensor([0.0, 0.4, 1.0], dtype=torch.float32)
    clipped = clip_anchor(anchor)
    assert bool((clipped > 0).all()) and bool((clipped < 1).all())
    u = torch.tensor([-0.5, 0.0, 0.5])
    local, logdet = anchored_cube_forward(u, anchor)
    assert torch.isfinite(local).all() and torch.isfinite(logdet)


def test_cat3_cube_sample_score_round_trip_with_perturbed_weights():
    torch.manual_seed(20)
    head = AnchoredCubeCat3Head(d_model=12, num_bins=24).double()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.add_(0.03 * torch.randn_like(parameter))
    context = torch.randn(128, 12, dtype=torch.float64)
    anchor = torch.sigmoid(torch.randn(128, 3, dtype=torch.float64))
    gen = torch.Generator().manual_seed(21)
    local, sampled = head.sample(context, anchor, width=2.6, gen=gen)
    scored = head.log_prob(context, local, anchor, width=2.6)
    assert torch.equal(local < 2.6, torch.ones_like(local, dtype=torch.bool))
    assert torch.allclose(sampled, scored, atol=2e-10, rtol=0)


def test_cat3_cube_density_numerically_integrates_to_one():
    torch.manual_seed(22)
    head = AnchoredCubeCat3Head(d_model=4, num_bins=8).double()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.add_(0.08 * torch.randn_like(parameter))
    n = 120_000
    context = torch.randn(1, 4, dtype=torch.float64).expand(n, -1)
    anchor = torch.tensor([0.19, 0.51, 0.86], dtype=torch.float64).expand(n, -1)
    points = torch.rand(n, 3, dtype=torch.float64) * 2.6
    density = head.log_prob(context, points, anchor, width=2.6).exp()
    integral = density.mean() * 2.6 ** 3
    assert abs(float(integral) - 1.0) < 0.025


def test_cat3_cube_rejects_points_outside_half_open_support():
    head = AnchoredCubeCat3Head(d_model=3, num_bins=4)
    context = torch.zeros(3, 3)
    anchor = torch.full((3, 3), 0.5)
    local = torch.tensor([[0.1, 0.2, 0.3], [-1e-5, 0.2, 0.3], [1.0, 0.2, 0.3]])
    score = head.log_prob(context, local, anchor, width=1.0)
    assert torch.isfinite(score[0])
    assert torch.isneginf(score[1:]).all()
