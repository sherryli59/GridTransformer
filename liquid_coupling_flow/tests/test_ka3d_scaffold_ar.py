import torch

from liquid_coupling_flow.ka3d_scaffold_ar import (
    KA3DScaffoldAR,
    assignment_recovery,
    ball_squash,
    ball_unsquash,
    fixed_ball_scaffold,
    hungarian_order,
)


def _model():
    torch.manual_seed(0)
    return KA3DScaffoldAR(
        d_model=32, n_head=2, n_layer=1, knn=4, knn_bnd=4,
        flow_bins=6, flow_tail=4.0,
    ).eval()


def test_fixed_scaffold_is_deterministic_and_strictly_inside_ball():
    a = fixed_ball_scaffold(37, 2.4)
    b = fixed_ball_scaffold(37, 2.4)
    assert torch.equal(a, b)
    assert a.shape == (37, 3)
    assert torch.all(a.norm(dim=-1) < 2.4)


def test_hungarian_returns_slot_to_particle_permutation():
    anchors = fixed_ball_scaffold(12, 2.0)
    storage = torch.tensor([7, 2, 10, 0, 5, 9, 1, 11, 4, 8, 3, 6])
    x = anchors[storage]
    order = hungarian_order(x, anchors)
    assert torch.equal(x[order], anchors)


def test_ball_map_roundtrip_support_and_analytic_jacobian():
    y = torch.tensor([0.7, -1.1, 0.4], dtype=torch.float64, requires_grad=True)
    x, logdet = ball_squash(y, 2.0)
    y2, inv_logdet = ball_unsquash(x, 2.0)
    assert x.norm() < 2.0
    assert torch.allclose(y, y2, atol=1e-10)
    assert torch.allclose(logdet + inv_logdet, logdet.new_zeros(()), atol=1e-10)
    jac = torch.autograd.functional.jacobian(lambda z: ball_squash(z, 2.0)[0], y)
    assert torch.allclose(torch.linalg.slogdet(jac).logabsdet, logdet, atol=1e-10)


def test_sample_density_matches_fresh_labeled_evaluation_and_has_ball_support():
    model = _model()
    bnd = torch.tensor([[1.9, 0.0, 0.0], [-1.9, 0.0, 0.0]])
    s_bnd = torch.tensor([0, 1])
    gen = torch.Generator().manual_seed(9)
    x, s, lq = model.sample_pair(bnd, s_bnd, 5, 2, 2.0, return_logq=True, gen=gen)
    fresh = model.log_prob_pair(x, s, bnd, s_bnd, 2.0, preordered=True)
    assert torch.all(x.norm(dim=-1) < 2.0)
    assert torch.allclose(lq, fresh, atol=3e-4), (lq, fresh)


def test_unlabeled_training_likelihood_is_storage_permutation_invariant():
    model = _model()
    anchors = fixed_ball_scaffold(8, 2.0)
    x = anchors * 0.92
    s = torch.tensor([0, 0, 1, 0, 1, 0, 0, 0])
    empty_x, empty_s = x.new_empty(0, 3), s.new_empty(0)
    lp = model.log_prob_pair(x, s, empty_x, empty_s, 2.0)
    p = torch.tensor([5, 1, 7, 2, 0, 6, 3, 4])
    lp2 = model.log_prob_pair(x[p], s[p], empty_x, empty_s, 2.0)
    assert torch.allclose(lp, lp2, atol=1e-5)


def test_assignment_recovery_identity_on_anchor_labeled_state():
    anchors = fixed_ball_scaffold(15, 2.0)
    frac, whole = assignment_recovery(anchors, 2.0)
    assert frac == 1.0 and whole
