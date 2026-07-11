import torch

from liquid_coupling_flow.ka3d_cavity_ar import (
    KA3DCavityAR,
    cavity_order,
    fibonacci_ball_scaffold,
)


def _tiny_model():
    torch.manual_seed(0)
    return KA3DCavityAR(
        d_model=32,
        n_head=2,
        n_layer=1,
        n_bins=16,
        arc_range=3.0,
        knn=4,
        knn_bnd=4,
    ).eval()


def test_sample_returns_exact_labeled_path_density():
    """The returned logq must score the actual generation order, not a re-sorted state."""
    model = _tiny_model()
    bnd = torch.tensor([[1.8, 0.0, 0.0], [-1.8, 0.0, 0.0]])
    s_bnd = torch.tensor([0, 1])
    torch.manual_seed(4)
    x, s, logq_sample = model.sample_pair(bnd, s_bnd, n_A=3, n_B=1, R=2.0, return_logq=True)
    logq_eval = model.log_prob_pair(x, s, bnd, s_bnd, R=2.0, preordered=True)
    assert logq_sample.ndim == 0
    assert torch.allclose(logq_sample, logq_eval, atol=2e-4), (logq_sample, logq_eval)


def test_unordered_data_likelihood_is_storage_permutation_invariant():
    model = _tiny_model()
    x = torch.tensor([
        [-0.8, -0.1, 0.2],
        [0.7, 0.2, -0.1],
        [0.0, 0.6, 0.3],
        [0.2, -0.7, -0.4],
    ])
    s = torch.tensor([0, 1, 0, 0])
    empty_x = torch.empty(0, 3)
    empty_s = torch.empty(0, dtype=torch.long)
    lp = model.log_prob_pair(x, s, empty_x, empty_s, R=2.0)
    p = torch.tensor([2, 0, 3, 1])
    lp_perm = model.log_prob_pair(x[p], s[p], empty_x, empty_s, R=2.0)
    assert torch.allclose(lp, lp_perm, atol=1e-5)


def test_slot_features_explicitly_encode_generation_progress_and_radius():
    scaffold = fibonacci_ball_scaffold(8, R=2.0)
    feat = KA3DCavityAR._slot_features(scaffold, torch.arange(8), 8, 2.0)
    assert feat.shape == (8, 3)
    assert feat[0, 0] == 0 and feat[-1, 0] == 1
    assert torch.all((feat[:, 1] >= 0) & (feat[:, 1] <= 1))
    assert torch.allclose(feat[:, 2], torch.full((8,), 0.8))


def test_canonical_order_is_a_permutation():
    x = torch.randn(17, 3)
    order = cavity_order(x, R=2.4)
    assert sorted(order.tolist()) == list(range(17))
