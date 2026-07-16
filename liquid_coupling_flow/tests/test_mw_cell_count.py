import torch

from liquid_coupling_flow.mw.mw_cell_count import (
    OctreeCountModel, count_factor_count, integer_compositions,
)


def _model(seed=30):
    torch.manual_seed(seed)
    model = OctreeCountModel(hidden=24).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.07 * torch.randn_like(parameter))
    return model


def test_every_small_node_composition_normalizes_exactly():
    model = _model()
    for parent in range(7):
        total = 0.0
        for composition in integer_compositions(parent, 8):
            children = torch.tensor(composition, dtype=torch.long)
            total += float(model.split_log_prob(children, N=max(parent, 6)).exp())
        assert abs(total - 1.0) < 2e-11


def test_sample_score_round_trip_for_g2_and_g4():
    model = _model(31)
    gen = torch.Generator().manual_seed(32)
    for N, G in ((8, 2), (64, 2), (64, 4), (512, 4)):
        for _ in range(5):
            counts, sampled = model.sample_counts(N, G, gen=gen)
            scored = model.count_log_prob(counts, N, G)
            assert int(counts.sum()) == N
            assert counts.numel() == G ** 3
            assert torch.allclose(sampled, scored, atol=2e-11, rtol=0)


def test_full_support_includes_all_particles_in_any_g2_leaf():
    model = _model(33)
    N, G = 64, 2
    for leaf in range(G ** 3):
        counts = torch.zeros(G ** 3, dtype=torch.long)
        counts[leaf] = N
        assert torch.isfinite(model.count_log_prob(counts, N, G))


def test_empty_nodes_and_zero_particle_configuration_are_normalized():
    model = _model(34)
    counts, sampled = model.sample_counts(0, 4, gen=torch.Generator().manual_seed(35))
    assert torch.equal(counts, torch.zeros(64, dtype=torch.long))
    scored = model.count_log_prob(counts, 0, 4)
    assert sampled == 0 and scored == 0


def test_bad_count_vectors_have_negative_infinite_density():
    model = _model(36)
    negative = torch.zeros(8, dtype=torch.long)
    negative[0] = -1
    wrong_sum = torch.zeros(8, dtype=torch.long)
    wrong_sum[0] = 7
    assert torch.isneginf(model.count_log_prob(negative, 8, 2))
    assert torch.isneginf(model.count_log_prob(wrong_sum, 8, 2))


def test_count_log_prob_is_differentiable():
    model = _model(37)
    counts = torch.tensor([2, 0, 1, 0, 3, 0, 1, 1])
    loss = -model.count_log_prob(counts, N=8, G=2)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_fresh_count_tree_starts_near_equal_cell_means():
    model = OctreeCountModel()
    gen = torch.Generator().manual_seed(38)
    draws = torch.stack([model.sample_counts(64, 2, gen=gen)[0] for _ in range(512)])
    assert torch.allclose(draws.float().mean(0), torch.full((8,), 8.0), atol=0.55)
    assert count_factor_count(2) == 7
    assert count_factor_count(4) == 63
