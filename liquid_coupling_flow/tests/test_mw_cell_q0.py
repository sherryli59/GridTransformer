import torch

from liquid_coupling_flow.mw.mw_cell_geom import cubic_orientations
from liquid_coupling_flow.mw.mw_cell_q0 import AugmentedCellState, MWCellQ0
from liquid_coupling_flow.mw.mw_cell_transformer import (
    DEFAULT_SELECTED_SCAFFOLD,
    warm_start_cell_from_scaffold,
)


def _model(seed=40):
    torch.manual_seed(seed)
    model = MWCellQ0(d_model=24, count_hidden=20, cat_bins=16, knn=5).double().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.025 * torch.randn_like(parameter))
    return model


def _identity_orientation_index(dtype=torch.float64):
    orientations = cubic_orientations(dtype=dtype)
    matches = (orientations == torch.eye(3, dtype=dtype)).all((1, 2)).nonzero().flatten()
    assert matches.numel() == 1
    return int(matches[0])


def test_joint_sample_score_round_trip_g2_and_g4():
    model = _model()
    gen = torch.Generator().manual_seed(41)
    for N, G, L in ((8, 2, 5.19531), (20, 4, 10.39062)):
        state, sampled = model.sample(2, N, L, G, gen=gen)
        scored = model.log_prob(state, L, G)
        assert torch.allclose(sampled, scored, atol=3e-9, rtol=0)
        assert state.debug_counts.shape == (2, G ** 3)
        assert torch.equal(state.debug_counts.sum(1), torch.full((2,), N))


def test_joint_sample_score_is_stable_in_production_float32():
    torch.manual_seed(49)
    model = MWCellQ0(d_model=24, count_hidden=20, cat_bins=16, knn=5).eval()
    state, sampled = model.sample(
        3, 20, 10.39062, 4, gen=torch.Generator().manual_seed(50)
    )
    scored = model.log_prob(state, 10.39062, 4)
    assert float((sampled - scored).abs().max()) < 2e-4


def test_joint_row_permutation_invariance_and_debug_metadata_is_untrusted():
    model = _model(42)
    gen = torch.Generator().manual_seed(43)
    state, _ = model.sample(2, 12, 5.19531, 2, gen=gen)
    original = model.log_prob(state, 5.19531, 2)
    x, priorities = [], []
    for b in range(2):
        permutation = torch.randperm(12, generator=gen)
        x.append(state.x[b, permutation])
        priorities.append(state.priorities[b, permutation])
    permuted = AugmentedCellState(
        torch.stack(x), torch.stack(priorities), state.shift.clone(),
        state.orientation_index.clone(), state.color_order.clone(),
        debug_counts=torch.full_like(state.debug_counts, -999),
    )
    assert torch.allclose(model.log_prob(permuted, 5.19531, 2), original, atol=2e-10, rtol=0)


def test_same_stage_other_cell_never_enters_target_cell_factor():
    model = _model(44)
    G, L, h, N = 4, 10.39062, 10.39062 / 4, 64
    ids = torch.arange(N)
    i = torch.div(ids, G * G, rounding_mode="floor")
    rem = ids - i * G * G
    j = torch.div(rem, G, rounding_mode="floor")
    k = rem - j * G
    x = (torch.stack((i, j, k), -1).double() + 0.5) * h
    priorities = (torch.arange(N, dtype=torch.float64) + 0.5) / N
    state = AugmentedCellState(
        x=x[None],
        priorities=priorities[None],
        shift=torch.zeros(1, 3, dtype=torch.float64),
        orientation_index=torch.tensor([_identity_orientation_index()]),
        color_order=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]]),
    )
    # Cell (0,0,0) and cell (2,0,0) have physical color zero.
    before = model.cell_position_log_prob(state, 0, 0, L, G)
    changed = AugmentedCellState(
        state.x.clone(), state.priorities.clone(), state.shift,
        state.orientation_index, state.color_order,
    )
    changed.x[0, 32, 1] += 0.2 * h
    after = model.cell_position_log_prob(changed, 0, 0, L, G)
    assert torch.equal(before, after)
    assert not torch.equal(
        model.cell_position_log_prob(state, 0, 32, L, G),
        model.cell_position_log_prob(changed, 0, 32, L, G),
    )


def test_all_particles_in_one_cell_has_finite_density_without_k_cap():
    model = _model(45)
    gen = torch.Generator().manual_seed(46)
    N, G, L, h = 24, 2, 5.19531, 5.19531 / 2
    x = torch.rand(N, 3, generator=gen, dtype=torch.float64) * (0.9 * h)
    priorities = torch.rand(N, generator=gen, dtype=torch.float64)
    state = AugmentedCellState(
        x=x[None], priorities=priorities[None],
        shift=torch.zeros(1, 3, dtype=torch.float64),
        orientation_index=torch.tensor([_identity_orientation_index()]),
        color_order=torch.arange(8)[None],
    )
    assert torch.isfinite(model.log_prob(state, L, G)).all()


def test_invalid_priority_support_is_rejected():
    model = _model(47)
    state, _ = model.sample(1, 8, 5.19531, 2, gen=torch.Generator().manual_seed(48))
    state.priorities[0, 0] = -0.1
    assert torch.isneginf(model.log_prob(state, 5.19531, 2)).all()


def test_cell_terms_recompose_joint_density_and_balanced_loss_is_finite():
    model = _model(52)
    state, joint = model.sample(2, 12, 5.19531, 2, gen=torch.Generator().manual_seed(53))
    recomposed = []
    for b in range(2):
        base, factors = model._score_one_terms(
            state.x[b], state.priorities[b], state.shift[b],
            int(state.orientation_index[b]), state.color_order[b], 5.19531, 2,
        )
        recomposed.append(base + sum((value for _, _, value in factors), base.new_zeros(())))
    assert torch.allclose(torch.stack(recomposed), joint, atol=3e-9, rtol=0)
    loss, diagnostics = model.balanced_nll(state, 5.19531, 2)
    assert torch.isfinite(loss)
    assert diagnostics["position_nll_by_K"]


def test_selected_july15_scaffold_warm_start_is_bit_exact():
    model = MWCellQ0(
        d_model=128, count_hidden=20, cat_bins=64, knn=20,
        n_head=4, n_layer=4, knn_bnd=20, knn_pot=24,
    ).eval()
    before_count = model.context_encoder.count_embed[-1].weight.detach().clone()
    metadata = warm_start_cell_from_scaffold(model, DEFAULT_SELECTED_SCAFFOLD)
    checkpoint = torch.load(DEFAULT_SELECTED_SCAFFOLD, map_location="cpu", weights_only=False)
    source = checkpoint["state_dict"]
    assert metadata["source_step"] == 400
    assert metadata["source_variant"] == "mw_scaffold_slotRL_v2"
    assert metadata["source_pos_temp"] == 0.55
    assert metadata["loaded_tensors"] == 89
    assert torch.equal(model.context_encoder.query.detach().cpu(), source["query"])
    assert torch.equal(
        model.context_encoder.tr.layers[3].linear2.weight.detach().cpu(),
        source["tr.layers.3.linear2.weight"],
    )
    assert torch.equal(
        model.position_head.base.head_c.weight.detach().cpu(),
        source["flow.head_c.weight"],
    )
    assert torch.equal(
        model.context_encoder.phi_a[2].weight.detach().cpu(),
        source["phi_a.2.weight"],
    )
    # Cell/count adapters are deliberately new and remain neutral.
    assert torch.equal(model.context_encoder.count_embed[-1].weight, before_count)
