import math

import torch

from liquid_coupling_flow.mw.mw_cell_geom import (
    LOG_COLOR_ORDERS,
    causal_cell_order,
    cell_colors,
    cell_indices,
    cell_local_coordinates,
    color_stages,
    cubic_orientations,
    earlier_stage_mask,
    flatten_cell_indices,
    from_oriented,
    group_by_cell_priority,
    physical_from_cell_local,
    sample_color_order,
    to_oriented,
    unflatten_cell_indices,
)


L = 10.39062
G = 4
H = L / G


def _orientation(index=0, dtype=torch.float64):
    return cubic_orientations(dtype=dtype)[index]


def test_full_cubic_group_has_48_unique_signed_permutations():
    orientations = cubic_orientations(dtype=torch.float64)
    assert orientations.shape == (48, 3, 3)
    assert torch.unique(orientations.reshape(48, -1), dim=0).shape[0] == 48
    eye = torch.eye(3, dtype=torch.float64).expand(48, -1, -1)
    assert torch.equal(orientations @ orientations.transpose(1, 2), eye)
    assert torch.equal(torch.linalg.det(orientations).abs(), torch.ones(48, dtype=torch.float64))


def test_oriented_chart_round_trip_and_translation_covariance():
    gen = torch.Generator().manual_seed(4)
    x = torch.rand(31, 3, generator=gen, dtype=torch.float64) * L
    shift = torch.tensor([0.3, 1.1, 2.0], dtype=torch.float64)
    orientation = _orientation(37)
    y = to_oriented(x, L, shift, orientation)
    recovered = from_oriented(y, L, shift, orientation)
    delta = recovered - x
    delta -= L * torch.round(delta / L)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=2e-12, rtol=0)

    translation = torch.tensor([1.7, -0.4, 3.2], dtype=torch.float64)
    translated_x = torch.remainder(x + translation, L)
    translated_shift = shift + translation @ orientation.T
    assert torch.allclose(
        to_oriented(translated_x, L, translated_shift, orientation), y, atol=3e-12, rtol=0
    )


def test_half_open_boundary_rule_is_total_and_deterministic():
    orientation = _orientation(9)
    shift = torch.tensor([0.17, 0.41, 0.83], dtype=torch.float64)
    eps = 256 * torch.finfo(torch.float64).eps * H
    y = torch.tensor(
        [[0.0, 0.0, 0.0], [H - eps, H, H + eps], [L - eps, 2 * H, 3 * H]],
        dtype=torch.float64,
    )
    x = from_oriented(y, L, shift, orientation)
    first = cell_indices(x, L, G, shift, orientation)
    second = cell_indices(x, L, G, shift, orientation)
    assert torch.equal(first, second)
    assert first.tolist() == [[0, 0, 0], [0, 1, 1], [3, 2, 3]]
    ids = flatten_cell_indices(first, G)
    assert torch.equal(unflatten_cell_indices(ids, G), first)


def test_cell_local_chart_reconstructs_physical_points():
    gen = torch.Generator().manual_seed(7)
    x = torch.rand(97, 3, generator=gen, dtype=torch.float64) * L
    shift = torch.rand(3, generator=gen, dtype=torch.float64) * H
    orientation = _orientation(22)
    indices, local = cell_local_coordinates(x, L, G, shift, orientation)
    assert bool((local >= 0).all()) and bool((local < H).all())
    recovered = physical_from_cell_local(indices, local, L, G, shift, orientation)
    delta = recovered - x
    delta -= L * torch.round(delta / L)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=3e-12, rtol=0)


def test_same_color_cells_have_h_gap_but_can_share_a_frozen_center():
    a = torch.tensor([0, 0, 0])
    b = torch.tensor([2, 0, 0])
    assert int(cell_colors(torch.stack((a, b)))[0]) == int(cell_colors(torch.stack((a, b)))[1])
    # [0,h) and [2h,3h) have a periodic face-to-face separation h.
    assert H > 1.8
    assert math.isclose(2 * H - H, H)


def test_random_color_order_and_causal_schedule_are_exact_permutations():
    gen = torch.Generator().manual_seed(11)
    order = sample_color_order(gen=gen)
    stages = color_stages(order)
    assert sorted(order.tolist()) == list(range(8))
    assert torch.equal(stages[order], torch.arange(8))
    assert math.isclose(LOG_COLOR_ORDERS, math.lgamma(9))

    schedule = causal_cell_order(G, order)
    assert sorted(schedule.tolist()) == list(range(G ** 3))
    scheduled_colors = cell_colors(unflatten_cell_indices(schedule, G))
    assert scheduled_colors.reshape(8, -1)[:, 0].tolist() == order.tolist()
    assert bool((scheduled_colors.reshape(8, -1) == order[:, None]).all())


def test_earlier_stage_mask_never_exposes_same_or_later_color():
    order = torch.tensor([5, 1, 7, 0, 3, 2, 6, 4])
    colors = torch.arange(8).repeat_interleave(3)
    active = 0
    mask = earlier_stage_mask(colors, active, order)
    stages = color_stages(order)
    assert torch.equal(mask, stages[colors] < stages[active])
    assert not bool(mask[colors == active].any())


def test_priority_grouping_is_invariant_to_joint_row_permutation():
    gen = torch.Generator().manual_seed(13)
    x = torch.rand(64, 3, generator=gen, dtype=torch.float64) * L
    priorities = torch.rand(64, generator=gen, dtype=torch.float64)
    shift = torch.rand(3, generator=gen, dtype=torch.float64) * H
    orientation = _orientation(31)
    a = group_by_cell_priority(x, priorities, L, G, shift, orientation)
    row_perm = torch.randperm(64, generator=gen)
    b = group_by_cell_priority(x[row_perm], priorities[row_perm], L, G, shift, orientation)
    assert torch.equal(a.counts, b.counts)
    assert torch.equal(a.colors.bincount(minlength=8), b.colors.bincount(minlength=8))
    assert torch.allclose(x[a.permutation], x[row_perm][b.permutation])
    assert torch.allclose(priorities[a.permutation], priorities[row_perm][b.permutation])
