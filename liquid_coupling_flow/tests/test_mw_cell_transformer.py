import torch

from liquid_coupling_flow.mw.mw_energy import (
    capped_mw_candidate_increment,
    capped_mw_candidate_pair_increment,
)
from liquid_coupling_flow.mw.mw_cell_transformer import MWCellTransformerContext


def test_candidate_increment_periodic_image_matches_unwrapped_local_cavity():
    dtype = torch.float64
    L = 5.0
    candidate = torch.tensor([[[0.08, 2.5, 2.5]]], dtype=dtype)
    cage_wrapped = torch.tensor([[[4.92, 2.5, 2.5], [0.7, 3.1, 2.8]]], dtype=dtype)
    cage_local = cage_wrapped.clone()
    cage_local[0, 0, 0] -= L
    periodic = capped_mw_candidate_increment(candidate, cage_wrapped, L=L)
    local = capped_mw_candidate_increment(candidate, cage_local)
    assert torch.allclose(periodic[0], local[0], atol=1e-12, rtol=0)
    assert torch.allclose(periodic[1], local[1], atol=1e-12, rtol=0)


def test_candidate_pair_fast_path_matches_full_increment_pair_term():
    torch.manual_seed(18)
    candidate = torch.rand(2, 7, 3, dtype=torch.float64) * 5.0
    cage = torch.rand(2, 5, 3, dtype=torch.float64) * 5.0
    full_pair, _ = capped_mw_candidate_increment(candidate, cage, L=5.0)
    fast_pair = capped_mw_candidate_pair_increment(candidate, cage, L=5.0)
    assert torch.allclose(fast_pair, full_pair, atol=1e-12, rtol=0)


def test_ragged_stage_batch_matches_individual_causal_contexts():
    torch.manual_seed(17)
    model = MWCellTransformerContext(d_model=24, n_head=4, n_layer=1, knn=4, knn_bnd=4).double().eval()
    rows = [
        {"earlier": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64),
         "prefix": torch.empty(0, 3, dtype=torch.float64),
         "origin": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
         "count_stencil": torch.zeros(27, dtype=torch.float64),
         "slot_features": torch.tensor([0.0, 0.1, 1.0], dtype=torch.float64),
         "cell_features": torch.zeros(6, dtype=torch.float64), "L": 5.0},
        {"earlier": torch.tensor([[0.1, 0.2, 0.3], [2.0, 2.0, 2.0]], dtype=torch.float64),
         "prefix": torch.tensor([[1.1, 1.2, 0.9]], dtype=torch.float64),
         "origin": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
         "count_stencil": torch.ones(27, dtype=torch.float64),
         "slot_features": torch.tensor([0.5, 0.2, 1.0], dtype=torch.float64),
         "cell_features": torch.ones(6, dtype=torch.float64), "L": 5.0},
    ]
    single = [model(**row) for row in rows]
    batched = model.forward_many(rows)
    assert torch.allclose(torch.stack(single), torch.stack(batched), atol=2e-12, rtol=0)
