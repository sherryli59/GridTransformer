import torch

from liquid_coupling_flow.mw.mw_cell_data import (
    frame_banks,
    fresh_augmented_state,
    load_or_build_raw_cache,
)


def test_raw_cache_never_bakes_augmented_auxiliaries(tmp_path):
    source = tmp_path / "source.pt"
    cfgs = torch.rand(12, 64, 3)
    torch.save({"cfgs": cfgs, "L": 5.19531}, source)
    cache = tmp_path / "cache.pt"
    payload = load_or_build_raw_cache(
        [source], cache, train_per_size=6, validation_per_size=3, seed=5,
    )
    assert cache.exists()
    bank = frame_banks(payload)[0]
    assert bank.N == 64
    assert len(bank.train) == 6
    assert len(bank.validation) == 2  # deterministic frame-held-out split of 12
    x = bank.train[:2]
    a = fresh_augmented_state(x, bank.L, 2, gen=torch.Generator().manual_seed(6))
    b = fresh_augmented_state(x, bank.L, 2, gen=torch.Generator().manual_seed(7))
    assert not torch.equal(a.priorities, b.priorities)
    assert a.shift.shape == (2, 3)
    assert a.color_order.shape == (2, 8)
