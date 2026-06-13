from __future__ import annotations

import torch
import h5py
import numpy as np
import pytest

from grid_transformer.data.lj_transferable import (
    LJTransferableCachedDataset,
    LJTransferableDataset,
    build_lj_transferable_cache,
)
from grid_transformer.models.curve_rail import CurveRailAttention


def _make(d_model=32, n_head=4, K=6, B=2, T=5):
    torch.manual_seed(0)
    mod = CurveRailAttention(d_model=d_model, n_head=n_head, n_rail=K)
    x = torch.randn(B, T, d_model)
    wp = torch.randn(B, T, K, 3)
    return mod, x, wp


def test_output_shape_and_noop_at_init() -> None:
    mod, x, wp = _make()
    y = mod(x, wp)
    assert y.shape == x.shape
    # out_proj is zero-initialized so the layer is an EXACT no-op at init (warm-start
    # probe protocol: a baseline state_dict loaded strict=False reproduces baseline
    # byte-for-byte at step 0). Any movement is then attributable to the rail.
    assert torch.allclose(y, x, atol=0.0)


def test_layer_becomes_active_once_out_proj_is_nonzero() -> None:
    mod, x, wp = _make()
    # Once out_proj moves off zero (as it does after the first optimizer step), the
    # rail modifies the hidden states.
    torch.nn.init.normal_(mod.out_proj.weight, std=0.1)
    y = mod(x, wp)
    assert not torch.allclose(y, x)


def test_rail_is_order_sensitive() -> None:
    """Permuting the waypoints along K must change the output (ordered, not perm-invariant)."""
    mod, x, wp = _make()
    # Activate the (zero-initialized) residual projection so the order dependence is
    # observable in the output, not masked by the no-op-at-init identity.
    torch.nn.init.normal_(mod.out_proj.weight, std=0.1)
    mod.eval()
    with torch.no_grad():
        y = mod(x, wp)
        perm = torch.tensor([5, 4, 3, 2, 1, 0])
        y_perm = mod(x, wp[:, :, perm, :])
    assert not torch.allclose(y, y_perm, atol=1e-5), "output should depend on rail order"


def test_batch_independence() -> None:
    """Each token attends only to its own waypoints (no cross-token/batch leakage)."""
    mod, x, wp = _make(B=2, T=4)
    mod.eval()
    with torch.no_grad():
        y_full = mod(x, wp)
        y0 = mod(x[:1], wp[:1])
    assert torch.allclose(y_full[:1], y0, atol=1e-5)


def test_rejects_bad_rail_size() -> None:
    mod, x, _ = _make(K=6)
    bad = torch.randn(x.shape[0], x.shape[1], 4, 3)  # K=4 != n_rail=6
    try:
        mod(x, bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError on rail-size mismatch")


def test_fixed_template_rail_repeats_for_factorized_tokens(tmp_path) -> None:
    h5_path = tmp_path / "lj4.h5"
    coords = np.array(
        [[
            [0.10, 0.10, 0.10],
            [0.65, 0.15, 0.10],
            [0.20, 0.70, 0.15],
            [0.75, 0.75, 0.75],
        ]],
        dtype=np.float32,
    )
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("traj", data=coords)
        h5f.attrs["boxlength"] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        h5f.attrs["density"] = 4.0

    ds = LJTransferableDataset(
        h5_path=str(h5_path),
        periodic=True,
        ordering="hilbert",
        hilbert_resolution=8,
        local_window=1.0,
        local_bins=8,
        factorized=True,
        use_curve_rail=True,
        curve_rail_mode="fixed_template",
        curve_rail_k=4,
    )
    item = ds[0]
    wp = item["curve_waypoints"]
    assert wp.shape == (9, 4, 3)
    assert item["target_idx"].shape[0] == 9
    assert torch.allclose(wp[0], wp[1])
    assert torch.allclose(wp[1], wp[2])
    assert torch.allclose(wp[3], wp[4])
    assert torch.allclose(wp[4], wp[5])
    assert not torch.allclose(wp[0], wp[3])

    with pytest.raises(ValueError, match="fixed_template"):
        LJTransferableDataset(
            h5_path=str(h5_path),
            periodic=True,
            ordering="hilbert",
            hilbert_resolution=8,
            factorized=True,
            use_curve_rail=True,
            curve_rail_mode="lookahead",
        )


def test_fixed_template_rail_cache_is_compact_and_generated(tmp_path) -> None:
    h5_path = tmp_path / "lj4.h5"
    cache_path = tmp_path / "cache.pt"
    coords = np.array(
        [[
            [0.10, 0.10, 0.10],
            [0.65, 0.15, 0.10],
            [0.20, 0.70, 0.15],
            [0.75, 0.75, 0.75],
        ]],
        dtype=np.float32,
    )
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("traj", data=coords)
        h5f.attrs["boxlength"] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        h5f.attrs["density"] = 4.0

    build_lj_transferable_cache(
        h5_path=str(h5_path),
        output_path=str(cache_path),
        periodic=True,
        ordering="hilbert",
        hilbert_resolution=8,
        local_window=1.0,
        local_bins=8,
        factorized=True,
        use_curve_rail=True,
        curve_rail_mode="fixed_template",
        curve_rail_k=4,
        num_augmentations=1,
        energy_chunk_size=4,
        cache_build_chunk_size=4,
    )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert "curve_waypoints" not in payload
    assert payload["metadata"]["curve_rail_storage"] == "fixed_template_generated"

    cached = LJTransferableCachedDataset(str(cache_path))
    item = cached[0]
    wp = item["curve_waypoints"]
    assert wp.shape == (9, 4, 3)
    assert torch.allclose(wp[0], wp[1])
    assert torch.allclose(wp[1], wp[2])
