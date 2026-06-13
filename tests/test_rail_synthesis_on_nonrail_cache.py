"""Fixed-template rail on a cache built WITHOUT rail geometry.

The fixed_template rail is sample-independent (a function of N, R, box only), so a
non-rail cache can serve rail-conditioned training by synthesizing waypoints in
__getitem__ — no 15GB cache rebuild. Lookahead mode genuinely needs per-sample
geometry and must still demand a rebuild.
"""
import numpy as np
import pytest
import torch

from grid_transformer.data.lj_transferable import (
    LJTransferableCachedDataset,
    build_lj_transferable_cache,
    fixed_template_waypoints,
)
from grid_transformer.training.lightning_module import LJTransferableDataModule

from test_cache_cell_size import CELL, _write_h5


@pytest.fixture()
def nonrail_cache(tmp_path):
    h5_l3 = tmp_path / "l3.h5"
    h5_l5 = tmp_path / "l5.h5"
    _write_h5(h5_l3, L=3.0, N=5, seed=0)
    _write_h5(h5_l5, L=5.0, N=9, seed=1)
    out = tmp_path / "mixed_cache.pt"
    build_lj_transferable_cache(
        file_paths=[str(h5_l3), str(h5_l5)],
        output_path=str(out),
        periodic=True,
        ordering="hilbert",
        hilbert_resolution=64,
        cell_size=CELL,
        local_window=3.0,
        local_bins=8,
        num_augmentations=1,
        augment_torus_shift=False,
        seed=0,
    )
    return str(out)


def test_fixed_template_synthesis_on_nonrail_cache(nonrail_cache):
    ds = LJTransferableCachedDataset(
        nonrail_cache,
        arc_repr=True,
        use_curve_rail=True,
        curve_rail_mode="fixed_template",
        curve_rail_k=4,
    )
    assert ds.use_curve_rail is True
    assert ds.curve_rail_mode == "fixed_template"
    assert ds.curve_rail_k == 4
    for i in range(len(ds)):
        item = ds[i]
        n = int(item["particle_length"])
        wp = item["curve_waypoints"]
        assert wp.shape == (n - 1, 4, 3)
        # synthesized waypoints must equal the template computed directly with the
        # per-size resolution rule (the multi-size load-bearing part)
        box = item["box_size"].numpy()
        R = ds._resolution_for_box(box)
        expected = fixed_template_waypoints(
            np.arange(1, n, dtype=np.int64), n, R, box.astype(np.float32),
            k=4, window_scale=ds.curve_rail_window, periodic=True,
            reference=ds.curve_rail_reference,
        )
        np.testing.assert_allclose(wp.numpy(), expected.astype(np.float32), atol=1e-6)


def test_lookahead_on_nonrail_cache_still_demands_rebuild(nonrail_cache):
    with pytest.raises(ValueError, match="fixed_template"):
        LJTransferableCachedDataset(
            nonrail_cache,
            arc_repr=True,
            use_curve_rail=True,
            curve_rail_mode="lookahead",
            curve_rail_k=4,
        )


def test_default_path_unchanged(nonrail_cache):
    ds = LJTransferableCachedDataset(nonrail_cache, arc_repr=True)
    assert ds.use_curve_rail is False
    assert "curve_waypoints" not in ds[0]


def test_datamodule_threads_rail_overrides(nonrail_cache):
    dm = LJTransferableDataModule(
        preprocessed_path=nonrail_cache,
        arc_repr=True,
        use_curve_rail=True,
        curve_rail_mode="fixed_template",
        curve_rail_k=4,
        batch_size=2,
        seed=0,
        val_frac=0.0,
    )
    dm.setup("fit")
    assert dm.use_curve_rail is True
    assert dm.curve_rail_k == 4
    assert dm.curve_rail_mode_eff == "fixed_template"
    batch = next(iter(dm.train_dataloader()))
    assert "curve_waypoints" in batch
