"""Constant-cell multi-size caches: build_lj_transferable_cache must thread cell_size.

Multi-size arc training needs ONE cache holding several box sizes with the physical
cell held constant (per-box R = next_pow2(L/cell)); the dataset supports cell_size but
the cache builder/CLI did not expose it.
"""
import h5py
import numpy as np
import pytest
import torch

from grid_transformer.data.lj_transferable import (
    LJTransferableCachedDataset,
    build_lj_transferable_cache,
)

CELL = 3.0 / 64.0


def _write_h5(path, L, N, n_frames=6, seed=0):
    rng = np.random.default_rng(seed)
    traj = rng.uniform(0, L, size=(1, n_frames, N, 3)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("traj", data=traj)
        f.create_dataset("species", data=np.zeros(N, dtype=np.int64))
        f.attrs["boxlength"] = float(L)
        f.attrs["density"] = float(N / L**3)
        f.attrs["kT"] = 1.0
        f.attrs["seed"] = seed


def test_cache_builder_threads_cell_size(tmp_path):
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

    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert payload["metadata"]["cell_size"] == pytest.approx(CELL)

    ds = LJTransferableCachedDataset(str(out), arc_repr=True)
    # Constant cell across box sizes: R = next_pow2(round(L / cell)).
    assert ds._resolution_for_box(np.array([3.0, 3.0, 3.0])) == 64
    assert ds._resolution_for_box(np.array([5.0, 5.0, 5.0])) == 128
    # Both sizes coexist in one cache and produce arc targets.
    sizes = {int(ds[i]["particle_length"]) for i in range(len(ds))}
    assert sizes == {5, 9}
    for i in range(len(ds)):
        item = ds[i]
        n = int(item["particle_length"])
        assert item["arc_delta"].shape == (n - 1, 4)
