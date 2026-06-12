import numpy as np
import pytest

from grid_transformer.data.curves import get_curve3d
from grid_transformer.data.lj_transferable import hilbert_arc_delta


def _random_sorted_config(curve, box, n, seed):
    """Random positions, Hilbert/gilbert-sorted, with their codes."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, box, size=(n, 3))
    cell = box / curve.R
    grid = np.clip((pos / cell).astype(np.int64), 0, curve.R - 1)
    codes = curve.encode(grid)
    order = np.argsort(codes, kind="stable")
    return pos[order], codes[order]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_arc_delta_with_hilbert_curve_matches_legacy(seed):
    R, box, n = 8, 3.0, 27
    curve = get_curve3d("hilbert", R)
    pos, codes = _random_sorted_config(curve, box, n, seed)
    box_arr = np.array([box, box, box])
    legacy = hilbert_arc_delta(pos, codes, box_arr, R, periodic=True)
    via_curve = hilbert_arc_delta(pos, codes, box_arr, R, periodic=True, curve=curve)
    np.testing.assert_array_equal(legacy, via_curve)  # byte-exact


@pytest.mark.parametrize("R", [6, 10])
def test_arc_delta_gilbert_roundtrip(R):
    """Encode (Δs, fine) with gilbert, decode with s-space advance -> exact recon.

    Mirrors the round-trip contract test in tests/test_arc_repr_fixes.py.
    """
    box, n = 3.0, 27
    curve = get_curve3d("gilbert", R)
    pos, codes = _random_sorted_config(curve, box, n, seed=3)
    box_arr = np.array([box, box, box])
    arc = hilbert_arc_delta(pos, codes, box_arr, R, periodic=True, curve=curve)
    assert arc.shape == (n - 1, 4)
    X = max(1, curve.ncells // n)
    cell = box / R

    # manual s-space decode, step by step. Exactness relies on cumlen gaps
    # (>= 1 cell) dwarfing the float32 Δs reconstruction error (~1e-4 at these
    # magnitudes); revisit if Δs storage precision or curve density changes.
    c = codes[0]
    for i in range(n - 1):
        s_next = curve.arc(np.array([c]))[0] + float(arc[i, 0]) * X
        c_next = curve.code_from_arc(np.array([s_next]))[0]
        assert c_next == codes[i + 1]  # Δs recovers the exact next code
        center = (curve.decode(np.array([c_next]))[0] + 0.5) * cell
        fine = arc[i, 1:4].astype(np.float64) * cell
        rec = center + fine
        d = rec - pos[i + 1]
        d -= np.round(d / box) * box  # min-image
        np.testing.assert_allclose(d, 0.0, atol=1e-5)
        c = c_next


def test_arc_delta_gilbert_fine_is_bounded():
    R, box, n = 6, 3.0, 27
    curve = get_curve3d("gilbert", R)
    pos, codes = _random_sorted_config(curve, box, n, seed=4)
    arc = hilbert_arc_delta(pos, codes, np.array([box] * 3), R, periodic=True, curve=curve)
    assert np.abs(arc[:, 1:4]).max() <= 0.5 + 1e-6


# ---------------------------------------------------------------------------
# Task 4: dataset integration tests (ordering="gilbert")
# ---------------------------------------------------------------------------

import h5py
import pytest
import torch

from grid_transformer.data.lj_transferable import (
    LJTransferableCachedDataset,
    LJTransferableDataset,
    build_lj_transferable_cache,
)


def _write_tiny_h5(path, L, N, n_frames=4, seed=0):
    """Minimal h5 in the schema expected by LJTransferableDataset."""
    rng = np.random.default_rng(seed)
    traj = rng.uniform(0, L, size=(1, n_frames, N, 3)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("traj", data=traj)
        f.create_dataset("species", data=np.zeros(N, dtype=np.int64))
        f.attrs["boxlength"] = float(L)
        f.attrs["density"] = float(N / L**3)
        f.attrs["kT"] = 1.0
        f.attrs["seed"] = seed


# ---------------------------------------------------------------------------
# Test 1 — _resolution_for_box: gilbert -> nearest-even; hilbert -> pow-2
# ---------------------------------------------------------------------------

def test_resolution_for_box_gilbert_is_nearest_even(tmp_path):
    h5 = tmp_path / "tiny.h5"
    _write_tiny_h5(h5, L=3.0, N=5)
    CELL = 0.046875  # 3.0/64 = 0.046875

    ds_g = LJTransferableDataset(
        str(h5),
        periodic=True,
        ordering="gilbert",
        hilbert_resolution=64,  # power-of-two default; no pow-2 check for gilbert
        cell_size=CELL,
        local_window=3.0,
        local_bins=8,
        random_grid_shift=False,
    )
    # gilbert rule: R = round(L/cell/2)*2, i.e. the nearest EVEN integer to the
    # real-valued L/cell (85.33 -> 86, 106.67 -> 106). Dividing by 2 BEFORE any
    # integer rounding is essential: pre-rounding L/cell to int (85, 107) puts
    # the halving on exact .5 ties where Python's banker's rounding gives the
    # FARTHER even number (85 -> 84, 107 -> 108).
    assert ds_g._resolution_for_box(np.array([3.0, 3.0, 3.0])) == 64
    assert ds_g._resolution_for_box(np.array([4.0, 4.0, 4.0])) == 86
    assert ds_g._resolution_for_box(np.array([5.0, 5.0, 5.0])) == 106

    ds_h = LJTransferableDataset(
        str(h5),
        periodic=True,
        ordering="hilbert",
        hilbert_resolution=64,
        cell_size=CELL,
        local_window=3.0,
        local_bins=8,
        random_grid_shift=False,
    )
    # hilbert: next_pow2(round(L/cell))
    assert ds_h._resolution_for_box(np.array([3.0, 3.0, 3.0])) == 64
    assert ds_h._resolution_for_box(np.array([4.0, 4.0, 4.0])) == 128
    assert ds_h._resolution_for_box(np.array([5.0, 5.0, 5.0])) == 128


# ---------------------------------------------------------------------------
# Test 2 — __getitem__ gilbert: non-pow-2 R works; arc_delta bounds hold
# ---------------------------------------------------------------------------

def test_dataset_getitem_gilbert(tmp_path):
    # Use cell_size=0.05 with L=3.0 -> n = 3.0/0.05/2 = 30 -> round(30)*2 = 60 (non-pow-2)
    # This is the key case: non-pow-2 R must not crash.
    L, N = 3.0, 5
    h5 = tmp_path / "tiny.h5"
    _write_tiny_h5(h5, L=L, N=N)

    out = tmp_path / "cache_gilbert.pt"
    build_lj_transferable_cache(
        file_paths=[str(h5)],
        output_path=str(out),
        periodic=True,
        ordering="gilbert",
        hilbert_resolution=64,  # pow-2 for construction; but cell_size overrides per-box R
        cell_size=0.05,  # -> R = round(3.0/0.05/2)*2 = round(30)*2 = 60 (non-pow-2!)
        local_window=3.0,
        local_bins=8,
        num_augmentations=1,
        augment_torus_shift=False,
        seed=0,
    )

    ds = LJTransferableCachedDataset(str(out), arc_repr=True)
    assert len(ds) > 0
    for i in range(len(ds)):
        item = ds[i]
        n = int(item["particle_length"])
        arc = item["arc_delta"]  # [N-1, 4]
        assert arc.shape == (n - 1, 4), f"item {i}: arc shape {arc.shape}"
        assert torch.isfinite(arc).all(), f"item {i}: non-finite arc"
        assert arc[:, 1:4].abs().max().item() <= 0.5 + 1e-6, (
            f"item {i}: fine offset out of range: {arc[:, 1:4].abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Test 3 — validation: gilbert+curve_rail raises; gilbert+non-pow-2 OK;
#           hilbert+non-pow-2 raises; DataModule accepts gilbert
# ---------------------------------------------------------------------------

def test_gilbert_validation(tmp_path):
    h5 = tmp_path / "tiny.h5"
    _write_tiny_h5(h5, L=3.0, N=5)

    # (a) gilbert + use_curve_rail=True -> NotImplementedError
    with pytest.raises((NotImplementedError, ValueError)):
        LJTransferableDataset(
            str(h5),
            periodic=True,
            ordering="gilbert",
            hilbert_resolution=64,
            use_curve_rail=True,
            local_window=3.0,
            local_bins=8,
            random_grid_shift=False,
        )

    # (b) gilbert with non-pow-2 hilbert_resolution does NOT raise
    ds = LJTransferableDataset(
        str(h5),
        periodic=True,
        ordering="gilbert",
        hilbert_resolution=60,  # non-pow-2: fine for gilbert
        local_window=3.0,
        local_bins=8,
        random_grid_shift=False,
    )
    assert ds.ordering == "gilbert"

    # (c) hilbert with non-pow-2 still raises ValueError
    with pytest.raises(ValueError, match="power of two"):
        LJTransferableDataset(
            str(h5),
            periodic=True,
            ordering="hilbert",
            hilbert_resolution=60,  # non-pow-2: must raise for hilbert
            local_window=3.0,
            local_bins=8,
            random_grid_shift=False,
        )

    # (d) DataModule accepts ordering="gilbert"
    from grid_transformer.training.lightning_module import LJTransferableDataModule

    dm = LJTransferableDataModule(
        data_path=str(h5),
        ordering="gilbert",
        hilbert_resolution=64,
    )
    assert dm.ordering == "gilbert"


# ---------------------------------------------------------------------------
# Task 5: s-space arc decode + --ordering plumbing
# ---------------------------------------------------------------------------


def test_arc_decode_positions_gilbert_roundtrip():
    """Sampler decode with a gilbert curve reconstructs encoder targets exactly."""
    from sample_lj import _arc_decode_positions

    R, box, n = 6, 3.0, 27
    curve = get_curve3d("gilbert", R)
    pos, codes = _random_sorted_config(curve, box, n, seed=7)
    box_arr = np.array([box] * 3)
    arc = hilbert_arc_delta(pos, codes, box_arr, R, periodic=True, curve=curve)

    box_t = torch.tensor([[box, box, box]], dtype=torch.float32)
    curr = torch.tensor(pos[:1], dtype=torch.float32)
    for i in range(n - 1):
        delta = torch.tensor(arc[i : i + 1], dtype=torch.float32)
        nxt, diag = _arc_decode_positions(
            curr, delta, box_t, R, n, curve=curve, return_diagnostics=True
        )
        target = torch.tensor(pos[i + 1], dtype=torch.float32)
        d = (nxt[0] - target).double().numpy()
        d -= np.round(d / box) * box
        np.testing.assert_allclose(d, 0.0, atol=1e-4)
        assert int(diag["c_next"][0]) == int(codes[i + 1])
        assert not bool(diag["clamp_hit"][0])
        curr = nxt


def test_arc_decode_positions_hilbert_default_unchanged():
    """curve=None keeps the historical pow-2 behavior (existing suites also
    cover this; this is the explicit equivalence check)."""
    from sample_lj import _arc_decode_positions

    R, box, n = 8, 3.0, 27
    rng = np.random.default_rng(11)
    curr = torch.tensor(rng.uniform(0, box, (5, 3)), dtype=torch.float32)
    delta = torch.tensor(rng.normal(0, 2, (5, 4)), dtype=torch.float32)
    box_t = torch.tensor([[box] * 3], dtype=torch.float32)
    a = _arc_decode_positions(curr, delta, box_t, R, n)
    b = _arc_decode_positions(curr, delta, box_t, R, n, curve=get_curve3d("hilbert", R))
    torch.testing.assert_close(a, b)


def test_rail_resolution_for_box_gilbert_rule():
    from sample_lj import _rail_resolution_for_box

    box = np.array([4.0, 4.0, 4.0])
    assert _rail_resolution_for_box(box, 64, 0.046875) == 128  # hilbert default
    assert _rail_resolution_for_box(box, 64, 0.046875, ordering="gilbert") == 86
