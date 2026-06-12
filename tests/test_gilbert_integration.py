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
    # gilbert: nearest_even(round(L/cell))
    # L=3.0: round(3.0/0.046875)=64 -> nearest_even(64)=64
    # L=4.0: round(4.0/0.046875)=85.3..->85 -> nearest_even(85)=round(85/2)*2=86
    # L=5.0: round(5.0/0.046875)=106.6..->107 -> nearest_even(107)=round(107/2)*2=106? No:
    #   round(107/2.0)=54 (half-even), but nearest even to 107 is 106 (even round-down)
    #   Spec: round(n/2)*2 where n=round(L/cell)=107 -> round(107/2.0)=54 (python rounds half to even)
    #   but 107/2=53.5 -> round to nearest even = 54 -> 54*2=108. Let me recalc:
    #   Wait, _resolution_for_box gilbert: max(2, int(round(n/2.0))*2)
    #   n=107: round(107/2.0)=round(53.5)=54 (banker's rounding) -> 54*2=108?
    #   But spec says 106 for L=5.0. Let me re-read the spec:
    #   "gilbert -> nearest EVEN integer". For 107, nearest even is 106 or 108.
    #   The plan says 106. The formula round(n/2)*2:
    #   107/2 = 53.5 -> Python round(53.5) = 54 (banker's rounds to even) -> 108?
    #   Hmm. Let me use math.floor(n/2)*2 for round-down behavior:
    #   floor(107/2)*2 = 53*2 = 106. That matches the spec!
    #
    # The plan says: max(2, int(round(n / 2.0)) * 2)
    # Python: round(53.5) = 54 (half-to-even), round(42.5) = 42, round(43.5) = 44
    # For n=85: round(85/2.0)=round(42.5)=42 (half-to-even) -> 84, not 86!
    # So round(n/2)*2 gives 84 for L=4.0, not 86.
    #
    # The spec says 86 for L=4.0. This must use a different formula.
    # nearest even to 85: could be 84 or 86, spec says 86 (round-up).
    # Use int(round(n / 2.0 + 0.5)) * 2? No...
    # Use (n + 1) // 2 * 2 (always round up to nearest even): 85->86, 107->108. But spec says 106.
    # Use n // 2 * 2 (always round down): 85->84, 107->106. But spec says 86 for L=4.0.
    #
    # Actually "nearest even": 85 is odd, nearest even is 84 (diff 1) or 86 (diff 1). Tie -> ?
    # 107 is odd, nearest even is 106 (diff 1) or 108 (diff 1). Tie -> ?
    # Spec says 86 for 85 but 106 for 107. That's inconsistent with any single rounding rule...
    # UNLESS n values are different:
    # L=4.0, cell=0.046875: 4.0/0.046875 = 85.333... -> round = 85 -> nearest even = 86
    # L=5.0, cell=0.046875: 5.0/0.046875 = 106.666... -> round = 107 -> nearest even = 106???
    # Wait: 106.666 rounds to 107, nearest even to 107 is 106 or 108.
    # But if we directly round to even: round(85.333/2)*2 = round(42.666)*2 = 43*2 = 86 ✓
    # And: round(106.666/2)*2 = round(53.333)*2 = 53*2 = 106 ✓
    # And: round(64/2)*2 = round(32)*2 = 32*2 = 64 ✓
    # So the formula is: max(2, int(round(n/2.0))*2) BUT n = round(L/cell) first?
    # NO — the plan says n = max(2, round(L/cell)), then for gilbert: max(2, round(n/2)*2)
    # Let me recheck with n = round(L/cell) directly (not pre-rounded):
    # L=4.0: n = max(2, round(4.0/0.046875)) = round(85.333) = 85
    # round(85/2) = round(42.5) = 42 (banker's) -> 84. Still 84.
    # But round(85.333/2) = round(42.666) = 43 -> 86! ✓
    # L=5.0: round(106.666/2) = round(53.333) = 53 -> 106 ✓
    # L=3.0: round(64/2) = round(32) = 32 -> 64 ✓
    # So the formula should use L/cell_size DIRECTLY, not pre-rounded:
    # n = L / cell_size; return max(2, int(round(n/2.0))*2)
    #
    # The plan's pseudocode: n = max(2, int(round(L / self.cell_size)))
    # then for gilbert: max(2, int(round(n / 2.0)) * 2)
    # With banker's rounding: n=85 -> round(42.5)=42 -> 84. Doesn't match spec.
    #
    # CONCLUSION: The plan's formula needs adjustment. The correct formula to match
    # the spec (64/86/106) is: max(2, int(round(L / cell_size / 2.0)) * 2)
    # i.e., round (L/cell)/2 directly without pre-rounding to integer.
    # I'll implement that and write the test to match.

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
