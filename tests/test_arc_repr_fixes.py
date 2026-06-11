"""Tests for the arc_repr correctness fixes from CODE_REVIEW_hilbert-arc-repr_2026-06-10.md.

Covers (report finding numbers):
- 3.4  fine-offset wrap at decode (no Hilbert-curve teleport from Gaussian tails)
- 3.5  particle-0 initialization in arc sampling
- 3.1  train.py ordering guard must read the real flag (lj_transfer_ordering)
- 3.3  cached dataset must reject non-Hilbert-ordered caches for arc_repr
- 3.2  datamodule must reject arc_repr without a preprocessed cache at init
- plus the encode->decode round-trip contract between hilbert_arc_delta (training)
  and _arc_decode_positions (sampling).
"""
import numpy as np
import pytest
import torch

from grid_transformer.data.lj_transferable import (
    _hilbert3d_decode,
    _hilbert3d_encode,
    _hilbert_bits,
    hilbert_arc_delta,
)


def _make_sorted_config(R: int, L: float, codes: np.ndarray, rng: np.random.Generator):
    """Positions at the given (sorted) Hilbert codes, offset inside each cell."""
    bits = _hilbert_bits(R)
    cell = L / R
    ax, ay, az = _hilbert3d_decode(codes.astype(np.int64), bits)
    centers = (np.stack([ax, ay, az], axis=1).astype(np.float64) + 0.5) * cell
    offsets = rng.uniform(-0.4, 0.4, size=centers.shape) * cell
    return centers + offsets


def test_arc_roundtrip_encode_decode():
    """Training-side encode and sampling-side decode must invert each other,
    including across large code gaps (turns) and box-boundary cells."""
    from sample_lj import _arc_decode_positions

    R, L = 8, 2.0
    N = 8
    X = R**3 // N  # 64
    box = np.array([L, L, L], dtype=np.float64)
    rng = np.random.default_rng(0)
    # Monotone codes with a non-uniform gap (a "turn"-like jump) and the last
    # cell of the curve (code R^3-1, a box-boundary cell).
    codes = np.array([3, 70, 128, 200, 250, 320, 400, R**3 - 1], dtype=np.int64)
    pos = _make_sorted_config(R, L, codes, rng)

    arc = hilbert_arc_delta(pos, codes, box, R, periodic=True)  # [N-1, 4]
    assert arc.shape == (N - 1, 4)

    box_t = torch.tensor([[L, L, L]], dtype=torch.float32)
    curr = torch.tensor(pos[0:1], dtype=torch.float32)
    for i in range(N - 1):
        delta = torch.tensor(arc[i : i + 1], dtype=torch.float32)
        curr = _arc_decode_positions(curr, delta, box_t, R, N)
        np.testing.assert_allclose(
            curr[0].numpy(), pos[i + 1], atol=1e-4,
            err_msg=f"round-trip diverged at step {i}",
        )


def test_arc_decode_wraps_fine_offset():
    """A Gaussian-tail fine offset (|fine| > 0.5) must be min-imaged into the
    intended cell, not allowed to cross into a (Hilbert-distant) neighbor."""
    from sample_lj import _arc_decode_positions

    R, L, N = 8, 2.0, 8
    box_t = torch.tensor([[L, L, L]], dtype=torch.float32)
    curr = torch.tensor([[0.9, 0.3, 0.3]], dtype=torch.float32)
    base = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    tail = base.clone()
    tail[0, 1] = 0.7  # out-of-range fine_x
    wrapped = base.clone()
    wrapped[0, 1] = 0.7 - 1.0  # its min-image, -0.3

    out_tail = _arc_decode_positions(curr, tail, box_t, R, N)
    out_wrapped = _arc_decode_positions(curr, wrapped, box_t, R, N)
    np.testing.assert_allclose(out_tail.numpy(), out_wrapped.numpy(), atol=1e-6)

    # And the decoded point must lie inside the predicted cell (within cell/2
    # of its center on every axis).
    cell = L / R
    out_centered = _arc_decode_positions(curr, base, box_t, R, N)
    np.testing.assert_array_less(
        np.abs(out_tail.numpy() - out_centered.numpy()), cell / 2 + 1e-6
    )


def test_arc_initial_positions_is_code0_cell_center():
    """Arc sampling must start particle 0 at the code-0 cell center (matching
    the training-side lowest-code cell), not the box corner (0,0,0)."""
    from sample_lj import _arc_initial_positions

    box_t = torch.tensor([[3.0, 3.0, 3.0]], dtype=torch.float32)
    R = 64
    out = _arc_initial_positions(box_t, R)
    expected = 3.0 / 64 / 2.0  # decode(0) == cell (0,0,0); center = cell/2
    np.testing.assert_allclose(out.numpy(), [[expected] * 3], atol=1e-7)


def test_train_arc_repr_ordering_guard_reads_real_flag():
    """The guard must read args.lj_transfer_ordering (the actual CLI flag);
    a namespace WITHOUT an 'ordering' attribute but with spectral
    lj_transfer_ordering must be rejected."""
    from types import SimpleNamespace

    from train import _validate_arc_repr_flags

    bad = SimpleNamespace(arc_repr=1, continuous_input=True, lj_transfer_ordering="spectral")
    with pytest.raises(ValueError, match="hilbert"):
        _validate_arc_repr_flags(bad)

    ok = SimpleNamespace(arc_repr=1, continuous_input=True, lj_transfer_ordering="hilbert")
    _validate_arc_repr_flags(ok)  # must not raise

    no_cont = SimpleNamespace(arc_repr=1, continuous_input=False, lj_transfer_ordering="hilbert")
    with pytest.raises(ValueError, match="continuous_input"):
        _validate_arc_repr_flags(no_cont)


def test_cache_arc_validation_rejects_non_hilbert_ordering():
    from grid_transformer.data.lj_transferable import _validate_arc_repr_cache

    # Valid combination passes.
    _validate_arc_repr_cache(ordering="hilbert", periodic=True, has_absolute_coords=True)

    with pytest.raises(ValueError, match="ordering"):
        _validate_arc_repr_cache(ordering="spectral", periodic=True, has_absolute_coords=True)
    with pytest.raises(ValueError, match="periodic"):
        _validate_arc_repr_cache(ordering="hilbert", periodic=False, has_absolute_coords=True)
    with pytest.raises(ValueError, match="absolute_coords"):
        _validate_arc_repr_cache(ordering="hilbert", periodic=True, has_absolute_coords=False)


def test_datamodule_arc_repr_requires_preprocessed_cache():
    from grid_transformer.training.lightning_module import LJTransferableDataModule

    with pytest.raises(ValueError, match="arc_repr"):
        LJTransferableDataModule(
            data_path="/nonexistent.h5",
            arc_repr=True,
            preprocessed_path=None,
        )
