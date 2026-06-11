"""Tests for sample_lj.py CLI fixes from the branch review:

- 3.7  --use_ida must default to None so --ar_arch auto can actually infer the
       architecture from checkpoint metadata (default True made every default
       invocation resolve to 'ida' and fail on standard/rail/arc checkpoints).
- 3.8  coord_dim inference must read 'ida_spatial_dim' (what the standard arch
       stores) before 'spatial_dim', and fail loudly instead of silently
       falling back to 2D.
- new  --cell_size / --hilbert_resolution overrides for size-transfer sampling
       (checkpoints trained without --lj_transfer_cell_size store cell_size=None,
       which pins R at the training resolution and lets the cell grow with the
       box — the K=1 study had to monkey-patch around this).
"""
from types import SimpleNamespace

import pytest

MINIMAL_ARGV = ["--ckpt", "x.ckpt", "--Lx", "3", "--Ly", "3"]


def test_use_ida_defaults_to_none_so_auto_arch_can_infer():
    from grid_transformer.models.ar_registry import resolve_ar_arch
    from sample_lj import build_parser

    args = build_parser().parse_args(MINIMAL_ARGV)
    assert args.use_ida is None
    assert resolve_ar_arch(args.ar_arch, use_ida=args.use_ida) == "auto"


def test_geometry_override_flags():
    from sample_lj import build_parser

    args = build_parser().parse_args(MINIMAL_ARGV)
    assert args.cell_size is None
    assert args.hilbert_resolution is None

    args = build_parser().parse_args(
        MINIMAL_ARGV + ["--cell_size", "0.046875", "--hilbert_resolution", "128"]
    )
    assert args.cell_size == pytest.approx(0.046875)
    assert args.hilbert_resolution == 128


def test_effective_rail_geometry_prefers_cli_over_checkpoint():
    from sample_lj import _effective_rail_geometry

    model = SimpleNamespace(hilbert_resolution=64, cell_size=None)

    # No overrides: checkpoint values pass through.
    res, cell = _effective_rail_geometry(model, None, None)
    assert (res, cell) == (64, None)

    # CLI cell_size override enables constant-cell size transfer.
    res, cell = _effective_rail_geometry(model, None, 0.046875)
    assert (res, cell) == (64, pytest.approx(0.046875))

    # CLI resolution override.
    res, cell = _effective_rail_geometry(model, 128, None)
    assert (res, cell) == (128, None)


def test_infer_coord_dim_reads_ida_spatial_dim():
    from sample_lj import _infer_coord_dim

    standard = SimpleNamespace(hparams=SimpleNamespace(ida_spatial_dim=3))
    legacy = SimpleNamespace(hparams=SimpleNamespace(spatial_dim=2))
    neither = SimpleNamespace(hparams=SimpleNamespace())

    assert _infer_coord_dim(standard, None) == 3
    assert _infer_coord_dim(legacy, None) == 2
    assert _infer_coord_dim(standard, 2) == 2  # explicit CLI wins
    with pytest.raises(ValueError, match="coord_dim"):
        _infer_coord_dim(neither, None)  # loud, not a silent 2D fallback
