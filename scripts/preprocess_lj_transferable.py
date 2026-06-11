#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from typing import Sequence

from grid_transformer.data.lj_transferable import (
    build_lj_transferable_cache,
    discretize_lj_transferable_cache,
)


def _resolve_h5_inputs(entries: Sequence[str]) -> list[str]:
    paths: list[str] = []
    for entry in entries:
        if os.path.isdir(entry):
            h5s = sorted(glob.glob(os.path.join(entry, "*.h5")))
            paths.extend(h5s)
        else:
            paths.append(entry)
    paths = [os.path.abspath(p) for p in paths]
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("No H5 files found from --data_h5 inputs.")
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Some --data_h5 entries do not exist: {missing[:5]}")
    return paths


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Precompute lj_transferable token sequences into a cache file so training "
            "can skip per-sample sorting/tokenization in the dataloader."
        )
    )
    ap.add_argument(
        "--data_h5",
        type=str,
        nargs="+",
        help="One or more H5 files and/or directories containing *.h5 files.",
    )
    ap.add_argument(
        "--source_cache",
        type=str,
        default=None,
        help="Existing continuous cache to discretize offline when used with --discrete.",
    )
    ap.add_argument("--output", type=str, required=True, help="Output .pt cache path.")
    ap.add_argument(
        "--periodic",
        dest="periodic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use periodic geometry. Disable for COM-centered nonperiodic snapshots.",
    )
    ap.add_argument("--ordering", type=str, choices=("hilbert", "spectral"), default="spectral")
    ap.add_argument("--spectral_sigma", type=float, default=1.0)
    ap.add_argument("--hilbert_resolution", type=int, default=128)
    ap.add_argument(
        "--cell_size",
        type=float,
        default=None,
        help="Hold the physical Hilbert cell constant: per-box R = next_pow2(round(L/cell_size)). "
        "Required for multi-size caches (overrides --hilbert_resolution per box).",
    )
    ap.add_argument("--window", type=float, default=3.0, help="Local displacement window W.")
    ap.add_argument("--bins", type=int, default=64, help="Number of bins per axis.")
    ap.add_argument("--no_long_jump", action="store_true", help="Disable dedicated long-jump token.")
    ap.add_argument(
        "--use_curve_rail",
        action="store_true",
        help="Precompute the Hilbert curve-rail (GPS-guide) look-ahead waypoints "
        "(3D periodic Hilbert ordering only).",
    )
    ap.add_argument(
        "--curve_rail_offsets",
        type=int,
        nargs="+",
        default=None,
        help="Custom Hilbert index offsets for the curve rail (lookahead mode; defaults "
        "to the log-spaced DEFAULT_CURVE_RAIL_OFFSETS when omitted).",
    )
    ap.add_argument(
        "--curve_rail_mode",
        type=str,
        choices=("lookahead", "fixed_template"),
        default="lookahead",
        help="lookahead: per-particle look-ahead from the actual cell (sample-dependent). "
        "fixed_template: sample-INDEPENDENT template indexed only by particle position.",
    )
    ap.add_argument("--curve_rail_window", type=float, default=1.0,
                    help="fixed_template half-window in units of X=R^3/N.")
    ap.add_argument("--curve_rail_k", type=int, default=8,
                    help="fixed_template number of waypoints per particle.")
    ap.add_argument("--curve_rail_reference", type=str, choices=("absolute", "prev_step"),
                    default="absolute", help="fixed_template waypoint frame.")
    ap.add_argument("--curve_rail_residual_target", action="store_true",
                    help="fixed_template: predict the rail-anchored residual pos_j - decode(j*X) "
                    "instead of the relative step pos_j - pos_{j-1} (absolute anchoring, no drift).")
    ap.add_argument(
        "--factorized",
        action="store_true",
        help="Flatten each D-dimensional relative jump into D sequential scalar tokens.",
    )
    ap.add_argument(
        "--polar",
        action="store_true",
        help="Store continuous relative-displacement targets in spherical coordinates (3D only).",
    )
    ap.add_argument(
        "--discrete",
        action="store_true",
        help="Quantize relative displacements with a learned 3D codebook instead of the windowed tokenizer.",
    )
    ap.add_argument(
        "--codebook_path",
        type=str,
        default="codebook.pt",
        help="Path to the saved discrete codebook tensor used when --discrete is enabled.",
    )
    ap.add_argument(
        "--random_shift",
        action="store_true",
        help="Apply one random global shift per cached sample before tokenization.",
    )
    ap.add_argument(
        "--no_augment_shift",
        action="store_true",
        help="Disable the batched 3D augmentations (continuous torus shift AND random "
        "90-degree rotation). Needed for cell-aligned data (e.g. the Hilbert-rail toy) "
        "where these move particles to different Hilbert cells and break exactness. "
        "Pair with --num_augmentations 1.",
    )
    ap.add_argument(
        "--augment_90deg_rotations",
        action="store_true",
        help="In 2D, expand each sample into four 90-degree rotations before sorting/tokenization.",
    )
    ap.add_argument(
        "--num_augmentations",
        type=int,
        default=5,
        help="Offline augmentation multiplier for the batched 3D Hilbert cache path.",
    )
    ap.add_argument(
        "--energy_chunk_size",
        type=int,
        default=2048,
        help="Chunk size used for cached LJ energy evaluation to avoid OOM on large H5 files.",
    )
    ap.add_argument(
        "--cache_build_chunk_size",
        type=int,
        default=16384,
        help="Chunk size used when building the 3D batched cache path to avoid file-sized intermediate arrays.",
    )
    ap.add_argument("--lj_epsilon", type=float, default=1.0, help="Lennard-Jones epsilon used for cached energy.")
    ap.add_argument("--lj_sigma", type=float, default=1.0, help="Lennard-Jones sigma used for cached energy.")
    ap.add_argument("--lj_cutoff", type=float, default=None, help="Optional Lennard-Jones cutoff used for cached energy.")
    ap.add_argument(
        "--lj_spring_constant",
        type=float,
        default=0.5,
        help="Harmonic spring constant used for nonperiodic cached energy.",
    )
    ap.add_argument("--target_system", type=str, choices=("lj", "dw"), default="lj", help="Target energy system.")
    ap.add_argument("--dw_a", type=float, default=0.9, help="Double-well quartic coefficient.")
    ap.add_argument("--dw_b", type=float, default=-4.0, help="Double-well quadratic coefficient.")
    ap.add_argument("--dw_c", type=float, default=0.0, help="Double-well constant offset.")
    ap.add_argument("--dw_offset", type=float, default=4.0, help="Double-well distance offset.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    if args.source_cache is not None:
        if not bool(args.discrete):
            raise ValueError("--source_cache is only supported together with --discrete.")
        source_cache = os.path.abspath(args.source_cache)
        if not os.path.isfile(source_cache):
            raise FileNotFoundError(f"Source cache does not exist: {source_cache}")
        summary = discretize_lj_transferable_cache(
            source_cache_path=source_cache,
            output_path=out,
            codebook_path=str(args.codebook_path),
            chunk_size=int(args.cache_build_chunk_size),
        )
    else:
        if not args.data_h5:
            raise ValueError("Either --data_h5 or --source_cache must be provided.")
        data_paths = _resolve_h5_inputs(args.data_h5)
        summary = build_lj_transferable_cache(
            file_paths=data_paths,
            output_path=out,
            periodic=bool(args.periodic),
            hilbert_resolution=int(args.hilbert_resolution),
            cell_size=args.cell_size,
            ordering=str(args.ordering),
            spectral_sigma=float(args.spectral_sigma),
            local_window=float(args.window),
            local_bins=int(args.bins),
            use_long_jump_token=(not args.no_long_jump),
            factorized=bool(args.factorized),
            polar=bool(args.polar),
            discrete=bool(args.discrete),
            codebook_path=(str(args.codebook_path) if args.discrete else None),
            random_grid_shift=bool(args.random_shift),
            augment_90deg_rotations=bool(args.augment_90deg_rotations),
            num_augmentations=int(args.num_augmentations),
            use_curve_rail=bool(args.use_curve_rail),
            curve_rail_offsets=args.curve_rail_offsets,
            curve_rail_mode=str(args.curve_rail_mode),
            curve_rail_window=float(args.curve_rail_window),
            curve_rail_k=int(args.curve_rail_k),
            curve_rail_reference=str(args.curve_rail_reference),
            curve_rail_residual_target=bool(args.curve_rail_residual_target),
            augment_torus_shift=(not bool(args.no_augment_shift)),
            energy_chunk_size=int(args.energy_chunk_size),
            cache_build_chunk_size=int(args.cache_build_chunk_size),
            limit=args.limit,
            seed=int(args.seed),
            target_system=str(args.target_system),
            lj_epsilon=float(args.lj_epsilon),
            lj_sigma=float(args.lj_sigma),
            lj_cutoff=args.lj_cutoff,
            lj_spring_constant=float(args.lj_spring_constant),
            dw_a=float(args.dw_a),
            dw_b=float(args.dw_b),
            dw_c=float(args.dw_c),
            dw_offset=float(args.dw_offset),
        )
    print(f"Saved cache: {summary['output_path']}")
    print(
        "base_n_samples={base_n_samples}, n_samples={n_samples}, max_particles={max_particles}, max_seq_len={max_seq_len}, "
        "vocab_size={vocab_size}, periodic={periodic}, ordering={ordering}, "
        "factorized={factorized}, polar={polar}, discrete={discrete}, random_grid_shift={random_grid_shift}, "
        "num_augmentations={num_augmentations}, energy_chunk_size={energy_chunk_size}, cache_build_chunk_size={cache_build_chunk_size}, rotation_count={rotation_count}, "
        "max_abs_relative_displacement={max_abs_relative_displacement:.6f}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
