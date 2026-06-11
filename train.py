import argparse
import glob
import os
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from grid_transformer.training.lightning_module import (
    LJPixelsDataModule,
    VQLatentTransformerModule,
    CIFARVQDataModule,
    LJCellDataModule,
    LJAbsoluteDataModule,
    LJTransferableDataModule,
)
from grid_transformer.models.ar_registry import AR_ARCH_CHOICES, resolve_ar_arch
from grid_transformer.models.transformer import GraphormerAR as GraphormerARStd
from grid_transformer.models.transformer_ida import GraphormerAR as GraphormerARIDA
from grid_transformer.models.vanilla_transformer import VanillaTransformerAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train binary/AR models with PyTorch Lightning.")
    parser.add_argument("--data_dir", type=str, required=True, help="Training data root directory/archive or LJ HDF5 file path.")
    parser.add_argument(
        "--dataset",
        choices=("lj", "cifar_vq", "lj_cell", "lj_transferable", "lj_abs"),
        default="cifar_vq",
        help=(
            "Dataset loader to use "
            "(lj=NPZ coords, cifar_vq=VQ-tokenised CIFAR-10, "
            "lj_cell=Hilbert-cell LJ tokens, lj_transferable=ordered local relative LJ tokens, "
            "lj_abs=absolute x/y coordinate bins per particle)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for AdamW.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--pixel_size", type=float, default=0.05, help="Pixel size for rasterization.")
    parser.add_argument("--mask_ratio", type=float, default=0.6, help="Minimum masking ratio for masked modeling.")
    parser.add_argument(
        "--mask_ratio_max",
        type=float,
        default=None,
        help="If set, sample masking ratios uniformly in [mask_ratio, mask_ratio_max] each iteration.",
    )
    parser.add_argument("--mask_pos_weight", type=float, default=10.0, help="Positive class weight in BCE loss on masked tokens.")
    parser.add_argument("--dim", type=int, default=64, help="Model hidden dimension.")
    parser.add_argument("--depth", type=int, default=6, help="Number of Swin blocks.")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads per block.")
    parser.add_argument("--window", type=int, default=8, help="Swin attention window size.")
    parser.add_argument("--shifted", action="store_true", help="Enable shifted windows.")
    parser.add_argument("--ckpt_dir", type=str, default="ckpts", help="Directory to store checkpoints.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--accelerator", type=str, default="auto", help="PyTorch Lightning accelerator setting.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use.")
    parser.add_argument("--precision", type=str, default="32", help="Numerical precision (e.g. 16, bf16, 32).")
    parser.add_argument("--log_every_n_steps", type=int, default=1, help="Logging frequency in training steps.")
    parser.add_argument(
        "--ckpt_every_n_steps",
        type=int,
        default=0,
        help="If >0, additionally save checkpoints every N training steps.",
    )
    parser.add_argument("--resume_from", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument(
        "--train_limit",
        type=int,
        default=None,
        help="Optional limit on number of training samples.",
    )
    parser.add_argument(
        "--lj_cell_resolution",
        type=int,
        default=64,
        help="Grid side length G for lj_cell tokenization (vocabulary size is G^2).",
    )
    parser.add_argument(
        "--lj_transfer_periodic",
        dest="lj_transfer_periodic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use periodic geometry for lj_transferable. Disable for COM-centered nonperiodic data.",
    )
    parser.add_argument(
        "--lj_transfer_hilbert_resolution",
        type=int,
        default=128,
        help="Hilbert sorting resolution (cells/axis) for lj_transferable; gives a CONSTANT cell "
        "COUNT (cell size = L/R varies with box). Ignored when --lj_transfer_cell_size is set.",
    )
    parser.add_argument(
        "--lj_transfer_cell_size",
        type=float,
        default=None,
        help="Constant physical Hilbert cell size (sigma units). When set, resolution is chosen "
        "PER BOX as next_pow2(L/cell_size), holding cell size ~constant across box lengths "
        "(more recursion in larger boxes) instead of a constant cell count.",
    )
    parser.add_argument(
        "--lj_transfer_ordering",
        type=str,
        choices=("hilbert", "spectral"),
        default="hilbert",
        help="Particle ordering used before relative-delta tokenization.",
    )
    parser.add_argument(
        "--lj_transfer_spectral_sigma",
        type=float,
        default=1.0,
        help="RBF width used by spectral ordering (used when --lj_transfer_ordering=spectral).",
    )
    parser.add_argument(
        "--lj_transfer_window",
        type=float,
        default=3.0,
        help="Local displacement window W for lj_transferable tokenization in physical units.",
    )
    parser.add_argument(
        "--lj_transfer_bins",
        type=int,
        default=64,
        help="Number of bins per axis for lj_transferable displacement tokenization.",
    )
    parser.add_argument(
        "--lj_transfer_use_curve_rail",
        action="store_true",
        default=False,
        help="Enable the Hilbert curve-rail (GPS-guide) look-ahead conditioning "
        "(3D periodic Hilbert ordering, --ar_arch standard only). Requires the cache "
        "to be built with the same flag.",
    )
    parser.add_argument(
        "--lj_transfer_curve_rail_offsets",
        type=int,
        nargs="+",
        default=None,
        help="Custom Hilbert index offsets for curve rail lookahead mode.",
    )
    parser.add_argument(
        "--lj_transfer_curve_rail_mode",
        type=str,
        choices=("lookahead", "fixed_template"),
        default="lookahead",
        help="Curve-rail geometry mode for on-the-fly lj_transferable training.",
    )
    parser.add_argument("--lj_transfer_curve_rail_window", type=float, default=1.0)
    parser.add_argument("--lj_transfer_curve_rail_k", type=int, default=8)
    parser.add_argument(
        "--lj_transfer_curve_rail_reference",
        type=str,
        choices=("absolute", "prev_step"),
        default="absolute",
    )
    parser.add_argument("--lj_transfer_curve_rail_residual_target", action="store_true")
    parser.add_argument(
        "--lj_transfer_no_long_jump",
        action="store_true",
        help="Disable dedicated long-jump token and clamp out-of-window displacements instead.",
    )
    parser.add_argument(
        "--factorized",
        action="store_true",
        help="Flatten each D-dimensional relative jump into D sequential scalar tokens.",
    )
    parser.add_argument(
        "--polar",
        action="store_true",
        help="Use spherical (r, theta, phi) continuous targets for 3D lj_transferable training.",
    )
    parser.add_argument(
        "--discrete",
        action="store_true",
        help="Use a learned discrete 3D codebook for lj_transferable relative-displacement targets.",
    )
    parser.add_argument(
        "--binned_discrete",
        action="store_true",
        help="Use the windowed relative-displacement tokenizer as an explicit discrete target space without a codebook.",
    )
    parser.add_argument(
        "--codebook_path",
        type=str,
        default="codebook.pt",
        help="Path to the saved discrete codebook tensor used when --discrete is enabled.",
    )
    parser.add_argument(
        "--lj_transfer_disable_random_shift",
        action="store_true",
        help="Disable random global shift augmentation for transferable LJ tokenization.",
    )
    parser.add_argument(
        "--lj_transfer_use_data_aug",
        action="store_true",
        help="Apply random right-angle rotations during on-the-fly lj_transferable training.",
    )
    parser.add_argument(
        "--lj_transfer_preprocessed_path",
        type=str,
        default=None,
        help="Optional path to a preprocessed lj_transferable .pt cache file.",
    )
    parser.add_argument(
        "--coord_dequant_width",
        type=float,
        default=0.0,
        help="Optional training-time coordinate dequantization width. If 0, disable coordinate dequantization.",
    )
    parser.add_argument(
        "--lj_abs_bins",
        type=int,
        default=512,
        help="Number of absolute coordinate bins per axis for dataset=lj_abs.",
    )
    parser.add_argument(
        "--lj_abs_disable_random_shift",
        action="store_true",
        help="Disable random global torus shift augmentation for dataset=lj_abs.",
    )
    parser.add_argument(
        "--lj_abs_ordering",
        type=str,
        choices=("raw", "hilbert"),
        default="raw",
        help="Particle ordering used before absolute-coordinate tokenization for dataset=lj_abs.",
    )
    parser.add_argument(
        "--lj_abs_hilbert_resolution",
        type=int,
        default=128,
        help="Hilbert grid resolution for dataset=lj_abs when --lj_abs_ordering=hilbert.",
    )
    parser.add_argument(
        "--vq_model",
        type=str,
        default=None,
        help="Hugging Face repo id or local path for the VQ model (required for cifar_vq).",
    )
    parser.add_argument("--vq_subfolder", type=str, default=None, help="Optional subfolder inside the VQ-GAN repository.")
    parser.add_argument("--vq_dtype", type=str, default=None, help="Optional torch dtype hint for VQ weights (e.g. float32, float16).")
    parser.add_argument("--vq_device", type=str, default="cpu", help="Device used to host the VQ encoder/decoder.")
    parser.add_argument("--vq_cache_latents", action="store_true", help="Cache encoded latents on the fly to speed up subsequent epochs.")
    parser.add_argument(
        "--vq_tmp_dir",
        type=str,
        default=None,
        help="Optional temporary directory to direct diffusers' temp files.",
    )
    parser.add_argument("--disable_hilbert", action="store_true", help="Use raster scan instead of Hilbert ordering for CIFAR AR tokens.")
    parser.add_argument("--ar_torus", action="store_true", help="Enable torus-aware minimum-image edge bias in AR attention.")
    parser.add_argument("--ar_no_edge_bias", action="store_true", help="Disable geometry-driven edge bias in AR attention.")
    parser.add_argument("--ar_no_dir_bias", action="store_true", help="Disable directional component in AR edge bias.")
    parser.add_argument(
        "--ar_arch",
        type=str,
        choices=AR_ARCH_CHOICES,
        default="auto",
        help=(
            "AR architecture selector: "
            "ida=transformer_ida.py, standard=transformer.py, vanilla=notebook-style transformer. "
            "When left as auto, legacy --use_ida decides between ida/standard."
        ),
    )
    parser.add_argument(
        "--ar_use_ida_pre",
        dest="ar_use_ida_pre",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable optional IDA pre-layer for transformer.py AR model (ignored when --use_ida is enabled).",
    )
    parser.add_argument(
        "--ar_ida_spatial_dim",
        type=int,
        default=2,
        help="Spatial dimension for transformer.py IDA pre-layer (typically 2 for LJ).",
    )
    parser.add_argument("--ar_dist_bins", type=int, default=32, help="Number of distance bins used by AR edge bias.")
    parser.add_argument(
        "--ar_max_dist",
        type=float,
        default=20.0,
        help="Distance saturation point for logarithmic edge-bias bucketing.",
    )
    parser.add_argument(
        "--use_rbf_bias",
        dest="use_rbf_bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use continuous RBF edge bias in transformer.py instead of discrete distance bins.",
    )
    parser.add_argument(
        "--use_deep_ida",
        dest="use_deep_ida",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use deep non-monotonic continuous distance bias in transformer.py.",
    )
    parser.add_argument(
        "--ar_disable_pos_emb",
        action="store_true",
        help="Deprecated alias; prefer default (no positional embeddings).",
    )
    parser.add_argument(
        "--ar_use_pos_emb",
        action="store_true",
        help="Enable absolute sequence-position embeddings (disabled by default).",
    )
    parser.add_argument(
        "--ar_use_rope",
        action="store_true",
        help="Deprecated; RoPE is disabled for this transferable setup.",
    )
    parser.add_argument(
        "--ar_rope_max_period",
        type=float,
        default=10000.0,
        help="RoPE base period parameter (larger means slower rotation with position).",
    )
    parser.add_argument("--ar_dropout", type=float, default=0.1, help="Dropout used by the AR transformer.")
    parser.add_argument(
        "--ar_permute_train",
        dest="ar_permute_train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For --ar_arch vanilla: randomly permute token groups after SOS during training only.",
    )
    parser.add_argument(
        "--ar_permute_group_size",
        type=int,
        default=1,
        help="For --ar_arch vanilla: number of consecutive tokens kept together when permuting.",
    )
    parser.add_argument(
        "--use_ida",
        dest="use_ida",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use IDA AR model from transformer_ida.py (default: True). Disable to use transformer.py.",
    )
    parser.add_argument(
        "--matmul_precision",
        type=str,
        default="high",
        choices=("highest", "high", "medium"),
        help="torch.set_float32_matmul_precision setting (use high/medium for Tensor Cores).",
    )
    parser.add_argument(
        "--use_continuous_head",
        action="store_true",
        help="Use a continuous MDN output head for lj_transferable autoregressive training.",
    )
    parser.add_argument(
        "--continuous_input",
        action="store_true",
        help="Feed the CONTINUOUS previous displacement as autoregressive input instead of the "
        "quantized discrete token (removes tokenizer bin-crossing drift; --ar_arch standard + "
        "--use_continuous_head, non-factorized, non-polar).",
    )
    parser.add_argument(
        "--arc_repr",
        type=int,
        default=0,
        help="Use Hilbert arc-length representation (Δs, fine_x, fine_y, fine_z) as AR target "
        "instead of (Δx, Δy, Δz). Requires --continuous_input, --ordering hilbert, --periodic. "
        "The 4D target is scale-invariant with constant cell size.",
    )
    parser.add_argument(
        "--num_mixtures",
        type=int,
        default=32,
        help="Number of Gaussian mixtures used by the continuous MDN head.",
    )
    parser.add_argument(
        "--full_covariance",
        action="store_true",
        help="Use a full-covariance Gaussian mixture head instead of independent per-dimension sigmas.",
    )
    parser.add_argument(
        "--lambda_var",
        type=float,
        default=0.0,
        help="Coefficient for the log-importance-weight variance penalty.",
    )
    parser.add_argument("--lj_kT", type=float, default=1.0, help="Lennard-Jones thermal energy kT.")
    parser.add_argument("--lj_epsilon", type=float, default=1.0, help="Lennard-Jones epsilon.")
    parser.add_argument("--lj_sigma", type=float, default=1.0, help="Lennard-Jones sigma.")
    parser.add_argument("--lj_cutoff", type=float, default=None, help="Optional Lennard-Jones cutoff radius.")
    parser.add_argument(
        "--lj_spring_constant",
        type=float,
        default=0.5,
        help="Harmonic spring constant used for nonperiodic Lennard-Jones evaluation.",
    )
    parser.add_argument(
        "--lj_boxlength",
        type=float,
        default=10.0,
        help="Box length used by the optional Lennard-Jones energy term.",
    )
    parser.add_argument(
        "--lj_periodic",
        action="store_true",
        help="Use periodic boundary conditions for the optional Lennard-Jones energy term.",
    )
    return parser.parse_args()


def build_data_module(args: argparse.Namespace):
    dataset = args.dataset
    if dataset == "cifar_vq":
        if not args.vq_model:
            raise ValueError("--vq_model must be provided when using dataset=cifar_vq")
        data_module = CIFARVQDataModule(
            data_root=args.data_dir,
            vq_model=args.vq_model,
            batch_size=args.batch_size,
            mask_ratio=args.mask_ratio,
            mask_ratio_max=args.mask_ratio_max,
            seed=args.seed,
            num_workers=args.num_workers,
            train_limit=args.train_limit,
            vq_subfolder=args.vq_subfolder,
            vq_dtype=args.vq_dtype,
            vq_device=args.vq_device,
            cache_latents=args.vq_cache_latents,
            tmp_dir=args.vq_tmp_dir,
            use_hilbert=(not args.disable_hilbert),
        )
        data_module.setup("fit")
        if data_module.latent_shape is None:
            raise RuntimeError("Failed to determine latent shape from CIFARVQDataModule.")
        c, h, w = data_module.latent_shape
        codebook_size = data_module.codebook_size
        if codebook_size is None:
            raise RuntimeError("Failed to determine VQ codebook size.")
        model_info = {
            "kind": "cifar_ar",
            "latent_shape": (c, h, w),
            "codebook_size": codebook_size,
        }
    elif dataset == "lj_cell":
        data_module = LJCellDataModule(
            data_path=args.data_dir,
            resolution=args.lj_cell_resolution,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            train_limit=args.train_limit,
        )
        data_module.setup("fit")
        vocab_size = data_module.vocab_size
        grid_size = data_module.grid_size
        if vocab_size is None or grid_size is None:
            raise RuntimeError("Failed to determine lj_cell vocabulary/grid size.")
        model_info = {
            "kind": "lj_cell_ar",
            "vocab_size": int(vocab_size),
            "grid_size": int(grid_size),
        }
    elif dataset == "lj_transferable":
        data_path = args.data_dir
        if os.path.isdir(data_path):
            h5_files = sorted(glob.glob(os.path.join(data_path, "*.h5")))
            if not h5_files:
                raise FileNotFoundError(f"No .h5 files found in directory: {data_path}")
            data_path = h5_files
        data_module = LJTransferableDataModule(
            data_path=data_path,
            periodic=bool(args.lj_transfer_periodic),
            hilbert_resolution=args.lj_transfer_hilbert_resolution,
            cell_size=args.lj_transfer_cell_size,
            ordering=args.lj_transfer_ordering,
            spectral_sigma=args.lj_transfer_spectral_sigma,
            local_window=args.lj_transfer_window,
            local_bins=args.lj_transfer_bins,
            use_long_jump_token=(not args.lj_transfer_no_long_jump),
            factorized=bool(args.factorized),
            polar=bool(args.polar),
            discrete=bool(args.discrete),
            codebook_path=(str(args.codebook_path) if args.discrete else None),
            random_grid_shift=(not args.lj_transfer_disable_random_shift),
            use_data_aug=bool(args.lj_transfer_use_data_aug),
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            train_limit=args.train_limit,
            preprocessed_path=args.lj_transfer_preprocessed_path,
            use_curve_rail=bool(args.lj_transfer_use_curve_rail),
            curve_rail_offsets=args.lj_transfer_curve_rail_offsets,
            curve_rail_mode=str(args.lj_transfer_curve_rail_mode),
            curve_rail_window=float(args.lj_transfer_curve_rail_window),
            curve_rail_k=int(args.lj_transfer_curve_rail_k),
            curve_rail_reference=str(args.lj_transfer_curve_rail_reference),
            curve_rail_residual_target=bool(args.lj_transfer_curve_rail_residual_target),
            arc_repr=bool(int(args.arc_repr)),
        )
        data_module.setup("fit")
        vocab_size = data_module.vocab_size
        if vocab_size is None:
            raise RuntimeError("Failed to determine lj_transferable vocabulary size.")
        model_info = {
            "kind": "lj_transferable_ar",
            "vocab_size": int(vocab_size),
            "coord_dim": int(data_module.coord_dim or 2),
            "factorized": bool(args.factorized),
            "polar": bool(args.polar),
            "discrete": bool(args.discrete),
            "binned_discrete": bool(args.binned_discrete),
        }
    elif dataset == "lj_abs":
        data_path = args.data_dir
        if os.path.isdir(data_path):
            h5_files = sorted(glob.glob(os.path.join(data_path, "*.h5")))
            if not h5_files:
                raise FileNotFoundError(f"No .h5 files found in directory: {data_path}")
            data_path = h5_files
        data_module = LJAbsoluteDataModule(
            data_path=data_path,
            bins=args.lj_abs_bins,
            ordering=args.lj_abs_ordering,
            hilbert_resolution=args.lj_abs_hilbert_resolution,
            random_grid_shift=(not args.lj_abs_disable_random_shift),
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            train_limit=args.train_limit,
        )
        data_module.setup("fit")
        vocab_size = data_module.vocab_size
        num_particles = data_module.num_particles
        if vocab_size is None or num_particles is None:
            raise RuntimeError("Failed to determine lj_abs vocabulary/particle count.")
        model_info = {
            "kind": "lj_abs_ar",
            "vocab_size": int(vocab_size),
            "num_particles": int(num_particles),
        }
    else:
        data_module = LJPixelsDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            pixel_size=args.pixel_size,
            mask_ratio=args.mask_ratio,
            mask_ratio_max=args.mask_ratio_max,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        model_info = {
            "kind": "lj_swin",
            "output_type": "binary",
            "latent_shape": None,
            "latent_downsample": 1.0,
            "input_channels": 1,
            "out_channels": 1,
            "pad_mode": "circular",
        }

    return data_module, model_info


def _validate_arc_repr_flags(args) -> None:
    """arc_repr targets are Hilbert-code differences: they require continuous input
    feedback and Hilbert ordering. Reads --lj_transfer_ordering (the actual CLI flag;
    an earlier version read the nonexistent args.ordering and never fired)."""
    if not int(args.arc_repr):
        return
    if not args.continuous_input:
        raise ValueError("--arc_repr requires --continuous_input.")
    if getattr(args, "lj_transfer_ordering", "hilbert") != "hilbert":
        raise ValueError("--arc_repr requires --lj_transfer_ordering hilbert.")


def main() -> None:
    args = parse_args()
    if args.discrete:
        args.factorized = False
        args.use_continuous_head = False
        args.full_covariance = False
    if args.binned_discrete:
        args.use_continuous_head = False
        args.full_covariance = False
    if args.use_rbf_bias and args.use_deep_ida:
        raise ValueError("--use_rbf_bias and --use_deep_ida are mutually exclusive.")
    if args.ar_permute_group_size <= 0:
        raise ValueError("--ar_permute_group_size must be positive.")
    if float(args.lambda_var) < 0.0:
        raise ValueError("--lambda_var must be non-negative.")
    if args.use_continuous_head:
        if args.dataset != "lj_transferable":
            raise ValueError("--use_continuous_head is only supported with --dataset lj_transferable.")
    if args.full_covariance and not args.use_continuous_head:
        raise ValueError("--full_covariance requires --use_continuous_head.")
    if args.continuous_input and not args.use_continuous_head:
        raise ValueError("--continuous_input currently requires --use_continuous_head.")
    _validate_arc_repr_flags(args)
    if args.discrete:
        if args.dataset != "lj_transferable":
            raise ValueError("--discrete is only supported with --dataset lj_transferable.")
        if args.polar:
            raise ValueError("--discrete is incompatible with --polar.")
    if args.binned_discrete:
        if args.dataset != "lj_transferable":
            raise ValueError("--binned_discrete is only supported with --dataset lj_transferable.")
        if args.discrete:
            raise ValueError("--binned_discrete is incompatible with --discrete.")
        if args.polar:
            raise ValueError("--binned_discrete is incompatible with --polar.")
    if args.polar:
        if args.dataset != "lj_transferable":
            raise ValueError("--polar is only supported with --dataset lj_transferable.")
        if not args.use_continuous_head:
            raise ValueError("--polar currently requires --use_continuous_head.")
        if args.factorized:
            raise ValueError(
                "--polar with --factorized is not supported in this checkout because "
                "spherical token-sequence conditioning is not wired."
            )
    if float(args.lambda_var) > 0.0 and args.dataset != "lj_transferable":
        raise ValueError("--lambda_var is currently supported only with --dataset lj_transferable.")
    if float(args.lambda_var) > 0.0 and not args.lj_transfer_preprocessed_path:
        raise ValueError(
            "--lambda_var requires --lj_transfer_preprocessed_path so target energies are cached "
            "during preprocessing."
        )

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.set_float32_matmul_precision(args.matmul_precision)

    data_module, model_info = build_data_module(args)

    if model_info["kind"] in ("cifar_ar", "lj_cell_ar", "lj_transferable_ar", "lj_abs_ar"):
        base_use_pos_emb = bool(args.ar_use_pos_emb) and (not args.ar_disable_pos_emb)
        if model_info["kind"] == "cifar_ar":
            latent_shape = model_info["latent_shape"]
            codebook_size = int(model_info["codebook_size"])
            d_model = int(latent_shape[0])
            id_coord_mode = None
            cell_grid_size = None
            coord_dequant_width = 0.0
            torus = args.ar_torus
            use_density_cond = False
            use_pos_emb = base_use_pos_emb
            use_rope = False
        elif model_info["kind"] == "lj_transferable_ar":
            codebook_size = int(model_info["vocab_size"])
            d_model = int(args.dim)
            id_coord_mode = None
            cell_grid_size = None
            coord_dequant_width = 0.0
            torus = bool(args.lj_transfer_periodic)
            use_density_cond = False
            use_pos_emb = False
            use_rope = False
        elif model_info["kind"] == "lj_abs_ar":
            codebook_size = int(model_info["vocab_size"])
            d_model = int(args.dim)
            id_coord_mode = None
            cell_grid_size = None
            coord_dequant_width = 0.0
            torus = False
            use_density_cond = False
            use_pos_emb = True
            use_rope = False
        else:
            codebook_size = int(model_info["vocab_size"])
            d_model = int(args.dim)
            id_coord_mode = "hilbert"
            cell_grid_size = int(model_info["grid_size"])
            coord_dequant_width = 0.0
            torus = True
            use_density_cond = False
            use_pos_emb = base_use_pos_emb
            use_rope = False
        if float(args.coord_dequant_width) > 0.0:
            coord_dequant_width = float(args.coord_dequant_width)
        ar_arch = resolve_ar_arch(args.ar_arch, use_ida=bool(args.use_ida))
        use_curve_rail = bool(getattr(data_module, "use_curve_rail", False))
        curve_rail_k = int(getattr(data_module, "curve_rail_k", 0))
        curve_rail_offsets = getattr(data_module, "curve_rail_offsets_eff", None)
        curve_rail_resolution = int(getattr(data_module, "hilbert_resolution_eff", 128))
        curve_rail_cell_size = getattr(data_module, "cell_size_eff", None)
        curve_rail_mode = str(getattr(data_module, "curve_rail_mode_eff", "lookahead"))
        curve_rail_window = float(getattr(data_module, "curve_rail_window_eff", 1.0))
        curve_rail_reference = str(getattr(data_module, "curve_rail_reference_eff", "absolute"))
        curve_rail_residual_target = bool(getattr(data_module, "curve_rail_residual_target_eff", False))
        if use_curve_rail and ar_arch != "standard":
            raise ValueError(
                "Curve-rail conditioning is currently supported only with --ar_arch standard, "
                f"got resolved ar_arch={ar_arch!r}."
            )
        if ar_arch == "ida":
            spatial_dim = int(model_info.get("coord_dim", args.ar_ida_spatial_dim))
            lit_module = GraphormerARIDA(
                K=codebook_size,
                d_model=d_model,
                n_layer=args.depth,
                n_head=args.heads,
                dropout=args.ar_dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                sos_id=codebook_size,
                input_vocab_size=codebook_size + 1,
                use_ida=(not args.ar_no_edge_bias),
                spatial_dim=spatial_dim,
                torus=torus,
                coord_dequantize_train=True,
                coord_dequant_width=float(coord_dequant_width),
                id_coord_mode=id_coord_mode,
                cell_grid_size=cell_grid_size,
                use_pos_emb=use_pos_emb,
                use_density_cond=use_density_cond,
                use_rope=use_rope,
                rope_max_period=args.ar_rope_max_period,
                is_factorized=bool(args.factorized),
                polar=bool(args.polar),
                discrete=bool(args.discrete),
                binned_discrete=bool(args.binned_discrete),
                use_continuous_head=bool(args.use_continuous_head),
                num_mixtures=int(args.num_mixtures),
                full_covariance=bool(args.full_covariance),
                lambda_var=float(args.lambda_var),
                lj_kT=float(args.lj_kT),
                lj_epsilon=float(args.lj_epsilon),
                lj_sigma=float(args.lj_sigma),
                lj_cutoff=args.lj_cutoff,
                lj_spring_constant=float(args.lj_spring_constant),
                lj_boxlength=float(args.lj_boxlength),
                lj_periodic=bool(args.lj_periodic),
            )
        elif ar_arch == "standard":
            lit_module = GraphormerARStd(
                K=codebook_size,
                d_model=d_model,
                n_layer=args.depth,
                n_head=args.heads,
                dropout=args.ar_dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                sos_id=codebook_size,
                input_vocab_size=codebook_size + 1,
                use_edge_bias=(not args.ar_no_edge_bias),
                use_ida_pre=bool(args.ar_use_ida_pre),
                ida_spatial_dim=int(model_info.get("coord_dim", args.ar_ida_spatial_dim)),
                torus=torus,
                use_dir_bias=(not args.ar_no_dir_bias),
                dist_bins=args.ar_dist_bins,
                edge_max_dist=args.ar_max_dist,
                use_rbf_bias=bool(args.use_rbf_bias),
                use_deep_ida=bool(args.use_deep_ida),
                coord_dequantize_train=True,
                coord_dequant_width=float(coord_dequant_width),
                id_coord_mode=id_coord_mode,
                cell_grid_size=cell_grid_size,
                use_pos_emb=use_pos_emb,
                use_density_cond=use_density_cond,
                use_rope=use_rope,
                rope_max_period=args.ar_rope_max_period,
                is_factorized=bool(args.factorized),
                polar=bool(args.polar),
                discrete=bool(args.discrete),
                binned_discrete=bool(args.binned_discrete),
                use_continuous_head=bool(args.use_continuous_head),
                continuous_input=bool(args.continuous_input),
                num_mixtures=int(args.num_mixtures),
                full_covariance=bool(args.full_covariance),
                lambda_var=float(args.lambda_var),
                lj_kT=float(args.lj_kT),
                lj_epsilon=float(args.lj_epsilon),
                lj_sigma=float(args.lj_sigma),
                lj_cutoff=args.lj_cutoff,
                lj_spring_constant=float(args.lj_spring_constant),
                lj_boxlength=float(args.lj_boxlength),
                lj_periodic=bool(args.lj_periodic),
                use_curve_rail=use_curve_rail,
                curve_rail_k=(curve_rail_k if use_curve_rail else 6),
                curve_rail_offsets=(curve_rail_offsets if use_curve_rail else None),
                hilbert_resolution=curve_rail_resolution,
                cell_size=curve_rail_cell_size,
                curve_rail_mode=curve_rail_mode,
                curve_rail_window=curve_rail_window,
                curve_rail_reference=curve_rail_reference,
                curve_rail_residual_target=curve_rail_residual_target,
                arc_repr=bool(int(args.arc_repr)),
            )
        elif ar_arch == "vanilla":
            if args.use_continuous_head:
                raise ValueError("--use_continuous_head is not supported with --ar_arch vanilla.")
            if float(args.lambda_var) > 0.0:
                raise ValueError("--lambda_var is not supported with --ar_arch vanilla.")
            lit_module = VanillaTransformerAR(
                K=codebook_size,
                d_model=d_model,
                n_layer=args.depth,
                n_head=args.heads,
                dropout=args.ar_dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                sos_id=codebook_size,
                input_vocab_size=codebook_size + 1,
                use_pos_emb=True,
                permute_train=bool(args.ar_permute_train),
                permute_group_size=int(args.ar_permute_group_size),
            )
        else:
            raise ValueError(f"Unsupported AR architecture '{ar_arch}'.")
        lit_module.hparams["ar_arch"] = ar_arch
    else:
        model_kwargs = {
            "in_ch": model_info["input_channels"],
            "out_ch": model_info["out_channels"],
            "dim": args.dim,
            "depth": args.depth,
            "heads": args.heads,
            "win": args.window,
            "shifted": args.shifted,
            "pad_mode": model_info["pad_mode"],
        }
        lit_module = VQLatentTransformerModule(
            model_kwargs=model_kwargs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            mask_pos_weight=args.mask_pos_weight,
            output_type=model_info["output_type"],
            latent_shape=model_info["latent_shape"],
            latent_downsample=model_info["latent_downsample"],
            cheat_linear=False,
        )

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="best",
        monitor="train/loss_epoch",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )

    callbacks = [checkpoint_cb]
    if args.ckpt_every_n_steps > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=args.ckpt_dir,
                filename="step-{step}",
                every_n_train_steps=args.ckpt_every_n_steps,
                save_top_k=-1,
                monitor=None,
                auto_insert_metric_name=False,
            )
        )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        default_root_dir=args.ckpt_dir,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
    )

    trainer.fit(lit_module, datamodule=data_module, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
