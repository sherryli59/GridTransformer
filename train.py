import argparse
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
)
from grid_transformer.training.ar import GraphormerAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train binary/AR models with PyTorch Lightning.")
    parser.add_argument("--data_dir", type=str, required=True, help="Training data root directory/archive or LJ HDF5 file path.")
    parser.add_argument(
        "--dataset",
        choices=("lj", "cifar_vq", "lj_cell"),
        default="cifar_vq",
        help="Dataset loader to use (lj=NPZ coords, cifar_vq=VQ-tokenised CIFAR-10, lj_cell=Hilbert-cell LJ tokens).",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for AdamW.")
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
    parser.add_argument("--ar_dist_bins", type=int, default=32, help="Number of distance bins used by AR edge bias.")
    parser.add_argument("--ar_dropout", type=float, default=0.1, help="Dropout used by the AR transformer.")
    parser.add_argument(
        "--matmul_precision",
        type=str,
        default="high",
        choices=("highest", "high", "medium"),
        help="torch.set_float32_matmul_precision setting (use high/medium for Tensor Cores).",
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


def main() -> None:
    args = parse_args()

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.set_float32_matmul_precision(args.matmul_precision)

    data_module, model_info = build_data_module(args)

    if model_info["kind"] in ("cifar_ar", "lj_cell_ar"):
        if model_info["kind"] == "cifar_ar":
            latent_shape = model_info["latent_shape"]
            codebook_size = int(model_info["codebook_size"])
            d_model = int(latent_shape[0])
            id_coord_mode = None
            cell_grid_size = None
            torus = args.ar_torus
        else:
            codebook_size = int(model_info["vocab_size"])
            d_model = int(args.dim)
            id_coord_mode = "hilbert"
            cell_grid_size = int(model_info["grid_size"])
            torus = True
        lit_module = GraphormerAR(
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
            torus=torus,
            use_dir_bias=(not args.ar_no_dir_bias),
            dist_bins=args.ar_dist_bins,
            id_coord_mode=id_coord_mode,
            cell_grid_size=cell_grid_size,
        )
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
