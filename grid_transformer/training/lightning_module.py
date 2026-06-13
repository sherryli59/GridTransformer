"""PyTorch Lightning modules for binary pixels and VQ latent transformers."""

from __future__ import annotations

import glob
import os
import warnings
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl

from ..data.lj_dataset import LJPixelsDataset
from ..data.cifar_vq_dataset import CIFARVQLatentDataset
from ..data.lj_cell_dataset import LJCellDataset
from ..data.lj_abs_dataset import LJAbsoluteDataset
from ..data.lj_transferable import (
    BucketedBatchSampler,
    LJTransferableCachedDataset,
    LJTransferableDataset,
)
from ..models.swin_pbc import BinarySwinPBC
import numpy as np
import math

def bce_on_mask(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pos_weight: float = 10.0) -> torch.Tensor:
    """Binary cross entropy computed only over masked spatial locations."""
    mask_bool = mask > 0.5
    if mask_bool.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logits_m = logits[mask_bool.expand_as(logits)]
    target_m = target[mask_bool.expand_as(target)]
    weight = torch.tensor(pos_weight, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits_m, target_m, pos_weight=weight)


def mse_on_mask(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error restricted to masked spatial locations."""
    mask_bool = mask > 0.5
    if mask_bool.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    mask_bool = mask_bool.expand(-1, pred.size(1), -1, -1)
    pred_sel = torch.masked_select(pred, mask_bool)
    target_sel = torch.masked_select(target, mask_bool)
    return F.mse_loss(pred_sel, target_sel)


def ce_on_mask(logits: torch.Tensor, target_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over masked positions for categorical logits."""
    if target_idx.dtype != torch.long:
        target_idx = target_idx.long()
    mask_bool = mask.squeeze(1) > 0.5
    if mask_bool.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.size(1))
    target_flat = target_idx.reshape(-1)
    mask_flat = mask_bool.reshape(-1)
    logits_sel = logits_flat[mask_flat]
    target_sel = target_flat[mask_flat]
    return F.cross_entropy(logits_sel, target_sel)



class VQLatentTransformerModule(pl.LightningModule):
    def __init__(self, model_kwargs: Optional[Dict[str, Any]] = None,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        mask_pos_weight: float = 10.0,
        output_type: str = "binary",
        latent_shape: Optional[Any] = None,
        latent_downsample: float = 1.0,
        cheat_linear: bool = False):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.mask_pos_weight = mask_pos_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.output_type = output_type
        self.latent_downsample = latent_downsample
        self.latent_shape = latent_shape
        # unpack shapes
        assert latent_shape is not None, "Need latent_shape (D,Hc,Wc)"
        D, Hc, Wc = latent_shape
        self.D, self.Hc, self.Wc = D, Hc, Wc
        self.K = model_kwargs["out_ch"]  # codebook size
        self.token_emb = torch.nn.Embedding(self.K + 1, self.D)
        self.mask_token = torch.nn.Parameter(torch.zeros(self.D))  # learned [D]
        self.logit_scale = torch.nn.Parameter(torch.tensor(10.0))   # optional sharpening for logits
        self.cheat_linear = cheat_linear
        model_kwargs = model_kwargs or {}
        model_kwargs["in_ch"] = self.D
        if not self.cheat_linear:
            # your normal backbone
            self.model = BinarySwinPBC(**model_kwargs)
        else:
              # ---- Token-ID path (cheat version) ----
            # light 1x1 projection + tiny context mixer
            self.input_proj = torch.nn.Conv2d(self.D, self.D, kernel_size=1)

            def conv3x3(cin, cout, dilation=1):
                pad = dilation
                return torch.nn.Conv2d(cin, cout, kernel_size=3, padding=pad, dilation=dilation)

            self.context = torch.nn.Sequential(
                conv3x3(self.D, self.D, dilation=1), torch.nn.GELU(),
                conv3x3(self.D, self.D, dilation=2), torch.nn.GELU(),
                conv3x3(self.D, self.D, dilation=3), torch.nn.GELU(),
            )

            self.cheat_head  = torch.nn.Conv2d(self.D, self.K, kernel_size=1, bias=True)
            self.logit_scale = torch.nn.Parameter(torch.tensor(5.0))
        self.register_buffer("E_weight", torch.empty(0), persistent=False)  # placeholder

    def on_fit_start(self) -> None:
        dm = self.trainer.datamodule
        E = getattr(dm, "codebook_weight", None)
        if E is None:
            raise RuntimeError("DataModule has no codebook_weight")
        # move to module device and save as buffer
        self.E_weight = E.to(self.device)  # [K, D]
        # (optional safety) check head size matches K
        if hasattr(self, "num_classes"):
            assert self.num_classes == self.E_weight.shape[0], "K mismatch vs head"

    def forward(self, input_idx: torch.Tensor):
        # input_idx: [B, Hc, Wc], Long in [0..K] (K = [MASK])
        B, H, W = input_idx.shape
        assert H == self.Hc and W == self.Wc, f"got {(H,W)}, expected {(self.Hc,self.Wc)}"

        # (1) embed tokens
        x = self.token_emb(input_idx)    # [B,H,W,D]
        x = x.permute(0,3,1,2).contiguous()  # [B,D,H,W]

        # (2) pass through model
        if self.cheat_linear:
            # 1×1 conv only
            h = self.input_proj(x)          # [B,D,H,W]
            h = self.context(h) + h         # residual context
            logits = self.logit_scale * self.cheat_head(h)   # [B,K,H,W]
            return logits

        else:
            # normal Swin backbone
            x_cat = x                        # [B,D,H,W]
            logits = self.model(x_cat)       # [B,K,H,W]
            return logits


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        preds = self.forward(batch["input_idx"])
        if self.output_type == "binary":
            loss = bce_on_mask(preds, batch["target_idx"], batch["mask"], pos_weight=self.mask_pos_weight)
        elif self.output_type == "latent":
            loss = mse_on_mask(preds, batch["target_idx"], batch["mask"])
        elif self.output_type == "latent_logits":
            loss = ce_on_mask(preds, batch["target_idx"], batch["mask"])
        else:
            raise ValueError(f"Unknown output_type '{self.output_type}'")
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch["input_idx"].size(0),
        )
        with torch.no_grad():
            # Shapes and basics
            K = preds.size(1)
            B, H, W = batch["target_idx"].shape
            mask_bool = (batch["mask"] > 0.5).squeeze(1)            # [B,H,W] boolean
            self.log("train/debug_K", float(K), on_step=True)
            self.log("train/debug_mask_frac", mask_bool.float().mean(), on_step=True)

            # Visible sites sanity (for IDs pipeline, visible input == target)
            vis_bool = ~mask_bool                                   # [B,H,W]
            vis_eq = (batch["input_idx"][vis_bool] == batch["target_idx"][vis_bool])
            if vis_eq.numel() > 0:
                self.log("train/debug_visible_match_rate", vis_eq.float().mean(), on_step=True)

            # Neighborhood visibility around masked positions (3x3)
            vis_float = vis_bool.float().unsqueeze(1)               # [B,1,H,W]
            vis_neighbors = F.avg_pool2d(vis_float, kernel_size=3, stride=1, padding=1) * 9.0  # [B,1,H,W]
            if mask_bool.any():
                self.log("train/debug_mean_vis_neighbors_masked",
                        vis_neighbors.squeeze(1)[mask_bool].mean(), on_step=True)

            # Softmax on logits
            prob = F.softmax(preds, dim=1)                          # [B,K,H,W]

            # Gather masked positions
            if mask_bool.any():
                # entropy over masked
                masked_prob = prob.permute(0,2,3,1)[mask_bool]      # [Nm, K]
                eps = 1e-12
                ent = -(masked_prob * (masked_prob.clamp_min(eps)).log()).sum(dim=1)  # [Nm]
                ent_norm = ent / math.log(K)
                self.log("train/debug_mask_entropy", ent.mean(), on_step=True)
                self.log("train/debug_mask_entropy_norm", ent_norm.mean(), on_step=True)

                # confidence and accuracy over masked
                top1p = masked_prob.max(dim=1).values.mean()
                pred_idx = masked_prob.argmax(dim=1)                # [Nm]
                target_masked = batch["target_idx"][mask_bool]      # [Nm]
                acc_masked = (pred_idx == target_masked).float().mean()

                self.log("train/debug_mask_top1p", top1p, on_step=True)
                self.log("train/debug_mask_acc", acc_masked, on_step=True)

            # Also log shapes (as scalars to avoid TB text clutter)
            self.log("train/debug_Hc", float(H), on_step=True)
            self.log("train/debug_Wc", float(W), on_step=True)
            self.log("train/debug_batch", float(B), on_step=True)

        if batch_idx == 0:
            n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            self.log("train/debug_n_params", float(n_params), on_step=True)
        return loss

    def on_after_backward(self):
        hot = []
        for name, p in self.named_parameters():
            if p.grad is not None and p.requires_grad:
                g = p.grad.detach().norm().item()
                hot.append((g, name))
        hot.sort(reverse=True)
        for g, name in hot[:8]:
            self.log(f"grad_top/{name}", g, on_step=True)
        self.log("train/debug_grad_norm", sum(g for g,_ in hot), on_step=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class LJPixelsDataModule(pl.LightningDataModule):
    """LightningDataModule that provides training batches for the LJ pixel dataset."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        pixel_size: float = 0.25,
        mask_ratio: float = 0.6,
        mask_ratio_max: Optional[float] = None,
        seed: int = 0,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.pixel_size = pixel_size
        self.mask_ratio = mask_ratio
        self.mask_ratio_max = mask_ratio_max
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._train_ds: Optional[LJPixelsDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self._train_ds is None:
            mask_ratio = self.mask_ratio
            if self.mask_ratio_max is not None:
                mask_ratio = (self.mask_ratio, self.mask_ratio_max)
            self._train_ds = LJPixelsDataset(
                self.data_dir,
                pixel_size=self.pixel_size,
                mask_ratio=mask_ratio,
                seed=self.seed,
            )

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            self.setup("fit")
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class LJCellDataModule(pl.LightningDataModule):
    """LightningDataModule for canonical Hilbert-sorted LJ cell-token sequences."""

    def __init__(
        self,
        data_path: str = "/mnt/ssd/mcmc/lj_N16_T1.h5",
        resolution: int = 64,
        batch_size: int = 128,
        seed: int = 0,
        num_workers: int = 0,
        pin_memory: bool = True,
        train_limit: Optional[int] = None,
    ) -> None:
        super().__init__()
        _ = seed  # reserved for future stochastic augmentation
        self.data_path = data_path
        self.resolution = int(resolution)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_limit = train_limit
        self._train_ds: Optional[LJCellDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self._train_ds is None:
            self._train_ds = LJCellDataset(
                h5_path=self.data_path,
                resolution=self.resolution,
                limit=self.train_limit,
            )

    @property
    def vocab_size(self) -> Optional[int]:
        if self._train_ds is None:
            return None
        return self._train_ds.vocab_size

    @property
    def grid_size(self) -> Optional[int]:
        if self._train_ds is None:
            return None
        return self._train_ds.resolution

    @property
    def sequence_length(self) -> Optional[int]:
        if self._train_ds is None:
            return None
        return self._train_ds.num_particles

    @property
    def sos_id(self) -> Optional[int]:
        if self._train_ds is None:
            return None
        return self._train_ds.sos_id

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            self.setup("fit")
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class LJTransferableDataModule(pl.LightningDataModule):
    """DataModule for relative-displacement LJ token sequences with configurable ordering."""

    def __init__(
        self,
        data_path: str | Sequence[str] = "/mnt/ssd/mcmc/lj_N16_T1.h5",
        periodic: bool = True,
        hilbert_resolution: int = 128,
        cell_size: Optional[float] = None,
        ordering: str = "hilbert",
        spectral_sigma: float = 1.0,
        local_window: float = 3.0,
        local_bins: int = 64,
        use_long_jump_token: bool = True,
        factorized: bool = False,
        polar: bool = False,
        discrete: bool = False,
        codebook_path: Optional[str] = None,
        random_grid_shift: bool = True,
        use_data_aug: bool = False,
        batch_size: int = 128,
        seed: int = 0,
        num_workers: int = 0,
        pin_memory: bool = True,
        train_limit: Optional[int] = None,
        drop_last: bool = False,
        preprocessed_path: Optional[str] = None,
        use_curve_rail: bool = False,
        curve_rail_offsets: Optional[Sequence[int]] = None,
        curve_rail_mode: str = "lookahead",
        curve_rail_window: float = 1.0,
        curve_rail_k: int = 8,
        curve_rail_reference: str = "absolute",
        curve_rail_residual_target: bool = False,
        arc_repr: bool = False,
        val_frac: float = 0.0,
    ) -> None:
        super().__init__()
        self.val_frac = float(val_frac)
        if not (0.0 <= self.val_frac < 1.0):
            raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
        self.data_path = data_path
        self.periodic = bool(periodic)
        self.hilbert_resolution = int(hilbert_resolution)
        self.cell_size = None if cell_size is None else float(cell_size)
        self.ordering = str(ordering).strip().lower()
        if self.ordering not in ("hilbert", "gilbert", "spectral"):
            raise ValueError(
                f"ordering must be 'hilbert', 'gilbert' or 'spectral', got {ordering!r}"
            )
        self.spectral_sigma = float(spectral_sigma)
        if self.spectral_sigma <= 0.0:
            raise ValueError(f"spectral_sigma must be > 0, got {self.spectral_sigma}")
        self.local_window = float(local_window)
        self.local_bins = int(local_bins)
        self.use_long_jump_token = bool(use_long_jump_token)
        self.factorized = bool(factorized)
        self.polar = bool(polar)
        self.discrete = bool(discrete)
        self.codebook_path = codebook_path
        self.random_grid_shift = bool(random_grid_shift)
        self.use_data_aug = bool(use_data_aug)
        self.batch_size = batch_size
        self.seed = int(seed)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_limit = train_limit
        self.drop_last = bool(drop_last)
        self.preprocessed_path = preprocessed_path
        self.use_curve_rail = bool(use_curve_rail)
        self.curve_rail_offsets = curve_rail_offsets
        self.curve_rail_mode = str(curve_rail_mode)
        self.curve_rail_window = float(curve_rail_window)
        self.curve_rail_k_cfg = int(curve_rail_k)
        self.curve_rail_reference = str(curve_rail_reference)
        self.curve_rail_residual_target = bool(curve_rail_residual_target)
        self.arc_repr = bool(arc_repr)
        if self.arc_repr and self.preprocessed_path is None:
            raise ValueError(
                "arc_repr=True requires a preprocessed cache (preprocessed_path): "
                "the on-the-fly LJTransferableDataset does not produce 'arc_delta' targets. "
                "Run preprocess_lj_transferable.py first."
            )
        if self.preprocessed_path is not None and self.random_grid_shift:
            warnings.warn(
                "preprocessed_path is set, but random_grid_shift=True. "
                "Cached datasets have fixed ordering/shift; disabling random_grid_shift for cached training.",
                stacklevel=2,
            )
            self.random_grid_shift = False
        if self.preprocessed_path is not None and self.use_data_aug:
            raise ValueError(
                "preprocessed_path is set, but use_data_aug=True. "
                "Right-angle rotation augmentation requires on-the-fly lj_transferable tokenization."
            )
        if self.discrete and self.factorized:
            raise ValueError("discrete=True requires factorized=False.")
        if self.discrete and self.polar:
            raise ValueError("discrete=True is incompatible with polar=True.")
        if self.discrete and self.preprocessed_path is None and not self.codebook_path:
            raise ValueError("discrete=True without a preprocessed cache requires codebook_path.")
        if (not self.periodic) and self.random_grid_shift:
            raise ValueError(
                "random_grid_shift=True is unsupported for nonperiodic lj_transferable training."
            )
        self._train_ds: Optional[LJTransferableDataset | LJTransferableCachedDataset] = None
        self._train_indices: Optional[np.ndarray] = None
        self._val_indices: Optional[np.ndarray] = None

    def _split_val(self) -> None:
        """Deterministic train/val split (seeded) for clean-input validation NLL."""
        if self.val_frac <= 0.0 or self._train_ds is None or self._train_indices is not None:
            return
        n = len(self._train_ds)
        n_val = int(round(n * self.val_frac))
        if n_val == 0:
            return
        perm = np.random.default_rng(self.seed).permutation(n)
        self._val_indices = np.sort(perm[:n_val])
        self._train_indices = np.sort(perm[n_val:])

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self._train_ds is None:
            if self.preprocessed_path is not None:
                self._train_ds = LJTransferableCachedDataset(
                    cache_path=self.preprocessed_path,
                    limit=self.train_limit,
                    discrete=self.discrete,
                    codebook_path=self.codebook_path,
                    arc_repr=self.arc_repr,
                    use_curve_rail=self.use_curve_rail,
                    curve_rail_mode=self.curve_rail_mode,
                    curve_rail_k=self.curve_rail_k_cfg,
                    curve_rail_window=self.curve_rail_window,
                    curve_rail_reference=self.curve_rail_reference,
                )
                cached_periodic = bool(getattr(self._train_ds, "periodic", True))
                if cached_periodic != self.periodic:
                    raise ValueError(
                        "Cached lj_transferable periodic setting does not match the datamodule: "
                        f"cache periodic={cached_periodic}, requested periodic={self.periodic}."
                    )
                cached_factorized = bool(getattr(self._train_ds, "factorized", False))
                if cached_factorized != self.factorized:
                    raise ValueError(
                        "Cached lj_transferable factorized setting does not match the datamodule: "
                        f"cache factorized={cached_factorized}, requested factorized={self.factorized}."
                    )
                cached_polar = bool(getattr(self._train_ds, "polar", False))
                if cached_polar != self.polar:
                    raise ValueError(
                        "Cached lj_transferable polar setting does not match the datamodule: "
                        f"cache polar={cached_polar}, requested polar={self.polar}."
                    )
                cached_rail = bool(getattr(self._train_ds, "use_curve_rail", False))
                if self.use_curve_rail and not cached_rail:
                    raise ValueError(
                        "use_curve_rail=True but the cache has no curve-rail geometry. "
                        "Rebuild the cache with use_curve_rail=True (RUN_PREPROCESS)."
                    )
                # Reflect what the cache actually provides so the model is configured to match.
                self.use_curve_rail = cached_rail
                self._split_val()
                return
            data_paths: str | Sequence[str]
            data_paths = self.data_path
            if isinstance(data_paths, str) and os.path.isdir(data_paths):
                h5_files = sorted(glob.glob(os.path.join(data_paths, "*.h5")))
                if not h5_files:
                    raise FileNotFoundError(f"No .h5 files found in directory: {data_paths}")
                data_paths = h5_files
            self._train_ds = LJTransferableDataset(
                file_paths=data_paths,
                periodic=self.periodic,
                hilbert_resolution=self.hilbert_resolution,
                cell_size=self.cell_size,
                ordering=self.ordering,
                spectral_sigma=self.spectral_sigma,
                local_window=self.local_window,
                local_bins=self.local_bins,
                use_long_jump_token=self.use_long_jump_token,
                factorized=self.factorized,
                polar=self.polar,
                discrete=self.discrete,
                codebook_path=self.codebook_path,
                random_grid_shift=self.random_grid_shift,
                use_data_aug=self.use_data_aug,
                limit=self.train_limit,
                seed=self.seed,
                use_curve_rail=self.use_curve_rail,
                curve_rail_offsets=self.curve_rail_offsets,
                curve_rail_mode=self.curve_rail_mode,
                curve_rail_window=self.curve_rail_window,
                curve_rail_k=self.curve_rail_k_cfg,
                curve_rail_reference=self.curve_rail_reference,
                curve_rail_residual_target=self.curve_rail_residual_target,
            )
            self._split_val()

    @property
    def vocab_size(self) -> Optional[int]:
        if self._train_ds is None:
            return None
        return self._train_ds.vocab_size

    @property
    def sequence_length(self) -> Optional[int]:
        if self._train_ds is None:
            return None
        if isinstance(self._train_ds, LJTransferableCachedDataset):
            uniq = np.unique(self._train_ds.sample_lengths)
            if uniq.size == 1:
                base_len = int(uniq[0])
                if getattr(self, "factorized", False) or getattr(self._train_ds, "factorized", False):
                    coord_dim = int(getattr(self._train_ds, "coord_dim", self.coord_dim or 3))
                    target_idx_all = getattr(self._train_ds, "target_idx_all", None)
                    if target_idx_all is not None and base_len == int(target_idx_all.shape[1]):
                        return int(base_len)
                    particle_length_all = getattr(self._train_ds, "particle_length_all", None)
                    if particle_length_all is not None:
                        particle_uniq = np.unique(particle_length_all.numpy())
                        if particle_uniq.size == 1 and base_len == int(particle_uniq[0]):
                            return max(0, base_len - 1) * coord_dim
                    delta_length_all = getattr(self._train_ds, "delta_length_all", None)
                    if delta_length_all is not None:
                        delta_uniq = np.unique(delta_length_all.numpy())
                        if delta_uniq.size == 1 and base_len == int(delta_uniq[0]):
                            return int(base_len) * coord_dim
                return int(base_len)
            return None
        uniq = np.unique(self._train_ds.sample_lengths)
        if uniq.size == 1:
            n_particles = int(uniq[0])
            n_predict = max(0, n_particles - 1)
            if bool(getattr(self._train_ds.tokenizer, "factorized", False)):
                return int(n_predict * int(getattr(self._train_ds, "coord_dim", 2)))
            return int(n_predict)
        return None

    @property
    def sos_id(self) -> Optional[int]:
        if self._train_ds is None:
            return None
        return self._train_ds.sos_id

    @property
    def density(self) -> Optional[float]:
        if self._train_ds is None:
            return None
        return None

    @property
    def coord_dim(self) -> Optional[int]:
        if self._train_ds is None:
            return None
        return int(getattr(self._train_ds, "coord_dim", 2))

    @property
    def curve_rail_k(self) -> int:
        """Number of rail waypoints K actually provided by the active dataset."""
        if self._train_ds is None or not self.use_curve_rail:
            return 0
        k = getattr(self._train_ds, "curve_rail_k", None)
        if k is not None:
            return int(k)
        wp = getattr(self._train_ds, "curve_waypoints_all", None)  # cached dataset
        if wp is not None:
            return int(wp.shape[2])
        offs = getattr(self._train_ds, "curve_rail_offsets", None)  # on-the-fly dataset
        return int(len(offs)) if offs is not None else 0

    @property
    def curve_rail_offsets_eff(self):
        """Hilbert index offsets actually used by the active dataset (or None)."""
        if self._train_ds is None or not self.use_curve_rail:
            return None
        offs = getattr(self._train_ds, "curve_rail_offsets", None)
        if offs is None:
            return None
        return [int(o) for o in offs]

    @property
    def hilbert_resolution_eff(self) -> int:
        if self._train_ds is None:
            return 128
        return int(getattr(self._train_ds, "hilbert_resolution", 128))

    @property
    def cell_size_eff(self):
        if self._train_ds is None:
            return None
        cs = getattr(self._train_ds, "cell_size", None)
        return None if cs is None else float(cs)

    @property
    def curve_rail_mode_eff(self) -> str:
        if self._train_ds is None:
            return self.curve_rail_mode
        return str(getattr(self._train_ds, "curve_rail_mode", self.curve_rail_mode))

    @property
    def curve_rail_window_eff(self) -> float:
        if self._train_ds is None:
            return self.curve_rail_window
        return float(getattr(self._train_ds, "curve_rail_window", self.curve_rail_window))

    @property
    def curve_rail_reference_eff(self) -> str:
        if self._train_ds is None:
            return self.curve_rail_reference
        return str(getattr(self._train_ds, "curve_rail_reference", self.curve_rail_reference))

    @property
    def curve_rail_residual_target_eff(self) -> bool:
        if self._train_ds is None:
            return self.curve_rail_residual_target
        return bool(getattr(self._train_ds, "curve_rail_residual_target", self.curve_rail_residual_target))

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            self.setup("fit")
        assert self._train_ds is not None
        ds: Any = self._train_ds
        lengths = np.asarray(self._train_ds.sample_lengths)
        if self._train_indices is not None:
            ds = Subset(self._train_ds, self._train_indices.tolist())
            lengths = lengths[self._train_indices]
        batch_sampler = BucketedBatchSampler(
            sample_lengths=lengths,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            seed=self.seed,
        )
        return DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self._train_ds is None:
            self.setup("fit")
        if self._val_indices is None:
            return None
        lengths = np.asarray(self._train_ds.sample_lengths)[self._val_indices]
        batch_sampler = BucketedBatchSampler(
            sample_lengths=lengths,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            seed=self.seed,
        )
        return DataLoader(
            Subset(self._train_ds, self._val_indices.tolist()),
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class LJAbsoluteDataModule(pl.LightningDataModule):
    """DataModule for absolute-coordinate LJ token sequences."""

    def __init__(
        self,
        data_path: str | Sequence[str] = "/mnt/ssd/mcmc/lj_N16_T1.h5",
        bins: int = 512,
        ordering: str = "raw",
        hilbert_resolution: int = 128,
        random_grid_shift: bool = False,
        batch_size: int = 128,
        seed: int = 0,
        num_workers: int = 0,
        pin_memory: bool = True,
        train_limit: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.bins = int(bins)
        self.ordering = str(ordering).strip().lower()
        self.hilbert_resolution = int(hilbert_resolution)
        self.random_grid_shift = bool(random_grid_shift)
        self.batch_size = batch_size
        self.seed = int(seed)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_limit = train_limit
        self._train_ds: Optional[LJAbsoluteDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self._train_ds is None:
            data_paths: str | Sequence[str]
            data_paths = self.data_path
            if isinstance(data_paths, str) and os.path.isdir(data_paths):
                h5_files = sorted(glob.glob(os.path.join(data_paths, "*.h5")))
                if not h5_files:
                    raise FileNotFoundError(f"No .h5 files found in directory: {data_paths}")
                data_paths = h5_files
            self._train_ds = LJAbsoluteDataset(
                file_paths=data_paths,
                bins=self.bins,
                ordering=self.ordering,
                hilbert_resolution=self.hilbert_resolution,
                random_grid_shift=self.random_grid_shift,
                limit=self.train_limit,
                seed=self.seed,
            )

    @property
    def vocab_size(self) -> Optional[int]:
        if self._train_ds is None:
            return None
        return self._train_ds.vocab_size

    @property
    def sequence_length(self) -> Optional[int]:
        if self._train_ds is None:
            return None
        uniq = np.unique(self._train_ds.sample_lengths)
        if uniq.size != 1:
            return None
        return int(uniq[0]) * 2

    @property
    def num_particles(self) -> Optional[int]:
        if self._train_ds is None:
            return None
        uniq = np.unique(self._train_ds.sample_lengths)
        if uniq.size != 1:
            return None
        return int(uniq[0])

    @property
    def sos_id(self) -> Optional[int]:
        if self._train_ds is None:
            return None
        return self._train_ds.sos_id

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            self.setup("fit")
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class CIFARVQDataModule(pl.LightningDataModule):
    """Lightning DataModule that serves VQ-tokenised CIFAR-10 batches."""

    def __init__(
        self,
        data_root: str,
        vq_model: str,
        batch_size: int = 128,
        mask_ratio: float = 0.9,
        mask_ratio_max: Optional[float] = None,
        seed: int = 0,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_limit: Optional[int] = None,
        vq_subfolder: Optional[str] = None,
        vq_dtype: Optional[str] = None,
        vq_device: str = "cpu",
        cache_latents: bool = False,
        tmp_dir: Optional[str] = None,
        use_hilbert: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.mask_ratio_max = mask_ratio_max
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_limit = train_limit
        self.vq_model = vq_model
        self.vq_subfolder = vq_subfolder
        self.vq_dtype = vq_dtype
        self.vq_device = vq_device
        self.cache_latents = cache_latents
        self.tmp_dir = tmp_dir
        self.use_hilbert = use_hilbert
        self._train_ds: Optional[CIFARVQLatentDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self._train_ds is None:
            self._train_ds = CIFARVQLatentDataset(
                data_root=self.data_root,
                vq_model=self.vq_model,
                train=True,
                limit=self.train_limit,
                mask_ratio=self.mask_ratio,
                mask_ratio_max=self.mask_ratio_max,
                seed=self.seed,
                vq_subfolder=self.vq_subfolder,
                vq_dtype=self.vq_dtype,
                vq_device=self.vq_device,
                cache_latents=self.cache_latents,
                tmp_dir=self.tmp_dir,
                use_hilbert=self.use_hilbert,
            )

    @property
    def codebook_weight(self) -> Optional[torch.Tensor]:
        if self._train_ds is None:
            return None
        return self._train_ds._codebook_weight  # [K, D] on CPU
    
    @property
    def latent_shape(self) -> Optional[tuple[int, int, int]]:
        if self._train_ds is None:
            return None
        return self._train_ds.latent_shape

    @property
    def input_channels(self) -> Optional[int]:
        if self._train_ds is None:
            return None
        return self._train_ds.input_channels

    @property
    def latent_downsample(self) -> Optional[float]:
        if self._train_ds is None:
            return None
        return self._train_ds.latent_downsample

    @property
    def codebook_size(self) -> Optional[int]:
        if self._train_ds is None:
            return None
        return self._train_ds.codebook_size

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            self.setup("fit")
        assert self._train_ds is not None
        num_workers = 0 if str(self.vq_device).startswith("cuda") else self.num_workers
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,   # has no effect when num_workers=0
        )
