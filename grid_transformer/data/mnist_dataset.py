from __future__ import annotations

from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from ..utils.rasterize import random_cyclic_roll


class MNISTMaskedDataset(Dataset):
    """Binary MNIST images with random masking for reconstruction training."""

    def __init__(
        self,
        data_root: str,
        train: bool = True,
        limit: Optional[int] = None,
        pixel_size: float = 1.0,
        mask_ratio: float = 0.9,
        mask_ratio_max: Optional[float] = None,
        seed: int = 0,
        threshold: float = 0.5,
    ) -> None:
        try:
            from torchvision import datasets, transforms
        except ImportError as exc:  # pragma: no cover - import error depends on env
            raise ImportError("torchvision is required to load the MNIST dataset.") from exc

        self.pixel_size = float(pixel_size)
        self.threshold = float(threshold)
        self.rng = np.random.default_rng(seed)

        transform = transforms.ToTensor()
        self._dataset = datasets.MNIST(
            root=data_root,
            train=train,
            download=True,
            transform=transform,
        )

        total = len(self._dataset)
        if limit is not None:
            limit_int = int(limit)
            if limit_int <= 0:
                raise ValueError("limit must be positive if provided")
            limit_int = min(limit_int, total)
            self._indices = list(range(limit_int))
        else:
            self._indices = list(range(total))

        if mask_ratio_max is not None:
            low = float(mask_ratio)
            high = float(mask_ratio_max)
            if not (0.0 <= low <= 1.0 and 0.0 <= high <= 1.0):
                raise ValueError("mask_ratio values must lie in [0, 1]")
            if high < low:
                low, high = high, low
            self._mask_ratio_scalar = None
            self._mask_ratio_range = (low, high)
        else:
            val = float(mask_ratio)
            if not (0.0 <= val <= 1.0):
                raise ValueError("mask_ratio must lie in [0, 1]")
            self._mask_ratio_scalar = val
            self._mask_ratio_range = None

    def __len__(self) -> int:
        return len(self._indices)

    def _next_mask_ratio(self) -> float:
        if self._mask_ratio_range is not None:
            low, high = self._mask_ratio_range
            return float(self.rng.uniform(low, high))
        assert self._mask_ratio_scalar is not None
        return self._mask_ratio_scalar

    def __getitem__(self, idx: int):
        real_idx = self._indices[idx]
        img, label = self._dataset[real_idx]

        grid = img.squeeze(0).numpy()
        grid = (grid > self.threshold).astype(np.float32, copy=False)
        grid, _ = random_cyclic_roll(grid, self.rng)

        mask_ratio = self._next_mask_ratio()
        mask = (self.rng.random(grid.shape) < mask_ratio).astype(np.float32, copy=False)

        x_in = grid.copy()
        x_in[mask == 1.0] = -1.0

        target = grid.astype(np.float32, copy=False)
        meta = np.array([28.0 * self.pixel_size, 28.0 * self.pixel_size, self.pixel_size], dtype=np.float32)

        return {
            "x_in": x_in[None, ...],
            "target": target[None, ...],
            "mask": mask[None, ...],
            "meta": meta,
            "label": np.array(label, dtype=np.int64),
        }
