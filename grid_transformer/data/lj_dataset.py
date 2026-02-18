import os, glob
from typing import Optional, Tuple

import numpy as np
from torch.utils.data import Dataset
from ..utils.rasterize import rasterize_binary, random_cyclic_roll

class LJPixelsDataset(Dataset):
    """
    Loads Lennard-Jones point configurations stored as .npz archives.

    Supports either per-snapshot files (coords (N,2), L (2,)) or a single
    batched archive holding coords shaped (num_samples, N, 2) and L either (2,)
    or (num_samples, 2). Returns masked binary occupancy grids plus metadata.

    The ``mask_ratio`` argument may be a float in [0, 1] (constant masking per
    sample) or a pair ``(min_ratio, max_ratio)`` in [0, 1], in which case the
    mask fraction is drawn uniformly for every retrieved sample. This is useful
    when training on tasks that need robustness to heavily masked inputs.
    """
    def __init__(self, data_dir, pixel_size=0.25, mask_ratio=0.6, seed=42):
        """
        Load Lennard-Jones snapshots stored as .npz archives.

        Supports either:
          1. A directory of per-snapshot files (legacy behaviour).
          2. A single archive with coords shaped (N_samples, N_particles, 2).
        """
        path = os.fspath(data_dir)
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.npz")))
        elif path.endswith(".npz") and os.path.isfile(path):
            files = [path]
        else:
            raise FileNotFoundError(f"No .npz files in {path}")

        if not files:
            raise FileNotFoundError(f"No .npz files in {path}")

        self.pixel_size = float(pixel_size)
        self.rng = np.random.default_rng(seed)
        self.files = files

        self._mask_ratio_scalar: Optional[float]
        self._mask_ratio_range: Optional[Tuple[float, float]]

        if isinstance(mask_ratio, (tuple, list, np.ndarray)):
            if len(mask_ratio) != 2:
                raise ValueError("mask_ratio sequence must have exactly two values (min, max)")
            low, high = float(mask_ratio[0]), float(mask_ratio[1])
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

        self._mode = "files"
        self._archive = None
        self._coords = None
        self._L = None
        self._L_per_sample = None
        self._length = len(files)

        if len(files) == 1:
            archive = np.load(files[0], allow_pickle=False)
            coords = archive.get("coords")
            if coords is None:
                archive.close()
                raise KeyError(f"coords key missing in {files[0]}")

            # Detect batched archives by dimensionality.
            if coords.ndim == 3 and coords.shape[-1] == 2:
                self._mode = "batched"
                self._archive = archive
                self._coords = coords
                self._length = coords.shape[0]

                L = archive.get("L")
                if L is None:
                    raise KeyError(f"L key missing in {files[0]}")
                if L.ndim == 1:
                    if L.shape[0] != 2:
                        raise ValueError(f"L must have 2 entries; got shape {L.shape}")
                    self._L = L.astype(np.float32, copy=False)
                    self._L_per_sample = None
                elif L.ndim == 2:
                    if L.shape[1] != 2 or L.shape[0] != self._length:
                        raise ValueError(
                            "Per-sample L must have shape (N_samples, 2); "
                            f"got {L.shape}"
                        )
                    self._L = L.astype(np.float32, copy=False)
                    self._L_per_sample = self._L
                else:
                    raise ValueError(f"Unexpected L shape {L.shape}")
            else:
                # Legacy single-snapshot archive -> treat like per-file dataset.
                archive.close()

    def __len__(self):
        if self._mode == "batched":
            return self._length
        return len(self.files)

    def _next_mask_ratio(self) -> float:
        if self._mask_ratio_range is not None:
            low, high = self._mask_ratio_range
            return float(self.rng.uniform(low, high))
        assert self._mask_ratio_scalar is not None
        return self._mask_ratio_scalar

    def _sample_from_batched(self, idx):
        coords = self._coords[idx].astype(np.float32, copy=False)
        if self._L_per_sample is None:
            Lx, Ly = map(float, self._L)
        else:
            Lx, Ly = map(float, self._L_per_sample[idx])
        return coords, Lx, Ly

    def __getitem__(self, idx):
        if self._mode == "batched":
            coords, Lx, Ly = self._sample_from_batched(idx)
        else:
            with np.load(self.files[idx]) as sample:
                coords = sample["coords"].astype(np.float32)
                Lx, Ly = map(float, sample["L"])
        grid = rasterize_binary(coords, Lx, Ly, self.pixel_size)  # (H,W)

        # Random cyclic shift for PBC equivariance
        grid, _ = random_cyclic_roll(grid, self.rng)

        H, W = grid.shape
        mask_ratio = self._next_mask_ratio()
        # Create random mask
        mask = (self.rng.random((H, W)) < mask_ratio).astype(np.float32)

        # Inputs: visible tokens (masked sites are filled with -1 "mask token")
        x_in = grid.copy()
        x_in[mask == 1.0] = -1.0  # sentinel for masked

        # BCE targets only on masked sites
        target = grid.astype(np.float32)

        # Per-sample metadata (could be used for conditioning later)
        meta = np.array([Lx, Ly, self.pixel_size], dtype=np.float32)

        return {
            "x_in": x_in[None, ...],      # (1,H,W)
            "target": target[None, ...],  # (1,H,W)
            "mask": mask[None, ...],      # (1,H,W)
            "meta": meta,
        }
