from __future__ import annotations

from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _is_power_of_two(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


def _hilbert_rot(side: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = side - 1 - x
            y = side - 1 - y
        x, y = y, x
    return x, y


def _hilbert_xy2d(side: int, x: int, y: int) -> int:
    """Map grid coordinate (x, y) to Hilbert index d for square side length `side`."""
    d = 0
    s = side // 2
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _hilbert_rot(s, x, y, rx, ry)
        s //= 2
    return d


class LJCellDataset(Dataset):
    """
    LJ snapshots mapped to canonical Hilbert-sorted discrete cell IDs.

    Each item returns:
      - sequence:  [N+1] full sequence with SOS prepended.
      - input_idx: [N]   shifted AR input (starts with SOS).
      - target_idx:[N]   canonical token IDs to predict.
      - seq:       [N]   alias of target_idx.
      - box_size:  [2]   physical box lengths (Lx, Ly).
    """

    def __init__(
        self,
        h5_path: str = "/mnt/ssd/mcmc/lj_N16_T1.h5",
        resolution: int = 64,
        limit: Optional[int] = None,
    ) -> None:
        self.h5_path = h5_path
        self.resolution = int(resolution)
        if not _is_power_of_two(self.resolution):
            raise ValueError(f"resolution must be a power of two for Hilbert mapping, got {self.resolution}")

        with h5py.File(self.h5_path, "r") as h5f:
            if "traj" not in h5f:
                raise KeyError(f"'traj' dataset not found in {self.h5_path}")
            traj = np.asarray(h5f["traj"], dtype=np.float32)
            if "boxlength" not in h5f.attrs:
                raise KeyError(f"'boxlength' attribute not found in {self.h5_path}")
            box_attr = np.asarray(h5f.attrs["boxlength"], dtype=np.float32).reshape(-1)

        if traj.ndim == 4:
            # [n_frames, batch_size, N, 2] -> [n_frames*batch_size, N, 2]
            n_frames, batch_size, n_particles, dim = traj.shape
            traj = traj.reshape(n_frames * batch_size, n_particles, dim)
        elif traj.ndim == 3:
            # [n_samples, N, 2]
            _, _, dim = traj.shape
        else:
            raise ValueError(f"Expected traj rank 3 or 4, got shape {traj.shape}")

        if dim < 2:
            raise ValueError(f"Expected at least 2 coordinate dimensions, got {dim}")

        if box_attr.size == 1:
            self.Lx = float(box_attr[0])
            self.Ly = float(box_attr[0])
        else:
            self.Lx = float(box_attr[0])
            self.Ly = float(box_attr[1])

        if self.Lx <= 0.0 or self.Ly <= 0.0:
            raise ValueError(f"Invalid box lengths: Lx={self.Lx}, Ly={self.Ly}")

        coords_xy = traj[..., :2]
        if limit is not None:
            lim = max(1, min(int(limit), coords_xy.shape[0]))
            coords_xy = coords_xy[:lim]

        self._coords = coords_xy
        self.num_particles = int(coords_xy.shape[1])
        self.vocab_size = int(self.resolution * self.resolution)
        self.sos_id = int(self.vocab_size)

    def __len__(self) -> int:
        return int(self._coords.shape[0])

    def _coords_to_cell_ids(self, coords: np.ndarray) -> np.ndarray:
        # Periodic wrap before discretization.
        x = np.mod(coords[:, 0], self.Lx)
        y = np.mod(coords[:, 1], self.Ly)

        ix = np.floor((x / self.Lx) * self.resolution).astype(np.int64)
        iy = np.floor((y / self.Ly) * self.resolution).astype(np.int64)
        ix = np.clip(ix, 0, self.resolution - 1)
        iy = np.clip(iy, 0, self.resolution - 1)

        hilbert_ids = np.empty(ix.shape[0], dtype=np.int64)
        for i in range(ix.shape[0]):
            hilbert_ids[i] = _hilbert_xy2d(self.resolution, int(ix[i]), int(iy[i]))

        return hilbert_ids

    def __getitem__(self, idx: int):
        coords = self._coords[idx]  # [N, 2]
        token_ids = torch.from_numpy(self._coords_to_cell_ids(coords))  # [N]
        token_ids, _ = torch.sort(token_ids)

        sequence = torch.empty(self.num_particles + 1, dtype=torch.long)
        sequence[0] = self.sos_id
        sequence[1:] = token_ids

        input_idx = sequence[:-1].clone()
        target_idx = sequence[1:].clone()
        return {
            "sequence": sequence,
            "input_idx": input_idx,
            "target_idx": target_idx,
            "seq": target_idx,
            "box_size": torch.tensor([self.Lx, self.Ly], dtype=torch.float32),
            "sos_id": torch.tensor(self.sos_id, dtype=torch.long),
            "grid_size": torch.tensor(self.resolution, dtype=torch.long),
        }
