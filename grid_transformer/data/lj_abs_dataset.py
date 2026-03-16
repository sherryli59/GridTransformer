from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize_file_paths(
    file_paths: Optional[Sequence[str]] = None,
    *,
    h5_path: Optional[str] = None,
) -> list[str]:
    if file_paths is None:
        if h5_path is None:
            raise ValueError("Provide `file_paths` (list) or `h5_path` (single file).")
        paths = [str(h5_path)]
    elif isinstance(file_paths, (str, bytes)):
        paths = [str(file_paths)]
    else:
        paths = [str(p) for p in file_paths]

    if not paths:
        raise ValueError("No H5 files provided.")
    return paths


@dataclass(frozen=True)
class AbsoluteCoordinateTokenizer:
    bins: int = 512

    @property
    def vocab_size(self) -> int:
        return int(self.bins)

    def encode(self, coords: np.ndarray, box: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32)
        box = np.asarray(box, dtype=np.float32).reshape(2)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be [N,2], got {tuple(coords.shape)}")
        wrapped = np.mod(coords, box[None, :])
        scaled = wrapped / np.maximum(box[None, :], 1e-8)
        tokens = np.floor(scaled * float(self.bins)).astype(np.int64)
        return np.clip(tokens, 0, self.bins - 1)

    def decode(self, token_ids: torch.LongTensor, box_size: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.long()
        box_size = torch.as_tensor(box_size, device=token_ids.device, dtype=torch.float32)
        if box_size.ndim == 1:
            box_size = box_size.view(1, 1, 2)
        elif box_size.ndim == 2:
            box_size = box_size[:, None, :]
        else:
            raise ValueError(f"box_size must be rank-1 or rank-2, got {tuple(box_size.shape)}")
        cell = box_size / float(self.bins)
        coords = (token_ids.to(torch.float32) + 0.5) * cell
        return torch.remainder(coords, box_size)


class LJAbsoluteDataset(Dataset):
    """
    LJ snapshots tokenized as absolute per-axis coordinate bins.

    Each particle contributes two tokens: x-bin then y-bin.
    The particle order is left intact so the model can learn with
    particle-index permutations applied in the training step.
    """

    def __init__(
        self,
        file_paths: Optional[Sequence[str]] = None,
        *,
        h5_path: Optional[str] = None,
        bins: int = 512,
        random_grid_shift: bool = False,
        limit: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.file_paths = _normalize_file_paths(file_paths, h5_path=h5_path)
        self.tokenizer = AbsoluteCoordinateTokenizer(bins=int(bins))
        self.random_grid_shift = bool(random_grid_shift)
        self.rng = np.random.default_rng(int(seed))

        self._coords_by_file: list[np.ndarray] = []
        self._box_by_file: list[np.ndarray] = []
        self._n_by_file: list[int] = []

        sample_to_file: list[np.ndarray] = []
        sample_to_local: list[np.ndarray] = []
        sample_lengths: list[np.ndarray] = []

        for file_idx, path in enumerate(self.file_paths):
            with h5py.File(path, "r") as h5f:
                if "traj" not in h5f:
                    raise KeyError(f"'traj' dataset not found in {path}")
                traj = np.asarray(h5f["traj"], dtype=np.float32)
                if "boxlength" not in h5f.attrs:
                    raise KeyError(f"'boxlength' attribute not found in {path}")
                box_attr = np.asarray(h5f.attrs["boxlength"], dtype=np.float32).reshape(-1)

            if traj.ndim == 4:
                n_frames, batch_size, n_particles, dim = traj.shape
                traj = traj.reshape(n_frames * batch_size, n_particles, dim)
            elif traj.ndim == 3:
                _, n_particles, dim = traj.shape
            else:
                raise ValueError(f"Expected traj rank 3 or 4, got shape {traj.shape} (file={path})")
            if dim < 2:
                raise ValueError(f"Expected at least 2 coordinate dims, got {dim} (file={path})")

            if box_attr.size == 1:
                Lx = float(box_attr[0])
                Ly = float(box_attr[0])
            else:
                Lx = float(box_attr[0])
                Ly = float(box_attr[1])
            if Lx <= 0.0 or Ly <= 0.0:
                raise ValueError(f"Invalid box lengths in {path}: Lx={Lx}, Ly={Ly}")

            coords_xy = traj[..., :2]
            n_samples_i = int(coords_xy.shape[0])

            self._coords_by_file.append(coords_xy)
            self._box_by_file.append(np.array([Lx, Ly], dtype=np.float32))
            self._n_by_file.append(int(n_particles))

            sample_to_file.append(np.full((n_samples_i,), file_idx, dtype=np.int64))
            sample_to_local.append(np.arange(n_samples_i, dtype=np.int64))
            sample_lengths.append(np.full((n_samples_i,), int(n_particles), dtype=np.int64))

        self.sample_id_to_file_index = np.concatenate(sample_to_file, axis=0)
        self.sample_id_to_local_index = np.concatenate(sample_to_local, axis=0)
        self.sample_lengths = np.concatenate(sample_lengths, axis=0)

        if limit is not None:
            lim = max(1, min(int(limit), int(self.sample_id_to_file_index.shape[0])))
            self.sample_id_to_file_index = self.sample_id_to_file_index[:lim]
            self.sample_id_to_local_index = self.sample_id_to_local_index[:lim]
            self.sample_lengths = self.sample_lengths[:lim]

        self.vocab_size = int(self.tokenizer.vocab_size)
        self.sos_id = int(self.vocab_size)

    def __len__(self) -> int:
        return int(self.sample_id_to_file_index.shape[0])

    def __getitem__(self, idx: int):
        file_idx = int(self.sample_id_to_file_index[idx])
        local_idx = int(self.sample_id_to_local_index[idx])

        coords = self._coords_by_file[file_idx][local_idx]
        box = self._box_by_file[file_idx]
        n_particles = int(self.sample_lengths[idx])

        if self.random_grid_shift:
            shift = self.rng.uniform(low=0.0, high=1.0, size=(2,)).astype(np.float32) * box
            coords = np.mod(coords + shift[None, :], box[None, :])
        else:
            coords = np.mod(coords, box[None, :])

        token_ids_np = self.tokenizer.encode(coords, box).reshape(-1)
        token_ids = torch.from_numpy(token_ids_np).long()

        sequence = torch.empty(token_ids.numel() + 1, dtype=torch.long)
        sequence[0] = int(self.sos_id)
        sequence[1:] = token_ids

        density = float(n_particles / max(1e-8, float(box[0] * box[1])))

        return {
            "sequence": sequence,
            "input_idx": sequence[:-1].clone(),
            "target_idx": sequence[1:].clone(),
            "seq": sequence[1:].clone(),
            "box_size": torch.from_numpy(box.copy()),
            "density": torch.tensor(density, dtype=torch.float32),
            "sample_length": torch.tensor(n_particles, dtype=torch.long),
            "particle_dim": torch.tensor(2, dtype=torch.long),
            "sos_id": torch.tensor(self.sos_id, dtype=torch.long),
            "vocab_size": torch.tensor(self.vocab_size, dtype=torch.long),
        }
