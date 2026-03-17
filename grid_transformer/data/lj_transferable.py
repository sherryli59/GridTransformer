from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..utils.spatial import spectral_argsort


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
    d = 0
    s = side // 2
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _hilbert_rot(s, x, y, rx, ry)
        s //= 2
    return d


def _hilbert3d_encode(x: np.ndarray, y: np.ndarray, z: np.ndarray, bits: int) -> np.ndarray:
    """
    Vectorized 3D Hilbert curve encoding based on John Skilling's 2004 algorithm.
    Transforms (x, y, z) coordinates into a 1D Hilbert integer.
    """
    if bits <= 0:
        return np.zeros_like(x, dtype=np.int64)
        
    X0 = x.astype(np.uint64, copy=True)
    X1 = y.astype(np.uint64, copy=True)
    X2 = z.astype(np.uint64, copy=True)

    M = 1 << (bits - 1)
    Q = M
    
    # 1. Inverse Undo (Axes to Transpose)
    while Q > 1:
        P = Q - 1
        
        cond0 = (X0 & Q) != 0
        X0 = np.where(cond0, X0 ^ P, X0)
        
        cond1 = (X1 & Q) != 0
        t1 = (X0 ^ X1) & P
        X0 = np.where(cond1, X0 ^ P, X0 ^ t1)
        X1 = np.where(cond1, X1, X1 ^ t1)
        
        cond2 = (X2 & Q) != 0
        t2 = (X0 ^ X2) & P
        X0 = np.where(cond2, X0 ^ P, X0 ^ t2)
        X2 = np.where(cond2, X2, X2 ^ t2)
        
        Q >>= 1

    # 2. Gray Encode
    X1 ^= X0
    X2 ^= X1

    t = np.zeros_like(X0)
    Q = M
    while Q > 1:
        cond = (X2 & Q) != 0
        t = np.where(cond, t ^ (Q - 1), t)
        Q >>= 1

    X0 ^= t
    X1 ^= t
    X2 ^= t

    # 3. Interleave Transposed Bits
    code = np.zeros_like(X0, dtype=np.int64)
    for b in range(bits):
        xb = (X0 >> b) & 1
        yb = (X1 >> b) & 1
        zb = (X2 >> b) & 1
        code |= (xb.astype(np.int64) << (3 * b + 2))
        code |= (yb.astype(np.int64) << (3 * b + 1))
        code |= (zb.astype(np.int64) << (3 * b))
        
    return code


def min_image_delta(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    return delta - box * np.round(delta / np.maximum(box, 1e-8))


def raw_delta(delta: np.ndarray, box: np.ndarray, *, periodic: bool) -> np.ndarray:
    if periodic:
        return min_image_delta(delta, box)
    return delta


def _load_h5_traj_and_box(path: str) -> tuple[np.ndarray, np.ndarray]:
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
    elif traj.ndim != 3:
        raise ValueError(f"Expected traj rank 3 or 4, got shape {traj.shape} (file={path})")
    dim = int(traj.shape[-1])
    if dim < 2:
        raise ValueError(
            f"Expected at least 2 coordinate dimensions, got {traj.shape[-1]} (file={path})"
        )

    if box_attr.size == 1:
        box = np.full((dim,), float(box_attr[0]), dtype=np.float32)
    elif box_attr.size >= dim:
        box = np.asarray(box_attr[:dim], dtype=np.float32)
    else:
        pad = np.full((dim - box_attr.size,), float(box_attr[-1]), dtype=np.float32)
        box = np.concatenate([np.asarray(box_attr, dtype=np.float32), pad], axis=0)
    if np.any(box <= 0.0):
        raise ValueError(f"Invalid box lengths in {path}: {box.tolist()}")

    return np.asarray(traj[..., :dim], dtype=np.float32), box


def _center_coords_to_com_zero(coords: np.ndarray) -> np.ndarray:
    return coords - coords.mean(axis=-2, keepdims=True)


def _rotation_matrix_2d(k: int) -> np.ndarray:
    kk = int(k) % 4
    if kk == 0:
        return np.eye(2, dtype=np.float32)
    if kk == 1:
        return np.array([[0, -1], [1, 0]], dtype=np.float32)
    if kk == 2:
        return np.array([[-1, 0], [0, -1]], dtype=np.float32)
    return np.array([[0, 1], [-1, 0]], dtype=np.float32)


def _rotation_matrix_3d(kx: int, ky: int, kz: int) -> np.ndarray:
    def _pow(mat: np.ndarray, p: int) -> np.ndarray:
        out = np.eye(3, dtype=np.float32)
        for _ in range(int(p) % 4):
            out = mat @ out
        return out

    rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    return _pow(rz, kz) @ _pow(ry, ky) @ _pow(rx, kx)


def _random_right_angle_rotation(
    rng: np.random.Generator,
    coords: np.ndarray,
) -> np.ndarray:
    dim = int(coords.shape[-1])
    if dim == 2:
        rot = _rotation_matrix_2d(int(rng.integers(0, 4)))
    elif dim == 3:
        rot = _rotation_matrix_3d(
            int(rng.integers(0, 4)),
            int(rng.integers(0, 4)),
            int(rng.integers(0, 4)),
        )
    else:
        raise ValueError(f"Right-angle data augmentation is only supported in 2D/3D, got dim={dim}")
    return np.asarray(coords @ rot.T, dtype=np.float32)


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
class RelativeDeltaTokenizer:
    window: float = 3.0
    bins: int = 64
    dim: int = 2
    use_long_jump_token: bool = True
    long_jump_delta: Tuple[float, ...] = ()
    factorized: bool = False

    def __post_init__(self) -> None:
        if int(self.dim) <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if int(self.bins) <= 0:
            raise ValueError(f"bins must be positive, got {self.bins}")
        if float(self.window) <= 0.0:
            raise ValueError(f"window must be > 0, got {self.window}")
        if len(self.long_jump_delta) == 0:
            object.__setattr__(self, "long_jump_delta", tuple(0.0 for _ in range(int(self.dim))))
        elif len(self.long_jump_delta) != int(self.dim):
            raise ValueError(
                f"long_jump_delta must have length dim={self.dim}, got {len(self.long_jump_delta)}"
            )

    @property
    def base_vocab_size(self) -> int:
        if self.factorized:
            return int(self.bins)
        return int(self.bins ** self.dim)

    @property
    def long_jump_id(self) -> Optional[int]:
        if not self.use_long_jump_token:
            return None
        return int(self.base_vocab_size)

    @property
    def vocab_size(self) -> int:
        return int(self.base_vocab_size + (1 if self.use_long_jump_token else 0))

    def encode(self, deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if deltas.ndim != 2 or deltas.shape[1] != int(self.dim):
            raise ValueError(f"deltas must be [N,{self.dim}], got {tuple(deltas.shape)}")
        w = float(self.window)
        b = int(self.bins)
        step = (2.0 * w) / float(b)

        clipped = np.clip(deltas, -w, w)
        scaled = (clipped + w) / max(step, 1e-8)
        coords = np.floor(scaled).astype(np.int64)
        coords = np.clip(coords, 0, b - 1)

        if self.factorized:
            tokens = coords.flatten()
            outside = (np.abs(deltas) > w).flatten()
            if self.use_long_jump_token:
                tokens[outside] = int(self.long_jump_id)
            return tokens.astype(np.int64, copy=False), outside.astype(np.bool_, copy=False)

        outside = np.any(np.abs(deltas) > w, axis=1)
        tokens = np.zeros((coords.shape[0],), dtype=np.int64)
        stride = 1
        for d in range(int(self.dim)):
            tokens += coords[:, d] * stride
            stride *= b
        if self.use_long_jump_token:
            tokens[outside] = int(self.long_jump_id)
        return tokens.astype(np.int64, copy=False), outside.astype(np.bool_, copy=False)

    def decode(self, token_ids: torch.LongTensor) -> torch.Tensor:
        if token_ids.ndim < 1:
            raise ValueError("token_ids must have rank >= 1")
        b = int(self.bins)
        w = float(self.window)
        step = (2.0 * w) / float(b)

        if self.factorized:
            if token_ids.shape[-1] % int(self.dim) != 0:
                raise ValueError(
                    f"Factorized token_ids last dimension must be divisible by dim={self.dim}, "
                    f"got shape {tuple(token_ids.shape)}"
                )
            flat = token_ids.reshape(-1).long()
            long_mask = (
                flat == int(self.long_jump_id)
                if self.use_long_jump_token
                else torch.zeros_like(flat, dtype=torch.bool)
            )
            if self.use_long_jump_token:
                flat = flat.clamp(0, self.base_vocab_size - 1)

            deltas_1d = -w + (flat.float() + 0.5) * step
            if self.use_long_jump_token and long_mask.any():
                deltas_1d[long_mask] = 0.0

            shape = list(token_ids.shape)
            shape[-1] = shape[-1] // int(self.dim)
            shape.append(int(self.dim))
            return deltas_1d.view(*shape)

        flat = token_ids.reshape(-1).long()
        long_mask = torch.zeros_like(flat, dtype=torch.bool)
        if self.use_long_jump_token:
            long_mask = flat == int(self.long_jump_id)
            flat = flat.clamp(0, self.base_vocab_size - 1)

        comps = []
        cur = flat
        for _ in range(int(self.dim)):
            comps.append(torch.remainder(cur, b).to(torch.float32))
            cur = torch.div(cur, b, rounding_mode="floor")
        deltas = torch.stack(
            [(-w + (comp + 0.5) * step) for comp in comps],
            dim=-1,
        )

        if self.use_long_jump_token and long_mask.any():
            lj = torch.tensor(self.long_jump_delta, dtype=deltas.dtype, device=deltas.device)
            deltas[long_mask] = lj

        return deltas.view(*token_ids.shape, int(self.dim))


class LJTransferableDataset(Dataset):
    """
    Lennard-Jones trajectories tokenized as ordered local relative moves.

    Per-sample pipeline:
      1. Random global torus shift.
      2. Sort shifted particles (Hilbert or spectral/Fiedler ordering).
      3. Minimum-image deltas between consecutive sorted positions (N-1 deltas).
      4. Quantize each delta on a local [-W, +W]^D grid.

    Supports multiple H5 files with potentially different particle counts.
    """

    def __init__(
        self,
        file_paths: Optional[Sequence[str]] = None,
        *,
        h5_path: Optional[str] = None,
        periodic: bool = True,
        hilbert_resolution: int = 128,
        ordering: str = "hilbert",
        spectral_sigma: float = 1.0,
        local_window: float = 3.0,
        local_bins: int = 64,
        use_long_jump_token: bool = True,
        factorized: bool = False,
        random_grid_shift: bool = True,
        use_data_aug: bool = False,
        limit: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.file_paths = _normalize_file_paths(file_paths, h5_path=h5_path)
        self.periodic = bool(periodic)
        self.ordering = str(ordering).strip().lower()
        if self.ordering not in ("hilbert", "spectral"):
            raise ValueError(f"ordering must be 'hilbert' or 'spectral', got {ordering!r}")
        self.spectral_sigma = float(spectral_sigma)
        if self.spectral_sigma <= 0.0:
            raise ValueError(f"spectral_sigma must be > 0, got {self.spectral_sigma}")
        self.hilbert_resolution = int(hilbert_resolution)
        if self.ordering == "hilbert" and not _is_power_of_two(self.hilbert_resolution):
            raise ValueError(
                f"hilbert_resolution must be a power of two, got {self.hilbert_resolution}"
            )
        self.tokenizer = RelativeDeltaTokenizer(
            window=float(local_window),
            bins=int(local_bins),
            dim=2,
            use_long_jump_token=bool(use_long_jump_token),
            factorized=bool(factorized),
        )
        self.random_grid_shift = bool(random_grid_shift)
        self.use_data_aug = bool(use_data_aug)
        self.factorized = bool(factorized)
        if (not self.periodic) and self.random_grid_shift:
            raise ValueError(
                "random_grid_shift=True is only supported for periodic lj_transferable data. "
                "Nonperiodic mode centers each sample to COM=0 and uses a fixed origin-anchored "
                "ordering."
            )
        self.rng = np.random.default_rng(int(seed))

        self._coords_by_file: list[np.ndarray] = []
        self._box_by_file: list[np.ndarray] = []
        self._n_by_file: list[int] = []
        self.coord_dim: Optional[int] = None

        sample_to_file: list[np.ndarray] = []
        sample_to_local: list[np.ndarray] = []
        sample_lengths: list[np.ndarray] = []

        for file_idx, path in enumerate(self.file_paths):
            traj, box_vec = _load_h5_traj_and_box(path)
            _, n_particles, dim = traj.shape

            if self.coord_dim is None:
                self.coord_dim = int(dim)
                self.tokenizer = RelativeDeltaTokenizer(
                    window=float(local_window),
                    bins=int(local_bins),
                    dim=int(dim),
                    use_long_jump_token=bool(use_long_jump_token),
                    factorized=bool(factorized),
                )
            elif int(dim) != int(self.coord_dim):
                raise ValueError(
                    f"All lj_transferable files must have the same coordinate dimension; "
                    f"expected {self.coord_dim}, got {dim} for {path}"
                )

            coords = traj
            if not self.periodic:
                coords = _center_coords_to_com_zero(coords)
            n_samples_i = int(coords.shape[0])

            self._coords_by_file.append(coords)
            self._box_by_file.append(box_vec)
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
        if self.coord_dim is None:
            raise RuntimeError("Failed to infer coordinate dimension from lj_transferable inputs.")

    def __len__(self) -> int:
        return int(self.sample_id_to_file_index.shape[0])

    def _space_filling_codes_2d(self, grid: np.ndarray) -> np.ndarray:
        h = np.empty(grid.shape[0], dtype=np.int64)
        for i in range(grid.shape[0]):
            h[i] = _hilbert_xy2d(self.hilbert_resolution, int(grid[i, 0]), int(grid[i, 1]))
        return h

    def _space_filling_codes_3d(self, grid: np.ndarray) -> np.ndarray:
        bits = int(math.ceil(math.log2(self.hilbert_resolution)))
        return _hilbert3d_encode(grid[:, 0], grid[:, 1], grid[:, 2], bits=bits)

    def _space_filling_codes(self, grid: np.ndarray) -> np.ndarray:
        if int(self.coord_dim) == 2:
            return self._space_filling_codes_2d(grid)
        if int(self.coord_dim) == 3:
            return self._space_filling_codes_3d(grid)
        raise ValueError(f"Hilbert-like ordering is only supported in 2D/3D, got dim={self.coord_dim}")

    def _grid_coords_periodic(self, coords: np.ndarray, box: np.ndarray) -> np.ndarray:
        wrapped = np.mod(coords, box[None, :])
        scaled = (wrapped / np.maximum(box[None, :], 1e-8)) * float(self.hilbert_resolution)
        grid = np.floor(scaled).astype(np.int64)
        return np.clip(grid, 0, self.hilbert_resolution - 1)

    def _grid_coords_nonperiodic(self, coords: np.ndarray) -> np.ndarray:
        extent = np.max(np.abs(coords), axis=0)
        extent = np.maximum(extent, 1e-8)
        shifted = (coords / (2.0 * extent[None, :])) + 0.5
        grid = np.floor(shifted * float(self.hilbert_resolution)).astype(np.int64)
        return np.clip(grid, 0, self.hilbert_resolution - 1)

    def _hilbert_sort_periodic(self, coords: np.ndarray, box: np.ndarray) -> np.ndarray:
        grid = self._grid_coords_periodic(coords, box)
        return np.argsort(self._space_filling_codes(grid), kind="stable")

    def _hilbert_sort_nonperiodic(self, coords: np.ndarray) -> np.ndarray:
        grid = self._grid_coords_nonperiodic(coords)
        return np.argsort(self._space_filling_codes(grid), kind="stable")

    def _spectral_sort_periodic(self, coords: np.ndarray, box: np.ndarray) -> np.ndarray:
        coords_t = torch.from_numpy(coords).to(dtype=torch.float32).unsqueeze(0)
        box_t = torch.from_numpy(box).to(dtype=torch.float32).view(1, -1)
        order_t = spectral_argsort(coords_t, box_size=box_t, sigma=self.spectral_sigma)[0]
        return order_t.detach().cpu().numpy().astype(np.int64, copy=False)

    def _spectral_sort_nonperiodic(self, coords: np.ndarray) -> np.ndarray:
        coords_t = torch.from_numpy(coords).to(dtype=torch.float32)
        n_points = int(coords_t.shape[0])
        if n_points <= 1:
            return np.arange(n_points, dtype=np.int64)

        dx = coords_t[:, None, :] - coords_t[None, :, :]
        dist_sq = torch.sum(dx * dx, dim=-1)
        gamma = 1.0 / (2.0 * float(self.spectral_sigma) * float(self.spectral_sigma))
        W = torch.exp(-dist_sq * gamma)
        W.fill_diagonal_(0.0)

        deg = W.sum(dim=-1).clamp_min(1e-8)
        eye = torch.eye(n_points, dtype=coords_t.dtype)
        L_rw = eye - (W / deg.unsqueeze(-1))
        L = 0.5 * (L_rw + L_rw.transpose(-1, -2))

        _, eigvecs = torch.linalg.eigh(L)
        u2 = eigvecs[:, 1]

        centroid = coords_t.mean(dim=0, keepdim=True)
        radius = torch.linalg.norm(coords_t - centroid, dim=-1)
        sign_score = torch.sum(u2 * (radius - radius.mean()))
        if sign_score < 0.0:
            u2 = -u2
        return torch.argsort(u2).cpu().numpy().astype(np.int64, copy=False)

    def __getitem__(self, idx: int):
        file_idx = int(self.sample_id_to_file_index[idx])
        local_idx = int(self.sample_id_to_local_index[idx])

        coords = self._coords_by_file[file_idx][local_idx]
        box = self._box_by_file[file_idx]
        n_particles = int(self.sample_lengths[idx])

        working = coords.copy()
        if self.use_data_aug:
            if self.periodic:
                center = 0.5 * box[None, :]
                working = _random_right_angle_rotation(self.rng, working - center) + center
            else:
                working = _random_right_angle_rotation(self.rng, working)

        if self.periodic and self.random_grid_shift:
            shift = self.rng.uniform(low=0.0, high=1.0, size=(int(self.coord_dim),)).astype(np.float32) * box
            shifted = np.mod(working + shift[None, :], box[None, :])
        elif self.periodic:
            shifted = np.mod(working, box[None, :])
        else:
            shifted = working

        if self.ordering == "hilbert":
            if self.periodic:
                order = self._hilbert_sort_periodic(shifted, box=box)
            else:
                order = self._hilbert_sort_nonperiodic(shifted)
        else:
            if self.periodic:
                order = self._spectral_sort_periodic(shifted, box=box)
            else:
                order = self._spectral_sort_nonperiodic(shifted)
                
        sorted_pos = shifted[order]

        # The first predicted token is always the displacement of sorted particle 1
        # relative to sorted particle 0. Particle 0 itself is implicit.
        n_predict = n_particles - 1
        deltas = np.empty((n_predict, int(self.coord_dim)), dtype=np.float32)
        for i in range(1, n_particles):
            raw = sorted_pos[i] - sorted_pos[i-1]
            deltas[i-1] = raw_delta(raw, box, periodic=self.periodic)

        token_ids_np, long_jump_mask_np = self.tokenizer.encode(deltas)
        token_ids = torch.from_numpy(token_ids_np).long()

        seq_len = len(token_ids_np)
        sequence = torch.empty(seq_len + 1, dtype=torch.long)
        sequence[0] = int(self.sos_id)
        sequence[1:] = token_ids

        input_idx = sequence[:-1].clone()
        target_idx = sequence[1:].clone()

        # Geometry context is expressed in the frame where sorted particle 0 sits at the
        # origin. The final absolute location of particle 0 is recovered only after a
        # full chain is generated and COM=0 is imposed.
        if self.periodic:
            shifted_coords = raw_delta(sorted_pos - sorted_pos[0], box, periodic=True)
        else:
            shifted_coords = sorted_pos - sorted_pos[0]

        if self.tokenizer.factorized:
            token_coords = np.repeat(shifted_coords[:-1], self.coord_dim, axis=0).astype(np.float32)
        else:
            token_coords = shifted_coords[:-1].astype(np.float32)

        density = float(n_particles / max(1e-8, float(np.prod(box, dtype=np.float64))))

        return {
            "sequence": sequence,
            "input_idx": input_idx,
            "target_idx": target_idx,
            "seq": target_idx,
            "box_size": torch.from_numpy(np.asarray(box, dtype=np.float32)),
            "density": torch.tensor(density, dtype=torch.float32),
            "token_coords": torch.from_numpy(token_coords),
            "long_jump_mask": torch.from_numpy(long_jump_mask_np),
            "sample_length": torch.tensor(seq_len, dtype=torch.long),
            "sos_id": torch.tensor(self.sos_id, dtype=torch.long),
            "vocab_size": torch.tensor(self.vocab_size, dtype=torch.long),
            "periodic": torch.tensor(self.periodic, dtype=torch.bool),
        }


class LJTransferableCachedDataset(Dataset):
    """
    Cached/tokenized lj_transferable dataset loaded from a preprocessed .pt file.

    The cache stores padded tensors and per-sample lengths; __getitem__ trims each
    sample to match the output schema of LJTransferableDataset.
    """

    def __init__(self, cache_path: str, *, limit: Optional[int] = None) -> None:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid cache format in {cache_path}: expected dict")

        required_keys = (
            "sequence",
            "input_idx",
            "target_idx",
            "token_coords",
            "long_jump_mask",
            "box_size",
            "density",
            "sample_length",
            "vocab_size",
            "sos_id",
        )
        missing = [k for k in required_keys if k not in payload]
        if missing:
            raise KeyError(f"Missing keys in cache {cache_path}: {missing}")

        self.sequence_all = torch.as_tensor(payload["sequence"], dtype=torch.long)
        self.input_idx_all = torch.as_tensor(payload["input_idx"], dtype=torch.long)
        self.target_idx_all = torch.as_tensor(payload["target_idx"], dtype=torch.long)
        self.token_coords_all = torch.as_tensor(payload["token_coords"], dtype=torch.float32)
        self.long_jump_mask_all = torch.as_tensor(payload["long_jump_mask"], dtype=torch.bool)
        self.box_size_all = torch.as_tensor(payload["box_size"], dtype=torch.float32)
        self.density_all = torch.as_tensor(payload["density"], dtype=torch.float32)
        self.sample_length_all = torch.as_tensor(payload["sample_length"], dtype=torch.long)

        if self.sample_length_all.ndim != 1:
            raise ValueError(f"sample_length must be rank-1, got {tuple(self.sample_length_all.shape)}")
        n_samples = int(self.sample_length_all.shape[0])
        for name, t in (
            ("sequence", self.sequence_all),
            ("input_idx", self.input_idx_all),
            ("target_idx", self.target_idx_all),
            ("token_coords", self.token_coords_all),
            ("long_jump_mask", self.long_jump_mask_all),
            ("box_size", self.box_size_all),
            ("density", self.density_all),
        ):
            if t.shape[0] != n_samples:
                raise ValueError(f"{name} has first dim {t.shape[0]} but expected {n_samples}")

        self.sample_lengths = self.sample_length_all.numpy().astype(np.int64, copy=False)
        if limit is not None:
            lim = max(1, min(int(limit), n_samples))
            self.sequence_all = self.sequence_all[:lim]
            self.input_idx_all = self.input_idx_all[:lim]
            self.target_idx_all = self.target_idx_all[:lim]
            self.token_coords_all = self.token_coords_all[:lim]
            self.long_jump_mask_all = self.long_jump_mask_all[:lim]
            self.box_size_all = self.box_size_all[:lim]
            self.density_all = self.density_all[:lim]
            self.sample_length_all = self.sample_length_all[:lim]
            self.sample_lengths = self.sample_lengths[:lim]

        self.vocab_size = int(payload["vocab_size"])
        self.sos_id = int(payload["sos_id"])
        self.metadata: dict[str, Any] = dict(payload.get("metadata", {}))
        self.periodic = bool(self.metadata.get("periodic", True))
        self.coord_dim = int(self.token_coords_all.shape[-1])
        self.factorized = bool(self.metadata.get("factorized", False))

    def __len__(self) -> int:
        return int(self.sample_length_all.shape[0])

    def __getitem__(self, idx: int):
        seq_len = int(self.sample_length_all[idx].item())
        return {
            "sequence": self.sequence_all[idx, : seq_len + 1],
            "input_idx": self.input_idx_all[idx, :seq_len],
            "target_idx": self.target_idx_all[idx, :seq_len],
            "seq": self.target_idx_all[idx, :seq_len],
            "box_size": self.box_size_all[idx],
            "density": self.density_all[idx],
            "token_coords": self.token_coords_all[idx, :seq_len, :],
            "long_jump_mask": self.long_jump_mask_all[idx, :seq_len],
            "sample_length": self.sample_length_all[idx],
            "sos_id": torch.tensor(self.sos_id, dtype=torch.long),
            "vocab_size": torch.tensor(self.vocab_size, dtype=torch.long),
            "periodic": torch.tensor(self.periodic, dtype=torch.bool),
        }


def build_lj_transferable_cache(
    *,
    file_paths: Optional[Sequence[str]] = None,
    h5_path: Optional[str] = None,
    output_path: str,
    periodic: bool = True,
    hilbert_resolution: int = 128,
    ordering: str = "hilbert",
    spectral_sigma: float = 1.0,
    local_window: float = 3.0,
    local_bins: int = 64,
    use_long_jump_token: bool = True,
    factorized: bool = False,
    random_grid_shift: bool = False,
    limit: Optional[int] = None,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Precompute tokenized lj_transferable samples and store them in a .pt cache.

    Returns a short summary dict.
    """
    dataset = LJTransferableDataset(
        file_paths=file_paths,
        h5_path=h5_path,
        periodic=bool(periodic),
        hilbert_resolution=int(hilbert_resolution),
        ordering=str(ordering),
        spectral_sigma=float(spectral_sigma),
        local_window=float(local_window),
        local_bins=int(local_bins),
        use_long_jump_token=bool(use_long_jump_token),
        factorized=bool(factorized),
        random_grid_shift=bool(random_grid_shift),
        limit=limit,
        seed=int(seed),
    )
    n_samples = len(dataset)
    if n_samples <= 0:
        raise RuntimeError("No samples found for cache build.")

    max_particles = int(dataset.sample_lengths.max())
    max_seq_len = int(max(0, max_particles - 1))
    if dataset.tokenizer.factorized:
        max_seq_len *= int(dataset.coord_dim)

    sequence = torch.empty((n_samples, max_seq_len + 1), dtype=torch.long)
    input_idx = torch.empty((n_samples, max_seq_len), dtype=torch.long)
    target_idx = torch.empty((n_samples, max_seq_len), dtype=torch.long)
    token_coords = torch.empty((n_samples, max_seq_len, int(dataset.coord_dim)), dtype=torch.float32)
    long_jump_mask = torch.empty((n_samples, max_seq_len), dtype=torch.bool)
    box_size = torch.empty((n_samples, int(dataset.coord_dim)), dtype=torch.float32)
    density = torch.empty((n_samples,), dtype=torch.float32)
    sample_length = torch.empty((n_samples,), dtype=torch.long)

    for i in range(n_samples):
        item = dataset[i]
        n = int(item["sample_length"].item())
        sequence[i].fill_(int(dataset.sos_id))
        input_idx[i].zero_()
        target_idx[i].zero_()
        token_coords[i].zero_()
        long_jump_mask[i].zero_()

        sequence[i, : n + 1] = item["sequence"]
        input_idx[i, :n] = item["input_idx"]
        target_idx[i, :n] = item["target_idx"]
        token_coords[i, :n, :] = item["token_coords"]
        long_jump_mask[i, :n] = item["long_jump_mask"]
        box_size[i] = item["box_size"]
        density[i] = item["density"]
        sample_length[i] = item["sample_length"]

    metadata = {
        "file_paths": list(dataset.file_paths),
        "periodic": bool(periodic),
        "ordering": str(ordering),
        "spectral_sigma": float(spectral_sigma),
        "hilbert_resolution": int(hilbert_resolution),
        "local_window": float(local_window),
        "local_bins": int(local_bins),
        "coord_dim": int(dataset.coord_dim),
        "use_long_jump_token": bool(use_long_jump_token),
        "factorized": bool(factorized),
        "random_grid_shift": bool(random_grid_shift),
        "use_data_aug": False,
        "seed": int(seed),
        "version": 4,
    }

    payload = {
        "sequence": sequence,
        "input_idx": input_idx,
        "target_idx": target_idx,
        "token_coords": token_coords,
        "long_jump_mask": long_jump_mask,
        "box_size": box_size,
        "density": density,
        "sample_length": sample_length,
        "vocab_size": int(dataset.vocab_size),
        "sos_id": int(dataset.sos_id),
        "metadata": metadata,
    }
    torch.save(payload, output_path)
    return {
        "output_path": output_path,
        "n_samples": n_samples,
        "max_particles": max_particles,
        "max_seq_len": max_seq_len,
        "vocab_size": int(dataset.vocab_size),
        "periodic": bool(periodic),
        "ordering": str(ordering),
        "factorized": bool(factorized),
        "random_grid_shift": bool(random_grid_shift),
    }


class BucketedBatchSampler(Sampler[list[int]]):
    """Yield batches of indices grouped by equal particle count, then globally shuffled."""

    def __init__(
        self,
        sample_lengths: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.sample_lengths = np.asarray(sample_lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.sample_lengths.ndim != 1:
            raise ValueError("sample_lengths must be a rank-1 sequence")

    def __len__(self) -> int:
        total = 0
        for n in np.unique(self.sample_lengths):
            count = int(np.sum(self.sample_lengths == n))
            if self.drop_last:
                total += count // self.batch_size
            else:
                total += int(math.ceil(count / self.batch_size))
        return total

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        all_batches: list[list[int]] = []
        for n in np.unique(self.sample_lengths):
            idxs = np.nonzero(self.sample_lengths == n)[0].astype(np.int64)
            if self.shuffle:
                rng.shuffle(idxs)

            n_full = len(idxs) // self.batch_size
            for i in range(n_full):
                s = i * self.batch_size
                e = s + self.batch_size
                all_batches.append(idxs[s:e].tolist())

            rem = len(idxs) % self.batch_size
            if (not self.drop_last) and rem > 0:
                all_batches.append(idxs[-rem:].tolist())

        if self.shuffle:
            rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch
