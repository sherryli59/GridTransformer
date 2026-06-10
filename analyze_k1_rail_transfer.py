"""Stage-1 distribution audit for K=1 rail size-transfer feasibility.

Compares the (sample-independent) K=1 fixed-template rail waypoint feature and the
(MCMC-derived) cartesian-delta target across N=27/64/125 under two R strategies, to decide
whether a zero-shot transfer of the trained K=1 checkpoint is worth running.

See docs/superpowers/specs/2026-06-10-k1-rail-size-transfer-feasibility-design.md
"""
from __future__ import annotations

import argparse
import math
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from grid_transformer.data.lj_transferable import (
    fixed_template_waypoints,
    min_image_delta,
    _hilbert3d_encode,
    _hilbert_bits,
)

CELL_TRAIN = 3.0 / 64.0  # 0.046875

SIZES = [
    dict(N=27, L=3.0, path="/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L3_rho1.0_N27_T1.0.h5"),
    dict(N=64, L=4.0, path="/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L4_rho1.0_N64_T1.0.h5"),
    dict(N=125, L=5.0, path="/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L5_rho1.0_N125_T1.0.h5"),
]


def choose_R(L: float, strategy: str) -> int:
    """Grid resolution for a box of side L holding cell_size ~= CELL_TRAIN.

    strategy="pow2":  next power of two of L/CELL_TRAIN (cell shrinks; validated path).
    strategy="exact": round(L/CELL_TRAIN) (cell ~constant; may be non-power-of-two).
    """
    n = max(2, int(round(L / CELL_TRAIN)))
    if strategy == "exact":
        return n
    if strategy == "pow2":
        return int(1 << int(math.ceil(math.log2(n))))
    raise ValueError(f"unknown strategy {strategy!r}")


def rail_waypoints_absolute(N: int, R: int, L: float) -> np.ndarray:
    """K=1 fixed-template rail waypoints the model is conditioned on: [N-1, 3].

    Exactly mirrors the training/sampling rail config (k=1, window_scale=1.0,
    reference="absolute"): waypoint for particle j is the box-frame cell center of
    decode((j+1)*X), X = R**3 // N. Sample-independent by construction.
    """
    box = np.array([L, L, L], dtype=np.float64)
    pred_idx = np.arange(1, N, dtype=np.int64)  # predicting particles 1..N-1
    wp = fixed_template_waypoints(
        pred_idx, N, int(R), box,
        k=1, window_scale=1.0, periodic=True, reference="absolute",
    )  # [N-1, 1, 3]
    return wp.reshape(-1, 3).astype(np.float64)


def in_box_fraction(points: np.ndarray, L: float, low: float = 0.0) -> float:
    """Fraction of points whose every coordinate lies in [low, low+L).

    Default low=0.0 checks the standard [0, L) box frame.
    Use low=-L/2 to check the min-image frame [-L/2, L/2), which is what
    reference='absolute' waypoints from fixed_template_waypoints live in.
    """
    inside = np.all((points >= low) & (points < low + L), axis=-1)
    return float(np.mean(inside))


def load_configs(path: str, N: int, n_configs: int, seed: int = 42) -> np.ndarray:
    """Return float32 [n_configs, N, 3] sampled from the h5 MCMC trajectory."""
    rng = np.random.default_rng(seed)
    with h5py.File(path) as f:
        traj = f["traj"]  # (n_chains, n_frames, N, 3)
        n_chains, n_frames = traj.shape[:2]
        total = n_chains * n_frames
        idx = rng.choice(total, size=min(n_configs, total), replace=False)
        ci, fi = idx // n_frames, idx % n_frames
        configs = np.stack([traj[c, f] for c, f in zip(ci, fi)]).astype(np.float32)
    return configs


def hilbert_sorted_codes(pos: np.ndarray, L: float, R: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_pos [N,3], order) by Hilbert code at resolution R."""
    box = np.array([L, L, L], dtype=np.float64)
    bits = _hilbert_bits(R)
    cell = box / float(R)
    grid = np.clip(np.floor(np.mod(pos, box) / cell).astype(np.int64), 0, R - 1)
    codes = _hilbert3d_encode(grid[:, 0], grid[:, 1], grid[:, 2], bits=bits)
    order = np.argsort(codes, kind="stable")
    return pos[order], order


def cartesian_delta_targets(configs: np.ndarray, L: float, R: int) -> np.ndarray:
    """Min-image (pos_{t+1} - pos_t) over Hilbert-sorted particles: [n*(N-1), 3]."""
    box = np.array([L, L, L], dtype=np.float64)
    out = []
    for pos in configs:
        sp, _ = hilbert_sorted_codes(pos.astype(np.float64), L, R)
        d = min_image_delta(sp[1:] - sp[:-1], box)
        out.append(d)
    return np.concatenate(out, axis=0)
