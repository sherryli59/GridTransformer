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
    _hilbert3d_decode,
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


def raw_cell_inbox_fraction(N: int, R: int) -> float:
    """Fraction of K=1 waypoints whose RAW decoded Hilbert cell lies fully in [0, R).

    Detects exact (non-power-of-two) R curve-length leakage. With bits=ceil(log2(R)) the
    Hilbert curve fills a 2**bits cube; decode((j+1)*X) can land on cells outside the
    physical [0, R) grid even though the model-visible (min-imaged) waypoint always appears
    in-box. Use this — not in_box_fraction on min-imaged waypoints — as the exact-R
    disqualifier. For power-of-two R this is always 1.0.
    """
    total = R ** 3
    X = max(1, total // int(N))
    bits = _hilbert_bits(R)
    j = np.arange(1, N, dtype=np.float64)            # predicting particles 1..N-1
    idx = np.clip(np.rint((j + 1.0) * X), 0, total - 1).astype(np.int64)  # (j+1)*X
    cx, cy, cz = _hilbert3d_decode(idx, bits=bits)
    cells = np.stack([cx, cy, cz], axis=-1)
    inside = np.all(cells < R, axis=-1)              # cells are >= 0 by construction
    return float(np.mean(inside))


def tile_to(arr: np.ndarray, n: int) -> np.ndarray:
    """Tile rows of arr up to at least n rows (for KS power on small waypoint sets)."""
    if len(arr) >= n:
        return arr
    reps = int(math.ceil(n / max(1, len(arr))))
    return np.tile(arr, (reps, 1))[:n]


def _ks_max(a: np.ndarray, b: np.ndarray) -> float:
    """Max per-axis KS statistic between two [*,3] sets."""
    return max(float(ks_2samp(a[:, d], b[:, d]).statistic) for d in range(3))


def main(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    MIN_SAMPLES = 300

    for strategy in ("pow2", "exact"):
        print(f"\n{'='*68}\nStrategy: {strategy}\n{'='*68}")
        rails, deltas, meta = {}, {}, {}
        for s in SIZES:
            N, L = s["N"], s["L"]
            R = choose_R(L, strategy)
            wp = rail_waypoints_absolute(N, R, L)
            ib = raw_cell_inbox_fraction(N, R)  # raw-cell leakage (not min-imaged waypoint)
            cfg = load_configs(s["path"], N, args.n_configs, seed=args.seed)
            dl = cartesian_delta_targets(cfg, L, R)
            rails[N], deltas[N], meta[N] = wp, dl, dict(R=R, cell=L / R, in_box=ib)
            print(f"  N={N:3d} L={L} R={R:4d} cell={L/R:.5f} in_box={ib:.3f} "
                  f"rail_n={len(wp)} delta_n={len(dl)}")

        ref = 27
        print(f"\n  Rail-input KS (vs N=27)         abs        /L(frac)   target-delta KS")
        ks_summary = {}
        for N in (64, 125):
            L_N = [s for s in SIZES if s["N"] == N][0]["L"]
            ra = _ks_max(tile_to(rails[ref], MIN_SAMPLES), tile_to(rails[N], MIN_SAMPLES))
            rf = _ks_max(
                tile_to(rails[ref] / SIZES[0]["L"], MIN_SAMPLES),
                tile_to(rails[N] / L_N, MIN_SAMPLES),
            )
            dk = _ks_max(deltas[ref], deltas[N])
            ks_summary[N] = dict(rail_abs=ra, rail_frac=rf, delta=dk)
            print(f"    N=27 vs N={N:3d}              {ra:8.4f}   {rf:8.4f}     {dk:8.4f}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"K=1 rail transfer audit — strategy={strategy}", fontsize=12)
        colors = {27: "#1f77b4", 64: "#ff7f0e", 125: "#2ca02c"}
        for N in (27, 64, 125):
            L = [s for s in SIZES if s["N"] == N][0]["L"]
            axes[0].hist((rails[N] / L)[:, 0], bins=40, density=True, alpha=0.5,
                         color=colors[N], label=f"N={N}")
            axes[1].hist(np.linalg.norm(rails[N], axis=1), bins=40, density=True, alpha=0.5,
                         color=colors[N], label=f"N={N}")
            axes[2].hist(np.linalg.norm(deltas[N], axis=1), bins=60, density=True, alpha=0.5,
                         color=colors[N], label=f"N={N}")
        axes[0].set_title("rail x / L (fractional)"); axes[0].legend()
        axes[1].set_title("|rail| (absolute)"); axes[1].legend()
        axes[2].set_title("|cartesian delta| target"); axes[2].legend()
        plt.tight_layout()
        p = os.path.join(args.out_dir, f"k1_rail_audit_{strategy}.png")
        plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
        print(f"  Saved {p}")

        max_rail = max(ks_summary[N]["rail_frac"] for N in (64, 125))
        min_ib = min(meta[N]["in_box"] for N in (64, 125))
        verdict = "PASS" if max_rail < 0.30 else "FAIL"
        print(f"  -> rail_frac maxKS={max_rail:.4f} (<0.30? {verdict}), "
              f"min in_box={min_ib:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_configs", type=int, default=500)
    ap.add_argument("--out_dir", type=str, default="reports/k1_rail_transfer")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
