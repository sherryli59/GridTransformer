"""Octahedral rotation NLL diagnostic (action-plan Phase 2, §2.1).

The Boltzmann ensemble in a periodic cubic box is invariant under the 24-element
proper octahedral group, but the Hilbert-ordered AR model need not be: rotating a
config changes its Hilbert sort, hence its factorization, hence its likelihood.
This script measures how non-invariant the learned density is:

    for each proper octahedral rotation R_k:
        rotate val configs about the box center, wrap, re-Hilbert-sort,
        rebuild the arc targets, evaluate per-config NLL.

Decision rule (action-plan §2.1): if the spread of mean NLL across the 24
rotations is << 0.5 (the per-config NLL decision gate used throughout), rotation
equivariance is a non-issue; otherwise octahedral augmentation (variant AUG) and
curve-gauge frames (variant D) enter the screening queue.

Usage:
    python octahedral_nll_diagnostic.py \
        --ckpt lj_ckpts_lj27_pbc_arc_repr_norm/lj27_pbc_arc_repr_norm_fullcov/best.ckpt \
        --data /mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L3_rho1.0_N27_T1.0.h5 \
        --nconfigs 512 --hilbert_resolution 64
"""
from __future__ import annotations

import argparse
from typing import Sequence

import h5py
import numpy as np
import torch

from grid_transformer.data.lj_transferable import (
    _hilbert3d_encode,
    _hilbert_bits,
    hilbert_arc_delta,
)
from grid_transformer.models.ar_registry import load_ar_checkpoint
from grid_transformer.training.ar import mdn_loss


def proper_octahedral_rotations() -> np.ndarray:
    """The 24 proper rotations of the cube: signed permutation matrices, det +1."""
    from itertools import permutations, product

    mats = []
    for perm in permutations(range(3)):
        for signs in product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                m[row, col] = s
            if np.linalg.det(m) > 0.5:
                mats.append(m)
    out = np.stack(mats)
    assert out.shape == (24, 3, 3)
    return out


def rotate_wrap(configs: np.ndarray, rot: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Rotate [M, N, 3] configs about the box center and wrap back into [0, L)."""
    box = np.asarray(box, dtype=np.float64)
    center = box / 2.0
    out = (configs - center) @ np.asarray(rot, dtype=np.float64).T + center
    return np.mod(out, box)


def arc_nll_for_configs(
    model,
    configs: np.ndarray,
    *,
    box_lengths: Sequence[float],
    R: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Per-config arc-repr NLL (summed over the N-1 predicted particles).

    Mirrors the training pipeline exactly: wrap -> Hilbert codes at resolution R ->
    stable sort -> hilbert_arc_delta targets (fine in cell units) -> teacher-forced
    forward with continuous input feedback -> MDN NLL. The token stream carries only
    the SOS embedding at position 0 (continuous_input mode ignores token content at
    t >= 1), so dummy SOS ids are correct here.
    """
    device = next(model.parameters()).device
    box = np.asarray(box_lengths, dtype=np.float64)
    bits = _hilbert_bits(R)
    cell = box / float(R)

    coords_list, arc_list = [], []
    for cfg in np.asarray(configs, dtype=np.float64):
        wrapped = np.mod(cfg, box)
        grid = np.clip(np.floor(wrapped / cell).astype(np.int64), 0, R - 1)
        codes = _hilbert3d_encode(grid[:, 0], grid[:, 1], grid[:, 2], bits=bits)
        order = np.argsort(codes, kind="stable")
        sorted_pos = wrapped[order]
        arc = hilbert_arc_delta(sorted_pos, codes[order], box, R, periodic=True)
        coords_list.append(sorted_pos.astype(np.float32))
        arc_list.append(arc)

    coords_all = torch.from_numpy(np.stack(coords_list))
    arc_all = torch.from_numpy(np.stack(arc_list))
    sos = int(model.sos_id)

    nlls = []
    with torch.no_grad():
        for i in range(0, coords_all.shape[0], batch_size):
            c = coords_all[i : i + batch_size].to(device)
            a = arc_all[i : i + batch_size].to(device)
            B, n_particles = c.shape[:2]
            T = n_particles - 1
            seq_in = torch.full((B, T), sos, dtype=torch.long, device=device)
            input_deltas = torch.cat([torch.zeros_like(a[:, :1]), a[:, :-1]], dim=1)
            box_t = torch.tensor(box, dtype=torch.float32, device=device).expand(B, -1)
            log_pi, mu, scale = model(
                seq_in, coords=c, box_size=box_t, input_deltas=input_deltas
            )
            _, seq_nll, _ = mdn_loss(log_pi, mu, scale, a, box_size=None)
            nlls.append(seq_nll.cpu().numpy())
    return np.concatenate(nlls).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="MCMC h5 with traj [chains, steps, N, 3]")
    ap.add_argument("--nconfigs", type=int, default=512)
    ap.add_argument("--hilbert_resolution", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--ar_arch", default="auto")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, arch = load_ar_checkpoint(args.ckpt, device, ar_arch=args.ar_arch)
    model = model.eval().to(device)
    if not bool(getattr(model, "arc_repr", False)):
        raise ValueError("This diagnostic currently supports arc_repr checkpoints only.")

    with h5py.File(args.data) as f:
        L = float(f.attrs["boxlength"])
        traj = f["traj"]
        # Last frame of each chain: well-decorrelated configs.
        configs = traj[: args.nconfigs, -1, :, :].astype(np.float64)
    box = np.array([L, L, L])
    n = configs.shape[1]
    print(f"ckpt={args.ckpt} (arch={arch})  configs={configs.shape}  L={L}  R={args.hilbert_resolution}")

    rows = []
    for k, rot in enumerate(proper_octahedral_rotations()):
        rotated = rotate_wrap(configs, rot, box)
        nll = arc_nll_for_configs(
            model, rotated, box_lengths=box, R=args.hilbert_resolution, batch_size=args.batch_size
        )
        per_coord = nll / (n - 1)
        rows.append((k, nll.mean(), per_coord.mean()))
        print(f"rotation {k:2d}: mean NLL/config = {nll.mean():9.4f}   per-particle = {per_coord.mean():7.4f}")

    means = np.array([r[1] for r in rows])
    spread = means.max() - means.min()
    print("\n=== Octahedral NLL diagnostic ===")
    print(f"identity NLL: {means[0]:.4f}")
    print(f"mean over 24: {means.mean():.4f}  std: {means.std():.4f}")
    print(f"spread (max-min): {spread:.4f}   gate: 0.5 per config")
    print("verdict:", "NON-ISSUE (skip AUG/D)" if spread < 0.5 else "TRIGGERED -> queue AUG (octahedral augmentation) and variant D")


if __name__ == "__main__":
    main()
