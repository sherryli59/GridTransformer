"""P4 closure audit: how much of the Hilbert curve does free-running generation traverse?

For each size, free-run-generate from a checkpoint and measure the closure fraction
    closure = arc_code(last particle) / ((N-1) * X)
which equals the mean Δs over the rollout. closure < 1 means the rollout ends short of
the curve end (underfilled box) — the mean-drift component of the OTgap. Reports the
distribution (mean, std, p10/p50/p90) per size, plus the implied "missing curve" fraction.

Sets the Phase-2 budget-guidance ceiling: budget guidance can recover at most the
mean-drift share, so 1-closure bounds its expected OTgap improvement.

CPU-only. Env: P4_CKPT (default baseline), P4_TAG, P4_SIZES (e.g. "L3,L4,L5").
"""
import os

import h5py
import numpy as np
import torch

from grid_transformer.data.lj_transferable import (
    RelativeDeltaTokenizer,
    _hilbert3d_encode,
    _hilbert_bits,
)
from grid_transformer.models.transformer import GraphormerAR
from sample_lj import _rail_resolution_for_box, autoregressive_relative_delta_sample

CKPT = os.environ.get("P4_CKPT", "lj_ckpts_multisize_arc_L3L5/multisize_arc_fullcov/best.ckpt")
TAG = os.environ.get("P4_TAG", "baseline")
NSAMP = int(os.environ.get("P4_NSAMP", "192"))
TEMPERATURE = float(os.environ.get("P4_TEMPERATURE", "0.9"))
CELL = 0.046875
SIZES = {
    "L3": ("/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L3_rho1.0_N27_T1.0.h5", 27, 3.0),
    "L4": ("/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L4_rho1.0_N64_T1.0.h5", 64, 4.0),
    "L5": ("/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L5_rho1.0_N125_T1.0.h5", 125, 5.0),
}
WANT = os.environ.get("P4_SIZES", "L3,L4,L5").split(",")


def p0_pool(h5, N, L, R, n):
    box = np.array([L, L, L], dtype=np.float64)
    bits = _hilbert_bits(R)
    cell = box / R
    with h5py.File(h5, "r") as f:
        traj = f["traj"][:].reshape(-1, N, 3)
    idx = np.random.RandomState(0).choice(traj.shape[0], min(n, traj.shape[0]), replace=False)
    p0 = []
    for i in idx:
        pos = traj[i]
        g = np.clip(np.floor(np.mod(pos, box[None]) / cell).astype(np.int64), 0, R - 1)
        codes = _hilbert3d_encode(g[:, 0], g[:, 1], g[:, 2], bits=bits)
        p0.append(pos[np.argsort(codes, kind="stable")][0])
    return np.stack(p0)


def main():
    torch.set_num_threads(8)
    print(f"checkpoint: {CKPT}\ntag: {TAG}  T={TEMPERATURE}  nsamp={NSAMP}\n")
    model = GraphormerAR.load_from_checkpoint(CKPT, map_location="cpu").eval()
    tok = RelativeDeltaTokenizer(window=3.0, bins=64, dim=3)

    print(f"{'size':5s} {'N':>5s} {'closure_mean':>13s} {'std':>6s} "
          f"{'p10':>6s} {'p50':>6s} {'p90':>6s} {'missing%':>9s}")
    rows = []
    for tag in WANT:
        h5, N, L = SIZES[tag]
        R = int(_rail_resolution_for_box(np.array([L, L, L], np.float32), 64, CELL, ordering="hilbert"))
        X = max(1, R**3 // N)
        p0 = p0_pool(h5, N, L, R, NSAMP)
        out = autoregressive_relative_delta_sample(
            model, n_particles=N, box_lengths=[L, L, L], nsamples=NSAMP, tokenizer=tok,
            seed=123, sample_mode="multinomial", temperature=TEMPERATURE,
            use_continuous_head=True, full_covariance=True, continuous_input=True,
            arc_repr=True, periodic=True, hilbert_resolution=64, cell_size=CELL,
            arc_p0_positions=torch.from_numpy(p0.astype(np.float32)), arc_p0_paired=True,
        )
        last_code = out["arc_codes"][:, -1].numpy().astype(np.float64)
        closure = last_code / ((N - 1) * X)
        m, s = closure.mean(), closure.std()
        p10, p50, p90 = np.percentile(closure, [10, 50, 90])
        missing = 100.0 * (1.0 - m)
        rows.append((tag, N, m, s, p10, p50, p90, missing))
        print(f"{tag:5s} {N:5d} {m:13.4f} {s:6.3f} {p10:6.3f} {p50:6.3f} {p90:6.3f} {missing:8.1f}%")

    print("\nInterpretation: closure = mean Δs over the rollout; (1-closure) is the curve")
    print("fraction left untraversed = the mean-drift share Phase-2 budget guidance can target.")


if __name__ == "__main__":
    main()
