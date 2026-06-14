#!/usr/bin/env python3
"""
Benchmark trained DW4 generative models by importance-sampling ESS.

The DW4 system (4 particles in 2D, double-well pair potential) is an EASY
benchmark: a faithful generative model should reach an importance-sampling
effective sample size (ESS) of at least ~40% against the DW4 Boltzmann target.
This script samples each trained architecture at temperature 1.0 (so the saved
model log-density is the model's TRUE density, not a tempered proposal),
evaluates the DW4 energy on the reconstructed positions, forms the importance
weights toward Boltzmann, and prints a comparison table.

It doubles as a sanity check on the whole sample+likelihood pipeline: if even
the best architecture cannot reach ~40% ESS, that points to a pipeline bug
(energy construction, logq sign, or coordinate mismatch) rather than a genuine
model failure.

ESS formula (mirrors benchmark_lj27.py):
    logw_i = -U(x_i)/kT - logq_i          # logq = model log-density (logp_*)
    w      = exp(logw - logw.max())       # stabilize
    ESS_fraction = (w.sum()**2) / (w**2).sum() / len(w)
    ESS%   = 100 * ESS_fraction

ESS is invariant to any constant added to logw, so the partition function, the
constant gauge between the model's relative-delta coordinates and the absolute
positions U is evaluated on, and any constant Jacobian all cancel. We only need
logq correct up to a global constant with x_base and U(x_base) in consistent
coordinates -- which the relative-delta reconstruction provides (DW4 energy is a
function of pair distances, so the COM gauge cancels).

Usage:
    python benchmark_dw4_ess.py                 # sample fresh + benchmark
    python benchmark_dw4_ess.py --reuse_existing # reuse cached T=1.0 npz if present
"""
import argparse
import os
import subprocess
import sys

import h5py
import numpy as np
import torch

from grid_transformer.physics.energy import DoubleWellPotential

# ---------------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------------- #
REPO = os.path.dirname(os.path.abspath(__file__))
PYTHON_BIN = sys.executable  # the env that imported torch successfully
SAMPLE_SCRIPT = os.path.join(REPO, "sample_lj.py")
CKPT_ROOT = os.path.join(REPO, "dw4_nopbc_0505")
CODEBOOK_PATH = os.path.join(CKPT_ROOT, "dw4_codebook_4096.pt")
TARGET_H5 = "/mnt/ssd/mcmc/dw_mcmc_sweep/dw4_N4_T1.0_dim2.h5"

# DW4 physical parameters (from train_sample_dw4.sh defaults, confirmed against
# the target h5 attrs).
DW_A, DW_B, DW_C, DW_OFFSET, KT = 0.9, -4.0, 0.0, 4.0, 1.0
N_PARTICLES = 4
SPATIAL_DIM = 2
BOX_L = 12.0          # box length from the target h5 (non-periodic; gauge only)
WINDOW = 8.0          # tokenizer displacement window (from training)
BINS = 64
AR_ARCH = "standard"

# Architectures: (label, run_dir, ckpt_basename, list-of-extra sample flags,
#                 logq_key)
ARCHS = [
    dict(
        label="continuous_nonfactorized (MDN, fullcov)",
        run="dw4_continuous_nonfactorized",
        ckpt="best_for_sampling.ckpt",
        flags=["--use_continuous_head", "--full_covariance"],
        logq_key="logp_continuous",
    ),
    dict(
        label="discrete_factorized (binned, factorized)",
        run="dw4_discrete_factorized",
        ckpt="best_for_sampling.ckpt",
        flags=["--binned_discrete", "--factorized"],
        logq_key="logp_discrete",
    ),
    dict(
        label="codebook_nonfactorized (codebook 4096)",
        run="dw4_codebook_nonfactorized",
        ckpt="best_for_sampling.ckpt",
        flags=["--discrete", "--codebook_path", CODEBOOK_PATH],
        logq_key="logp_discrete",
    ),
    dict(
        label="discrete_factorized_bins256 (binned-256, factorized)",
        run="dw4_discrete_factorized_bins256",
        ckpt="best.ckpt",  # no best_for_sampling.ckpt present
        flags=["--binned_discrete", "--factorized"],
        logq_key="logp_discrete",
        bins=256,
    ),
]


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #
def build_energy():
    """DoubleWellPotential with the flattened dim = n_particles * spatial_dims."""
    return DoubleWellPotential(
        a=DW_A, b=DW_B, c=DW_C, offset=DW_OFFSET,
        dim=N_PARTICLES * SPATIAL_DIM, n_particles=N_PARTICLES,
    )


def sanity_check_energy():
    """Evaluate the target energy on real MCMC data; abort if it looks wrong."""
    dw = build_energy()
    with h5py.File(TARGET_H5, "r") as f:
        traj = f["traj"]
        flat = traj[:200].reshape(-1, traj.shape[-2], traj.shape[-1])
    x = torch.as_tensor(np.asarray(flat[:5000], dtype=np.float32))
    U = dw.potential(x).numpy()
    mean, std = float(U.mean()), float(U.std())
    print(f"# energy sanity (real MCMC data, {len(U)} configs): "
          f"mean U = {mean:.3f}, std = {std:.3f}, "
          f"range [{U.min():.2f}, {U.max():.2f}]")
    if not np.isfinite(mean) or abs(mean) > 1e4:
        raise SystemExit(
            f"ABORT: target energy looks wrong on real data (mean={mean}). "
            "Fix energy construction before computing any ESS."
        )
    return mean


def sample_path(run, reuse):
    """Where we store the freshly-sampled T=1.0 npz for this run."""
    return os.path.join(CKPT_ROOT, run, "samples_dw4_ess_T1.npz")


def run_sampler(arch, nsamples, batch, seed, reuse):
    out = sample_path(arch["run"], reuse)
    if reuse and os.path.isfile(out):
        d = np.load(out)
        if d["x_base"].shape[0] >= nsamples and float(d["temperature"]) == 1.0:
            print(f"#   reusing cached {os.path.relpath(out, REPO)}")
            return out
    ckpt = os.path.join(CKPT_ROOT, arch["run"], arch["ckpt"])
    if not os.path.isfile(ckpt):
        raise SystemExit(f"ABORT: checkpoint not found: {ckpt}")
    bins = arch.get("bins", BINS)
    cmd = [
        PYTHON_BIN, SAMPLE_SCRIPT,
        "--mode", "relative",
        "--ckpt", ckpt,
        "--Lx", str(BOX_L), "--Ly", str(BOX_L),
        "--coord_dim", str(SPATIAL_DIM),
        "--num_particles", str(N_PARTICLES),
        "--sample_mode", "multinomial",
        "--temperature", "1.0",           # CRITICAL: model's true density
        "--relative_window", str(WINDOW),
        "--relative_bins", str(bins),
        "--nsamples", str(nsamples),
        "--sample_batch_size", str(batch),
        "--no-periodic",
        "--ar_arch", AR_ARCH,
        "--seed", str(seed),
        "--save", out,
    ] + arch["flags"]
    print(f"#   sampling {arch['label']} -> {os.path.relpath(out, REPO)}")
    subprocess.run(cmd, check=True, cwd=REPO,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out


def compute_ess(npz_path, logq_key, dw):
    d = np.load(npz_path)
    x = torch.as_tensor(np.asarray(d["x_base"], dtype=np.float32))
    if logq_key not in d.files:
        raise SystemExit(f"ABORT: {logq_key} missing from {npz_path}; "
                         f"have {list(d.files)}")
    logq = np.asarray(d[logq_key], dtype=np.float64)
    U = dw.potential(x).numpy().astype(np.float64)
    # Importance weights toward the Boltzmann target. ESS is invariant to a
    # constant offset in logw, so partition function / gauge / Jacobian cancel.
    logw = -U / KT - logq
    logw = logw - logw.max()
    w = np.exp(logw)
    ess = (w.sum() ** 2) / (np.sum(w ** 2) + 1e-300) / len(w) * 100.0
    return len(w), float(U.mean()), float(ess)


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=10000)
    ap.add_argument("--sample_batch_size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse_existing", action="store_true",
                    help="reuse a previously written T=1.0 npz if it has enough samples")
    args = ap.parse_args()

    print(f"# DW4 ESS benchmark")
    print(f"# DW params: a={DW_A} b={DW_B} c={DW_C} offset={DW_OFFSET} kT={KT} "
          f"(dim={N_PARTICLES * SPATIAL_DIM}, n_particles={N_PARTICLES})")
    print(f"# target data: {TARGET_H5}")
    target_meanU = sanity_check_energy()
    dw = build_energy()
    print(f"# sampling at temperature 1.0, nsamples={args.nsamples}\n")

    rows = []
    for arch in ARCHS:
        try:
            npz = run_sampler(arch, args.nsamples, args.sample_batch_size,
                              args.seed, args.reuse_existing)
            n, meanU, ess = compute_ess(npz, arch["logq_key"], dw)
            rows.append((arch["label"], arch["ckpt"], n, meanU, ess))
        except subprocess.CalledProcessError as e:
            print(f"#   FAILED to sample {arch['label']}: {e}")
            rows.append((arch["label"], arch["ckpt"], 0, float("nan"), float("nan")))

    # Table
    print("\n" + "=" * 88)
    print(f"{'architecture':46s} {'ckpt':22s} {'n':>6s} {'meanU':>8s} {'ESS%':>7s}")
    print("-" * 88)
    print(f"{'TARGET (real MCMC data)':46s} {'-':22s} {'-':>6s} "
          f"{target_meanU:8.3f} {'100.0':>7s}")
    best = 0.0
    for label, ckpt, n, meanU, ess in rows:
        ess_s = f"{ess:7.2f}" if np.isfinite(ess) else f"{'n/a':>7s}"
        mu_s = f"{meanU:8.3f}" if np.isfinite(meanU) else f"{'n/a':>8s}"
        print(f"{label:46s} {ckpt:22s} {n:6d} {mu_s} {ess_s}")
        if np.isfinite(ess):
            best = max(best, ess)
    print("=" * 88)
    print(f"# best architecture ESS = {best:.2f}%  "
          f"({'CLEARS' if best >= 40 else 'BELOW'} the 40% bar)")
    if best < 40:
        print("# WARNING: no architecture cleared ~40% ESS on this EASY benchmark.")
        print("#          Treat this as a likely pipeline bug (energy / logq sign /")
        print("#          coordinate gauge), not a real model result, before trusting it.")


if __name__ == "__main__":
    main()
