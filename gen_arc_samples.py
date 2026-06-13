"""Generate arc_repr (optionally rail-conditioned) samples to .npz for benchmark_lj27.

Saves out["x_base"] (n, N, 3) absolute positions under each size, in the format
benchmark_lj27.load_candidates expects (d["x_base"]). Rail-aware: a rail checkpoint is
evaluated WITH the rail active (forward skips rail_attn when waypoints are None).

Env: GEN_CKPT, GEN_OUT (dir), GEN_SIZES (default L3,L4,L5), GEN_NSAMP (default 256),
GEN_TEMPERATURE (default 0.9).
"""
import os

import numpy as np
import torch

from grid_transformer.data.lj_transferable import (
    RelativeDeltaTokenizer,
    _hilbert3d_encode,
    _hilbert_bits,
)
from grid_transformer.models.transformer import GraphormerAR
from sample_lj import _rail_resolution_for_box, autoregressive_relative_delta_sample

CKPT = os.environ["GEN_CKPT"]
OUT = os.environ.get("GEN_OUT", "reports/multisize_arc/samples")
NSAMP = int(os.environ.get("GEN_NSAMP", "256"))
TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.9"))
CELL = 0.046875
SIZES = {
    "L3": ("/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L3_rho1.0_N27_T1.0.h5", 27, 3.0),
    "L4": ("/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L4_rho1.0_N64_T1.0.h5", 64, 4.0),
    "L5": ("/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L5_rho1.0_N125_T1.0.h5", 125, 5.0),
}
WANT = os.environ.get("GEN_SIZES", "L3,L4,L5").split(",")


def p0_pool(h5, N, L, R, n):
    import h5py

    box = np.array([L, L, L], dtype=np.float64)
    bits = _hilbert_bits(R)
    cell = box / R
    with h5py.File(h5, "r") as f:
        traj = f["traj"][:].reshape(-1, N, 3)
    idx = np.random.RandomState(1).choice(traj.shape[0], min(n, traj.shape[0]), replace=False)
    p0 = []
    for i in idx:
        pos = traj[i]
        g = np.clip(np.floor(np.mod(pos, box[None]) / cell).astype(np.int64), 0, R - 1)
        codes = _hilbert3d_encode(g[:, 0], g[:, 1], g[:, 2], bits=bits)
        p0.append(pos[np.argsort(codes, kind="stable")][0])
    return np.stack(p0)


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
    os.makedirs(OUT, exist_ok=True)
    model = GraphormerAR.load_from_checkpoint(CKPT, map_location="cpu").eval()
    tok = RelativeDeltaTokenizer(window=3.0, bins=64, dim=3)

    rail_kwargs = {}
    if bool(getattr(model, "use_curve_rail", False)) and model.rail_attn is not None:
        rail_kwargs = dict(
            use_curve_rail=True,
            curve_rail_mode=str(getattr(model, "curve_rail_mode", "fixed_template")),
            curve_rail_k=int(getattr(model, "curve_rail_k", model.rail_attn.n_rail)),
            curve_rail_window=float(getattr(model, "curve_rail_window", 1.0)),
            curve_rail_reference=str(getattr(model, "curve_rail_reference", "absolute")),
            curve_rail_offsets=None,
        )
        print(f"[rail-aware] k={rail_kwargs['curve_rail_k']}")

    for tag in WANT:
        h5, N, L = SIZES[tag]
        R = int(_rail_resolution_for_box(np.array([L, L, L], np.float32), 64, CELL, ordering="hilbert"))
        p0 = p0_pool(h5, N, L, R, NSAMP)
        out = autoregressive_relative_delta_sample(
            model, n_particles=N, box_lengths=[L, L, L], nsamples=NSAMP, tokenizer=tok,
            seed=7, sample_mode="multinomial", temperature=TEMPERATURE,
            use_continuous_head=True, full_covariance=True, continuous_input=True,
            arc_repr=True, periodic=True, hilbert_resolution=64, cell_size=CELL,
            arc_p0_positions=torch.from_numpy(p0.astype(np.float32)), arc_p0_paired=True,
            **rail_kwargs,
        )
        x = out["x_base"].numpy().astype(np.float32)
        path = os.path.join(OUT, f"arc_{tag}_N{N}.npz")
        np.savez(path, x_base=x)
        nonc = float(out["arc_noncanonical"].float().mean()) if "arc_noncanonical" in out else float("nan")
        print(f"{tag} N={N}: saved {x.shape} -> {path}  noncanonical={nonc:.4f}")


if __name__ == "__main__":
    main()
