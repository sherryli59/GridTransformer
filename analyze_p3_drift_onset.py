"""P3 probe: where does held-out free-running Δs drift begin?

Teacher-force a data prefix of k steps at held-out L4 (N=64), free-run the rest from the
multisize_arc_fullcov checkpoint, and measure the generated suffix's Δs statistics as a
function of k and of steps-since-handoff.

Distinguishes:
  - "compounds from step 1": suffix drifts the same way regardless of k, starting right
    at the handoff -> per-step bias amplification; iid input noise is the right corruption.
  - "late attractor": drift only appears after enough self-generated context accumulates
    -> context-distribution shift; scheduled-sampling-style corruption is the right form.

Built-in implementation check: forced prefix steps must reproduce the data codes exactly
(the arc decode of an exact data Δs round-trips through code_from_arc).

CPU-friendly: batched KV-cache generation, ~4 conditions x 63 steps x 192 samples.
"""
import os

import h5py
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

from grid_transformer.data.lj_transferable import (
    RelativeDeltaTokenizer,
    _hilbert3d_encode,
    _hilbert_bits,
    hilbert_arc_delta,
)
from grid_transformer.models.transformer import GraphormerAR
from sample_lj import _rail_resolution_for_box, autoregressive_relative_delta_sample

CKPT = os.environ.get("P3_CKPT", "lj_ckpts_multisize_arc_L3L5/multisize_arc_fullcov/best.ckpt")
TAG = os.environ.get("P3_TAG", "")  # appended to the output figure name
H5 = "/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L4_rho1.0_N64_T1.0.h5"
OUT = "reports/multisize_arc/figs"
N, L = 64, 4.0
NSAMP = 192
KS = [0, 16, 32, 48]
TEMPERATURE = float(os.environ.get("P3_TEMPERATURE", "0.9"))  # 0.9 = diagnosed production sampling
SEED = 123


def build_data_arcs():
    box = np.array([L, L, L], dtype=np.float64)
    R = int(_rail_resolution_for_box(box.astype(np.float32), 64, 0.046875, ordering="hilbert"))
    bits = _hilbert_bits(R)
    cell = box / R
    with h5py.File(H5, "r") as f:
        traj = f["traj"][:].reshape(-1, N, 3)
    idx = np.random.RandomState(0).choice(traj.shape[0], NSAMP, replace=False)
    arcs, p0s = [], []
    for i in idx:
        pos = traj[i]
        g = np.clip(np.floor(np.mod(pos, box[None]) / cell).astype(np.int64), 0, R - 1)
        codes = _hilbert3d_encode(g[:, 0], g[:, 1], g[:, 2], bits=bits)
        order = np.argsort(codes, kind="stable")
        sp = pos[order]
        arcs.append(hilbert_arc_delta(sp, codes[order], box, R, periodic=True))
        p0s.append(sp[0])
    return np.stack(arcs), np.stack(p0s), R  # [NSAMP, N-1, 4], [NSAMP, 3]


def main():
    torch.set_num_threads(8)
    data_arc, p0s, R = build_data_arcs()
    X = max(1, R**3 // N)
    data_ds = data_arc[:, :, 0]  # [NSAMP, N-1]
    print(f"R={R}  X={X}  data Δs mean={data_ds.mean():.4f}")

    model = GraphormerAR.load_from_checkpoint(CKPT, map_location="cpu")
    model.eval()
    tok = RelativeDeltaTokenizer(window=3.0, bins=64, dim=3)
    assert int(tok.vocab_size) == int(model.K), (tok.vocab_size, model.K)

    results = {}
    for k in KS:
        out = autoregressive_relative_delta_sample(
            model,
            n_particles=N,
            box_lengths=[L, L, L],
            nsamples=NSAMP,
            tokenizer=tok,
            seed=SEED,
            sample_mode="multinomial",
            temperature=TEMPERATURE,
            use_continuous_head=True,
            full_covariance=True,
            continuous_input=True,
            arc_repr=True,
            periodic=True,
            hilbert_resolution=64,
            cell_size=0.046875,
            arc_p0_positions=torch.from_numpy(p0s.astype(np.float32)),
            arc_p0_paired=True,
            teacher_prefix_deltas=(
                torch.from_numpy(data_arc[:, :k].astype(np.float32)) if k > 0 else None
            ),
        )
        codes = out["arc_codes"].numpy().astype(np.float64)
        gen_ds = np.diff(codes, axis=1) / X  # hilbert: arc == code index
        if k > 0:  # implementation check: forced prefix must reproduce data Δs
            err = np.abs(gen_ds[:, :k] - data_ds[:, :k]).max()
            assert err < 1e-6, f"k={k}: forced prefix diverged from data (max err {err})"
        sfx_gen, sfx_dat = gen_ds[:, k:], data_ds[:, k:]
        w1 = wasserstein_distance(sfx_gen.ravel(), sfx_dat.ravel())
        results[k] = dict(gen_ds=gen_ds, sfx_mean=sfx_gen.mean(), dat_mean=sfx_dat.mean(), w1=w1)
        print(f"k={k:2d}: suffix Δs mean gen={sfx_gen.mean():.4f} data={sfx_dat.mean():.4f}  "
              f"W1={w1:.4f}  noncanonical={float(out['arc_noncanonical'].float().mean()):.4f}")

    # per-step means
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    steps = np.arange(N - 1)
    ax[0].plot(steps, data_ds.mean(0), "k--", lw=1.5, label="data")
    for k in KS:
        m = results[k]["gen_ds"].mean(0)
        ax[0].plot(steps, m, label=f"k={k}")
        if k > 0:
            ax[0].axvline(k, color="gray", lw=0.5, ls=":")
    ax[0].set_title("per-step mean Δs vs index (vertical lines: handoff)")
    ax[0].set_xlabel("step t"); ax[0].set_ylabel("mean Δs"); ax[0].legend(fontsize=8)
    for k in KS:
        m = results[k]["gen_ds"].mean(0)[k:]
        ax[1].plot(np.arange(len(m)), m, label=f"k={k}")
    ax[1].plot(np.arange(N - 1), np.full(N - 1, data_ds.mean()), "k--", lw=1, label="data mean")
    ax[1].set_title("mean Δs vs steps since handoff (aligned)")
    ax[1].set_xlabel("t − k"); ax[1].legend(fontsize=8)
    fig.suptitle(f"P3: drift onset at held-out L4 (T={TEMPERATURE}, {NSAMP} samples)")
    fig.tight_layout()
    fname = f"{OUT}/p3_drift_onset{('_' + TAG) if TAG else ''}_T{TEMPERATURE:g}.png"
    fig.savefig(fname, dpi=110)
    print(f"\nwrote {fname}")

    # verdict heuristic: compare drift in the first 8 post-handoff steps across k
    print("\n=== verdict ===")
    early = {k: results[k]["gen_ds"].mean(0)[k:k + 8].mean() for k in KS}
    dat_ref = {k: data_ds.mean(0)[k:k + 8].mean() for k in KS}
    for k in KS:
        print(f"  k={k:2d}: first-8-post-handoff gen mean {early[k]:.4f} (data {dat_ref[k]:.4f})")
    biases = [early[k] - dat_ref[k] for k in KS if k > 0]
    if all(b < -0.02 for b in biases):
        print("  -> drift starts immediately at every handoff: per-step bias, compounds from step 1 "
              "(iid input noise is the matched corruption)")
    else:
        print("  -> drift needs accumulated self-generated context: late attractor "
              "(scheduled-sampling corruption is the matched form)")


if __name__ == "__main__":
    main()
