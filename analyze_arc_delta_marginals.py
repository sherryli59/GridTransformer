"""Compare arc_repr delta marginals (Δs, fine_x/y/z): generated vs data.

Reproduces the exact training target by recomputing Hilbert codes from positions
(LJTransferableCachedDataset.__getitem__ arc block), then hilbert_arc_delta.
Plots aggregate marginals and per-particle-index structure.
"""
import argparse, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp
from grid_transformer.data.lj_transferable import (
    hilbert_arc_delta, _hilbert3d_encode, _hilbert_bits,
)

COMP = ["Δs", "fine_x", "fine_y", "fine_z"]


def arc_delta_from_positions(pos, box, R, periodic=True):
    bits = _hilbert_bits(R)
    cell = box.astype(np.float64) / float(R)
    grid = np.floor(np.mod(pos, box[None, :]).astype(np.float64) / cell).astype(np.int64)
    grid = np.clip(grid, 0, R - 1)
    codes = _hilbert3d_encode(grid[:, 0], grid[:, 1], grid[:, 2], bits=bits)
    return hilbert_arc_delta(pos.astype(np.float64), codes, box, R, periodic=periodic)


def batch_arc(positions, box, R):
    return np.stack([arc_delta_from_positions(positions[b], box, R) for b in range(positions.shape[0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="reports/multisize_arc/samples_N125_gpu2k.npz")
    ap.add_argument("--cache", default="lj_caches_arc_transfer/lj125_L5_R128_cache.pt")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--out", default="reports/multisize_arc/figs")
    ap.add_argument("--tag", default="N125_L5")
    args = ap.parse_args()
    import os; os.makedirs(args.out, exist_ok=True)

    c = torch.load(args.cache, map_location="cpu", weights_only=False)
    R = int(c["metadata"]["hilbert_resolution"])
    box = c["box_size"][0].numpy().astype(np.float64)
    absc = c["absolute_coords"]
    idx = np.random.RandomState(0).choice(absc.shape[0], min(args.n, absc.shape[0]), replace=False)
    data = batch_arc(absc[idx].numpy(), box, R)            # (n,124,4)

    g = np.load(args.gen, allow_pickle=True)
    gen = batch_arc(g["x_base"].astype(np.float64), box, R)  # (m,124,4)
    print(f"data {data.shape}  gen {gen.shape}  R={R} box={box.tolist()}")

    # ---------- 1. aggregate marginals ----------
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for j, ax in enumerate(axes):
        d = data[..., j].ravel(); gj = gen[..., j].ravel()
        lo, hi = np.percentile(np.concatenate([d, gj]), [0.2, 99.8])
        bins = np.linspace(lo, hi, 80)
        ax.hist(d, bins=bins, density=True, alpha=0.5, label="data", color="C0")
        ax.hist(gj, bins=bins, density=True, alpha=0.5, label="generated", color="C1")
        w1 = wasserstein_distance(d, gj); ks = ks_2samp(d, gj).statistic
        ax.set_title(f"{COMP[j]}  W1={w1:.4f} KS={ks:.3f}")
        ax.set_yscale("log"); ax.legend()
    fig.suptitle(f"Aggregate arc-delta marginals — {args.tag} (all particle indices pooled)")
    fig.tight_layout(); fig.savefig(f"{args.out}/arc_delta_aggregate_{args.tag}.png", dpi=110)
    print("wrote", f"{args.out}/arc_delta_aggregate_{args.tag}.png")

    # ---------- 2. per-index mean ± std ----------
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    T = data.shape[1]; xs = np.arange(T)
    for j, ax in enumerate(axes):
        dm, ds = data[..., j].mean(0), data[..., j].std(0)
        gm, gs = gen[..., j].mean(0), gen[..., j].std(0)
        ax.plot(xs, dm, color="C0", label="data mean")
        ax.fill_between(xs, dm - ds, dm + ds, color="C0", alpha=0.2)
        ax.plot(xs, gm, color="C1", label="gen mean")
        ax.fill_between(xs, gm - gs, gm + gs, color="C1", alpha=0.2)
        ax.set_title(f"{COMP[j]} vs particle index"); ax.set_xlabel("particle index t"); ax.legend()
    fig.suptitle(f"Per-index arc-delta mean ± std — {args.tag}")
    fig.tight_layout(); fig.savefig(f"{args.out}/arc_delta_perindex_meanstd_{args.tag}.png", dpi=110)
    print("wrote", f"{args.out}/arc_delta_perindex_meanstd_{args.tag}.png")

    # ---------- 3. per-index 2D density heatmaps for Δs ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    vb = np.linspace(np.percentile(data[..., 0], 0.5), np.percentile(data[..., 0], 99.5), 60)
    for ax, arr, name in [(axes[0], data, "data"), (axes[1], gen, "generated")]:
        H = np.stack([np.histogram(arr[:, t, 0], bins=vb, density=True)[0] for t in range(T)])
        im = ax.imshow(H.T, origin="lower", aspect="auto",
                       extent=[0, T, vb[0], vb[-1]], cmap="viridis")
        ax.set_title(f"Δs distribution vs index — {name}"); ax.set_xlabel("particle index t")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("Δs")
    fig.tight_layout(); fig.savefig(f"{args.out}/arc_delta_ds_heatmap_{args.tag}.png", dpi=110)
    print("wrote", f"{args.out}/arc_delta_ds_heatmap_{args.tag}.png")

    # ---------- 4. example per-index histograms ----------
    sample_idx = sorted({1, T // 4, T // 2, (3 * T) // 4, T - 1})
    fig, axes = plt.subplots(2, len(sample_idx), figsize=(4 * len(sample_idx), 8))
    for col, t in enumerate(sample_idx):
        for row, j in enumerate([0, 1]):  # Δs and fine_x
            ax = axes[row, col]
            d = data[:, t, j]; gj = gen[:, t, j]
            lo, hi = np.percentile(np.concatenate([d, gj]), [0.5, 99.5])
            bins = np.linspace(lo, hi, 40)
            ax.hist(d, bins=bins, density=True, alpha=0.5, color="C0", label="data")
            ax.hist(gj, bins=bins, density=True, alpha=0.5, color="C1", label="gen")
            w1 = wasserstein_distance(d, gj)
            ax.set_title(f"{COMP[j]} @ t={t}  W1={w1:.3f}")
            if col == 0: ax.set_ylabel(COMP[j])
            if row == 0 and col == 0: ax.legend()
    fig.suptitle(f"Per-index marginals at selected indices — {args.tag}")
    fig.tight_layout(); fig.savefig(f"{args.out}/arc_delta_examples_{args.tag}.png", dpi=110)
    print("wrote", f"{args.out}/arc_delta_examples_{args.tag}.png")

    # ---------- quantitative summary ----------
    print("\n=== aggregate W1 / KS (gen vs data) ===")
    for j in range(4):
        d = data[..., j].ravel(); gj = gen[..., j].ravel()
        print(f"  {COMP[j]:7s}  W1={wasserstein_distance(d, gj):.4f}  KS={ks_2samp(d, gj).statistic:.4f}  "
              f"mean d/g={d.mean():.3f}/{gj.mean():.3f}  std d/g={d.std():.3f}/{gj.std():.3f}")
    # monotonicity: fraction of gen configs with any Δs<0 step
    gen_backward = (gen[..., 0] < 0).any(1).mean()
    print(f"\n  gen configs with >=1 backward (Δs<0) step: {gen_backward*100:.1f}%   "
          f"per-step Δs<0: data={ (data[...,0]<0).mean()*100:.3f}%  gen={(gen[...,0]<0).mean()*100:.3f}%")


if __name__ == "__main__":
    main()
