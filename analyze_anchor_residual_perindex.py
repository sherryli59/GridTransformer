"""Per-index distribution of the Hilbert-center (template) anchor residual.

For each particle index j, characterize the TARGET the model would have to predict under
two anchoring schemes:
  absolute  : residual_j = minimage(pos_j - decode(j*X))   [no drift, but loose?]
  prev      : residual_j = minimage(pos_j - pos_{j-1})     [tight, but drifts]
Data only (MCMC). Shows whether the absolute residual is tight/learnable per index and how
it scales with system size.
"""
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grid_transformer.data.lj_transferable import (
    fixed_template_anchors, _hilbert3d_encode, _hilbert_bits, min_image_delta)

SIZES = {  # tag: (h5, N, L, R)
    "L3_N27":   ("/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L3_rho1.0_N27_T1.0.h5",   27,  3.0,  64),
    "L5_N125":  ("/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L5_rho1.0_N125_T1.0.h5", 125,  5.0, 128),
    "L10_N1000":("/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L10_rho1.0_N1000_T1.0.h5",1000,10.0,256),
}
OUT = "reports/multisize_arc/figs"


def residuals(pos, L, R):
    box = np.array([L, L, L], dtype=np.float64)
    bits = _hilbert_bits(R); cell = box / R
    g = np.clip(np.floor(np.mod(pos, box[None, :]) / cell).astype(np.int64), 0, R - 1)
    codes = _hilbert3d_encode(g[:, 0], g[:, 1], g[:, 2], bits=bits)
    order = np.argsort(codes, kind="stable")
    sp = pos[order].astype(np.float64)
    N = len(sp)
    anch = fixed_template_anchors(np.arange(1, N), N, R, box)          # decode(j*X)
    r_abs = min_image_delta(sp[1:] - anch, box)                       # [N-1,3]
    r_prev = min_image_delta(sp[1:] - sp[:-1], box)                   # [N-1,3]
    return r_abs, r_prev


def collect(tag, n=600):
    p, N, L, R = SIZES[tag]
    with h5py.File(p, "r") as f:
        traj = f["traj"][:].reshape(-1, N, 3)
    idx = np.random.RandomState(0).choice(traj.shape[0], min(n, traj.shape[0]), replace=False)
    A = np.stack([residuals(traj[i], L, R)[0] for i in idx])   # [n,N-1,3]
    P = np.stack([residuals(traj[i], L, R)[1] for i in idx])
    return A, P, L


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    data = {tag: collect(tag) for tag in SIZES}

    # ---- per-size: distribution of residual vs index (x-component heatmap + |r| heatmap) ----
    for tag, (A, P, L) in data.items():
        N1 = A.shape[1]
        fig, ax = plt.subplots(2, 2, figsize=(16, 8))
        # (0,0) abs residual x-comp distribution vs index
        vb = np.linspace(-L/2, L/2, 60)
        H = np.stack([np.histogram(A[:, t, 0], bins=vb, density=True)[0] for t in range(N1)])
        im = ax[0,0].imshow(H.T, origin="lower", aspect="auto", extent=[0, N1, -L/2, L/2], cmap="viridis")
        ax[0,0].set_title(f"{tag}: ABS anchor residual x-comp vs index"); ax[0,0].set_ylabel("pos_j - decode(j·X)  [x]")
        fig.colorbar(im, ax=ax[0,0])
        # (0,1) abs |residual| distribution vs index
        magA = np.linalg.norm(A, axis=-1)
        mb = np.linspace(0, L/2*np.sqrt(3), 60)
        H2 = np.stack([np.histogram(magA[:, t], bins=mb, density=True)[0] for t in range(N1)])
        im2 = ax[0,1].imshow(H2.T, origin="lower", aspect="auto", extent=[0, N1, 0, L/2*np.sqrt(3)], cmap="magma")
        ax[0,1].set_title(f"{tag}: ABS |residual| vs index"); ax[0,1].set_ylabel("|pos_j - decode(j·X)|")
        fig.colorbar(im2, ax=ax[0,1])
        # (1,0) prev-particle x-comp distribution vs index (the tight reference)
        vbp = np.linspace(-L/4, L/4, 60)
        Hp = np.stack([np.histogram(P[:, t, 0], bins=vbp, density=True)[0] for t in range(N1)])
        imp = ax[1,0].imshow(Hp.T, origin="lower", aspect="auto", extent=[0, N1, -L/4, L/4], cmap="viridis")
        ax[1,0].set_title(f"{tag}: PREV-particle delta x-comp vs index (tight ref)"); ax[1,0].set_ylabel("pos_j - pos_{j-1} [x]")
        fig.colorbar(imp, ax=ax[1,0])
        # (1,1) per-index std comparison
        ax[1,1].plot(np.arange(N1), A[..., 0].std(0), color="C1", label="abs anchor std/axis")
        ax[1,1].plot(np.arange(N1), P[..., 0].std(0), color="C0", label="prev-particle std/axis")
        ax[1,1].axhline(L/np.sqrt(12), ls="--", color="gray", label=f"uniform std ({L/np.sqrt(12):.2f})")
        ax[1,1].set_title(f"{tag}: per-axis std vs index"); ax[1,1].set_xlabel("index j"); ax[1,1].legend()
        fig.suptitle(f"Anchor-residual target distribution per index — {tag}")
        fig.tight_layout(); fig.savefig(f"{OUT}/anchor_residual_perindex_{tag}.png", dpi=105); plt.close(fig)
        print("wrote", f"{OUT}/anchor_residual_perindex_{tag}.png")

    # ---- cross-size summary: std vs FRACTIONAL index ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for tag, (A, P, L) in data.items():
        N1 = A.shape[1]; frac = np.arange(N1) / N1
        ax[0].plot(frac, A[..., 0].std(0), label=f"{tag} (L={L:.0f})")
        ax[1].plot(frac, P[..., 0].std(0), label=f"{tag} (L={L:.0f})")
    ax[0].set_title("ABS anchor: per-axis std vs fractional index\n(size-invariant target ⇒ curves overlap)")
    ax[1].set_title("PREV-particle: per-axis std vs fractional index")
    for a in ax: a.set_xlabel("j / N"); a.set_ylabel("std/axis"); a.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/anchor_residual_std_vs_size.png", dpi=110); plt.close(fig)
    print("wrote", f"{OUT}/anchor_residual_std_vs_size.png")

    print("\n=== summary: is the target size-invariant? (per-axis std, pooled) ===")
    for tag, (A, P, L) in data.items():
        print(f"  {tag:10s} abs std={A.reshape(-1,3).std(0).mean():.3f}  prev std={P.reshape(-1,3).std(0).mean():.3f}  "
              f"abs/prev ratio={A.reshape(-1,3).std(0).mean()/P.reshape(-1,3).std(0).mean():.2f}")


if __name__ == "__main__":
    main()
