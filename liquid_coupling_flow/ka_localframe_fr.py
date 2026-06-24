"""FREE-RUNNING (FR) generation quality of the geometry-invariant local-frame model at the trained
N=100 and the HELD-OUT N=36/256. The conditional transfers (TF-clash << random); THIS gates whether the
generative ROLLOUT transfers -- overlaps, g_AA/g_AB/g_BB, species composition -- vs the reference data.
Exposure-bias drift lives only in the rollout, and g_BB has broken in every prior FR sample, so this is
the decisive next test. Uses the protected snapshot ka_localframe_N100_20k.pt."""
from __future__ import annotations
import os, torch, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from liquid_coupling_flow.ka_localframe import KALocalFrameModel
from liquid_coupling_flow.ka_observables import partial_gr
from liquid_coupling_flow.ka_gridformer_train import overlap_frac

ART = os.path.join(os.path.dirname(__file__), "artifacts")
RAD = {0: 0.50, 1: 0.44}; FC = {0: "#4878CF", 1: "#D6322E"}
SIG = {(0, 0): 1.0, (0, 1): 0.8, (1, 1): 0.88}


def _np(a):
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def overlaps(x, L, thr=0.7):
    df = x[:, None, :] - x[None, :, :]; df = df - L * torch.round(df / L)
    d = (df ** 2).sum(-1).sqrt() + torch.eye(x.shape[0], device=x.device) * 1e3
    return (d.min(1).values < thr)


def draw(ax, x, sp, L):
    ov = overlaps(x, L).cpu().numpy(); xx, spn = x.cpu().numpy(), sp.cpu().numpy()
    for i in range(x.shape[0]):
        ax.add_patch(Circle(xx[i], RAD[int(spn[i])], facecolor=FC[int(spn[i])], alpha=0.65,
                            edgecolor="k" if ov[i] else "none", lw=1.6 if ov[i] else 0))
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return float(ov.mean())


@torch.no_grad()
def main(device="cuda" if torch.cuda.is_available() else "cpu", B=1536):
    ck = torch.load(os.path.join(ART, "ka_localframe_N100_20k.pt"), map_location=device, weights_only=False)
    m = KALocalFrameModel(rho=ck["rho"], n_bins=ck["n_bins"], knn=ck["knn"]).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    print(f"loaded ka_localframe_N100_20k.pt (step {ck.get('step')}, knn {ck['knn']})", flush=True)
    sizes = (36, 100, 256)
    R = {}
    for N in sizes:
        ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=device, weights_only=False)
        s, L, data = ref["s"].to(device).long(), ref["L"], ref["x"].to(device)
        nB = int((s == 1).sum()) if s.dim() == 1 else int((s[0] == 1).sum())
        sd = s.expand(data.shape[0], N) if s.dim() == 1 else s
        xg, sg = m.sample(B, N, n_B=nB, device=device)
        R[N] = dict(L=L, data=data, sd=sd, xg=xg, sg=sg, nB=nB)
        tag = "(train)" if N == 100 else "(HELDOUT)"
        print(f"  N={N:3d} {tag:9s} nB={nB}/{N} ({nB/N:.2f} B) | overlap: data {overlap_frac(data[:512], L, N):.4f}  "
              f"FR-gen {overlap_frac(xg, L, N):.4f}", flush=True)

    # ---- snapshots: per N, 2 data + 2 FR-gen ----
    fig, ax = plt.subplots(3, 4, figsize=(13, 10))
    for r, N in enumerate(sizes):
        d = R[N]; sel = torch.randperm(d["data"].shape[0])[:2]
        for c in range(2):
            ov = draw(ax[r, c], d["data"][sel[c]], d["sd"][sel[c]], d["L"]); ax[r, c].set_title(f"N={N} data (ov {ov:.2f})", fontsize=9)
        for c in range(2):
            ov = draw(ax[r, 2 + c], d["xg"][c], d["sg"][c], d["L"]); ax[r, 2 + c].set_title(f"N={N} FR-gen (ov {ov:.2f})", fontsize=9)
    fig.suptitle("Local-frame FR generation: data vs free-run, N=36/100/256 (A=blue B=red, black ring=overlap)", fontsize=12)
    fig.tight_layout(); o1 = os.path.join(ART, "ka_localframe_fr_snapshots.png"); fig.savefig(o1, dpi=110); print(f"saved {o1}", flush=True)

    # ---- g(r): rows=N, cols=AA/AB/BB ----
    fig2, ax2 = plt.subplots(3, 3, figsize=(16, 13))
    for r, N in enumerate(sizes):
        d = R[N]; rmax = min(d["L"] / 2, 4.0); tag = "train" if N == 100 else "HELDOUT"
        for c, pair in enumerate(((0, 0), (0, 1), (1, 1))):
            rc, g = partial_gr(d["data"][:2000], d["sd"][:2000], d["L"], rmax, 100, pair)
            rc2, g2 = partial_gr(d["xg"], d["sg"], d["L"], rmax, 100, pair)
            ax2[r, c].plot(_np(rc), _np(g), "k", lw=2.2, label="data")
            ax2[r, c].plot(_np(rc2), _np(g2), "C3", lw=1.6, label="FR-gen")
            ax2[r, c].axvline(2 ** (1 / 6) * SIG[pair], color="grey", ls=":", lw=1); ax2[r, c].set_xlim(0, 3)
            ax2[r, c].set_title(f"N={N} ({tag})  g_{['A','B'][pair[0]]}{['A','B'][pair[1]]}(r)"); ax2[r, c].legend(fontsize=8); ax2[r, c].grid(alpha=0.3)
    fig2.suptitle("Local-frame FR g(r): data vs free-run. g_BB (col 3) = the historically-broken pair", fontsize=13)
    fig2.tight_layout(); o2 = os.path.join(ART, "ka_localframe_fr_gr.png"); fig2.savefig(o2, dpi=120); print(f"saved {o2}", flush=True)

    # ---- quantitative g_BB: peak position/height + small-r spurious contacts ----
    print("\n  g_BB diagnostics (data vs FR-gen)  [data peak ~2.43@1.73, g_BB(r<0.88)~0.02]:", flush=True)
    for N in sizes:
        d = R[N]; rmax = min(d["L"] / 2, 4.0)
        rc, g = partial_gr(d["data"][:2000], d["sd"][:2000], d["L"], rmax, 100, (1, 1)); rc, g = _np(rc), _np(g)
        rc2, g2 = partial_gr(d["xg"], d["sg"], d["L"], rmax, 100, (1, 1)); g2 = _np(g2)
        small = rc < 0.88
        print(f"    N={N:3d}: peak data {g.max():.2f}@{rc[g.argmax()]:.2f}  FR {g2.max():.2f}@{rc[g2.argmax()]:.2f}  | "
              f"g_BB(r<0.88) data {g[small].mean():.3f}  FR {g2[small].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
