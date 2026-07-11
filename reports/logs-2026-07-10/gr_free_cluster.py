"""g(r) structural comparison: free-cluster generator (FR samples) vs true held clusters.

For each held cluster (radius R), generate one matched cluster (same n_A/n_B) and pool pairwise
distances. Normalize by the exact ball-line-picking reference (distance pdf of two uniform points
in a ball of radius R) so g->1 for a structureless ball. Species-resolved (total, AA, AB, BB).
A clash tail shows as spurious density at r<0.9; real structure = first/second shells.
"""
import shutil, statistics as st
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ar import KA3DScaffoldAR

dev = "cuda" if torch.cuda.is_available() else "cpu"
SCR = Path("/tmp/claude-1002/-mnt-ssd-GridTransformer/271167b9-117a-400e-b11f-c2c1bfa0b6df/scratchpad")
SCR.mkdir(parents=True, exist_ok=True)
src = Path("liquid_coupling_flow/artifacts/ka3d_free_scaffold_ar_best.pt")
ckp = SCR / "gr_ckpt.pt"; shutil.copy(src, ckp)          # copy so we don't read a checkpoint mid-save
ck = torch.load(ckp, map_location=dev, weights_only=False)
m = KA3DScaffoldAR().to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
step = ck["step"]; print(f"loaded scaffold ckpt step={step} held_nll(best)={ck.get('best_held_nll'):+.3f}", flush=True)

d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
R = 2.0; r_ctx = 2.5; g = torch.Generator(device=dev).manual_seed(11)
empty_x, empty_s = torch.zeros(0, 3, device=dev), torch.zeros(0, dtype=torch.long, device=dev)

true_c, gen_c = [], []
with torch.no_grad():
    for frame in range(900, 1024):
        center = torch.rand(3, generator=g, device=dev) * L
        p = carve(X[frame], S[frame], center, R, L)
        if p["n_in"] < 8:
            continue
        xr = _mic(p["x_in"], center, L); s = p["s_in"]
        nB = int((s == 1).sum()); nA = p["n_in"] - nB
        true_c.append((xr, s))
        gx, gs = m.sample_pair(empty_x, empty_s, nA, nB, R)
        gen_c.append((gx, gs))

n_true_particles = sum(x.shape[0] for x, _ in true_c)
print(f"clusters={len(true_c)} at R={R}; particles/cluster median={st.median([x.shape[0] for x,_ in true_c])}; "
      f"total generated particles={sum(x.shape[0] for x,_ in gen_c)}", flush=True)


def pair_hist(clusters, edges, sa=None, sb=None):
    counts = np.zeros(len(edges) - 1); npair = 0
    for x, s in clusters:
        n = x.shape[0]
        iu = torch.triu_indices(n, n, 1, device=x.device)
        dij = (x[iu[0]] - x[iu[1]]).norm(dim=-1)
        if sa is not None:
            si, sj = s[iu[0]], s[iu[1]]
            mask = ((si == sa) & (sj == sb)) | ((si == sb) & (sj == sa))
            dij = dij[mask]; npair += int(mask.sum())
        else:
            npair += iu.shape[1]
        counts += np.histogram(dij.cpu().numpy(), bins=edges)[0]
    return counts, npair


def ball_pdf(r, R):                                       # distance pdf of two uniform pts in a 3-ball
    s = r / R
    return (3 * s ** 2 / R) * (1.0 - 0.75 * s + s ** 3 / 16.0) * (s <= 2.0)


edges = np.linspace(0.01, 4.0, 120); ctr = 0.5 * (edges[:-1] + edges[1:]); dr = edges[1] - edges[0]
ideal = ball_pdf(ctr, R) * dr                             # per-pair prob mass in each bin

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (sa, sb, name) in zip(axes.flat, [(None, None, "total"), (0, 0, "AA"), (0, 1, "AB"), (1, 1, "BB")]):
    for clusters, lab, col in [(true_c, "true (held)", "k"), (gen_c, f"generated (step {step})", "tab:red")]:
        cnt, npair = pair_hist(clusters, edges, sa, sb)
        gr = cnt / np.maximum(npair * ideal, 1e-9)
        ax.plot(ctr, gr, color=col, lw=1.6, label=lab)
    ax.axvline(0.9, color="gray", ls=":", lw=0.8)
    ax.set_title(f"g_{name}(r)"); ax.set_xlabel("r / sigma"); ax.set_ylabel("g(r)")
    ax.set_xlim(0, 4); ax.legend(fontsize=8)
fig.suptitle(f"Free-cluster g(r): generated vs true (R={R}, {len(true_c)} clusters) — clash tail = spurious r<0.9")
fig.tight_layout()
out = Path("reports/logs-2026-07-10/gr_free_cluster.png"); fig.savefig(out, dpi=110)
print(f"saved {out}", flush=True)
