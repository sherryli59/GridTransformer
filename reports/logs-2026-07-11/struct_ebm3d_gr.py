"""3D cavity-block species-resolved g(r) -- the structural statistic (replaces the clash cutoff).
For each regenerated block particle (species a), distances to ALL other cavity particles (interior +
boundary, species b), normalised by the BULK ideal-gas expectation rho_b*4pi r^2 dr (rho_b = rho*x_b so
g->1 at large r). Ground truth = the TRUE carved interior block. Compares data / base(no potential) /
c-only / multi-axis. Prints first-peak height + position + L2-to-data, and saves a 3-panel plot."""
import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ka3d_scaffold_ebm import KA3DScaffoldEBM
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; R = 2.0; R_CTX = 2.5; K = 8; NCONF = 80; RMAX, NB = 3.0, 60; RHO = 1.2
ART = "liquid_coupling_flow/artifacts"
CKPTS = {"base": f"{ART}/ka3d_cavity_base.pt", "c-only": f"{ART}/ka3d_cavity_ebm.pt",
         "multi-axis": f"{ART}/ka3d_cavity_ebm3ax.pt"}


def load(p):
    ck = torch.load(p, map_location=dev, weights_only=False)
    m = KA3DScaffoldEBM(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    return m


models = {k: load(v) for k, v in CKPTS.items()}
d = torch.load(f"{ART}/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
xB = float((S == 1).float().mean()); xA = 1.0 - xB                    # global species fractions
rho = {0: RHO * xA, 1: RHO * xB}
gen = torch.Generator(device=dev).manual_seed(3)
edges = torch.linspace(0, RMAX, NB + 1, device=dev); dr = float(edges[1] - edges[0]); rc = (edges[:-1] + edges[1:]) / 2
shell = 4 * math.pi * rc ** 2 * dr


def cavity(ci, c):
    p = carve(X[ci], S[ci], c, R, L)
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + R_CTX)
    return xo, so, xout[bm], p["s_out"][bm], p["n_in"]


def blob(n):
    a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    m = torch.zeros(n, dtype=torch.bool, device=dev); m[(a - a[seed]).norm(dim=-1).topk(min(K, n), largest=False).indices] = True
    return m


def accum(cidx, bs, ax, as_, H, C):
    """cidx = indices of the block centres WITHIN ax (so self is excluded by index, not by a distance
    threshold -- cdist self-distance is ~1e-3 from cancellation and would otherwise leak into bin 0)."""
    dfull = torch.cdist(ax[cidx], ax)                               # [n_blk, P]
    dfull[torch.arange(cidx.shape[0], device=dev), cidx] = 9.0      # exclude self by index
    for a in (0, 1):
        rows = (bs == a)
        if int(rows.sum()) == 0:
            continue
        C[a] += int(rows.sum())
        dmat = dfull[rows]
        for b in (0, 1):
            dv = dmat[:, as_ == b]; dv = dv[dv < RMAX]
            H[a][b] += torch.histc(dv, NB, 0, RMAX)


def run(model):
    H = {a: {b: torch.zeros(NB, device=dev) for b in (0, 1)} for a in (0, 1)}; C = {0: 0, 1: 0}
    for ci in range(900, 900 + NCONF):
        c = torch.rand(3, generator=gen, device=dev) * L
        xo, so, bnd, s_bnd, n = cavity(ci, c)
        if n < 10:
            continue
        blk = blob(n)
        if model is not None:
            xo, so, _ = model.sample_block(xo, so, blk, bnd, s_bnd, R, gen=gen)
        allx, alls = torch.cat([xo, bnd]), torch.cat([so, s_bnd])
        cidx = torch.nonzero(blk).squeeze(1)                        # block centres = interior indices in allx
        accum(cidx, so[blk], allx, alls, H, C)
    return {(a, b): (H[a][b] / max(C[a], 1)) / (rho[b] * shell) for a in (0, 1) for b in (0, 1)}


g = {"data": run(None)}
for k, m in models.items():
    g[k] = run(m)
names = {(0, 0): "g_AA", (0, 1): "g_AB", (1, 1): "g_BB"}; rc_np = rc.cpu().numpy()
cols = {"data": "k", "base": "C3", "c-only": "C2", "multi-axis": "C0"}
win = (rc > 0.7) & (rc < 1.8); pmask = rc > 0.5                      # physical first-shell region


def peak(gr):
    sub = gr[pmask]; j = int(sub.argmax()); return float(sub[j]), float(rc[pmask][j])


fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for i, key in enumerate([(0, 0), (0, 1), (1, 1)]):
    ph, pr = peak(g["data"][key])
    print(f"\n{names[key]}  (data peak {ph:.2f} @ r={pr:.2f})", flush=True)
    for k in ("data", "base", "c-only", "multi-axis"):
        style = "-" if k in ("data", "multi-axis") else "--"
        lw = 2.4 if k == "data" else 1.6
        ax[i].plot(rc_np, g[k][key].cpu(), cols[k] + style, lw=lw, label=k)
        if k != "data":
            l2 = float(((g[k][key][win] - g["data"][key][win]) ** 2).mean().sqrt())
            ph, pr = peak(g[k][key])
            print(f"    {k:11s} peak {ph:.2f} @ r={pr:.2f}   L2(0.7-1.8)={l2:.3f}", flush=True)
    ax[i].set_title(names[key]); ax[i].set_xlabel("r"); ax[i].axhline(1, color="gray", lw=0.5); ax[i].legend(fontsize=8)
ax[0].set_ylabel("g(r)"); plt.tight_layout()
out = "reports/logs-2026-07-11/ebm3d_cavity_gr.png"
plt.savefig(out, dpi=110); print(f"\nsaved {out}", flush=True)
