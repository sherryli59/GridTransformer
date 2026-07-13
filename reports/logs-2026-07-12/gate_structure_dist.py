"""TASK 6 structure-fix gate -- DISTRIBUTION version (per user directive: never judge by a single
number/median; plot the full distribution + structural statistics).

Collects the FULL per-(cavity,chain) population for AR / FLOW / DATA blocks and saves + plots:
  (1) inherent-structure FLOOR distribution (overlaid histograms) -- the make-or-break structural metric;
  (2) per-chain before->after floor scatter (x=AR floor, y=FLOW floor; below y=x = improved) -- shows the
      WHERE-IT-HELPS-vs-HURTS structure a median erases;
  (3) g(r) (all) and g_BB(r) curves for AR / FLOW / DATA -- peak location/height/shape;
  (4) clash-rate, cross-chain diversity, and raw dE distributions.
Held reference chains only (zero leakage), K=8, pos_temp=0.4. Saves the raw population to a .pt.
"""
import torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; BIGL = 100.0; K = 8; M = 8; POS_TEMP = 0.4; N_CAGE = 48; DUMMY_R = 50.0
FLOW_CKPT = "liquid_coupling_flow/artifacts/ka3d_cavity_egnn_flow_best.pt"


def fixed_size_cage(cage_x, sp_cage, centroid, n_cage):
    Mn, m = cage_x.shape[0], cage_x.shape[1]
    if m >= n_cage:
        idx = torch.topk((cage_x[0] - centroid).norm(dim=-1), n_cage, largest=False).indices
        return cage_x[:, idx], sp_cage[:, idx]
    pad = n_cage - m
    dpos = torch.tensor([DUMMY_R, 0., 0.], device=dev, dtype=cage_x.dtype)[None, None].expand(Mn, pad, 3)
    return torch.cat([cage_x, dpos], 1), torch.cat([sp_cage, torch.zeros(Mn, pad, dtype=sp_cage.dtype, device=dev)], 1)


def local_cage(cage_x, cage_s, centroid, rmax=5.0):
    keep = (cage_x[0] - centroid).norm(dim=-1) < rmax
    return cage_x[:, keep], cage_s[:, keep]


def block_E(xb, sb, cage_x, cage_s):
    allx = torch.cat([xb, cage_x], 1).double(); alls = torch.cat([sb, cage_s], 1)
    return ka_energy(allx, alls.long(), BIGL)


def lbfgs_floor(xb0, sb, cage_x, cage_s, iters=80):
    x = xb0.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([x], lr=0.3, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-6)

    def closure():
        opt.zero_grad(); e = block_E(x, sb, cage_x, cage_s).sum(); e.backward(); return e
    opt.step(closure)
    with torch.no_grad():
        return block_E(x, sb, cage_x, cage_s)                                   # [M]


def clash_per_chain(xb, cage_x, cut=0.8):
    Kk = xb.shape[1]; allx = torch.cat([xb, cage_x], 1)
    dm = torch.cdist(xb, allx); dm[:, torch.arange(Kk), torch.arange(Kk)] = 9.0
    return (dm.min(2).values < cut).float().mean(1).cpu().numpy()                # [M] per-chain clash frac


def rdf_accum(bx, bs, cage_x, cage_s, r_edges, spec=None):
    """Accumulate block-particle -> {block+cage} pair-distance counts (optionally same-species) into r_edges bins."""
    counts = np.zeros(len(r_edges) - 1)
    for m in range(bx.shape[0]):
        ax = torch.cat([bx[m], cage_x[m]]); asp = torch.cat([bs[m], cage_s[m]])
        src = bx[m] if spec is None else bx[m][bs[m] == spec]
        tgt = ax if spec is None else ax[asp == spec]
        if src.shape[0] == 0 or tgt.shape[0] == 0:
            continue
        d = torch.cdist(src, tgt); d = d[d > 1e-4].cpu().numpy()
        counts += np.histogram(d, bins=r_edges)[0]
    return counts


ck = torch.load(FLOW_CKPT, map_location=dev, weights_only=False); a = ck["args"]
flow = CavityBlockFlow(k=a["k"], n_cage=a["n_cage"], r_c=a.get("r_c", 2.5), hidden_nf=a["hidden_nf"],
                       n_layers=a["n_layers"], n_species=2, max_neighbors=a["max_neighbors"]).to(dev).double()
flow.load_state_dict(ck["state_dict"]); flow.eval()
print(f"flow step {ck['step']} held_fm {ck['held_loss']:.4f}", flush=True)
ar_ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt", map_location=dev, weights_only=False)
ar = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
ar.load_state_dict(ar_ck["state_dict"], strict=False); ar.eval(); ar.use_frame = False
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(0)

pop = {k: [] for k in ["fl_ar", "fl_flow", "fl_data", "dE_ar", "dE_flow", "cl_ar", "cl_flow"]}
div = {"ar": [], "flow": []}
r_edges = np.linspace(0.5, 3.0, 51); rc = 0.5 * (r_edges[:-1] + r_edges[1:])
gr = {f"{g}_{s}": np.zeros(len(rc)) for g in ("ar", "flow", "data") for s in ("all", "bb")}
ncav = 0
for ci in range(16):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, 2.0, L)
    if p["n_in"] < K + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], 2.0)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (2.0 + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; anch = fixed_ball_scaffold(n, 2.0, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    block_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    block_mask[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    xo_b = xo[None].expand(M, n, 3).contiguous(); so_b = so[None].expand(M, n).contiguous()
    xo_ar, so_ar, logq_ar = ar.sample_block_b(xo_b, so_b, block_mask, bnd, sb, 2.0, gen=gen, pos_temp=POS_TEMP)
    ar_blk = xo_ar[:, block_mask]; sp_blk = so_ar[:, block_mask]
    cage_full = torch.cat([bnd[None].expand(M, -1, -1), xo_ar[:, ~block_mask]], 1)
    cage_s_full = torch.cat([sb[None].expand(M, -1), so_ar[:, ~block_mask]], 1)
    centroid = xo[block_mask].mean(0)
    cage_en, cage_s_en = local_cage(cage_full, cage_s_full, centroid)
    cage_fix, cage_s_fix = fixed_size_cage(cage_full, cage_s_full, centroid, N_CAGE)
    flow_blk = flow.composed_logq(ar_blk.double(), logq_ar.double(), cage_fix.double(), sp_blk, cage_s_fix, R=2.0)[0].float()
    data_blk = xo[None, block_mask].expand(M, K, 3).contiguous(); data_s = so[None, block_mask].expand(M, K)
    Ed = block_E(data_blk, data_s, cage_en, cage_s_en).mean().item()
    pop["fl_ar"] += (((lbfgs_floor(ar_blk, sp_blk, cage_en, cage_s_en) - Ed) / K).cpu().tolist())
    pop["fl_flow"] += (((lbfgs_floor(flow_blk, sp_blk, cage_en, cage_s_en) - Ed) / K).cpu().tolist())
    pop["fl_data"] += (((lbfgs_floor(data_blk, data_s, cage_en, cage_s_en) - Ed) / K).cpu().tolist())
    pop["dE_ar"] += (((block_E(ar_blk, sp_blk, cage_en, cage_s_en) - Ed) / K).cpu().tolist())
    pop["dE_flow"] += (((block_E(flow_blk, sp_blk, cage_en, cage_s_en) - Ed) / K).cpu().tolist())
    pop["cl_ar"] += clash_per_chain(ar_blk, cage_en).tolist()
    pop["cl_flow"] += clash_per_chain(flow_blk, cage_en).tolist()

    def xdiv(bx):
        return [float((torch.cdist(bx[i], bx[j]).min(1).values < 0.3).float().mean()) for i in range(M) for j in range(i + 1, M)]
    div["ar"] += xdiv(ar_blk); div["flow"] += xdiv(flow_blk)
    for tag, bx, bs in (("ar", ar_blk, sp_blk), ("flow", flow_blk, sp_blk), ("data", data_blk, data_s)):
        gr[f"{tag}_all"] += rdf_accum(bx, bs, cage_en, cage_s_en, r_edges, None)
        gr[f"{tag}_bb"] += rdf_accum(bx, bs, cage_en, cage_s_en, r_edges, 1)
    ncav += 1
    print(f"  cav {ncav} done (pop so far {len(pop['fl_ar'])})", flush=True)
    if ncav >= 10:
        break

torch.save({"pop": pop, "div": div, "gr": gr, "rc": rc, "ncav": ncav, "K": K, "pos_temp": POS_TEMP},
           "liquid_coupling_flow/artifacts/gate_structure_pop.pt")

# ---- PLOTS ----
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
fa, ff, fd = np.array(pop["fl_ar"]), np.array(pop["fl_flow"]), np.array(pop["fl_data"])
b = np.linspace(min(fa.min(), ff.min(), fd.min()) - 0.5, np.percentile(np.r_[fa, ff], 95) + 1, 40)
ax[0, 0].hist(fa, b, alpha=0.5, label=f"AR (med {np.median(fa):+.2f})", color="tab:red")
ax[0, 0].hist(ff, b, alpha=0.5, label=f"FLOW (med {np.median(ff):+.2f})", color="tab:blue")
ax[0, 0].hist(fd, b, alpha=0.5, label=f"DATA (med {np.median(fd):+.2f})", color="tab:green")
ax[0, 0].set_title("Inherent-structure FLOOR distribution /particle"); ax[0, 0].set_xlabel("floor above data"); ax[0, 0].legend(fontsize=8)

ax[0, 1].scatter(fa, ff, s=14, alpha=0.6)
lim = [min(fa.min(), ff.min()) - 0.3, max(np.percentile(fa, 97), np.percentile(ff, 97)) + 0.3]
ax[0, 1].plot(lim, lim, "k--", lw=1, label="no change (y=x)")
ax[0, 1].axhline(np.median(fd), color="tab:green", ls=":", label=f"data median {np.median(fd):+.2f}")
ax[0, 1].set_xlim(lim); ax[0, 1].set_ylim(lim); ax[0, 1].set_xlabel("AR floor"); ax[0, 1].set_ylabel("FLOW floor")
ax[0, 1].set_title("Per-chain before->after (below y=x = improved)"); ax[0, 1].legend(fontsize=8)

impr = fa - ff
ax[0, 2].scatter(fa, impr, s=14, alpha=0.6, color="tab:purple"); ax[0, 2].axhline(0, color="k", ls="--", lw=1)
ax[0, 2].set_xlabel("AR floor (how strained)"); ax[0, 2].set_ylabel("floor improvement (AR-FLOW)")
ax[0, 2].set_title("Where it helps: improvement vs AR strain")

for key, c_, lab in (("ar", "tab:red", "AR"), ("flow", "tab:blue", "FLOW"), ("data", "tab:green", "DATA")):
    norm = gr[f"{key}_all"] / max(gr[f"{key}_all"].sum(), 1)
    ax[1, 0].plot(rc, norm / (rc ** 2), c_, label=lab)
    nbb = gr[f"{key}_bb"] / max(gr[f"{key}_bb"].sum(), 1)
    ax[1, 1].plot(rc, nbb / (rc ** 2), c_, label=lab)
ax[1, 0].set_title("g(r) all-species (block->all)"); ax[1, 0].set_xlabel("r"); ax[1, 0].legend(fontsize=8); ax[1, 0].axvline(0.8, color="gray", ls=":", lw=0.8)
ax[1, 1].set_title("g_BB(r) (the persistently-wrong shell)"); ax[1, 1].set_xlabel("r"); ax[1, 1].legend(fontsize=8)

ax[1, 2].hist(pop["cl_ar"], np.linspace(0, 1, 25), alpha=0.5, label="AR clash", color="tab:red")
ax[1, 2].hist(pop["cl_flow"], np.linspace(0, 1, 25), alpha=0.5, label="FLOW clash", color="tab:blue")
ax[1, 2].set_title("Per-chain clash-rate distribution"); ax[1, 2].set_xlabel("clash fraction"); ax[1, 2].legend(fontsize=8)
fig.suptitle(f"EGNN cavity-flow structure gate -- {ncav} held cavities x M={M} (K={K}, pos_temp={POS_TEMP}), FM held-loss {ck['held_loss']:.3f}", fontsize=12)
fig.tight_layout()
out = "reports/logs-2026-07-12/gate_structure_dist.png"; fig.savefig(out, dpi=140)
print(f"\nsaved plot: {out}", flush=True)
print(f"floor: AR med {np.median(fa):+.2f} [{np.percentile(fa,25):+.2f},{np.percentile(fa,75):+.2f}]  "
      f"FLOW med {np.median(ff):+.2f} [{np.percentile(ff,25):+.2f},{np.percentile(ff,75):+.2f}]  "
      f"DATA med {np.median(fd):+.2f}", flush=True)
print(f"frac chains FLOW floor within 0.5 of data: {np.mean(ff < np.median(fd) + 0.5):.2f} (AR: {np.mean(fa < np.median(fd) + 0.5):.2f})", flush=True)
print(f"frac chains FLOW improved over AR (impr>0.2): {np.mean(impr > 0.2):.2f}  worsened (impr<-0.2): {np.mean(impr < -0.2):.2f}", flush=True)
print("DONE", flush=True)
