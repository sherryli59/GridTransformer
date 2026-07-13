"""Snapshot figures of the cavity generation pipeline: context -> AR order -> AR-vs-flow overlay.

Two held cavities (K=8, R=2.0). Per cavity, one row of three slab-projection panels (PCA plane of the
data block; context within |depth|<1.2 shown, alpha by depth; particles drawn at PHYSICAL radius
sigma_s/2 so overlaps are visible):
  P1 context   : boundary shell (gray), retained interior (slate), DATA block (teal), cavity circle.
  P2 AR order  : scaffold anchors (x) + the AR draw numbered 1..8 in actual generation order,
                 faint path arrows 1->2->..., anchor->placement tethers.
  P3 overlay   : DATA (teal) vs AR proposal (orange) vs EGNN-flow-corrected (purple), arrows AR->flow;
                 panel title carries dE/particle for AR and FLOW and clash counts.
Colors: Dark2 triple #1B9E77/#D95F02/#7570B3 (validated: CVD dE 39.6); species by SIZE (A sigma=1.0,
B sigma=0.88), context recessive.
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; BIGL = 100.0; K = 8; R = 2.0; PT = 0.4; N_CAGE = 48; DUMMY_R = 50.0
C_DATA, C_AR, C_FLOW = "#1B9E77", "#D95F02", "#7570B3"
C_BND, C_RET = "#C4C7CC", "#7E9CC4"
SIG = {0: 1.0, 1: 0.88}

ar = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
ar.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                              map_location=dev, weights_only=False)["state_dict"], strict=False)
ar.eval(); ar.use_frame = False
ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_egnn_flow_gated_big_best.pt",
                map_location=dev, weights_only=False)
a = ck["args"]
flow = CavityBlockFlow(k=a["k"], n_cage=a["n_cage"], r_c=a.get("r_c", 2.5), hidden_nf=a["hidden_nf"],
                       n_layers=a["n_layers"], n_species=2, max_neighbors=a["max_neighbors"],
                       ode_rtol=1e-5, ode_atol=1e-5, max_steps=1000).to(dev)
flow.load_state_dict(ck["state_dict"]); flow.eval()
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
gen = torch.Generator(device=dev).manual_seed(31)


def fixed_size_cage(cage_x, sp_cage, centroid, n_cage):
    Mn, m = cage_x.shape[0], cage_x.shape[1]
    if m >= n_cage:
        idx = torch.topk((cage_x[0] - centroid).norm(dim=-1), n_cage, largest=False).indices
        return cage_x[:, idx], sp_cage[:, idx]
    pad = n_cage - m
    dpos = torch.tensor([DUMMY_R, 0., 0.], device=dev, dtype=cage_x.dtype)[None, None].expand(Mn, pad, 3)
    return torch.cat([cage_x, dpos], 1), torch.cat([sp_cage, torch.zeros(Mn, pad, dtype=sp_cage.dtype, device=dev)], 1)


def block_E(xb, sb, cage_x, cage_s):
    allx = torch.cat([xb, cage_x], 1).double(); alls = torch.cat([sb, cage_s], 1)
    return ka_energy(allx, alls.long(), BIGL)


def snapshot(ci):
    while True:
        c = torch.rand(3, generator=gen, device=dev) * L
        p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] >= K + 6:
            break
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX)
    bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; anch_full = fixed_ball_scaffold(n, R, dev)
    seedp = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev)
    blk[(anch_full - anch_full[seedp]).norm(dim=-1).topk(K, largest=False).indices] = True
    order = torch.argsort(blk.to(torch.uint8), stable=True)
    gen_seq = order[n - K:]                                   # original slot ids, in generation order
    anchors_gen = anch_full[gen_seq]
    xo_b = xo[None].expand(4, n, 3).contiguous(); so_b = so[None].expand(4, n).contiguous()
    xo_ar, so_ar, _ = ar.sample_block_b(xo_b, so_b, blk, bnd, sb, R, gen=gen, pos_temp=PT)
    m0 = 0                                                    # draw shown in the figure
    cage_x = torch.cat([bnd[None].expand(4, -1, -1), xo_ar[:, ~blk]], 1)
    cage_s = torch.cat([sb[None].expand(4, -1), so_ar[:, ~blk]], 1)
    cfx, cfs = fixed_size_cage(cage_x, cage_s, xo[blk].mean(0), N_CAGE)
    y_blk, _ = flow.flow(xo_ar[:, blk], cfx, so_ar[:, blk], cfs, reverse=False)
    Ed = block_E(xo[None, blk], so[None, blk], cage_x[:1], cage_s[:1])[0]
    dE_ar = float((block_E(xo_ar[:, blk], so_ar[:, blk], cage_x, cage_s)[m0] - Ed) / K)
    dE_fl = float((block_E(y_blk, so_ar[:, blk], cage_x, cage_s)[m0] - Ed) / K)
    # AR order within the shown draw: block positions of draw m0, in generation order
    ar_pos = xo_ar[m0][gen_seq]; ar_sp = so_ar[m0][gen_seq]
    fl_pos = y_blk[m0]                                        # slot-aligned with xo_ar[:, blk]
    blk_ids = torch.nonzero(blk, as_tuple=True)[0]
    # map slot-aligned flow/data arrays into generation order for consistent pairing
    pos_in_blk = torch.tensor([int((blk_ids == g).nonzero()) for g in gen_seq], device=dev)
    return {"bnd": bnd.cpu(), "sb": sb.cpu(), "ret": xo[~blk].cpu(), "sret": so[~blk].cpu(),
            "data": xo[blk][pos_in_blk].cpu(), "sdata": so[blk][pos_in_blk].cpu(),
            "ar": ar_pos.cpu(), "sar": ar_sp.cpu(), "flow": fl_pos[pos_in_blk].cpu(),
            "anch": anchors_gen.cpu(), "dE_ar": dE_ar, "dE_fl": dE_fl, "ci": ci}


def slab_axes(snap):
    """PCA plane of the data block; returns projector (3->2) and depth fn."""
    Xb = snap["data"] - snap["data"].mean(0)
    _, _, V = torch.linalg.svd(Xb)
    P2, nrm = V[:2], V[2]
    ctr = snap["data"].mean(0)
    proj = lambda x: ((x - ctr) @ P2.T)
    depth = lambda x: ((x - ctr) @ nrm)
    return proj, depth


EC_B = "#1A1A1A"          # species-B ring (dark, thick) -- composite species channel on any fill

def draw_set(ax, xy, dep, sp, color, slab=1.2, lw=0.8, ec="white", zbase=2, alpha_base=0.85, filled=True,
             numbers=None):
    """Species encoding: A = large disc + light edge; B = small disc + dark thick ring. `numbers` (list
    of str aligned to xy rows) prints a centered label per in-slab particle."""
    for i in range(xy.shape[0]):
        dz = abs(float(dep[i]))
        if dz > slab:
            continue
        al = alpha_base * max(0.25, 1 - 0.55 * dz / slab)
        is_b = int(sp[i]) == 1
        r = SIG[int(sp[i])] / 2
        e_col = EC_B if is_b else ec
        e_lw = (lw + 1.4) if is_b else lw
        z = zbase + (1 - dz / slab)
        ax.add_patch(Circle((float(xy[i, 0]), float(xy[i, 1])), r,
                            facecolor=color if filled else "none", edgecolor=e_col if filled else color,
                            linewidth=e_lw, alpha=al, zorder=z))
        if numbers is not None and numbers[i] is not None:
            ax.annotate(numbers[i], (float(xy[i, 0]), float(xy[i, 1])), ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="white", zorder=z + 0.5)


def panel(ax, snap, mode):
    proj, depth = slab_axes(snap)
    bnd2, bndd = proj(snap["bnd"]), depth(snap["bnd"])
    ret2, retd = proj(snap["ret"]), depth(snap["ret"])
    dat2, datd = proj(snap["data"]), depth(snap["data"])
    ar2, ard = proj(snap["ar"]), depth(snap["ar"])
    fl2, fld = proj(snap["flow"]), depth(snap["flow"])
    an2 = proj(snap["anch"])
    draw_set(ax, bnd2, bndd, snap["sb"], C_BND, lw=0.4, zbase=1, alpha_base=0.55)
    draw_set(ax, ret2, retd, snap["sret"], C_RET, lw=0.5, zbase=2, alpha_base=0.6)
    nums = [str(t + 1) for t in range(K)]                    # canonical (Morton) generation index 1..8
    if mode == "context":
        draw_set(ax, dat2, datd, snap["sdata"], C_DATA, zbase=4, numbers=nums)
        ax.add_patch(Circle((float(proj(torch.zeros(1, 3))[0, 0]), float(proj(torch.zeros(1, 3))[0, 1])),
                            R, facecolor="none", edgecolor="#666", linestyle="--", linewidth=1.0, zorder=6))
    elif mode == "order":
        draw_set(ax, dat2, datd, snap["sdata"], C_DATA, zbase=3, alpha_base=0.22)
        ax.scatter(an2[:, 0], an2[:, 1], marker="x", s=42, c="#555", zorder=6, linewidths=1.2)
        for t in range(K):
            ax.add_patch(FancyArrowPatch((float(an2[t, 0]), float(an2[t, 1])),
                                         (float(ar2[t, 0]), float(ar2[t, 1])),
                                         arrowstyle="-", linestyle=":", color="#999", lw=0.8, zorder=5))
            if t > 0:
                ax.add_patch(FancyArrowPatch((float(ar2[t - 1, 0]), float(ar2[t - 1, 1])),
                                             (float(ar2[t, 0]), float(ar2[t, 1])),
                                             arrowstyle="->", mutation_scale=9, color=C_AR,
                                             lw=1.0, alpha=0.55, zorder=7))
        draw_set(ax, ar2, ard, snap["sar"], C_AR, zbase=8, numbers=nums)   # numbers on top of arrows
    else:  # overlay
        draw_set(ax, dat2, datd, snap["sdata"], C_DATA, zbase=3, alpha_base=0.45)
        draw_set(ax, ar2, ard, snap["sar"], C_AR, zbase=4, alpha_base=0.5)
        draw_set(ax, fl2, fld, snap["sar"], C_FLOW, zbase=5)
        for t in range(K):
            ax.add_patch(FancyArrowPatch((float(ar2[t, 0]), float(ar2[t, 1])),
                                         (float(fl2[t, 0]), float(fl2[t, 1])),
                                         arrowstyle="->", mutation_scale=10, color="#444",
                                         lw=1.1, zorder=7))
    ax.set_xlim(-4.7, 4.7); ax.set_ylim(-4.7, 4.7); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#DDD")


snaps = [snapshot(0), snapshot(5)]
fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.4))
for r, snap in enumerate(snaps):
    panel(axes[r, 0], snap, "context")
    panel(axes[r, 1], snap, "order")
    panel(axes[r, 2], snap, "overlay")
    axes[r, 0].set_ylabel(f"held chain ci={snap['ci']}", fontsize=11)
    axes[r, 2].set_title(f"overlay -- dE/particle: AR {snap['dE_ar']:+.1f}, flow {snap['dE_fl']:+.1f}",
                         fontsize=10)
axes[0, 0].set_title("cavity & context (data block numbered\nin canonical Morton generation order 1-8)", fontsize=10)
axes[0, 1].set_title("transformer (AR) generation order\n(x scaffold anchors - numbers match the truth's index)", fontsize=10)
h = [plt.Line2D([], [], marker="o", ls="", mfc=c, mec="white", ms=10, label=l)
     for c, l in ((C_BND, "boundary (frozen context)"), (C_RET, "retained interior"),
                  (C_DATA, "data block (truth)"), (C_AR, "AR proposal"), (C_FLOW, "EGNN-flow corrected"))]
h += [plt.Line2D([], [], marker="o", ls="", mfc="#BBB", mec="white", ms=11, label="species A (large)"),
      plt.Line2D([], [], marker="o", ls="", mfc="#BBB", mec=EC_B, mew=2.2, ms=8, label="species B (small, dark ring)")]
fig.legend(handles=h, loc="lower center", ncol=7, frameon=False, fontsize=9)
fig.suptitle("KA3D cavity block pipeline: context -> AR placement order -> flow correction "
             f"(K={K}, R={R}, pos_temp={PT}; slab |depth|<1.2, radius = sigma/2; A large, B small)",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0.045, 1, 0.96])
out = "reports/logs-2026-07-13/viz_cavity_pipeline.png"
fig.savefig(out, dpi=160)
print(f"saved -> {out}", flush=True)
