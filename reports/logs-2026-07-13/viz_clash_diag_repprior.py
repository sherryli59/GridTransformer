"""Clash diagnostic: prove the 'clean g(r)' hides the core overlaps that dominate energy.

Same setup as viz_gr_species (K=8, R=2.0, block particles = centers, block+FULL cage = neighbors).
Three views the aggregate g(r) cannot show:
  (1) g(r) extended DOWN to r=0.25 (core region) -- clash weight below the 0.6 floor of the earlier plot.
  (2) per-block-particle MIN sigma-scaled gap  min_j r_ij / sigma_(s_i,s_j)  distribution: a clash is a
      particle whose nearest neighbour sits inside its hard core (ratio < ~0.9). This is the quantity
      r^-12 energy explodes on; ONE such particle per block dominates the block energy.
  (3) clash FRACTION (particles with min-gap < cut) per arm at cuts 0.80/0.85/0.90, + worst-cavity.
Species-scaled throughout (sigma_AA=1.0, sigma_AB=0.8, sigma_BB=0.88). AR vs EGNN-flow vs DATA.
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA

dev = "cuda"; RCTX = 2.5; K = 8; R = 2.0; PT = 0.4; M = 8; N_CAGE = 48; DUMMY_R = 50.0; NCAV = 12
C_DATA, C_AR, C_FLOW = "#1B9E77", "#D95F02", "#7570B3"
sig = torch.tensor(SIGMA, device=dev)

ar = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
ar.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                              map_location=dev, weights_only=False)["state_dict"], strict=False)
ar.eval(); ar.use_frame = False
ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_egnn_flow_repprior_best.pt",
                map_location=dev, weights_only=False)
a = ck["args"]
flow = CavityBlockFlow(k=a["k"], n_cage=a["n_cage"], r_c=a.get("r_c", 2.5), hidden_nf=a["hidden_nf"],
                       n_layers=a["n_layers"], n_species=2, max_neighbors=a["max_neighbors"], rep_prior=a.get("rep_prior", False),
                       ode_rtol=1e-5, ode_atol=1e-5, max_steps=1000).to(dev)
flow.load_state_dict(ck["state_dict"]); flow.eval()
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
rho = X.shape[1] / L ** 3
xB = float((S == 1).float().mean()); rho_sp = {0: rho * (1 - xB), 1: rho * xB}
gen = torch.Generator(device=dev).manual_seed(7)   # same seed/cavities as viz_gr_species

edges = torch.linspace(0.25, R + RCTX, 70); ctr = 0.5 * (edges[1:] + edges[:-1]); dr = float(edges[1] - edges[0])
shell = 4 * torch.pi * ctr ** 2 * dr
PAIRS = [(0, 0, "AA"), (0, 1, "AB"), (1, 1, "BB")]
hist = {t: {ab: torch.zeros(len(ctr)) for _, _, ab in PAIRS} for t in ("data", "ar", "flow")}
nctr = {t: {ab: 0 for _, _, ab in PAIRS} for t in ("data", "ar", "flow")}
mingap = {t: [] for t in ("data", "ar", "flow")}        # per (cavity,draw,particle) min sigma-scaled gap
worst = {t: [] for t in ("data", "ar", "flow")}          # per-cavity worst-particle min gap


def fixed_size_cage(cx, cs, cen, ncg):
    Mn, m = cx.shape[0], cx.shape[1]
    if m >= ncg:
        idx = torch.topk((cx[0] - cen).norm(dim=-1), ncg, largest=False).indices
        return cx[:, idx], cs[:, idx]
    pad = ncg - m
    dp = torch.tensor([DUMMY_R, 0., 0.], device=dev, dtype=cx.dtype)[None, None].expand(Mn, pad, 3)
    return torch.cat([cx, dp], 1), torch.cat([cs, torch.zeros(Mn, pad, dtype=cs.dtype, device=dev)], 1)


def collect(tag, blk_x, blk_s, nbr_x, nbr_s):
    for sa, sb, ab in PAIRS:
        for mm in range(blk_x.shape[0]):
            ca = blk_x[mm][blk_s[mm] == sa]; nb = nbr_x[mm][nbr_s[mm] == sb]
            if ca.shape[0] == 0 or nb.shape[0] == 0:
                continue
            dd = torch.cdist(ca, nb).flatten(); dd = dd[dd > 0.05]   # 0.05 >> cdist self float-error, << any real pair
            hist[tag][ab] += torch.histc(dd.cpu(), bins=len(ctr), min=float(edges[0]), max=float(edges[-1]))
            nctr[tag][ab] += ca.shape[0]
    # min sigma-scaled gap per block particle over ALL neighbours (block+cage), species-correct sigma.
    # nbr_x = [block(K), cage] so each center's SELF is at nbr index i -> exclude by INDEX (not a distance
    # threshold: cdist self-distances carry ~1e-4 float error that slips a `d<1e-6` mask, AND a threshold
    # would wrongly erase a GENUINE two-particle overlap, undercounting exactly the clashes we measure).
    Kk = blk_x.shape[1]
    for mm in range(blk_x.shape[0]):
        d = torch.cdist(blk_x[mm], nbr_x[mm])                       # [K, K+P]
        d[torch.arange(Kk), torch.arange(Kk)] = float("inf")       # block self-diagonal only
        sgm = sig[blk_s[mm][:, None], nbr_s[mm][None, :]]           # [K, K+P]
        mn = (d / sgm).min(1).values                                # [K]
        mingap[tag] += mn.cpu().tolist()
        worst[tag].append(float(mn.min()))


ncav = 0
for ci in range(16):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < K + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; anch = fixed_ball_scaffold(n, R, dev)
    seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev)
    blk[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    xo_b = xo[None].expand(M, n, 3).contiguous(); so_b = so[None].expand(M, n).contiguous()
    xo_ar, so_ar, _ = ar.sample_block_b(xo_b, so_b, blk, bnd, sb, R, gen=gen, pos_temp=PT)
    cage_x = torch.cat([bnd[None].expand(M, -1, -1), xo_ar[:, ~blk]], 1)
    cage_s = torch.cat([sb[None].expand(M, -1), so_ar[:, ~blk]], 1)
    cfx, cfs = fixed_size_cage(cage_x, cage_s, xo[blk].mean(0), N_CAGE)
    y_blk, _ = flow.flow(xo_ar[:, blk], cfx, so_ar[:, blk], cfs, reverse=False)
    ar_blk, sp_blk = xo_ar[:, blk], so_ar[:, blk]
    dblk = xo[None, blk].expand(M, K, 3); ds = so[None, blk].expand(M, K)
    collect("data", dblk, ds, torch.cat([dblk, cage_x], 1), torch.cat([ds, cage_s], 1))
    collect("ar", ar_blk, sp_blk, torch.cat([ar_blk, cage_x], 1), torch.cat([sp_blk, cage_s], 1))
    collect("flow", y_blk, sp_blk, torch.cat([y_blk, cage_x], 1), torch.cat([sp_blk, cage_s], 1))
    ncav += 1
    if ncav >= NCAV:
        break


def gr(t, ab, sb):
    return hist[t][ab] / (nctr[t][ab] * rho_sp[sb] * shell) if nctr[t][ab] else torch.zeros(len(ctr))


fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
# panel 1: g_AA/g_AB/g_BB extended into the core, AR only, to show clash weight below 0.6
ax = axes[0]
for (sa, sb, ab), ls in zip(PAIRS, ("-", "--", ":")):
    ax.plot(ctr.numpy(), gr("ar", ab, sb).numpy(), color=C_AR, ls=ls, lw=1.6, label=f"AR $g_{{{ab}}}$")
ax.axvspan(0.25, 0.6, color="#C0392B", alpha=0.08)
ax.annotate("below the earlier\nplot's r=0.6 floor", (0.28, ax.get_ylim()[1] * 0.6), fontsize=8, color="#C0392B")
ax.set_title("AR g(r) into the core\n(clash weight the r>0.6 plot hid)", fontsize=10)
ax.set_xlabel("r / $\\sigma_{AA}$"); ax.set_ylabel("g(r)"); ax.legend(frameon=False, fontsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
# panel 2: min sigma-scaled gap distribution
ax = axes[1]
ge = torch.linspace(0.4, 1.6, 49)
for t, col in (("data", C_DATA), ("ar", C_AR), ("flow", C_FLOW)):
    h = torch.histc(torch.tensor(mingap[t]).clamp(0.4, 1.6), bins=48, min=0.4, max=1.6)
    h = h / h.sum()
    ax.plot(0.5 * (ge[1:] + ge[:-1]).numpy(), h.numpy(), color=col, lw=1.9,
            label={"data": "DATA", "ar": "AR", "flow": "flow"}[t])
ax.axvspan(0.4, 0.9, color="#C0392B", alpha=0.07)
ax.axvline(0.9, color="#C0392B", ls=":", lw=1)
ax.set_title("per-particle min $r/\\sigma$ to any neighbour\n(<0.9 shaded = hard-core clash)", fontsize=10)
ax.set_xlabel("min $r_{ij}/\\sigma_{ij}$"); ax.set_ylabel("fraction of placed particles")
ax.legend(frameon=False, fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
# panel 3: clash fraction bars at cuts
ax = axes[2]
cuts = [0.80, 0.85, 0.90]
xs = torch.arange(len(cuts)).numpy(); w = 0.26
for i, (t, col) in enumerate((("data", C_DATA), ("ar", C_AR), ("flow", C_FLOW))):
    mg = torch.tensor(mingap[t])
    fr = [100 * float((mg < cut).float().mean()) for cut in cuts]
    b = ax.bar(xs + (i - 1) * w, fr, w, color=col, label={"data": "DATA", "ar": "AR", "flow": "flow"}[t])
    ax.bar_label(b, fmt="%.0f", fontsize=7.5, padding=1)
ax.set_xticks(xs); ax.set_xticklabels([f"< {c}" for c in cuts])
ax.set_title("clash rate: % placed particles with\na core overlap tighter than cut", fontsize=10)
ax.set_xlabel("min $r/\\sigma$ threshold"); ax.set_ylabel("% of placed particles")
ax.legend(frameon=False, fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.suptitle(f"Clash diagnostic: the clean g(r) hides core overlaps  ({ncav} cavities x M={M}, K={K}, "
             f"R={R}, pos_temp={PT}; flow=gated_big step {ck['step']})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = "reports/logs-2026-07-13/viz_clash_diag_repprior.png"
fig.savefig(out, dpi=160)
torch.save({"mingap": mingap, "worst": worst, "hist": hist, "nctr": nctr, "ctr": ctr}, "reports/logs-2026-07-13/viz_clash_diag_repprior.pt")
print(f"saved -> {out}", flush=True)
for t in ("data", "ar", "flow"):
    mg = torch.tensor(mingap[t]); wf = torch.tensor(worst[t])
    print(f"  {t:4s}: clash%<0.9 {100*float((mg<0.9).float().mean()):5.1f}  <0.85 {100*float((mg<0.85).float().mean()):5.1f}  "
          f"<0.8 {100*float((mg<0.8).float().mean()):5.1f}  | worst-cavity min r/sig {float(wf.min()):.2f}  "
          f"median-cavity worst {float(wf.median()):.2f}", flush=True)
