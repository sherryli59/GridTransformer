"""Decompose cavity block clashes by PARTNER category -> tests AR future-blindness directly.

A placed block particle (Morton generation rank t) can clash with:
  earlier-block : block particle with gen rank < t   (AR SAW it when placing -> inexcusable)
  later-block   : block particle with gen rank > t   (AR was BLIND to it -> future-blindness)
  retained      : retained interior (frozen, AR saw it)
  boundary      : boundary shell    (frozen, AR saw it)
Per center, min sigma-scaled gap to EACH category; a clash (<0.9) is attributed to the category holding
the binding (overall-min) partner. Reported for AR and EGNN-flow (data = control). Prediction: AR clashes
dominated by LATER-block; flow (joint, sees whole block) should fix those first.
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

dev = "cuda"; RCTX = 2.5; K = 8; R = 2.0; PT = 0.4; M = 8; N_CAGE = 48; DUMMY_R = 50.0; NCAV = 12; CLASH = 0.9
CATS = ["earlier-block", "later-block", "retained", "boundary"]
COL = {"earlier-block": "#D95F02", "later-block": "#7570B3", "retained": "#66A61E", "boundary": "#E7298A"}
sig = torch.tensor(SIGMA, device=dev)

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
gen = torch.Generator(device=dev).manual_seed(7)

# attrib[arm][cat] = # placed particles whose BINDING (min) partner is in cat AND is a clash (<0.9)
attrib = {arm: {cat: 0 for cat in CATS} for arm in ("ar", "flow")}
# percat[arm][cat] = # placed particles that clash with SOMETHING in cat (categories not exclusive)
percat = {arm: {cat: 0 for cat in CATS} for arm in ("ar", "flow")}
nplaced = {"ar": 0, "flow": 0}


def fixed_size_cage(cx, cs, cen, ncg):
    Mn, m = cx.shape[0], cx.shape[1]
    if m >= ncg:
        idx = torch.topk((cx[0] - cen).norm(dim=-1), ncg, largest=False).indices
        return cx[:, idx], cs[:, idx]
    pad = ncg - m
    dp = torch.tensor([DUMMY_R, 0., 0.], device=dev, dtype=cx.dtype)[None, None].expand(Mn, pad, 3)
    return torch.cat([cx, dp], 1), torch.cat([cs, torch.zeros(Mn, pad, dtype=cs.dtype, device=dev)], 1)


def cat_min_gaps(blk_x, blk_s, gen_rank, ret_x, ret_s, bnd_x, bnd_s):
    """Return [M,K,4] min sigma-gap of each block particle to each category (earlier/later/retained/boundary)."""
    Mn, Kk = blk_x.shape[0], blk_x.shape[1]
    out = torch.full((Mn, Kk, 4), float("inf"), device=dev)
    for mm in range(Mn):
        # block-block
        dbb = torch.cdist(blk_x[mm], blk_x[mm]) / sig[blk_s[mm][:, None], blk_s[mm][None, :]]
        earlier = gen_rank[None, :] < gen_rank[:, None]        # [K,K] col is earlier than row
        later = gen_rank[None, :] > gen_rank[:, None]
        out[mm, :, 0] = dbb.masked_fill(~earlier, float("inf")).min(1).values
        out[mm, :, 1] = dbb.masked_fill(~later, float("inf")).min(1).values
        # block-retained
        if ret_x.shape[1] > 0:
            dr = torch.cdist(blk_x[mm], ret_x[mm]) / sig[blk_s[mm][:, None], ret_s[mm][None, :]]
            out[mm, :, 2] = dr.min(1).values
        # block-boundary
        db = torch.cdist(blk_x[mm], bnd_x) / sig[blk_s[mm][:, None], bnd_s[None, :]]
        out[mm, :, 3] = db.min(1).values
    return out


def tally(arm, cg):
    Mn, Kk, _ = cg.shape
    nplaced[arm] += Mn * Kk
    binding = cg.argmin(-1)                                    # [M,K] category of overall-min partner
    gmin = cg.min(-1).values                                  # [M,K]
    isclash = gmin < CLASH
    for ci_ in range(4):
        attrib[arm][CATS[ci_]] += int(((binding == ci_) & isclash).sum())
        percat[arm][CATS[ci_]] += int((cg[..., ci_] < CLASH).sum())


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
    # generation order: sampler places block as the Morton-ordered suffix after retained
    order = torch.argsort(blk.to(torch.uint8), stable=True)
    gen_seq = order[n - K:]                                    # original slot ids, in generation order
    blk_ids = torch.nonzero(blk, as_tuple=True)[0]             # order used by xo_ar[:,blk] (increasing idx)
    gen_rank = torch.tensor([int((gen_seq == j).nonzero()) for j in blk_ids], device=dev)  # rank per blk col
    xo_b = xo[None].expand(M, n, 3).contiguous(); so_b = so[None].expand(M, n).contiguous()
    xo_ar, so_ar, _ = ar.sample_block_b(xo_b, so_b, blk, bnd, sb, R, gen=gen, pos_temp=PT)
    ret_x, ret_s = xo_ar[:, ~blk], so_ar[:, ~blk]
    cage_x = torch.cat([bnd[None].expand(M, -1, -1), ret_x], 1)
    cage_s = torch.cat([sb[None].expand(M, -1), ret_s], 1)
    cfx, cfs = fixed_size_cage(cage_x, cage_s, xo[blk].mean(0), N_CAGE)
    y_blk, _ = flow.flow(xo_ar[:, blk], cfx, so_ar[:, blk], cfs, reverse=False)
    tally("ar", cat_min_gaps(xo_ar[:, blk], so_ar[:, blk], gen_rank, ret_x, ret_s, bnd, sb))
    tally("flow", cat_min_gaps(y_blk, so_ar[:, blk], gen_rank, ret_x, ret_s, bnd, sb))
    ncav += 1
    if ncav >= NCAV:
        break

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
# panel 1: clash rate by BINDING partner category (stacked), AR vs flow
ax = axes[0]
arms = ["ar", "flow"]
bottom = {arm: 0.0 for arm in arms}
xs = [0, 1]
for cat in CATS:
    vals = [100 * attrib[arm][cat] / nplaced[arm] for arm in arms]
    b = ax.bar(xs, vals, 0.6, bottom=[bottom[arm] for arm in arms], color=COL[cat], label=cat)
    for k2, arm in enumerate(arms):
        if vals[k2] > 0.8:
            ax.annotate(f"{vals[k2]:.0f}", (xs[k2], bottom[arm] + vals[k2] / 2), ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        bottom[arm] += vals[k2]
for k2, arm in enumerate(arms):
    ax.annotate(f"{bottom[arm]:.0f}% total", (xs[k2], bottom[arm] + 0.4), ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(["AR proposal", "EGNN-flow"])
ax.set_ylabel("% of placed particles that clash (<0.9$\\sigma$)")
ax.set_title("clash rate by BINDING (nearest) partner", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
# panel 2: per-category clash rate (a particle can clash with several) -> where the overlaps live
ax = axes[1]
xc = torch.arange(len(CATS)).numpy(); w = 0.38
for i, arm in enumerate(arms):
    vals = [100 * percat[arm][cat] / nplaced[arm] for cat in CATS]
    b = ax.bar(xc + (i - 0.5) * w, vals, w, color="#D95F02" if arm == "ar" else "#7570B3",
               label={"ar": "AR", "flow": "flow"}[arm])
    ax.bar_label(b, fmt="%.0f", fontsize=8, padding=1)
ax.set_xticks(xc); ax.set_xticklabels(CATS, rotation=15, fontsize=9)
ax.set_ylabel("% placed particles clashing with this category")
ax.set_title("per-category clash rate (categories overlap)", fontsize=11)
ax.legend(frameon=False, fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.suptitle(f"Clash decomposition by partner: AR future-blindness test  ({ncav} cavities x M={M}, K={K}, "
             f"R={R}, pos_temp={PT}; flow=gated_big step {ck['step']})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = "reports/logs-2026-07-13/viz_clash_decomp.png"
fig.savefig(out, dpi=160)
torch.save({"attrib": attrib, "percat": percat, "nplaced": nplaced}, "reports/logs-2026-07-13/viz_clash_decomp.pt")
print(f"saved -> {out}", flush=True)
for arm in arms:
    print(f"  {arm:4s} binding: " + "  ".join(f"{c} {100*attrib[arm][c]/nplaced[arm]:.1f}%" for c in CATS), flush=True)
    print(f"  {arm:4s} percat : " + "  ".join(f"{c} {100*percat[arm][c]/nplaced[arm]:.1f}%" for c in CATS), flush=True)
