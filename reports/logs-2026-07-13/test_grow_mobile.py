"""TEST 1: does enlarging the mobile set (absorb nearest retained neighbours) cut clashes?

Hypothesis (from the clash decomposition: AR block particles clash MOST with FROZEN retained interior,
which a K=8 move can't push aside). For each cavity, grow the mobile block from the base K=8 by adding
the G nearest RETAINED particles to the block centroid (G in 0,2,4,6,8), AR-resample the enlarged block,
and measure the sigma-scaled clash rate (<0.9) of (a) the FULL mobile set and (b) the ORIGINAL 8 only.
If the original-8 clash rate falls with G, freeing their frozen neighbours is the lever; if it's flat
(they just re-clash as block-block), it is not. Retrain-free (AR base only)."""
import torch, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA

dev = "cuda"; RCTX = 2.5; K = 8; R = 2.0; PT = 0.4; M = 8; NCAV = 12; CLASH = 0.9
GROW = [0, 2, 4, 6, 8]
sig = torch.tensor(SIGMA, device=dev)

ar = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
ar.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                              map_location=dev, weights_only=False)["state_dict"], strict=False)
ar.eval(); ar.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
gen = torch.Generator(device=dev).manual_seed(7)


def clash_rate(mob_x, mob_s, nbr_x, nbr_s, subset=None):
    """% of (subset of) mobile particles whose min sigma-gap to any neighbour (excl self) < CLASH.
    mob_x[M,Km,3] centers; nbr_x[M,P,3] = mobile + frozen-cage. Index self-exclusion (mobile is nbr[:Km])."""
    Mn, Km = mob_x.shape[0], mob_x.shape[1]
    hits = 0; tot = 0
    for mm in range(Mn):
        d = torch.cdist(mob_x[mm], nbr_x[mm])
        d[torch.arange(Km), torch.arange(Km)] = float("inf")
        mn = (d / sig[mob_s[mm][:, None], nbr_s[mm][None, :]]).min(1).values
        sel = mn if subset is None else mn[subset]
        hits += int((sel < CLASH).sum()); tot += sel.numel()
    return hits, tot


full = {g: [0, 0] for g in GROW}      # [hits, tot] over full mobile set
orig = {g: [0, 0] for g in GROW}      # [hits, tot] over the original 8 only

ncav = 0
for ci in range(16):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < K + 10:            # need slack for G up to 8 retained
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; anch = fixed_ball_scaffold(n, R, dev)
    seed = int(torch.randint(n, (), generator=gen, device=dev))
    base = torch.zeros(n, dtype=torch.bool, device=dev)
    base_idx = (anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices
    base[base_idx] = True
    centroid = xo[base].mean(0)
    # rank RETAINED (non-base) interior particles by distance to block centroid
    ret_ids = torch.nonzero(~base, as_tuple=True)[0]
    ret_order = ret_ids[(xo[ret_ids] - centroid).norm(dim=-1).argsort()]
    for g in GROW:
        mask = base.clone()
        if g > 0:
            mask[ret_order[:g]] = True
        Km = int(mask.sum())
        xo_b = xo[None].expand(M, n, 3).contiguous(); so_b = so[None].expand(M, n).contiguous()
        xo_g, so_g, _ = ar.sample_block_b(xo_b, so_b, mask, bnd, sb, R, gen=gen, pos_temp=PT)
        mob_x, mob_s = xo_g[:, mask], so_g[:, mask]
        cage_x = torch.cat([bnd[None].expand(M, -1, -1), xo_g[:, ~mask]], 1)
        cage_s = torch.cat([sb[None].expand(M, -1), so_g[:, ~mask]], 1)
        nbr_x = torch.cat([mob_x, cage_x], 1); nbr_s = torch.cat([mob_s, cage_s], 1)
        h, t = clash_rate(mob_x, mob_s, nbr_x, nbr_s)
        full[g][0] += h; full[g][1] += t
        # subset = which mobile columns are the original 8 (mask is boolean over n; mobile order = increasing idx)
        mob_ids = torch.nonzero(mask, as_tuple=True)[0]
        is_orig = torch.tensor([bool(base[j]) for j in mob_ids], device=dev)
        h2, t2 = clash_rate(mob_x, mob_s, nbr_x, nbr_s, subset=is_orig)
        orig[g][0] += h2; orig[g][1] += t2
    ncav += 1
    if ncav >= NCAV:
        break

fig, ax = plt.subplots(figsize=(7.2, 5))
gx = GROW
yf = [100 * full[g][0] / full[g][1] for g in gx]
yo = [100 * orig[g][0] / orig[g][1] for g in gx]
ax.plot(gx, yf, "-o", color="#D95F02", lw=2, label="full mobile set")
ax.plot(gx, yo, "-s", color="#7570B3", lw=2, label="original 8 particles only")
for x, y in zip(gx, yo):
    ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color="#7570B3")
for x, y in zip(gx, yf):
    ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, color="#D95F02")
ax.axhline(0, color="#BBB", lw=0.8, ls=":")
ax.set_xlabel("G = nearest retained particles absorbed into the mobile set")
ax.set_ylabel("clash rate (<0.9$\\sigma$), %")
ax.set_title(f"TEST 1: enlarging the mobile set vs clash rate\n({ncav} cavities x M={M}, AR base, K={K}+G)", fontsize=11)
ax.legend(frameon=False, fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = "reports/logs-2026-07-13/test_grow_mobile.png"
fig.savefig(out, dpi=160)
torch.save({"full": full, "orig": orig, "GROW": GROW, "ncav": ncav}, "reports/logs-2026-07-13/test_grow_mobile.pt")
print(f"saved -> {out}", flush=True)
for g in GROW:
    print(f"  G={g}: full mobile clash {100*full[g][0]/full[g][1]:5.1f}%   original-8 clash {100*orig[g][0]/orig[g][1]:5.1f}%", flush=True)
