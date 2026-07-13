"""Validate the hard minimum-separation c-axis mask: (1) EXACTNESS -- sample_block_b(min_sep) scored by
block_log_prob_b(min_sep) must round-trip (as the tempered path does, ~1.5e-5); (2) EFFECT -- clash rate
of sampled blocks vs no mask; (3) UNBIASED -- data blocks must still score finite (never in a forbidden
c-bin) at a safe cut (data min-gap >= 0.95 > cut). CPU (GPU busy with the rep_prior retrain)."""
import torch
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA

dev = "cpu"; RCTX = 2.5; K = 8; R = 2.0; M = 16; PT = 0.4; CUT = 0.85
sig = torch.tensor(SIGMA)

ar = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
ar.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                              map_location=dev, weights_only=False)["state_dict"], strict=False)
ar.eval(); ar.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].float(), D["s"].long(), float(D["L"])
gen = torch.Generator(device=dev).manual_seed(3)


def clash_rate(mob_x, mob_s, cage_x, cage_s):
    hits = tot = 0
    nbr_x = torch.cat([mob_x, cage_x], 1); nbr_s = torch.cat([mob_s, cage_s], 1)
    Km = mob_x.shape[1]
    for mm in range(mob_x.shape[0]):
        d = torch.cdist(mob_x[mm], nbr_x[mm]); d[torch.arange(Km), torch.arange(Km)] = float("inf")
        mn = (d / sig[mob_s[mm][:, None], nbr_s[mm][None, :]]).min(1).values
        hits += int((mn < 0.9).sum()); tot += Km
    return hits, tot


c = torch.rand(3, generator=gen, device=dev) * L
p = carve(X[0], S[0], c, R, L)
xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
n = xo.shape[0]; anch = fixed_ball_scaffold(n, R, dev)
seed = int(torch.randint(n, (), generator=gen, device=dev))
blk = torch.zeros(n, dtype=torch.bool); blk[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
xo_b = xo[None].expand(M, n, 3).contiguous(); so_b = so[None].expand(M, n).contiguous()

# (1) EXACTNESS round-trip WITH the mask
g2 = torch.Generator(device=dev).manual_seed(9)
xs, ss, lq_sample = ar.sample_block_b(xo_b, so_b, blk, bnd, sb, R, gen=g2, pos_temp=PT, min_sep=CUT)
lq_score = ar.block_log_prob_b(xs, ss, blk, bnd, sb, R, pos_temp=PT, min_sep=CUT)
diff = (lq_score - lq_sample).abs()
print(f"(1) EXACTNESS  median|score-sample| {diff.median():.2e}  match(<1e-3) {100*float((diff<1e-3).float().mean()):.0f}%  max {diff.max():.2e}")
# control: scoring the masked sample WITHOUT the mask must differ (proves the mask actually acts)
lq_nomask = ar.block_log_prob_b(xs, ss, blk, bnd, sb, R, pos_temp=PT, min_sep=None)
print(f"    mask-vs-nomask score gap (should be > 0): median {float((lq_nomask-lq_sample).abs().median()):.3f}")

# (2) CLASH EFFECT: no mask vs mask
g3 = torch.Generator(device=dev).manual_seed(4)
xn, sn, _ = ar.sample_block_b(xo_b, so_b, blk, bnd, sb, R, gen=g3, pos_temp=PT, min_sep=None)
g4 = torch.Generator(device=dev).manual_seed(4)
xm, sm, _ = ar.sample_block_b(xo_b, so_b, blk, bnd, sb, R, gen=g4, pos_temp=PT, min_sep=CUT)
cage_x = torch.cat([bnd[None].expand(M, -1, -1), xn[:, ~blk]], 1); cage_s = torch.cat([sb[None].expand(M, -1), sn[:, ~blk]], 1)
h0, t0 = clash_rate(xn[:, blk], sn[:, blk], cage_x, cage_s)
cage_xm = torch.cat([bnd[None].expand(M, -1, -1), xm[:, ~blk]], 1); cage_sm = torch.cat([sb[None].expand(M, -1), sm[:, ~blk]], 1)
h1, t1 = clash_rate(xm[:, blk], sm[:, blk], cage_xm, cage_sm)
print(f"(2) CLASH<0.9  no-mask {100*h0/t0:.1f}%   min_sep={CUT} {100*h1/t1:.1f}%")

# (3) UNBIASED: the DATA block must score finite under the mask (never lands in a forbidden bin)
data_b = xo[None].expand(M, n, 3).contiguous(); datas_b = so[None].expand(M, n).contiguous()
lq_data = ar.block_log_prob_b(data_b, datas_b, blk, bnd, sb, R, pos_temp=PT, min_sep=CUT)
print(f"(3) UNBIASED   data-block logq finite: {bool(torch.isfinite(lq_data).all())}  (min {float(lq_data.min()):.1f})")
