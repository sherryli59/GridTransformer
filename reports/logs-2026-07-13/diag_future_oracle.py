"""Would future info help the overstretch? VALID oracle only (an earlier anchor/noise proxy was RETRACTED:
placing context particles at anchors or data+noise is OOD for a model never trained on it -> nonsense
scores). Here we compare the TRUE-data position NLL of each block member under two REAL conditioning sets,
both in-distribution:
  CAUSAL  : member r sees only the prefix 1..r-1 (data)   -- current AR (full K-block mask).
  FULLCAGE: member r sees ALL other members at DATA        -- single-site block, everyone else retained.
Both are legitimate conditionals the trained model handles. GAP = causal - fullcage at early ranks = how
much the model WOULD sharpen early placements if it knew the future. LARGE early gap => embedding
future-anchor/reservation info (then RETRAINING) is a real lever (the current model can't use it zero-shot).
FT knn24, K=8 data blob, R=2.5, 16 cav."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ebm import ball_unsquash

dev = "cuda"; RCTX = 2.5; R = 2.5; K = 8; NCAV = 16
FT = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt"
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24).to(dev)
m.load_state_dict(torch.load(FT, map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False


@torch.no_grad()
def per_slot_posnll(xo, so, block_mask, bnd, s_bnd):
    xo, so = xo[None], so[None]
    order, anchors, anchor_y, rblock, kind, valid, slot_feat, mm, n = m._prep(xo, so, block_mask, bnd, s_bnd, R)
    xo, so = xo[:, order], so[:, order]
    combined = torch.cat([bnd[None], xo], 1); scomb = torch.cat([s_bnd[None], so], 1)
    h = m._fc_batched(combined, scomb, kind, valid, anchors, slot_feat) + m._r_bias(R, xo.device, xo.dtype)
    y, logdet = ball_unsquash(xo, R); u = y - anchor_y[None]
    cage_x, cage_s, cage_v = m._cage_knn_b(combined, scomb, valid, anchors)
    lp_u = m._tilted_lp_u_b(h + m.sp_out_emb(so), u, anchor_y, cage_x, cage_s, cage_v, so, R)
    return -(lp_u[0] + logdet[0]), rblock


causal = [[] for _ in range(K)]; fullc = [[] for _ in range(K)]
g = torch.Generator(device=dev).manual_seed(0); ncav = 0
for ci in range(16):
    c = torch.rand(3, generator=g, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < K + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    seed = int(torch.randint(n, (), generator=g, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev)
    blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    bidx = blk.nonzero().squeeze(1)
    nllc, rblock = per_slot_posnll(xo, so, blk, bnd, sb)
    for r, sl in enumerate(rblock.nonzero().squeeze(1).tolist()):
        causal[r].append(float(nllc[sl]))
    for r, gi in enumerate(bidx.tolist()):
        mk = torch.zeros(n, dtype=torch.bool, device=dev); mk[gi] = True
        nllf, rb2 = per_slot_posnll(xo, so, mk, bnd, sb)
        fullc[r].append(float(nllf[int(rb2.nonzero().squeeze(1)[0])]))
    ncav += 1
    if ncav >= NCAV:
        break

print(f"=== future-info ORACLE (VALID): TRUE-data position NLL, causal vs full-DATA-cage (K={K}, {ncav} cav) ===", flush=True)
print(f"{'rank':>4} | {'CAUSAL':>8} {'FULLCAGE':>9} {'gap':>7}", flush=True)
for r in range(K):
    cc, ff = st.mean(causal[r]), st.mean(fullc[r])
    print(f"{r+1:>4} | {cc:>8.3f} {ff:>9.3f} {cc-ff:>7.3f}", flush=True)
eg = st.mean([st.mean(causal[r]) - st.mean(fullc[r]) for r in range(4)])
print(f"\n  EARLY (rank 1-4) mean gap = {eg:+.3f} nats => future info sharpens early placements by this much;"
      f"\n  a model that EMBEDS future-anchor info (and is RETRAINED) could aim to recover it. Zero-shot proxies"
      f"\n  are confounded (OOD) and were retracted.", flush=True)
