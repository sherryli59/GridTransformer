"""STEP-0 for the reservation-tilt proposal: verify its PREMISE. The rank-rising clash creation
(0.11->1.10) has two candidate causes the rank curve cannot separate:
  (b) OBSTRUCTION: early block members consume space that future anchors need -> a future-demand
      reservation tilt on EARLY placements is the right fix;
  (a) COMPOUNDING: early members are fine; rank-r conditions on r-1 model-placed (slightly off) members
      and errors snowball -> the fix is exposure-bias-side (scheduled sampling / relaxed targets), and a
      reservation tilt attacks the wrong term.
Direct test of (b): every block slot generates its residual around a KNOWN anchor, so future demand is
'one particle within a bounded residual of each remaining anchor a_k'. Compare, per placement rank r,
the MODEL's placed particle vs the DATA particle at the same slot:
  d_fut(r)  = min_k>r |x_r - a_k|          (distance to nearest future anchor)
  n_09(r)   = #{k>r : |x_r - a_k| < 0.9}   (future anchors it sits on)
  |u|(r)    = |x_r - a_r|                  (residual size: does the model wander off its anchor?)
If model d_fut approximately equals data d_fut -> NO obstruction -> premise (b) fails -> compounding (a) drives the
rank rise. If model d_fut < data d_fut / n_09 higher on EARLY ranks -> obstruction confirmed, tilt well-aimed.
K=8 blob, R=2.5, production settings, 12 cav x 16 samples."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; RCTX = 2.5; R = 2.5; M = 16; NCAV = 12; K = 8
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])

md, dd = [[] for _ in range(K - 1)], [[] for _ in range(K - 1)]        # d_fut model / data, per rank
mn, dn = [[] for _ in range(K - 1)], [[] for _ in range(K - 1)]        # n_09
mu, du = [[] for _ in range(K)], [[] for _ in range(K)]                # |u| residual, per rank
gen = torch.Generator(device=dev).manual_seed(0)
ncav = 0
with torch.no_grad():
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
        seed = int(torch.randint(n, (), generator=gen, device=dev))
        blk = torch.zeros(n, dtype=torch.bool, device=dev)
        blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
        Xg, Sg, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                     blk, bnd, sb, R, gen=gen)
        bi = blk.nonzero().squeeze(1)                                   # ascending = generation order
        anc = a[bi]                                                     # block anchors [K,3]
        for r in range(K):
            du[r].append(float((xo[bi[r]] - anc[r]).norm()))
            for k in range(M):
                mu[r].append(float((Xg[k][bi[r]] - anc[r]).norm()))
            if r < K - 1:
                fut = anc[r + 1:]                                       # remaining future anchors
                dfd = (xo[bi[r]] - fut).norm(dim=-1)
                dd[r].append(float(dfd.min())); dn[r].append(float((dfd < 0.9).sum()))
                for k in range(M):
                    dfm = (Xg[k][bi[r]] - fut).norm(dim=-1)
                    md[r].append(float(dfm.min())); mn[r].append(float((dfm < 0.9).sum()))
        ncav += 1
        if ncav >= NCAV:
            break

print(f"=== future-anchor OBSTRUCTION test (K={K}, R={R}): model vs data, per placement rank ===", flush=True)
print(f"{'rank':>4} | {'d_fut model':>11} {'d_fut DATA':>10} | {'n<0.9 model':>11} {'n<0.9 DATA':>10} | "
      f"{'|u| model':>9} {'|u| DATA':>8}", flush=True)
for r in range(K):
    if r < K - 1:
        print(f"{r+1:>4} | {st.mean(md[r]):>11.3f} {st.mean(dd[r]):>10.3f} | {st.mean(mn[r]):>11.2f} "
              f"{st.mean(dn[r]):>10.2f} | {st.mean(mu[r]):>9.3f} {st.mean(du[r]):>8.3f}", flush=True)
    else:
        print(f"{r+1:>4} | {'-':>11} {'-':>10} | {'-':>11} {'-':>10} | "
              f"{st.mean(mu[r]):>9.3f} {st.mean(du[r]):>8.3f}", flush=True)
print("\nREAD: model d_fut << data d_fut (esp. early ranks) => obstruction real, reservation tilt well-aimed."
      "\n      model ~= data => early placements do NOT over-consume capacity; rank rise = compounding"
      "\n      (exposure bias) => the lever is scheduled-sampling/relaxed targets, not future-demand tilts.", flush=True)
