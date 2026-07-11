"""GATING check: is the large-block clash DRIFT (exposure bias) or structural HALF-CAGE (AR blind to future
block particles)? Generate a K-block three ways and compare per-particle clash (min dist < 0.8 to any other):
  FR    : each particle sampled given its own SAMPLED prefix        (half-cage + drift)
  clean : each particle sampled given the DATA prefix (no drift)     (half-cage, no drift)
  full  : each particle sampled given ALL OTHERS at data (full cage) (no half-cage, no drift)
Verdict: FR ~ clean  => NOT drift.  clean >> full => HALF-CAGE is the wall (fix = full-cage corrector/refine).
Uses the batched EBM context (_fc_batched) with a controllable visibility mask + positions."""
import torch, statistics as st
import torch.nn.functional as F
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold, ball_squash, ball_unsquash
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; R = 2.0; RCTX = 2.5; Kblk = 12; M = 16; CUT = 0.8
ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax.pt", map_location=dev, weights_only=False)
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"]); gen = torch.Generator(device=dev).manual_seed(0)


@torch.no_grad()
def gen_variant(xo, so, blk, bnd, s_bnd, variant):
    fl = m.flow; n, mm = xo.shape[0], bnd.shape[0]
    anchors = fixed_ball_scaffold(n, R, dev, xo.dtype)
    order = torch.argsort(blk.to(torch.uint8), stable=True)
    xo, so, anchors = xo[order], so[order], anchors[order]; n_ret = int((~blk).sum())
    anchor_y, _ = ball_unsquash(anchors, R)
    kind = torch.cat([torch.ones(mm, dtype=torch.long, device=dev), torch.zeros(n, dtype=torch.long, device=dev)])
    slot_feat = m._slot_features(anchors, torch.arange(n, device=dev), n, R)
    # M chains; combined starts all-DATA (boundary + reordered interior data)
    comb0 = torch.cat([bnd, xo], 0)[None].expand(M, mm + n, 3).clone()
    scomb = torch.cat([s_bnd, so], 0)[None].expand(M, mm + n).clone()
    idxp = torch.arange(mm + n, device=dev)
    sampled = xo[None].expand(M, n, 3).clone()
    combined = comb0.clone()                                            # positions the model SEES
    rbias = m._r_bias(R, dev, xo.dtype)
    for jj in range(n_ret, n):
        if variant == "full":
            vis = (idxp != (mm + jj))[None]                            # everything except self (full cage, data)
            cc = comb0                                                 # all-data context
        else:                                                          # FR / clean: causal prefix only
            vis = (idxp < (mm + jj))[None]
            cc = combined                                              # FR: prefix=sampled; clean: prefix=data
        h = m._fc_batched(cc, scomb, kind, vis, anchors[jj:jj + 1], slot_feat[jj:jj + 1])[:, 0] + rbias
        ls = F.log_softmax(m.head_species(h), -1); sj = so[jj].expand(M)   # fix species = data (isolate positions)
        he = h + m.sp_out_emb(sj)
        cage_x, cage_s, cage_v = m._cage_knn_b(cc, scomb, vis, anchors[jj:jj + 1])

        def Vax(uf, ax, net, emb):
            return m._V_axis_b(uf[:, None], ax, anchor_y[jj:jj + 1], cage_x, cage_s, cage_v, sj[:, None], R, net, emb)[:, 0]
        z = torch.zeros(M, 3, device=dev)
        la = F.log_softmax(fl.head_a(he) - Vax(z, 0, m.phi_a, m.pair_emb_a), -1); ba = torch.multinomial(la.exp(), 1, generator=gen).squeeze(1)
        ub = z.clone(); ub[:, 0] = fl._ctr(ba)
        lb = F.log_softmax(fl.head_b(he + fl.bin_a_emb(ba)) - Vax(ub, 1, m.phi_b, m.pair_emb_b), -1); bb = torch.multinomial(lb.exp(), 1, generator=gen).squeeze(1)
        uc = ub.clone(); uc[:, 1] = fl._ctr(bb)
        lc = F.log_softmax(fl.head_c(he + fl.bin_a_emb(ba) + fl.bin_b_emb(bb)) - Vax(uc, 2, m.phi, m.pair_emb), -1); bc = torch.multinomial(lc.exp(), 1, generator=gen).squeeze(1)
        dith = (torch.rand(M, 3, device=dev, generator=gen) - 0.5) * fl.bw
        u = torch.stack([fl._ctr(ba), fl._ctr(bb), fl._ctr(bc)], -1) + dith
        pos, _ = ball_squash(anchor_y[jj][None] + u, R)
        sampled[:, jj] = pos
        if variant == "FR":
            combined[:, mm + jj] = pos                                  # next slot sees the SAMPLED particle
        # clean/full: leave combined at data (already)
    # clash: each sampled BLOCK particle vs all OTHER particles (block+retained sampled, boundary)
    blk_re = blk[order]
    allx = torch.cat([sampled, bnd[None].expand(M, mm, 3)], 1)          # [M, n+m, 3]
    bi_idx = torch.nonzero(blk_re).squeeze(1)
    dcl = []
    for M_i in range(M):
        A = allx[M_i]; bi = sampled[M_i][blk_re]
        dmat = torch.cdist(bi, A)
        for r, gi in enumerate(bi_idx):
            dmat[r, gi] = 9.0
        dcl.append(float((dmat.min(1).values < CUT).float().mean()))
    return st.mean(dcl)


res = {"FR": [], "clean": [], "full": []}
nc = 0
for ci in range(900, 940):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < Kblk + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev); blk[(a - a[seed]).norm(dim=-1).topk(Kblk, largest=False).indices] = True
    for v in ("FR", "clean", "full"):
        res[v].append(gen_variant(xo, so, blk, bnd, sb, v))
    nc += 1
    if nc >= 12:
        break
print(f"K={Kblk} block, {nc} cavities, per-block-particle clash rate (<{CUT}):")
for v in ("FR", "clean", "full"):
    print(f"  {v:6s}: {100*st.mean(res[v]):.1f}%", flush=True)
print(f"\n  DRIFT effect (FR - clean): {100*(st.mean(res['FR'])-st.mean(res['clean'])):+.1f}%   "
      f"HALF-CAGE effect (clean - full): {100*(st.mean(res['clean'])-st.mean(res['full'])):+.1f}%", flush=True)
