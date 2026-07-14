"""Does 'remove only K, regenerate conditioned on all-else-fixed' (retained interior naturally in front)
escape the half-cage? Test at fixed R=2.5: seed from the DATA interior, freeze all-but-K, regenerate a
CLUSTERED K-block (blob) via sample_block_b (block_mask -> _prep puts retained FIRST, so block particles
see ALL retained interior + boundary). Decompose each BLOCK particle's clashes (r<0.85 sigma) into:
  - vs RETAINED interior  (the user's fully-visible fixed cage -- should be LOW)
  - vs OTHER BLOCK        (block-internal half-cage -- the RESIDUAL)
  - vs BOUNDARY
Compare to full-regen per-particle interior clash at the same R (0.45). If block-vs-retained << full-regen
and block-vs-block carries the residual, the user's framing IS the right design and already in use; the
ceiling is the block-internal packing of a clustered large-K block, NOT the retained-interior conditioning.
K in {1,4,8,16}: K=1 has NO block-internal term (full cage) -> the cleanest control."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA

dev = "cuda"; RCTX = 2.5; R = 2.5; M = 16; NCAV = 12; CUT = 0.85
SIG = torch.tensor(SIGMA, device=dev)
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def blob(n, K, a, gen):
    seed = int(torch.randint(n, (), generator=gen, device=dev))
    mk = torch.zeros(n, dtype=torch.bool, device=dev)
    mk[(a - a[seed]).norm(dim=-1).topk(min(K, n), largest=False).indices] = True
    return mk


@torch.no_grad()
def kblock_clash(K, gen):
    cret, cblk, cbnd = [], [], []
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
        blk = blob(n, K, a, gen)                                            # clustered block
        Xg, Sg, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                     blk, bnd, sb, R, gen=gen)              # retained = data, regen block
        bi = blk.nonzero().squeeze(1); ri = (~blk).nonzero().squeeze(1)     # block / retained indices
        for k in range(M):
            xi, si = Xg[k], Sg[k]
            xb_, sb_ = xi[bi], si[bi]                                        # block particles
            xr_, sr_ = xi[ri], si[ri]                                        # retained interior
            # block vs retained
            d = torch.cdist(xb_, xr_); sg = SIG[sb_[:, None], sr_[None, :]]
            cret.append(float((d < CUT * sg).sum(1).float().mean()))
            # block vs block
            d = torch.cdist(xb_, xb_); sg = SIG[sb_[:, None], sb_[None, :]]
            eye = torch.eye(len(bi), dtype=torch.bool, device=dev)
            cblk.append(float(((d < CUT * sg) & ~eye).sum(1).float().mean()))
            # block vs boundary
            d = torch.cdist(xb_, bnd); sg = SIG[sb_[:, None], sb[None, :]]
            cbnd.append(float((d < CUT * sg).sum(1).float().mean()))
        ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(cret), st.mean(cblk), st.mean(cbnd)


print(f"=== K-block regen from DATA (retained frozen+visible), R={R}, clash/BLOCK-particle (r<{CUT}sig) ===", flush=True)
print(f"    reference: FULL-regen interior clash/particle at R=2.5 was 0.453 (all interior grown AR)", flush=True)
print(f"{'K':>3} | {'vs RETAINED (visible)':>21} {'vs BLOCK (residual)':>19} {'vs boundary':>12} {'TOTAL':>7}", flush=True)
for K in (1, 4, 8, 16):
    gen = torch.Generator(device=dev).manual_seed(0)
    cr, cbk, cbd = kblock_clash(K, gen)
    print(f"{K:>3} | {cr:>21.3f} {cbk:>19.3f} {cbd:>12.3f} {cr+cbk+cbd:>7.3f}", flush=True)
print("\nEXPECT: vs-RETAINED low+~flat (fully-visible fixed cage); vs-BLOCK ~0 at K=1 (full cage) and "
      "GROWS with K (clustered block-internal half-cage) = the real acceptance ceiling.", flush=True)
