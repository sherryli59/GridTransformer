"""Prove WHY pad-12-keep-8 fails: decompose the kept-8's clash in the usable move (first-8 fresh, tail-4
left at their present data positions) into (i) partners the 8 SAW during generation (each other + retained
+ boundary) vs (ii) the 4 tail slots they did NOT see (hidden as future, but PRESENT in the config). If the
excess over the 'ideal all-fresh' case (0.247) is concentrated in (ii), the failure is exactly 'placed
blind to 4 present particles'. FT knn24, R=2.5, 12 cav x 16 samp."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA

dev = "cuda"; RCTX = 2.5; R = 2.5; M = 16; NCAV = 12; CUT = 0.85
SIG = torch.tensor(SIGMA, device=dev)
FT = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt"
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24).to(dev)
m.load_state_dict(torch.load(FT, map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False


def clashes(xi, si, xj, sj):
    d = torch.cdist(xi, xj); sg = SIG[si[:, None], sj[None, :]]
    return (d < CUT * sg)


@torch.no_grad()
def run():
    seen, hidden4 = [], []
    g = torch.Generator(device=dev).manual_seed(0); ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=g, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < 18:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
        seed = int(torch.randint(n, (), generator=g, device=dev))
        near = (a - a[seed]).norm(dim=-1).topk(12, largest=False).indices
        b12 = torch.zeros(n, dtype=torch.bool, device=dev); b12[near] = True
        idx = b12.nonzero().squeeze(1); first8, last4 = idx[:8], idx[8:]
        Xp, Sp, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                     b12, bnd, sb, R, gen=g)
        # "not-block" retained (slots the 8 DID see as fixed context) = interior minus the 12-block
        ret = (~b12).nonzero().squeeze(1)
        for k in range(M):
            x8, s8 = Xp[k][first8], Sp[k][first8]                       # kept-8 (fresh)
            # (i) SEEN partners: other kept-8 + retained-data + boundary
            seen_x = torch.cat([Xp[k][first8], xo[ret], bnd]); seen_s = torch.cat([Sp[k][first8], so[ret], sb])
            cl = clashes(x8, s8, seen_x, seen_s)
            for j in range(8):
                cl[j, j] = False                                        # self
            seen.append(float(cl.sum(1).float().mean()))
            # (ii) HIDDEN-but-present: the 4 tail slots at DATA positions
            ch = clashes(x8, s8, xo[last4], so[last4])
            hidden4.append(float(ch.sum(1).float().mean()))
        ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(seen), st.mean(hidden4)


sm, hm = run()
print(f"=== kept-8 clash attribution in the USABLE pad-12 move (FT knn24, R={R}, {NCAV} cav) ===", flush=True)
print(f"  vs SEEN partners (kept-8 + retained + boundary): {sm:.3f}", flush=True)
print(f"  vs HIDDEN-but-present 4 tail-data slots        : {hm:.3f}", flush=True)
print(f"  total {sm+hm:.3f}  (matches usable-move 0.413; ideal-all-fresh was 0.247)", flush=True)
print(f"\n  => {100*hm/(sm+hm):.0f}% of the kept-8's clash is with the 4 particles it was generated BLIND to.", flush=True)
