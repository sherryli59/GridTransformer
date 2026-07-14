"""Does the RL+demand declash TRANSFER to K>8? RL trained on K in {4,8}; K=12/16/24 are OOD. If it learned
a general local 'place better' behavior (+ K-aware demand feat K/n, rem/K), it should transfer; if it
memorized K=8 placements, it won't. Compare base (block-cond) vs RL+demand across a K-ladder: total
clash/member, capped-repulsive-E/member, block spread (under-pack guard). R=2.5, M=12, 10 cav."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA, EPS

dev = "cuda"; RCTX = 2.5; R = 2.5; M = 12; NCAV = 10; CUT = 0.85; CAP = 5.0
SIG = torch.tensor(SIGMA, device=dev); EP = torch.tensor(EPS, device=dev)
BASE = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt"
RLD = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_rl_demand_knn24.pt"
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def load(path, use_demand):
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=use_demand).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["state_dict"], strict=False)
    m.eval(); m.use_frame = False
    return m


@torch.no_grad()
def metrics(m, K, gen):
    cl, en, spread = [], [], []
    ncav = 0
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
        bidx = blk.nonzero().squeeze(1)
        Xg, Sg, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                     blk, bnd, sb, R, gen=gen)
        for k in range(M):
            xb, sbk = Xg[k][bidx], Sg[k][bidx]
            allx = torch.cat([Xg[k], bnd]); alls = torch.cat([Sg[k], sb])
            d = torch.cdist(xb, allx); sg = SIG[sbk[:, None], alls[None, :]]; ccl = d < CUT * sg
            for j, ii in enumerate(bidx.tolist()):
                ccl[j, ii] = False
            cl.append(float(ccl.sum(1).float().mean()))
            sig = SIG[sbk[:, None], alls[None, :]]; eps = EP[sbk[:, None], alls[None, :]]
            d2 = d.clamp_min(1e-6) ** 2
            for j, ii in enumerate(bidx.tolist()):
                d2[j, ii] = 1e12
            inv6 = (sig ** 2 / d2) ** 3
            en.append(float((4 * eps * (inv6 ** 2 - inv6)).clamp(0, CAP).sum(1).mean()))
            spread.append(float((xb - xb.mean(0)).norm(dim=-1).mean()))
        ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(cl), st.mean(en), st.mean(spread)


mb = load(BASE, False); mr = load(RLD, True)
print(f"=== RL+demand K-transfer (trained K in 4,8; K>8 = OOD), R={R} ===", flush=True)
print(f"{'K':>3} | {'clash base':>10} {'clash RL':>9} {'d%':>5} | {'E base':>7} {'E RL':>7} {'d%':>5} | {'spread b/RL':>12}", flush=True)
for K in (8, 12, 16, 24):
    cb, eb, spb = metrics(mb, K, torch.Generator(device=dev).manual_seed(0))
    cr, er, spr = metrics(mr, K, torch.Generator(device=dev).manual_seed(0))
    tag = "  (in-dist)" if K == 8 else "  (OOD)"
    print(f"{K:>3} | {cb:>10.3f} {cr:>9.3f} {100*(cr-cb)/cb:>+4.0f}% | {eb:>7.2f} {er:>7.2f} {100*(er-eb)/eb:>+4.0f}% | "
          f"{spb:.2f}/{spr:.2f}{tag}", flush=True)
