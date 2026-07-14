"""Test the (uncommitted, in-tree) virtual_tail knob: shift a K-block's slot_index down by vt so its frac
features look like the first-K of a (K+vt) block -- an EXACT, zero-shot realization of 'pretend it's bigger,
leave room', reusing the trained frac conditioning (no retrain, no hidden particles). Hypothesis: telling
the block 'more members are coming' makes it place looser -> fewer clashes, BUT under-packs the region
(lower local density = wrong structure). Measure vs virtual_tail: (1) EXACTNESS sample==score roundtrip
(must hold, or the knob is unusable); (2) clash/particle; (3) dE/particle; (4) 5-seed MTM acceptance;
(5) block spread RMS |x-centroid| (the under-packing cost). FT knn24, K=8 blob, R=2.5, M=16."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA

dev = "cuda"; RCTX = 2.5; R = 2.5; BIGL = 100.0; BETA = 2.0; M = 16; NCAV = 12; CUT = 0.85; K = 8
SIG = torch.tensor(SIGMA, device=dev)
FT = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt"
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24).to(dev)
m.load_state_dict(torch.load(FT, map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False


def Efn(Xi, Si): return ka_energy(Xi.double(), Si.long(), BIGL).float()


def blob(n, seed, a):
    b = torch.zeros(n, dtype=torch.bool, device=dev)
    b[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    return b


@torch.no_grad()
def sweep(vt, gen, check_exact=False):
    cl, de, acc, spread = [], [], [], []
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
        seed = int(torch.randint(n, (), generator=gen, device=dev)); blk = blob(n, seed, a)
        bidx = blk.nonzero().squeeze(1)
        Xp, Sp, lqf = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                       blk, bnd, sb, R, gen=gen, virtual_tail=vt)
        if check_exact:
            lqs = m.block_log_prob_b(Xp, Sp, blk, bnd, sb, R, virtual_tail=vt)
            assert float((lqf - lqs).abs().max()) < 5e-2, f"roundtrip broke at vt={vt}: {(lqf-lqs).abs().max()}"
        E0 = Efn(torch.cat([xo, bnd])[None], torch.cat([so, sb])[None])[0]
        Ep = Efn(torch.cat([Xp, bnd[None].expand(M, -1, 3)], 1), torch.cat([Sp, sb[None].expand(M, -1)], 1))
        u0 = -BETA * E0 - m.block_log_prob_b(xo[None], so[None], blk, bnd, sb, R, virtual_tail=vt)[0]
        up = -BETA * Ep - lqf; sf = torch.logsumexp(up, 0)
        J = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen)); lr = up.clone(); lr[J] = u0
        acc.append(float(torch.rand((), device=dev, generator=gen).log() < (sf - torch.logsumexp(lr, 0))))
        de.append(float(((Ep - E0) / K).median()))
        for k in range(M):
            xb = Xp[k][bidx]
            allx = torch.cat([Xp[k], bnd]); alls = torch.cat([Sp[k], sb])
            d = torch.cdist(xb, allx); sg = SIG[Sp[k][bidx][:, None], alls[None, :]]; ccl = d < CUT * sg
            for j, ii in enumerate(bidx.tolist()):
                ccl[j, ii] = False
            cl.append(float(ccl.sum(1).float().mean()))
            spread.append(float((xb - xb.mean(0)).norm(dim=-1).mean()))
        ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(cl), st.median(de), 100 * st.mean(acc), st.mean(spread)


print(f"=== virtual_tail sweep (FT knn24, K={K} blob, R={R}) ===", flush=True)
# exactness check first
sweep(4, torch.Generator(device=dev).manual_seed(0), check_exact=True)
print("  exactness (sample==score, vt=4): PASS", flush=True)
print(f"{'vt':>3} | {'clash/p':>8} {'dE/p':>9} {'MTM(5seed)':>18} {'block spread':>12}", flush=True)
for vt in (0, 2, 4, 6, 8):
    accs = []
    cl = de = spr = None
    for s in range(5):
        c, d, ac, sp = sweep(vt, torch.Generator(device=dev).manual_seed(s))
        accs.append(ac)
        if s == 0:
            cl, de, spr = c, d, sp
    print(f"{vt:>3} | {cl:>8.3f} {de:>+9.1f} {st.mean(accs):>6.1f} +/- {st.pstdev(accs):>4.1f}%   {spr:>12.3f}", flush=True)
