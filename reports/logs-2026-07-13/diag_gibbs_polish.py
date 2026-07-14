"""GIBBS POLISH (user idea, TF-motivated): the spike is compounding, and TF proved the conditional is
EXCELLENT given a complete cage. So after generating a K=8 block AR (half-built cage -> tail spike),
re-place the compounded members via SINGLE-SITE resampling -- each sees the FINISHED block (all other
members + retained + boundary = full cage). Single-site block: the member is last in the reorder, so its
causal context is everything else. Coherent (no hiding present particles), unlike pad-and-discard.
CAVEAT (full-cage-lever-needs-energy): iterating the learned conditional as pure Gibbs eventually COLLAPSES
(overlap->1) without an energy guard -> measure the TRAJECTORY over sweeps to find help-vs-collapse.

Arms (FT knn24, K=8 blob, R=2.5, M=16, 12 cav): baseline AR; then polish TAIL-3 (ranks 6-8) and ALL-8,
pure-Gibbs sweeps 1/2/3. Metrics per moved-8: clash/particle and dE/particle (the acceptance-relevant one).
Also the rank-resolved clash curve after 1 tail-polish (did the spike flatten?)."""
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


def clash_block(Xf, Sf, bidx, bnd, sb):
    xi, si = Xf[bidx], Sf[bidx]
    allx = torch.cat([Xf, bnd]); alls = torch.cat([Sf, sb])
    d = torch.cdist(xi, allx); sg = SIG[si[:, None], alls[None, :]]; cl = d < CUT * sg
    for j, ii in enumerate(bidx.tolist()):
        cl[j, ii] = False
    return cl.sum(1).float()                       # [K] per-member


@torch.no_grad()
def polish(Xb, Sb, slots, bnd, sb, gen):
    """One Gibbs sweep: sequentially single-site resample each global slot in `slots`."""
    for i in slots:
        mk = torch.zeros(Xb.shape[1], dtype=torch.bool, device=dev); mk[i] = True
        Xb, Sb, _ = m.sample_block_b(Xb, Sb, mk, bnd, sb, R, gen=gen)
    return Xb, Sb


@torch.no_grad()
def run():
    out = {}  # label -> (clash list, de list)
    rankcurve = {"base": [[] for _ in range(K)], "tail1": [[] for _ in range(K)]}
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
        bidx = blk.nonzero().squeeze(1)                       # ascending morton = rank order
        tail = bidx[-3:].tolist(); allk = bidx.tolist()
        E0 = Efn(torch.cat([xo, bnd])[None], torch.cat([so, sb])[None])[0]
        Xg, Sg, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                     blk, bnd, sb, R, gen=g)

        def record(label, Xf, Sf):
            cl = torch.stack([clash_block(Xf[k], Sf[k], bidx, bnd, sb) for k in range(M)])  # [M,K]
            Ef = Efn(torch.cat([Xf, bnd[None].expand(M, -1, 3)], 1), torch.cat([Sf, sb[None].expand(M, -1)], 1))
            out.setdefault(label, ([], []))
            out[label][0].append(float(cl.mean())); out[label][1].append(float(((Ef - E0) / K).median()))
            return cl

        clb = record("0 baseline AR", Xg, Sg)
        for r in range(K):
            rankcurve["base"][r].append(float(clb[:, r].mean()))
        # tail polish
        Xt, St = Xg.clone(), Sg.clone()
        for s in (1, 2, 3):
            Xt, St = polish(Xt, St, tail, bnd, sb, g)
            clt = record(f"tail3 polish x{s}", Xt, St)
            if s == 1:
                for r in range(K):
                    rankcurve["tail1"][r].append(float(clt[:, r].mean()))
        # all polish
        Xa, Sa = Xg.clone(), Sg.clone()
        for s in (1, 2, 3):
            Xa, Sa = polish(Xa, Sa, allk, bnd, sb, g)
            record(f"all8  polish x{s}", Xa, Sa)
        ncav += 1
        if ncav >= NCAV:
            break
    return out, rankcurve


out, rc = run()
print(f"=== Gibbs polish trajectory (FT knn24, K={K} blob, R={R}, {NCAV} cav) ===", flush=True)
print(f"{'arm':>18} | {'clash/p':>8} {'dE/particle':>12}", flush=True)
for label in ["0 baseline AR", "tail3 polish x1", "tail3 polish x2", "tail3 polish x3",
              "all8  polish x1", "all8  polish x2", "all8  polish x3"]:
    cl, de = out[label]
    print(f"{label:>18} | {st.mean(cl):>8.3f} {st.median(de):>+12.1f}", flush=True)
print("\n  rank-resolved clash: baseline vs after 1 tail-polish", flush=True)
print("   rank:  " + " ".join(f"{r+1:>5}" for r in range(K)), flush=True)
print("   base:  " + " ".join(f"{st.mean(rc['base'][r]):>5.2f}" for r in range(K)), flush=True)
print("   tail1: " + " ".join(f"{st.mean(rc['tail1'][r]):>5.2f}" for r in range(K)), flush=True)
