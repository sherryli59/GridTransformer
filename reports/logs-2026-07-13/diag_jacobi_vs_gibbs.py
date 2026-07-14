"""User idea: 'regenerate EVERYTHING conditioning on the imperfect first pass'. That is a SIMULTANEOUS
(Jacobi) full-cage update: each member re-placed at q(x_i | ALL others from pass-1), all applied at once.
Contrast with the SEQUENTIAL (Gauss-Seidel/Gibbs) polish already tested (each sees the latest update).
Prediction: Jacobi can COLLIDE -- two neighbours both move toward the same freed gap simultaneously ->
new clash -- so 'regenerate everything at once' may be WORSE than sequential. Also: the stochastic 2nd pass
hurts exact likelihood (product of full conditionals = pseudo-likelihood, not a normalized joint -> no exact
log_q for IS/MTM), which is the user's own worry. This quantifies the declash ceiling of the literal idea.
FT knn24, K=8 blob, R=2.5, M=16, 12 cav."""
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


def clash_p(Xf, Sf, bidx, bnd, sb):
    xi, si = Xf[:, bidx], Sf[:, bidx]
    allx = torch.cat([Xf, bnd[None].expand(M, -1, 3)], 1); alls = torch.cat([Sf, sb[None].expand(M, -1)], 1)
    d = torch.cdist(xi, allx); sg = SIG[si[:, :, None], alls[:, None, :]]; cl = d < CUT * sg
    for j, ii in enumerate(bidx.tolist()):
        cl[:, j, ii] = False
    return float(cl.sum(2).float().mean())


@torch.no_grad()
def run():
    o = {k: ([], []) for k in ("base", "jacobi1", "gibbs1")}
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
        E0 = Efn(torch.cat([xo, bnd])[None], torch.cat([so, sb])[None])[0]
        Xg, Sg, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                     blk, bnd, sb, R, gen=g)

        def rec(key, Xf, Sf):
            Ef = Efn(torch.cat([Xf, bnd[None].expand(M, -1, 3)], 1), torch.cat([Sf, sb[None].expand(M, -1)], 1))
            o[key][0].append(clash_p(Xf, Sf, bidx, bnd, sb)); o[key][1].append(float(((Ef - E0) / K).median()))

        rec("base", Xg, Sg)
        # JACOBI: each member resampled from PASS-1 (frozen Xg), all applied together
        Xj, Sj = Xg.clone(), Sg.clone()
        newx, news = Xg.clone(), Sg.clone()
        for i in bidx.tolist():
            mk = torch.zeros(n, dtype=torch.bool, device=dev); mk[i] = True
            xi_, si_, _ = m.sample_block_b(Xg.clone(), Sg.clone(), mk, bnd, sb, R, gen=g)  # cond on PASS-1
            newx[:, i] = xi_[:, i]; news[:, i] = si_[:, i]
        rec("jacobi1", newx, news)
        # GIBBS: sequential, each sees latest
        Xs, Ss = Xg.clone(), Sg.clone()
        for i in bidx.tolist():
            mk = torch.zeros(n, dtype=torch.bool, device=dev); mk[i] = True
            Xs, Ss, _ = m.sample_block_b(Xs, Ss, mk, bnd, sb, R, gen=g)
        rec("gibbs1", Xs, Ss)
        ncav += 1
        if ncav >= NCAV:
            break
    return o


o = run()
print(f"=== 'regenerate everything on pass-1' (JACOBI) vs sequential (GIBBS), 1 sweep (FT knn24, K={K}) ===", flush=True)
print(f"{'arm':>28} | {'clash/p':>8} {'dE/particle':>12}", flush=True)
for k, name in (("base", "0 baseline AR (pass-1)"), ("jacobi1", "regen-all-on-pass1 (Jacobi)"),
                ("gibbs1", "sequential (Gibbs)")):
    print(f"{name:>28} | {st.mean(o[k][0]):>8.3f} {st.median(o[k][1]):>+12.1f}", flush=True)
