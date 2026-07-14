"""STACK-AGREEMENT convergence test (the CORRECT experiment; replaces the 'over-mixing' misread). Run BOTH
seeds -- alien (AR proposal, rises from below) and data (reference, falls from above) -- at escalating mutation
budget n_mut, at fixed R. A valid kernel that is ALSO ergodic makes the two seeds MEET; the meeting value is
the true equilibrium overlap in THIS observable (no borrowed cross-observable 'truth'). Read:
  - gap |data - alien| -> 0 as n_mut grows        => converged; report the shared value as q(R).
  - data plateaus HIGH, alien stays LOW (gap persists) => kernel is valid but NON-ERGODIC across basins
    (local block-MTM can't cross); need basin-crossing (exchange/PT). This is the decisive local-move-
    insufficiency proof.
  - data falls all the way to alien's low value    => data LEFT the reference basin (block regen jumps out);
    both sampling wrong-basin equilibrium.
K=8, T=20, J=3, M=16, ncav=3. lam05 Rext, l=0.368. R in {1.6, 2.0}."""
import sys, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-14")
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b, blob_mask, ess
from smc_pts_sweep_hocky import box_overlap
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept

dev = "cuda"; RCTX = 2.5; BETA = 2.0; ART = "liquid_coupling_flow/artifacts"; L_BOX = 0.368
RHO = 1.149; BULK = RHO * L_BOX ** 3; K = 8
CK = f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(CK, map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
J, M, T, NCAV = 3, 16, 20, 3
NMUTS = [3, 8, 16, 32]


@torch.no_grad()
def island_seeded(xo, so, bnd, sb, R, n_mut, gen, seed_mode):
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    if seed_mode == "alien":
        Xb, Sb, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(), allm, bnd, sb, R, gen=gen)
    else:
        Xb = xo[None].expand(M, n, 3).clone(); Sb = so[None].expand(M, n).clone()
    Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
    lam = torch.linspace(0, 1, T + 1, device=dev) ** 4
    a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    logw = torch.zeros(M, device=dev)
    for t in range(1, T + 1):
        logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur - q0)
        if ess(logw) < 0.5:
            w = torch.softmax(logw, 0); idx = torch.multinomial(w, M, replacement=True, generator=gen)
            Xb, Sb, Ecur, q0 = Xb[idx].clone(), Sb[idx].clone(), Ecur[idx].clone(), q0[idx].clone(); logw = torch.zeros(M, device=dev)
        lt = float(lam[t])
        for _ in range(n_mut):
            blk = blob_mask(n, K, a, gen)
            lqr = m.block_log_prob_b(Xb, Sb, blk, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xb, Sb, blk, bnd, sb, R, gen=gen)
            En = energy_b(Xn, Sn, bnd, sb); q0n = None if lt == 1.0 else m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
            la = geometric_bridge_log_accept(log_q0_current=q0, log_q0_proposed=q0n, energy_current=Ecur,
                                             energy_proposed=En, log_r_reverse=lqr, log_r_forward=lqf, lam=lt, beta=BETA)
            acc = torch.rand(M, device=dev, generator=gen).log() < la
            Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb)
            Ecur = torch.where(acc, En, Ecur)
            if q0n is not None:
                q0 = torch.where(acc, q0n, q0)
    return Xb, Sb


def measure(R, n_mut, seed_mode):
    gen = torch.Generator(device=dev).manual_seed(0); ncav = 0; qs = []
    for ci in range(24):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xin = _mic(p["x_in"], c, L)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        qc = []
        for j in range(J):
            g = torch.Generator(device=dev).manual_seed((8000 if seed_mode == "data" else 1000) + 31 * ci + j)
            Xd, Sd = island_seeded(xo, so, bnd, sb, R, n_mut, g, seed_mode)
            qc.append(st.mean([box_overlap(xin, Xd[k], R, L_BOX) for k in range(Xd.shape[0])]) - BULK)
        qs.append(st.mean(qc)); ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(qs)


print(f"=== STACK-AGREEMENT: do alien(up) and data(down) seeds MEET as n_mut grows? (lam05, l={L_BOX}) ===", flush=True)
print(f"K={K} T={T} J={J} M={M} ncav={NCAV}. gap->0 = converged; gap persists = non-ergodic (need basin-crossing)", flush=True)
for R in (1.6, 2.0):
    print(f"\n R={R}:  {'n_mut':>6} {'alien q~':>9} {'data q~':>9} {'gap':>7}", flush=True)
    for nm in NMUTS:
        qa = measure(R, nm, "alien"); qd = measure(R, nm, "data")
        print(f"        {nm:>6} {qa:>+9.3f} {qd:>+9.3f} {qd - qa:>7.3f}", flush=True)
