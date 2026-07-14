"""DUAL-SEED confirmation of the basin trap. The island-SMC currently seeds every island from the AR proposal
(ALIEN basin). Here we ALSO seed islands from the DATA/reference interior (scaffold-ordered xo), run the SAME
tempered path + block-MTM mutation, and compare the final sample-vs-reference overlap q~_ref.

DIAGNOSIS by seed-dependence (a converged sampler is seed-INDEPENDENT):
  - data-seeded q~ HOLDS high while alien-seeded collapses  => TRAPPED: reference basin is STABLE under our
    SMC but UNREACHABLE from alien (local mutation can't cross). Green light for a dual-seed/PT estimator.
  - data-seeded ALSO collapses                              => the mutation actively DESTROYS the reference
    basin (deeper problem; mutation too aggressive / lam=1 move melts it).
  - data ~ alien and BOTH ~ truth (~0.63)                   => actually converged (would contradict the trap).
l=0.368 (paper box). lam05 Rext model. R in {1.6, 2.0}."""
import sys, math, statistics as st
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
RHO = 1.149; BULK = RHO * L_BOX ** 3
CK = f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(CK, map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
J, M, T, NMUT, NCAV, K = 4, 16, 20, 3, 3, 8
cfg = {"move": "1blob", "K": K, "swap": False}


@torch.no_grad()
def island_seeded(xo, so, bnd, sb, R, M, T, n_mut, gen, seed_mode):
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    if seed_mode == "alien":
        Xb, Sb, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(), allm, bnd, sb, R, gen=gen)
    else:  # data: start every walker AT the reference interior (scaffold-ordered)
        Xb = xo[None].expand(M, n, 3).clone(); Sb = so[None].expand(M, n).clone()
    Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
    lam = torch.linspace(0, 1, T + 1, device=dev) ** 4
    a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    logw = torch.zeros(M, device=dev); esses = []
    for t in range(1, T + 1):
        logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur - q0); esses.append(ess(logw))
        if ess(logw) < 0.5:
            w = torch.softmax(logw, 0); idx = torch.multinomial(w, M, replacement=True, generator=gen)
            Xb, Sb, Ecur, q0 = Xb[idx].clone(), Sb[idx].clone(), Ecur[idx].clone(), q0[idx].clone(); logw = torch.zeros(M, device=dev)
        lt = float(lam[t])
        for _ in range(n_mut):
            blk = blob_mask(n, cfg["K"], a, gen)
            lqr = m.block_log_prob_b(Xb, Sb, blk, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xb, Sb, blk, bnd, sb, R, gen=gen)
            En = energy_b(Xn, Sn, bnd, sb); q0n = None if lt == 1.0 else m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
            la = geometric_bridge_log_accept(log_q0_current=q0, log_q0_proposed=q0n, energy_current=Ecur,
                                             energy_proposed=En, log_r_reverse=lqr, log_r_forward=lqf, lam=lt, beta=BETA)
            acc = torch.rand(M, device=dev, generator=gen).log() < la
            Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb)
            Ecur = torch.where(acc, En, Ecur)
            if q0n is not None:
                q0 = torch.where(acc, q0n, q0)
    return Xb, Sb, esses


print(f"=== DUAL-SEED trap confirmation (lam05, l={L_BOX}) ===  seed-independent => converged", flush=True)
print(f"J={J} M={M} T={T} n_mut={NMUT} ncav={NCAV}  PT-ref true q(R)~0.63-0.66", flush=True)
print(f"{'R':>4} | {'alien q~':>9} {'data q~':>9} | {'data/alien':>10} | verdict", flush=True)
for R in (1.6, 2.0):
    gen = torch.Generator(device=dev).manual_seed(0); ncav = 0
    qa_all, qd_all = [], []
    for ci in range(24):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xin = _mic(p["x_in"], c, L)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        qa, qd = [], []
        for j in range(J):
            ga = torch.Generator(device=dev).manual_seed(1000 + 31 * ci + j)
            gd = torch.Generator(device=dev).manual_seed(5000 + 31 * ci + j)
            Xa, Sa, _ = island_seeded(xo, so, bnd, sb, R, M, T, NMUT, ga, "alien")
            Xd, Sd, _ = island_seeded(xo, so, bnd, sb, R, M, T, NMUT, gd, "data")
            qa.append(st.mean([box_overlap(xin, Xa[k], R, L_BOX) for k in range(Xa.shape[0])]) - BULK)
            qd.append(st.mean([box_overlap(xin, Xd[k], R, L_BOX) for k in range(Xd.shape[0])]) - BULK)
        qa_all.append(st.mean(qa)); qd_all.append(st.mean(qd)); ncav += 1
        if ncav >= NCAV:
            break
    qa = st.mean(qa_all); qd = st.mean(qd_all); ratio = qd / max(qa, 1e-3)
    verdict = ("TRAPPED: ref basin STABLE but unreachable from alien" if qd > 2 * qa and qd > 0.3
               else ("data ALSO collapses: mutation destroys ref basin" if qd < 0.2
                     else "seed-independent-ish (check vs truth)"))
    print(f"{R:>4} | {qa:>+9.3f} {qd:>+9.3f} | {ratio:>10.1f} | {verdict}", flush=True)
