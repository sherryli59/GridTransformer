"""ACCEPTANCE-AS-IT-GOES: stream the per-rung acceptance of the two-blob redistribution I-MTM move (and the
local block move) as the SMC anneals lambda 0->1. Shows WHERE on the schedule the (clashy) big redistribution
move is accepted -- expected high at low lambda (soft target), collapsing as lambda->1. One island, alien seed,
R=2.0, multi-try N_try=8, redist fired at every rung here (no lam cutoff) so the full profile is visible.
Prints t, lambda, local-move acc, redist acc, running mean -- flushed per rung."""
import sys, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-14")
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b, blob_mask, ess
from smc_pts_sweep_hocky import box_overlap
from diag_twoblob_imtm import two_blob_mask, redist_imtm  # reuse the exact I-MTM move
import diag_twoblob_imtm as IM
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept

dev = "cuda"; RCTX = 2.5; BETA = 2.0; ART = "liquid_coupling_flow/artifacts"; L_BOX = 0.368
RHO = 1.149; BULK = RHO * L_BOX ** 3; K = 8; K_R = 12; R = 2.0
M = IM.M; T = IM.T; N_TRY = IM.N_TRY
m = IM.m; X, S, L = IM.X, IM.S, IM.L


@torch.no_grad()
def stream_island(xo, so, bnd, sb, R, gen, n_try):
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    Xb, Sb, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(), allm, bnd, sb, R, gen=gen)
    Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
    xin = _mic(carve_p["x_in"], cc, L)
    lam = torch.linspace(0, 1, T + 1, device=dev) ** 4
    a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    logw = torch.zeros(M, device=dev); Kr = min(K_R, (n - 2) // 2); rd_all = []
    print(f"{'t':>3} {'lambda':>7} | {'local_acc':>9} {'redist_acc':>10} {'run_mean_rd':>11} {'q~_now':>7}", flush=True)
    for t in range(1, T + 1):
        logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur - q0)
        if ess(logw) < 0.5:
            w = torch.softmax(logw, 0); idx = torch.multinomial(w, M, replacement=True, generator=gen)
            Xb, Sb, Ecur, q0 = Xb[idx].clone(), Sb[idx].clone(), Ecur[idx].clone(), q0[idx].clone(); logw = torch.zeros(M, device=dev)
        lt = float(lam[t]); loc_acc = []
        for _ in range(2):
            blk = blob_mask(n, K, a, gen)
            lqr = m.block_log_prob_b(Xb, Sb, blk, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xb, Sb, blk, bnd, sb, R, gen=gen)
            En = energy_b(Xn, Sn, bnd, sb); q0n = None if lt == 1.0 else m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
            la = geometric_bridge_log_accept(log_q0_current=q0, log_q0_proposed=q0n, energy_current=Ecur,
                                             energy_proposed=En, log_r_reverse=lqr, log_r_forward=lqf, lam=lt, beta=BETA)
            acc = torch.rand(M, device=dev, generator=gen).log() < la
            Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb); Ecur = torch.where(acc, En, Ecur)
            if q0n is not None:
                q0 = torch.where(acc, q0n, q0)
            loc_acc.append(float(acc.float().mean()))
        rd_here = []
        if Kr >= 4:
            for _ in range(3):
                U = two_blob_mask(n, a, gen, Kr)
                Xb, Sb, q0, Ecur, af = redist_imtm(Xb, Sb, bnd, sb, R, lt, q0, Ecur, allm, U, gen, n_try)
                rd_here.append(af); rd_all.append(af)
        qnow = st.mean([box_overlap(xin, Xb[k], R, L_BOX) for k in range(M)]) - BULK
        print(f"{t:>3} {lt:>7.4f} | {st.mean(loc_acc):>9.3f} "
              f"{(st.mean(rd_here) if rd_here else 0):>10.3f} {(st.mean(rd_all) if rd_all else 0):>11.3f} {qnow:>7.3f}", flush=True)
    return Xb, Sb


gen = torch.Generator(device=dev).manual_seed(0); found = False
for ci in range(24):
    cc = torch.rand(3, generator=gen, device=dev) * L; carve_p = carve(X[ci], S[ci], cc, R, L)
    if carve_p["n_in"] < K + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(carve_p["x_in"], cc, L), carve_p["s_in"], R)
    xout = _mic(carve_p["x_out"], cc, L); bmask = xout.norm(dim=-1) < (R + RCTX)
    bnd, sb = xout[bmask], carve_p["s_out"][bmask]
    print(f"=== redist acceptance-as-it-goes (R={R}, n={xo.shape[0]}, N_try={N_TRY}, alien seed) ===", flush=True)
    stream_island(xo, so, bnd, sb, R, torch.Generator(device=dev).manual_seed(1234), N_TRY)
    found = True; break
