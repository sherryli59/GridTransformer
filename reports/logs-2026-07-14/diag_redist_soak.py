"""SWEET-SPOT SOAK: the acceptance profile showed the redistribution I-MTM move fires ONLY in a narrow window
lambda~0.05-0.09 (16.7% via multi-try), then freezes. So concentrate MANY multi-try attempts in that window
(+ extra schedule rungs there) instead of spreading them across lambda<0.6. Sweep soak intensity N_soak;
measure whether ALIEN-seed q~ rises past the single-burst 0.032 toward the data bracket. Schedule = quartic
with EXTRA linspace rungs inserted in [SOAK_LO, SOAK_HI]. lam05, l=0.368, R=2.0, N_try=8."""
import sys, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-14")
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b, blob_mask, ess
from smc_pts_sweep_hocky import box_overlap
from diag_twoblob_imtm import two_blob_mask, redist_imtm   # guarded now; safe import
import diag_twoblob_imtm as IM
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept

dev = "cuda"; RCTX = 2.5; BETA = 2.0; L_BOX = 0.368; RHO = 1.149; BULK = RHO * L_BOX ** 3
K = 8; K_R = 12; R = 2.0; N_TRY = 8; SOAK_LO, SOAK_HI = 0.04, 0.12
m = IM.m; X, S, L = IM.X, IM.S, IM.L; M = 16; T = 20; J = 3; NCAV = 2; NMUT = 2


def schedule(n_extra):
    """quartic base + n_extra evenly-spaced rungs inside the sweet-spot window, sorted+unique."""
    base = (torch.linspace(0, 1, T + 1, device=dev) ** 4)
    if n_extra > 0:
        extra = torch.linspace(SOAK_LO, SOAK_HI, n_extra + 2, device=dev)[1:-1]
        base = torch.cat([base, extra])
    return torch.sort(torch.unique(base)).values


@torch.no_grad()
def island(xo, so, bnd, sb, R, gen, n_soak, n_extra):
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    Xb, Sb, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(), allm, bnd, sb, R, gen=gen)
    Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
    lam = schedule(n_extra); a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    logw = torch.zeros(M, device=dev); Kr = min(K_R, (n - 2) // 2); rd_win = []
    for t in range(1, len(lam)):
        logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur - q0)
        if ess(logw) < 0.5:
            w = torch.softmax(logw, 0); idx = torch.multinomial(w, M, replacement=True, generator=gen)
            Xb, Sb, Ecur, q0 = Xb[idx].clone(), Sb[idx].clone(), Ecur[idx].clone(), q0[idx].clone(); logw = torch.zeros(M, device=dev)
        lt = float(lam[t])
        for _ in range(NMUT):
            blk = blob_mask(n, K, a, gen)
            lqr = m.block_log_prob_b(Xb, Sb, blk, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xb, Sb, blk, bnd, sb, R, gen=gen)
            En = energy_b(Xn, Sn, bnd, sb); q0n = None if lt == 1.0 else m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
            la = geometric_bridge_log_accept(log_q0_current=q0, log_q0_proposed=q0n, energy_current=Ecur,
                                             energy_proposed=En, log_r_reverse=lqr, log_r_forward=lqf, lam=lt, beta=BETA)
            acc = torch.rand(M, device=dev, generator=gen).log() < la
            Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb); Ecur = torch.where(acc, En, Ecur)
            if q0n is not None:
                q0 = torch.where(acc, q0n, q0)
        n_here = n_soak if (SOAK_LO <= lt <= SOAK_HI) else 0            # SOAK only in the window
        if Kr >= 4:
            for _ in range(n_here):
                U = two_blob_mask(n, a, gen, Kr)
                Xb, Sb, q0, Ecur, af = redist_imtm(Xb, Sb, bnd, sb, R, lt, q0, Ecur, allm, U, gen, N_TRY)
                rd_win.append(af)
    return Xb, Sb, (st.mean(rd_win) if rd_win else 0.0)


def run(n_soak, n_extra):
    gen = torch.Generator(device=dev).manual_seed(0); ncav = 0; qs = []; accs = []
    for ci in range(24):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xin = _mic(p["x_in"], c, L)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        qc = []
        for j in range(J):
            g = torch.Generator(device=dev).manual_seed(1000 + 31 * ci + j)
            Xd, Sd, af = island(xo, so, bnd, sb, R, g, n_soak, n_extra); accs.append(af)
            qc.append(st.mean([box_overlap(xin, Xd[k], R, L_BOX) for k in range(Xd.shape[0])]) - BULK)
        qs.append(st.mean(qc)); ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(qs), st.mean(accs)


print(f"=== SWEET-SPOT SOAK sweep (alien, R={R}, window lam in [{SOAK_LO},{SOAK_HI}], N_try={N_TRY}) ===", flush=True)
print(f"prior: 1 island single-burst q~=0.032. Does soaking the window raise alien q~?", flush=True)
print(f"{'n_soak':>7} {'n_extra_rungs':>13} | {'alien q~':>9} {'win_rd_acc':>10}", flush=True)
for n_soak, n_extra in [(3, 0), (15, 0), (15, 6), (40, 6)]:
    q, af = run(n_soak, n_extra)
    print(f"{n_soak:>7} {n_extra:>13} | {q:>9.3f} {af:>10.3f}", flush=True)
