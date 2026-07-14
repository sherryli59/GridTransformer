"""WITHIN-BASIN MIXING test (isolate the sub-problem the dual-seed exposed). DATA-seed the SMC (start IN the
reference basin) and escalate the mutation kernel; watch whether q~ relaxes from the frozen ~0.98 (R=2.0) DOWN
toward the converged truth ~0.63. Outcomes:
  - relaxes to ~0.63 and plateaus         => stronger moves SOLVE within-basin mixing (green light for PT).
  - stuck near 0.98                        => block-MTM cannot mix a pinned interior (need a different move).
  - drops below ~0.63 toward alien floor   => moves too aggressive, replicas LEAVE the reference basin.
Levers: n_mut (mutation sweeps/rung), K (block size), swap (species exchange). lam05 Rext model, l=0.368.
R=2.0 (frozen case) + R=1.6 control (data-seed already ~=truth there). J=3, M=16, T=20, ncav=3."""
import sys, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-14")
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b, blob_mask, ess, swap_sweep
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
J, M, T, NCAV = 3, 16, 20, 3


@torch.no_grad()
def island_data_seed(xo, so, bnd, sb, R, K, n_mut, swap, gen):
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    Xb = xo[None].expand(M, n, 3).clone(); Sb = so[None].expand(M, n).clone()      # DATA seed
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
        if swap:
            Sb, q0, Ecur = swap_sweep(m, Xb, Sb, bnd, sb, R, lt, q0, Ecur, allm, gen)
    return Xb, Sb


def blob_mask_wrap(n, K, a, gen):  # helper unused; blob_mask imported handles it
    return blob_mask(n, K, a, gen)


CONFIGS = [("K8 nm3 (base)", 8, 3, False), ("K8 nm8", 8, 8, False), ("K8 nm16", 8, 16, False),
           ("K16 nm8", 16, 8, False), ("K8 nm8 +swap", 8, 8, True)]
print(f"=== within-basin mixing (DATA-seed; target relax q~ -> truth ~0.63 @R2.0, ~0.66 @R1.6) ===", flush=True)
print(f"lam05, l={L_BOX}, J={J} M={M} T={T} ncav={NCAV}. base R2.0 data-seed was 0.98 (frozen).", flush=True)
print(f"{'config':>16} | {'R1.6 q~':>8} {'R2.0 q~':>8} | note", flush=True)
for name, K, nm, sw in CONFIGS:
    row = {}
    for R in (1.6, 2.0):
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
                g = torch.Generator(device=dev).manual_seed(7000 + 31 * ci + j)
                Xd, Sd = island_data_seed(xo, so, bnd, sb, R, K, nm, sw, g)
                qc.append(st.mean([box_overlap(xin, Xd[k], R, L_BOX) for k in range(Xd.shape[0])]) - BULK)
            qs.append(st.mean(qc)); ncav += 1
            if ncav >= NCAV:
                break
        row[R] = st.mean(qs)
    note = "relaxing toward truth" if row[2.0] < 0.9 else "still frozen"
    print(f"{name:>16} | {row[1.6]:>+8.3f} {row[2.0]:>+8.3f} | {note}", flush=True)
