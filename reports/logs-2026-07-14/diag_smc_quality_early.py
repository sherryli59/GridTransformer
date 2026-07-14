"""EARLY SMC-QUALITY diagnostic: does the RL declashing BUY SMC efficiency, net of the +0.24-nat NLL drift?
The tempered island-SMC uses the block proposal q0 as the geometric-bridge MTM density. The ONE metric that
folds in both forces is the BLOCK-MUTATION ACCEPTANCE RATE: la = geometric_bridge_log_accept(...) contains
(i) the energy term (declash -> higher accept) and (ii) the q0-rescore term (NLL drift -> lower accept). We
also track mean ESS (weight quality) and final q(R) + per-island spread (mixing).

Compare base-Rext (lam=0, best NLL, more clash) vs RL-Rext (declashed, +0.24 NLL). If RL raises acceptance/
ESS/mixing at the SAME q(R), declash wins net and lam=0.5 is justified; if acceptance DROPS (NLL cost
dominates), lower lam. Split acceptance by lam regime: low-lam (energy-light, q0 dominates) vs high-lam
(energy-heavy, declash dominates). J=3, M=16, T=12, K=8, R in {2.5 in-dist, 3.5 large-R}. Shares GPU."""
import sys, time, math, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-14")
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b, blob_mask, overlap, ess
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept

dev = "cuda"; RCTX = 2.5; BETA = 2.0; ART = "liquid_coupling_flow/artifacts"
BASE = f"{ART}/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_Rext_best.pt"       # lam=0 (no RL), best NLL
RL = f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_best.pt"                # lam=0.5 declashed
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def load(path):
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["state_dict"], strict=False)
    m.eval(); m.use_frame = False
    return m


@torch.no_grad()
def island_instrumented(m, xo, so, bnd, sb, R, K, M, T, n_mut, gen):
    """island() with per-mutation acceptance instrumentation, split by lam regime."""
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    Xb, Sb, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(), allm, bnd, sb, R, gen=gen)
    Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
    lam = torch.linspace(0, 1, T + 1, device=dev) ** 4
    a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    logw = torch.zeros(M, device=dev); esses = []
    acc_lo = acc_hi = tot_lo = tot_hi = 0.0
    for t in range(1, T + 1):
        logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur - q0); esses.append(ess(logw))
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
            ar = float(acc.float().mean())
            if lt < 0.5:
                acc_lo += ar; tot_lo += 1
            else:
                acc_hi += ar; tot_hi += 1
            Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb)
            Ecur = torch.where(acc, En, Ecur)
            if q0n is not None:
                q0 = torch.where(acc, q0n, q0)
    return Xb, Sb, esses, (acc_lo / max(tot_lo, 1), acc_hi / max(tot_hi, 1))


def run(m, R, K, J=3, M=16, T=12, ncav=3):
    accs_lo, accs_hi, esss, qs_all, qstd = [], [], [], [], []
    gen = torch.Generator(device=dev).manual_seed(0); nc = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xin = _mic(p["x_in"], c, L); sin = p["s_in"]
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        qs = []
        for j in range(J):
            g = torch.Generator(device=dev).manual_seed(1000 + 31 * ci + j)
            Xp, Sp, esses, (alo, ahi) = island_instrumented(m, xo, so, bnd, sb, R, K, M, T, 3, g)
            accs_lo.append(alo); accs_hi.append(ahi); esss.append(st.mean(esses)); qs.append(overlap(Xp, Sp, xin, sin))
        qs_all.append(st.mean(qs)); qstd.append(st.pstdev(qs)); nc += 1
        if nc >= ncav:
            break
    return (st.mean(accs_lo), st.mean(accs_hi), st.mean(esss), st.mean(qs_all), st.mean(qstd))


mb, mr = load(BASE), load(RL)
print("=== EARLY SMC quality: base-Rext (lam=0) vs RL-Rext (lam=0.5) ===", flush=True)
print("acc=block-mutation MH acceptance (lo lam / hi lam); ESS=mean tempered ESS; q(R)+/-isl-std=overlap/mixing", flush=True)
print(f"{'R':>4} {'model':>5} | {'acc_lo':>6} {'acc_hi':>6} | {'ESS':>5} | {'q(R)':>5} {'isl-std':>7}", flush=True)
for R in (2.5, 3.5):
    for name, m in (("base", mb), ("RL", mr)):
        t0 = time.time()
        alo, ahi, es, q, qsd = run(m, R, K=8)
        print(f"{R:>4} {name:>5} | {alo:>6.3f} {ahi:>6.3f} | {es:>5.3f} | {q:>5.3f} {qsd:>7.3f}  ({time.time()-t0:.0f}s)", flush=True)
