"""Is the Morton-TAIL (suffix) resample a clean q0-Gibbs move? A suffix has NO particles after it in the AR
order, so q0 factors cleanly: q0 = q0(prefix)*q(suffix|prefix); resampling the suffix from the model IS the
exact conditional -> MH ratio telescopes to 1 -> acceptance 1.0 at lambda=0. Contrast with an interior blob
(KNN) whose regen shifts the context of later particles -> structural reject at lambda=0. Measure single-try
block-MTM acceptance for TAIL vs BLOB masks at lambda in {0, 0.02, 0.1}. Fresh alien MB=32, 3 cavities."""
import sys, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b, blob_mask
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; BETA = 2.0; R = 2.0; MB = 32; NCAV = 3; ART = "liquid_coupling_flow/artifacts"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
LAMS = [0.0, 0.02, 0.1]; KBLK = 10


def acc_move(Xb, Sb, bnd, sb, R, lt, q0, Ecur, allm, mask, gen):
    lqr = m.block_log_prob_b(Xb, Sb, mask, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xb, Sb, mask, bnd, sb, R, gen=gen)
    En = energy_b(Xn, Sn, bnd, sb); q0n = None if lt == 1.0 else m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
    la = geometric_bridge_log_accept(log_q0_current=q0, log_q0_proposed=q0n, energy_current=Ecur,
                                     energy_proposed=En, log_r_reverse=lqr, log_r_forward=lqf, lam=lt, beta=BETA)
    return float((torch.rand(Xb.shape[0], device=dev, generator=gen).log() < la).float().mean())


gen = torch.Generator(device=dev).manual_seed(0)
tail = {l: [] for l in LAMS}; blob = {l: [] for l in LAMS}; ncav = 0
for ci in range(24):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < KBLK + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)   # xo/so are in SCAFFOLD (Morton) order
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order); g = torch.Generator(device=dev).manual_seed(100 + ci)
    X0, S0, _ = m.sample_block_b(xo[None].expand(MB, n, 3).clone(), so[None].expand(MB, n).clone(), allm, bnd, sb, R, gen=g)
    E0 = energy_b(X0, S0, bnd, sb); Q0 = m.block_log_prob_b(X0, S0, allm, bnd, sb, R)
    tmask = torch.zeros(n, dtype=torch.bool, device=dev); tmask[n - KBLK:] = True      # Morton SUFFIX (last KBLK slots)
    bmask = blob_mask(n, KBLK, a, g)                                                     # interior KNN blob (same size)
    for lam in LAMS:
        tail[lam].append(acc_move(X0, S0, bnd, sb, R, lam, Q0, E0, allm, tmask, g))
        blob[lam].append(acc_move(X0, S0, bnd, sb, R, lam, Q0, E0, allm, bmask, g))
    ncav += 1
    if ncav >= NCAV:
        break

print(f"=== Morton-TAIL (clean q0-Gibbs?) vs interior BLOB, single-try acceptance, K={KBLK} ===", flush=True)
print(f"{'lambda':>7} | {'tail_acc':>8} {'blob_acc':>8}", flush=True)
for lam in LAMS:
    print(f"{lam:>7.3f} | {st.mean(tail[lam]):>8.3f} {st.mean(blob[lam]):>8.3f}", flush=True)
print(f"\nTail ~1.0 at lambda=0 => clean conditional resample (principled); blob ~0 => AR-context tax.", flush=True)
