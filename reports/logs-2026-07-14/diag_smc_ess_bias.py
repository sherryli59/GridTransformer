"""WHY is our PTS estimator biased, and HOW MANY effective samples do we have? Two orthogonal measures:
  (A) SMC WEIGHT-ESS -> effective sample count = mean(ESS_frac) * M * J. Measures weight degeneracy WITHIN
      the SMC. Reported as the ESS trajectory over tempering rungs + the run-mean.
  (B) POPULATION DIVERSITY qc_pair = mean pairwise sample-vs-sample box-overlap. If qc_pair ~ rho (self)
      the M samples are near-copies (one basin); if ~ q_bulk they are diverse. This is the BASIN measure ESS
      can't see.
Bias signature: q~(vs reference) collapses to ~0 while qc_pair stays HIGH (>> q~) and ESS looks healthy =
population is internally consistent but trapped in a NON-reference basin (local block-MTM can't cross). The
PT-stack reference on THIS N=4096 rho=1.149 data (memory ka3d-pts-smc) converged to G_PTS~0.63 at R=2.0 -> the
true overlap is large there, so our q~~0 is sampler bias, not physics. l=0.368 (paper box). R in {1.6, 2.0}."""
import sys, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-14")
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import island
from smc_pts_sweep_hocky import box_overlap
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; RCTX = 2.5; ART = "liquid_coupling_flow/artifacts"; L_BOX = 0.368
CK = f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(CK, map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
rho = X.shape[1] / L ** 3
J, M, T, NMUT, NCAV, K = 6, 16, 20, 3, 3, 8
cfg = {"move": "1blob", "K": K, "swap": False}


def qc_pair(Xp, R):                          # mean pairwise sample-vs-sample box overlap (basin degeneracy)
    Mn = Xp.shape[0]; ov = []
    for i in range(Mn):
        for j in range(i + 1, Mn):
            ov.append(box_overlap(Xp[i], Xp[j], R, L_BOX))
    return st.mean(ov)


print(f"=== ESS + effective-samples + basin-diversity (why biased) ===  model={CK.split('/')[-1]}", flush=True)
print(f"rho={rho:.3f}  J={J} M={M} T={T} n_mut={NMUT} ncav={NCAV}  l_box={L_BOX}", flush=True)
print(f"self-overlap ceiling q(self)=rho={rho:.2f};  bulk floor rho*l^3={rho*L_BOX**3:.3f}", flush=True)
print(f"{'R':>4} | {'ESS_frac':>8} {'eff_samp':>8} | {'q~_ref':>7} {'qc_pair':>7} | {'isl-std':>7} | read", flush=True)
for R in (1.6, 2.0):
    gen = torch.Generator(device=dev).manual_seed(0); ncav = 0
    ess_all, qtil_all, qcp_all, qisl_std = [], [], [], []
    for ci in range(24):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xin = _mic(p["x_in"], c, L)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        qb = rho * L_BOX ** 3
        qbox_isl, qc_isl, ess_isl = [], [], []
        for j in range(J):
            g = torch.Generator(device=dev).manual_seed(1000 + 31 * ci + j)
            Xp, Sp, _, esses = island(m, xo, so, bnd, sb, R, cfg, M, T, NMUT, g)
            qbox_isl.append(st.mean([box_overlap(xin, Xp[k], R, L_BOX) for k in range(Xp.shape[0])]))
            qc_isl.append(qc_pair(Xp, R)); ess_isl.append(st.mean(esses))
        ess_all.append(st.mean(ess_isl))
        qtil_all.append(st.mean(qbox_isl) - qb); qcp_all.append(st.mean(qc_isl) - qb)
        qisl_std.append(st.pstdev(qbox_isl)); ncav += 1
        if ncav >= NCAV:
            break
    essf = st.mean(ess_all); effs = essf * M * J
    qtil = st.mean(qtil_all); qcp = st.mean(qcp_all); islstd = st.mean(qisl_std)
    read = "TRAPPED: samples agree (qc_pair) but far from ref (q~~0)" if (qcp > 2 * max(qtil, 0.001)) else "diverse"
    print(f"{R:>4} | {essf:>8.3f} {effs:>8.1f} | {qtil:>+7.3f} {qcp:>+7.3f} | {islstd:>7.3f} | {read}", flush=True)
print(f"\nESS_frac = weight-based effective fraction (WITHIN-SMC degeneracy); eff_samp = ESS_frac*M*J.", flush=True)
print(f"qc_pair (sample-vs-sample, bulk-subtracted) >> q~_ref => population trapped in a NON-reference basin;", flush=True)
print(f"ESS cannot see this (it measures weights, not distance-to-truth). PT-ref (memory) q(R=2.0)~0.63 true.", flush=True)
