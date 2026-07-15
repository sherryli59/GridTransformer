"""DIRECT q0-coverage evidence (user challenge: 'no single-chain kernel can cross' was inferred from FLAWED
kernels -- interior=q0-tax, suffix=frozen-head -- so measure q0 itself). For each cavity at R=2.0:
  logq0(REFERENCE interior)  vs  logq0 of 64 ALIEN full draws (allmask = true full-marginal draws)
  U(reference)               vs  U(alien draws)
  lambda=1 tempered log-weight  s = -beta*U - logq0  for both  =>  if s(ref) >> s(alien), an independence
  full-regen MH move WOULD accept a reference-basin proposal essentially always once drawn -- the only barrier
  is DRAW FREQUENCY (basin mass under q0), not acceptance. Decompose:
  - logq0(ref) WITHIN alien logq0 distribution  => q0 density on ref configs is healthy; wall = basin MASS/
    entropy (many alien basins) + kernel reachability. Coverage-by-density NOT refuted.
  - logq0(ref) tens of nats BELOW               => true pointwise under-weighting.
lam05 Rext ckpt, R=2.0, 4 cavities."""
import sys, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; BETA = 2.0; R = 2.0; ART = "liquid_coupling_flow/artifacts"; M = 64
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])

gen = torch.Generator(device=dev).manual_seed(0); ncav = 0
print(f"=== q0 coverage: score the REFERENCE under q0 vs alien full draws (R={R}, M={M}) ===", flush=True)
print(f"{'cav':>4} {'n':>4} | {'logq0_ref':>9} {'logq0_alien (mean+-sd)':>22} {'z':>6} | {'U_ref':>7} {'U_alien':>8} | {'s_ref - s_alien':>15}", flush=True)
zs, sgaps = [], []
for ci in range(24):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < 14:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    lq_ref = float(m.block_log_prob_b(xo[None], so[None], allm, bnd, sb, R)[0])
    U_ref = float(energy_b(xo[None], so[None], bnd, sb)[0])
    g = torch.Generator(device=dev).manual_seed(100 + ci)
    Xa, Sa, lq_a = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                    allm, bnd, sb, R, gen=g)
    Ua = energy_b(Xa, Sa, bnd, sb)
    lam, lsd = float(lq_a.mean()), float(lq_a.std())
    z = (lq_ref - lam) / max(lsd, 1e-6)
    s_ref = -BETA * U_ref - lq_ref
    s_alien = (-BETA * Ua - lq_a)
    sgap = s_ref - float(s_alien.max())          # vs the BEST alien draw (what MH would compare against)
    zs.append(z); sgaps.append(sgap)
    print(f"{ci:>4} {n:>4} | {lq_ref:>9.1f} {lam:>10.1f} +- {lsd:>5.1f} {z:>6.1f} | {U_ref:>7.1f} "
          f"{float(Ua.mean()):>8.1f} | {sgap:>+15.1f}", flush=True)
    ncav += 1
    if ncav >= 4:
        break
print(f"\nz = (logq0_ref - mean)/sd of alien logq0. z within ~+-2 => ref configs are NOT pointwise", flush=True)
print(f"under-weighted (wall = basin MASS + kernel reach). s_ref - max s_alien >> 0 => an independence", flush=True)
print(f"full-regen MH WOULD accept a ref proposal once drawn; barrier = draw frequency only.", flush=True)
print(f"summary: z mean {st.mean(zs):+.1f}; (s_ref - best alien s) mean {st.mean(sgaps):+.1f} nats", flush=True)
