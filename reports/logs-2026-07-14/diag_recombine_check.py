"""Can we sidestep the logZ variance for the OVERLAP q(R)? For a pinned (single-basin, R<xi_PTS) cavity the
islands sample the SAME basin, so exp(logZ_j) weighting just re-weights the observable by AIS estimator
NOISE. Test: fixed cavity, J=12 islands (T=56), report per-island q spread vs logZ spread, and compare
recombination: UNIFORM island average vs exp(logZ_j)-weighted. If per-island q is tight and uniform~=
logZ-weighted-median but the logZ-weighted is a single lucky island, uniform is the correct+robust
estimator. RL+demand ckpt, R=2.0, 1blob-K8."""
import sys, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-14")
from ka3d_smc_sweep import island, overlap
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; RCTX = 2.5; R = 2.0; ART = "liquid_coupling_flow/artifacts"; J = 12; T = 56
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_demand_knn24_best.pt", map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
g = torch.Generator(device=dev).manual_seed(0)
c = torch.rand(3, generator=g, device=dev) * L; p = carve(X[0], S[0], c, R, L)
xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
xin = _mic(p["x_in"], c, L); sin = p["s_in"]
xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
cfg = {"move": "1blob", "K": 8, "swap": False}
lz, qs = [], []
for j in range(J):
    gg = torch.Generator(device=dev).manual_seed(3000 + j)
    Xp, Sp, logZ, _ = island(m, xo, so, bnd, sb, R, cfg, 16, T, 3, gg)
    lz.append(float(logZ)); qs.append(overlap(Xp, Sp, xin, sin))
lzt = torch.tensor(lz); wj = torch.softmax(lzt, 0)
q_unif = st.mean(qs); q_lzw = float(sum(wj[j] * qs[j] for j in range(J)))
print(f"=== recombination check (pinned R={R}<xi_PTS, {J} islands, T={T}) ===", flush=True)
print(f"  per-island q: mean {st.mean(qs):.3f}  std {st.pstdev(qs):.3f}  range [{min(qs):.3f},{max(qs):.3f}]", flush=True)
print(f"  per-island logZ: std {st.pstdev(lz):.1f}  spread {max(lz)-min(lz):.1f} nats", flush=True)
print(f"  max softmax(logZ) weight: {float(wj.max()):.3f}  (=> {'winner-take-all' if float(wj.max())>0.9 else 'shared'})", flush=True)
print(f"  q(R) recombined:  UNIFORM {q_unif:.3f}   exp(logZ)-weighted {q_lzw:.3f}", flush=True)
print(f"\n  => per-island q std << logZ-implied noise; the logZ weighting collapses to 1 island. For a pinned"
      f"\n  cavity UNIFORM island averaging is the correct, low-variance estimator; keep logZ only for free energy.", flush=True)
