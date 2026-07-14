"""Is the island logZ spread NUMERICAL (under-annealing -> fixable) or PHYSICAL (basin-evidence diffs)?
Fix ONE cavity (so true logZ is a single number => island spread = pure AIS estimator variance) and run
J=8 independent islands, sweeping the annealing resolution T (and n_mut). Report std(logZ_j), spread
(max-min), mean ESS, time. If std(logZ_j) SHRINKS with T => numerical (more rungs / adaptive schedule fixes
the winner-take-all). If it plateaus large => physical/mixing-limited (need better mutation or PT-coupling,
or report per-island q +/- spread). Uses the sweep driver's island(). RL+demand ckpt, R=2.0, 1blob-K8."""
import sys, statistics as st, time
import torch
sys.path.insert(0, "reports/logs-2026-07-14")
from ka3d_smc_sweep import island, energy_b
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; RCTX = 2.5; R = 2.0; ART = "liquid_coupling_flow/artifacts"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_demand_knn24_best.pt", map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])

# fix ONE cavity
g = torch.Generator(device=dev).manual_seed(0)
c = torch.rand(3, generator=g, device=dev) * L; p = carve(X[0], S[0], c, R, L)
xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
print(f"=== logZ estimator variance vs annealing resolution (fixed cavity, n_in={xo.shape[0]}, J=8 islands) ===", flush=True)
cfg = {"move": "1blob", "K": 8, "swap": False}
print(f"{'T':>4} {'n_mut':>5} | {'mean logZ':>10} {'std logZ':>9} {'spread':>8} {'mean ESS':>9} {'time':>6}", flush=True)
for T, nmut in [(14, 3), (28, 3), (56, 3), (112, 3), (56, 8)]:
    t0 = time.time(); lz = []; es = []
    for j in range(8):
        gg = torch.Generator(device=dev).manual_seed(2000 + j)
        _, _, logZ, esses = island(m, xo, so, bnd, sb, R, cfg, 16, T, nmut, gg)
        lz.append(float(logZ)); es.append(st.mean(esses))
    print(f"{T:>4} {nmut:>5} | {st.mean(lz):>10.1f} {st.pstdev(lz):>9.1f} {max(lz)-min(lz):>8.1f} "
          f"{st.mean(es):>9.2f} {time.time()-t0:>5.0f}s", flush=True)
print("\n  std logZ SHRINKS with T => numerical (fixable: more rungs / adaptive schedule); PLATEAUS => mixing"
      "\n  limited (better mutation / PT-couple / report per-island +/- spread).", flush=True)
