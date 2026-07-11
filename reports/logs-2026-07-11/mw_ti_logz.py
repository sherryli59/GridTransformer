"""Independent logZ arbiter via THERMODYNAMIC INTEGRATION: logZ(beta) = 3N log L + int_0^beta <-U>_b db.
Plain single-site MCMC (pi_b-invariant, well-mixed) at a dense b-grid, warm-started rung to rung;
trapezoid + refinement estimate. No weights, no resampling, no adaptive schedule -- methodologically
orthogonal to BOTH SMC estimates (certified uniform-path 1198.1 vs flow-path ~1222)."""
import math, sys, torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.mw.mw_smc import mutation_sweeps
from liquid_coupling_flow.mw.mw_base import UniformBase
from liquid_coupling_flow.mw.mw_energy import mw_energy, T_STAR, RHO_STAR

DEV = "cuda"; N, B = 64, 64
L = (N / RHO_STAR) ** (1 / 3); beta_t = 1.0 / T_STAR
base = UniformBase(N, L)
# dense at low b where <U> varies fastest
bs = [0.0, 0.02, 0.05, 0.1, 0.15, 0.22, 0.32, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 3.9, 5.4, 7.5, beta_t]
EQ_SW, ME_SW = 50, 30
res = {}
for seed in (1, 2):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = base.sample(B, g)
    us = []
    for b in bs:
        if b > 0:
            x, U, _, _ = mutation_sweeps(x, base, 1.0, b, L, EQ_SW, 0.0783 if b > 0.5 else 0.15, g)
            acc = []
            for _ in range(ME_SW):
                x, U, _, _ = mutation_sweeps(x, base, 1.0, b, L, 1, 0.0783 if b > 0.5 else 0.15, g)
                acc.append(float(U.mean()))
            u = sum(acc) / len(acc)
        else:
            u = float(mw_energy(x, L).mean())
        us.append(u)
        print(f"seed {seed} b {b:6.3f}  <U> {u:+9.3f}", flush=True)
    # trapezoid of <-U> db
    I = sum(0.5 * (bs[i+1] - bs[i]) * (-(us[i]) - us[i+1]) for i in range(len(bs) - 1))
    lz = 3 * N * math.log(L) + I
    res[seed] = lz
    print(f"seed {seed}: logZ(TI) = {lz:.2f}   [certified-SMC 1198.1 | flow-path SMC ~1222]", flush=True)
print(f"\nTI verdict: {res}", flush=True)
