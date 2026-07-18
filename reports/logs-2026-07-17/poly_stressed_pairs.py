"""Discriminator dataset: STRESSED-state relaxation pairs (half-lambda instant jump + 50-sweep
settle => state with measured 41% reproducible displacement signal) in poly_train_swapflow.py's
exact pair format (identity sigma path). If FM learns here but not on adiabatic escort pairs,
the escort failure is DISTRIBUTION (signal-poor checkpoints), not model capacity."""
import sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
sys.path.insert(0, "/mnt/ssd/GridTransformer/reports/logs-2026-07-17")
from poly_ncmc_v2 import local_sweep, build_local_set
from liquid_coupling_flow.poly.model import seed_numba

NP_ = int(sys.argv[1]) if len(sys.argv) > 1 else 800
D = torch.load("reports/logs-2026-07-17/poly_bank_fixed_run1.pt", weights_only=False)
D2 = torch.load("reports/logs-2026-07-17/poly_bank_fixed_run2.pt", weights_only=False)
recs = [D[(1, 0.085)], D2[(2, 0.085)]]
L = float(recs[0]["L"]); beta = 1 / 0.085
buf = np.zeros(300, dtype=np.int64)
rng = np.random.default_rng(11)
pairs = []
m = 0
while len(pairs) < NP_:
    m += 1
    rec = recs[m % 2]
    fr = rec["x"][(m // 2) % 16].astype(np.float64); s0 = rec["sigs"][(m // 2) % 16].astype(np.float64)
    i = int(rng.integers(300)); j = int(rng.integers(299)); j += (j >= i)
    if not (0.1 <= abs(s0[i] - s0[j]) < 0.9):
        continue
    x0 = fr.copy(); sg = s0.copy(); si, sj = sg[i], sg[j]
    lam = rng.uniform(0.3, 0.7)
    sg[i] = (1 - lam) * si + lam * sj; sg[j] = (1 - lam) * sj + lam * si
    seed_numba(20000 + m)
    nS = build_local_set(x0, i, j, 3.0, L, buf)
    for _ in range(50):
        local_sweep(x0, sg, L, beta, i, j, 3.0, 0.12, buf, nS)
    xr = x0.copy()
    seed_numba(30000 + m)
    nS2 = build_local_set(xr, i, j, 3.0, L, buf)
    for _ in range(100):
        local_sweep(xr, sg, L, beta, i, j, 3.0, 0.12, buf, nS2)
    mid = 0.5 * (x0[i] + x0[j]); dd = x0 - mid; dd -= L * np.round(dd / L)
    idx = np.argsort((dd ** 2).sum(1))[:8].astype(np.int64)
    mask = np.zeros(300, bool); mask[idx] = True
    cen = x0[idx].mean(0)
    def cent(a):
        b = a - cen
        return b - L * np.round(b / L)
    env_all = np.where(~mask)[0]
    de = x0[env_all] - cen; de -= L * np.round(de / L)
    env_idx = env_all[np.argsort((de ** 2).sum(1))[:64]]
    pairs.append({"x_old": cent(x0[idx]).astype(np.float32), "x_new": cent(xr[idx]).astype(np.float32),
                  "sig_start": sg[idx].copy(), "sig_end": sg[idx].copy(),
                  "env_x": cent(x0[env_idx]).astype(np.float32), "env_sig": sg[env_idx].copy(),
                  "ds": abs(si - sj), "L": L, "beta": beta})
torch.save({"pairs": pairs}, "reports/logs-2026-07-17/poly_stressedpairs_T0.085_k8.pt")
print(f"saved {len(pairs)} stressed pairs")
