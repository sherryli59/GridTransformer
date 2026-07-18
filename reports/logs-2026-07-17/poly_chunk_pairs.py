"""Deployment-matched CHUNK-STRESSED escort training pairs (PoC track, cost-blind).
State = immediately after an instant Delta-lambda=0.25 chunk switch, with realistic prior
protocol history (earlier chunks + R sweeps each); target = 100 local sweeps of relaxation at
the new fixed lambda. Identity sigma path (fixed-lambda kernel). Trainer format.
Usage: poly_chunk_pairs.py NPAIRS [R_sweeps_per_chunk]
"""
import sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
sys.path.insert(0, "/mnt/ssd/GridTransformer/reports/logs-2026-07-17")
from poly_ncmc_v2 import local_sweep, build_local_set
from liquid_coupling_flow.poly.model import seed_numba

NP_ = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
R = int(sys.argv[2]) if len(sys.argv) > 2 else 50
CHUNKS = [0.25, 0.5, 0.75, 1.0]
D1 = torch.load("reports/logs-2026-07-17/poly_bank_fixed_run1.pt", weights_only=False)
D2 = torch.load("reports/logs-2026-07-17/poly_bank_fixed_run2.pt", weights_only=False)
recs = [D1[(1, 0.085)], D2[(2, 0.085)]]
L = float(recs[0]["L"]); beta = 1 / 0.085
buf = np.zeros(300, dtype=np.int64)
rng = np.random.default_rng(21)
pairs = []
m = 0
while len(pairs) < NP_:
    m += 1
    rec = recs[m % 2]
    fr = rec["x"][(m // 2) % 16].astype(np.float64); s0 = rec["sigs"][(m // 2) % 16].astype(np.float64)
    i = int(rng.integers(300)); j = int(rng.integers(299)); j += (j >= i)
    if not (0.1 <= abs(s0[i] - s0[j]) < 0.9):
        continue
    dv = fr[i] - fr[j]; dv = dv - L * np.round(dv / L)
    if float((dv ** 2).sum()) ** 0.5 > 2.0:
        continue
    x = fr.copy(); sg = s0.copy(); si, sj = sg[i], sg[j]
    seed_numba(40000 + m)
    # pick which chunk boundary this pair samples (uniform over the 4)
    stop_at = int(rng.integers(len(CHUNKS)))
    ok = True
    for c, lam in enumerate(CHUNKS[:stop_at + 1]):
        sg[i] = (1 - lam) * si + lam * sj
        sg[j] = (1 - lam) * sj + lam * si
        if c < stop_at:                                   # history chunks: switch + R sweeps
            nS = build_local_set(x, i, j, 3.0, L, buf)
            for _ in range(R):
                local_sweep(x, sg, L, beta, i, j, 3.0, 0.12, buf, nS)
    # now: state IMMEDIATELY after chunk `stop_at`'s switch (no sweeps yet) = fresh stress
    x_before = x.copy()
    xr = x.copy()
    seed_numba(50000 + m)
    nS = build_local_set(xr, i, j, 3.0, L, buf)
    for _ in range(100):
        local_sweep(xr, sg, L, beta, i, j, 3.0, 0.12, buf, nS)
    mid = 0.5 * (x_before[i] + x_before[j]); dd = x_before - mid; dd -= L * np.round(dd / L)
    order = np.argsort((dd ** 2).sum(1))
    others = [m for m in order if m != i and m != j][:6]
    idx = np.array([i, j] + others, dtype=np.int64)          # pair always in block, no dupes                            # ensure pair in block
    mask = np.zeros(300, bool); mask[idx] = True
    cen = x_before[idx].mean(0)
    def cent(a):
        b = a - cen
        return b - L * np.round(b / L)
    env_all = np.where(~mask)[0]
    de = x_before[env_all] - cen; de -= L * np.round(de / L)
    env_idx = env_all[np.argsort((de ** 2).sum(1))[:64]]
    pairs.append({"x_old": cent(x_before[idx]).astype(np.float32),
                  "x_new": cent(xr[idx]).astype(np.float32),
                  "sig_start": sg[idx].copy(), "sig_end": sg[idx].copy(),
                  "env_x": cent(x_before[env_idx]).astype(np.float32),
                  "env_sig": sg[env_idx].copy(),
                  "ds": abs(si - sj), "chunk": stop_at, "L": L, "beta": beta})
    if len(pairs) % 500 == 0:
        torch.save({"pairs": pairs}, "reports/logs-2026-07-17/poly_chunkpairs_T0.085_k8.pt")
        print(f"{len(pairs)}/{NP_}", flush=True)
torch.save({"pairs": pairs}, "reports/logs-2026-07-17/poly_chunkpairs_T0.085_k8.pt")
print(f"CHUNK PAIRS DONE {len(pairs)}", flush=True)
