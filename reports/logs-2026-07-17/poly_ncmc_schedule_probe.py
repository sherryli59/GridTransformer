"""Dissipation-equalized lambda-schedule A/B probe for NCMC v2.

v2 measured dW(lambda) strongly BACK-LOADED (~0.7% of work in the first decile -> ~17% in the
last). Optimal-protocol theory (linear response): dissipation is minimized when each step incurs
equal work -> schedule = inverse of the cumulative mean dW(lambda) curve. Build that schedule
from v2's measured profiles (r=3.0, n=800, dead bins pooled) and A/B it against the linear
schedule at the same n on fresh attempts.
"""
import sys, time
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
sys.path.insert(0, "/mnt/ssd/GridTransformer/reports/logs-2026-07-17")
from poly_ncmc_v2 import ncmc_work_local  # noqa: E402
from liquid_coupling_flow.poly.model import seed_numba  # noqa: E402

T = 0.085
BETA = 1.0 / T
BINS = [(0.2, 0.3), (0.3, 0.45), (0.45, 0.9)]
N_STEPS = 800
R_LOC = 3.0
PER_BIN = 90

# --- build the equalized schedule from measured profiles (r=3.0, n=800, dead bins pooled) ---
V = torch.load("reports/logs-2026-07-17/poly_ncmc_v2_T0.085.pt", weights_only=False)
profs = []
for b in BINS:
    cell = V["grid"][(3.0, 800)][str(b)]
    profs.append(np.asarray(cell["dW_profile"]).mean(axis=0))
mean_prof = np.mean(profs, axis=0)                     # mean dW per linear step, len 800
mean_prof = np.maximum(mean_prof, 1e-12)
lam_lin = np.arange(1, 801) / 800.0                    # linear grid the profile lives on
cum = np.cumsum(mean_prof); cum /= cum[-1]             # cumulative work fraction vs lambda
# equal-dissipation schedule: pick lambdas at equal cumulative-work quantiles
q = (np.arange(1, N_STEPS + 1)) / N_STEPS
sched_eq = np.interp(q, cum, lam_lin)
sched_eq[-1] = 1.0
sched_eq = np.maximum.accumulate(sched_eq)             # enforce monotone
sched_lin = np.arange(1, N_STEPS + 1) / N_STEPS
print(f"equalized schedule: lambda at step 400/800 = {sched_eq[400]:.3f} (linear: 0.500); "
      f"first decile covers lambda 0..{sched_eq[N_STEPS//10]:.3f}", flush=True)

# --- A/B on fresh attempts ---
D = torch.load("reports/logs-2026-07-17/poly_bank_fixed_run3.pt", weights_only=False)
rec = D[(3, T)]
L = float(rec["L"]); N = rec["x"].shape[1]
rng = np.random.default_rng(77)
out = {"sched_eq": sched_eq, "results": {}}
for name, sched in (("linear", sched_lin), ("equalized", sched_eq)):
    t0 = time.time()
    res = {b: [] for b in BINS}
    need = {b: PER_BIN for b in BINS}
    seed_numba(6100 + (0 if name == "linear" else 1))
    dW_buf = np.zeros(N_STEPS); nS_buf = np.zeros(N_STEPS, dtype=np.int64)
    guard = 0
    while any(v > 0 for v in need.values()) and guard < 30000:
        guard += 1
        fr = guard % 16
        i = int(rng.integers(N)); j = int(rng.integers(N - 1))
        if j >= i:
            j += 1
        s0 = rec["sigs"][fr].astype(np.float64)
        ds = abs(float(s0[i] - s0[j]))
        bb = next((b for b in BINS if b[0] <= ds < b[1]), None)
        if bb is None or need[bb] <= 0:
            continue
        need[bb] -= 1
        x = rec["x"][fr].astype(np.float64).copy()
        sig = s0.copy()
        W = ncmc_work_local(x, sig, L, BETA, i, j, sched, 1, R_LOC, 0.12, dW_buf, nS_buf)
        res[bb].append(W)
    out["results"][name] = {str(b): np.array(v) for b, v in res.items()}
    line = [f"{name:>10}"]
    for b in BINS:
        W = np.array(res[b])
        acc = np.minimum(1, np.exp(np.clip(-BETA * W, -700, 0)))
        line.append(f"{b}: Wmed {np.median(W):+.2f} acc {acc.mean():.4f} (n={len(W)})")
    print(f"[{time.time()-t0:5.0f}s] " + " | ".join(line), flush=True)
    torch.save(out, "reports/logs-2026-07-17/poly_ncmc_schedule_probe_T0.085.pt")
print("SCHEDULE PROBE DONE", flush=True)
