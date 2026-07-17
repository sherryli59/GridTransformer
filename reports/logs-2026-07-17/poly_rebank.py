"""Regenerate CONSISTENT (x, sig) frame banks from the original poly_tau_curves_run{r}.pt rungs.

WHY: the original bank collected 16 x-frames while swap_sweep kept permuting sig between
captures, but saved only the END-of-rung sig -> frames 0-14 pair positions with a stale
sigma-assignment (measured U/N +13 vs +0.32 equilibrium; only frame 15 is consistent).
Fix: restart each rung's MC from its one consistent pair (x[15], sig), re-collect 16
swap-decorrelated frames SAVING sig PER FRAME.

Usage: poly_rebank.py RUN            (one process per run; incremental save per rung)
Output: poly_bank_fixed_run{RUN}.pt with D[(run,T)] = {'x':[16,N,3], 'sigs':[16,N],
        'L','T','tau_swap','U_N_frames'}; consumers pair x[j] with sigs[j].
"""
import sys, time
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.poly.model import disp_sweep, swap_sweep, total_U, seed_numba

RUN = int(sys.argv[1])
SRC = f"reports/logs-2026-07-17/poly_tau_curves_run{RUN}.pt"
OUT = f"reports/logs-2026-07-17/poly_bank_fixed_run{RUN}.pt"
TEMPS = [0.2, 0.1, 0.085, 0.075, 0.065, 0.058, 0.052, 0.047]   # gate-relevant rungs
STEP = 0.12
N_FRAMES = 16

src = torch.load(SRC, map_location="cpu", weights_only=False)
out = {}
for T in TEMPS:
    key = (RUN, T)
    if key not in src:
        print(f"T={T}: not in {SRC} yet -- skipped", flush=True)
        continue
    rec = src[key]
    L = float(rec["L"])
    N = rec["x"].shape[1]
    x = rec["x"][-1].astype(np.float64).copy()      # the ONE consistent frame
    sig = rec["sig"].astype(np.float64).copy()
    beta = 1.0 / T
    u0 = total_U(x, sig, L) / N
    if not np.isfinite(u0) or u0 > 2.0:
        print(f"T={T}: SANITY FAIL last-frame U/N={u0:+.3f} -- skipped", flush=True)
        continue
    tau_s = int(rec["tau_swap"])
    spacing = max(1000, 3 * max(tau_s, 1))
    seed_numba(1234 + RUN * 100 + int(T * 1000))
    frames, fsigs, us = [], [], []
    t0 = time.time()
    for j in range(N_FRAMES):
        for _ in range(spacing):
            disp_sweep(x, sig, L, beta, STEP)
            swap_sweep(x, sig, L, beta, N)
        frames.append(x.copy())
        fsigs.append(sig.copy())
        us.append(total_U(x, sig, L) / N)
    out[key] = {"x": np.stack(frames), "sigs": np.stack(fsigs), "L": L, "T": T,
                "tau_swap": tau_s, "U_N_frames": np.array(us)}
    torch.save(out, OUT)
    print(f"run {RUN} T={T}: 16 frames @ spacing {spacing}, U/N {np.min(us):+.4f}..{np.max(us):+.4f} "
          f"(start {u0:+.4f}) ({time.time()-t0:.0f}s)", flush=True)
print(f"REBANK DONE run {RUN} -> {OUT}", flush=True)
