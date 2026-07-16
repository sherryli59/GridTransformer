# reports/logs-2026-07-17/poly_bank.py
"""Descending-T swap-equilibrated banks for the NBC polydisperse model + tau_alpha(T) for
local-only vs local+swap dynamics. Locates the OPERATIONAL SWAP ARREST T* = lowest rung where
tau_swap exceeds BUDGET sweeps. Each rung hands its final configs to the next (swap-equilibrated
cooling). Incremental save per rung. Usage: poly_bank.py [N] [BUDGET]"""
import sys, time
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.poly.model import (draw_sigmas, disp_sweep, swap_sweep, total_U,
                                             seed_numba)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000_000
LADDER = [0.30, 0.20, 0.15, 0.12, 0.10, 0.085, 0.075, 0.065, 0.058, 0.052, 0.047]
N_RUNS = 4                       # independent runs (90/10 held-out split is BY RUN)
EQ_SW = 200_000                  # per-rung equilibration before tau measurement (adaptive: 20*tau_prev)
STEP = 0.12
A_OV = 0.3
L = N ** (1.0 / 3.0)
seed_numba(11)


def q_self(x, x0, L):
    d = x - x0
    d -= L * np.round(d / L)
    return float((np.sqrt((d ** 2).sum(1)) < A_OV).mean())


def tau_alpha(x, sig, L, beta, use_swap, budget):
    """Return first t (sweeps) with Q(t) < 1/e, or -1 if not reached within budget. Logs Q(t)."""
    x0 = x.copy()
    t = 0
    ts, qs = [], []
    next_meas = 1
    while t < budget:
        disp_sweep(x, sig, L, beta, STEP)
        if use_swap:
            swap_sweep(x, sig, L, beta, N)
        t += 1
        if t >= next_meas:
            q = q_self(x, x0, L)
            ts.append(t); qs.append(q)
            if q < np.exp(-1.0):
                return t, ts, qs
            next_meas = int(next_meas * 1.3) + 1
    return -1, ts, qs


out = {"LADDER": LADDER, "N": N, "BUDGET": BUDGET}
first_rung = True
for run in range(N_RUNS):
    rng = np.random.default_rng(run)
    x = rng.random((N, 3)) * L
    sig = draw_sigmas(N, seed=100 + run)
    for T in LADDER:
        beta = 1.0 / T
        t0 = time.time()
        for _ in range(EQ_SW):
            disp_sweep(x, sig, L, beta, STEP)
            swap_sweep(x, sig, L, beta, N)
        eq_elapsed = time.time() - t0
        if first_rung:
            # sweeps/sec at the first rung, so the controller can sanity the wall-clock math for
            # the full-bank launch (BUDGET is already enforced in sweeps by tau_alpha itself).
            rate = EQ_SW / max(eq_elapsed, 1e-9)
            print(f"[rate probe] eq sweeps/sec (local+swap) = {rate:.1f} "
                  f"(BUDGET={BUDGET} sweeps ~= {BUDGET / max(rate, 1e-9) / 3600.0:.2f}h at this rate)",
                  flush=True)
            first_rung = False
        frames = []
        # tau_alpha operates on copies of x/sig -- these are side probes off the main ladder
        # trajectory; the ladder's real x/sig keep evolving below through frame collection.
        tau_s, ts_s, qs_s = tau_alpha(x.copy(), sig.copy(), L, beta, True, BUDGET)
        tau_l, ts_l, qs_l = tau_alpha(x.copy(), sig.copy(), L, beta, False, min(BUDGET, 2_000_000))
        for _ in range(16):                              # bank frames, swap-decorrelated
            for _ in range(max(1000, 3 * max(tau_s, 1))):
                disp_sweep(x, sig, L, beta, STEP)
                swap_sweep(x, sig, L, beta, N)
            frames.append(x.copy())
        out[(run, T)] = {"x": np.stack(frames), "sig": sig.copy(), "L": L, "T": T,
                         "tau_swap": tau_s, "tau_local": tau_l,
                         "Q_swap": (ts_s, qs_s), "Q_local": (ts_l, qs_l),
                         "U_N": total_U(x, sig, L) / N}
        torch.save(out, "reports/logs-2026-07-17/poly_tau_curves.pt")
        print(f"run {run} T={T}: tau_swap={tau_s} tau_local={tau_l} U/N={out[(run,T)]['U_N']:+.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if tau_s < 0:
            print(f"run {run}: SWAP ARREST at T={T} (tau_swap > {BUDGET})", flush=True)
            break
print("BANK DONE", flush=True)
