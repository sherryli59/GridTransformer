"""NCMC diameter-swap probe (v1, no NN): does nonequilibrium-candidate MC dissolve the
frozen-env insertion wall that capped ALL one-shot local proposals (phase-1 verdict)?

Move: pick pair (i,j) UNIFORMLY (state-independent selection => exactly symmetric); drive
sigma_i,sigma_j linearly to their swapped values over n_steps; after each lambda-switch,
propagate the WHOLE system with sweeps_per_step Metropolis displacement sweeps at the current
lambda-Hamiltonian (full-system propagation avoids mobile-set selection subtleties -- every
sub-kernel is detailed-balanced at its lambda). W = sum of instantaneous switching energies.
Acceptance of the composite (palindromic protocol, self-inverse endpoint swap, symmetric
selection) = min(1, e^{-beta W})  [Nilmeier et al., PNAS 2011].

n_steps=0 row = classical one-shot swap (W = vertical dU) -- the anchor that measured
0.44/0.026/0/0/0 across |dsig| bins. Phase-1 ORACLE (frozen-env relaxed endpoint) capped at
~1-2% in the dead bins; NCMC can exceed it because the environment relaxes INSIDE the move.

Per-attempt measurement from equilibrium starts (clean held-out bank frames), no chaining.
Records every attempt: ds, pair distance, W, acc_prob=min(1,e^{-bW}). Saves full arrays.
"""
import sys, time
import numpy as np
import torch
from numba import njit
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.poly.model import row_e, pair_v, sigma_ij, disp_sweep, seed_numba

BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.9)]
STEP = 0.12


@njit(cache=True)
def _pair_e(x, i, j, si, sj, L):
    dx = x[i, 0] - x[j, 0]; dy = x[i, 1] - x[j, 1]; dz = x[i, 2] - x[j, 2]
    dx -= L * np.round(dx / L); dy -= L * np.round(dy / L); dz -= L * np.round(dz / L)
    return pair_v(dx * dx + dy * dy + dz * dz, sigma_ij(si, sj))


@njit(cache=True)
def _pair_env_e(x, sig, i, j, L):
    """Energy of everything touching i or j (pair counted once)."""
    return (row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
            + row_e(x, sig, j, x[j, 0], x[j, 1], x[j, 2], L)
            - _pair_e(x, i, j, sig[i], sig[j], L))


@njit(cache=True)
def ncmc_work(x, sig, L, beta, i, j, n_steps, sweeps_per_step, step):
    """Drive sigma_i,sigma_j to swapped values over n_steps; return accumulated switching work.
    n_steps==0: instantaneous swap (classical vertical work)."""
    si0 = sig[i]; sj0 = sig[j]
    if n_steps == 0:
        e0 = _pair_env_e(x, sig, i, j, L)
        sig[i] = sj0; sig[j] = si0
        e1 = _pair_env_e(x, sig, i, j, L)
        return e1 - e0
    W = 0.0
    for k in range(n_steps):
        lam = (k + 1.0) / n_steps
        e0 = _pair_env_e(x, sig, i, j, L)
        sig[i] = (1.0 - lam) * si0 + lam * sj0
        sig[j] = (1.0 - lam) * sj0 + lam * si0
        e1 = _pair_env_e(x, sig, i, j, L)
        W += e1 - e0
        for _ in range(sweeps_per_step):
            disp_sweep(x, sig, L, beta, step)
    return W


if __name__ == "__main__":
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.085
    beta = 1.0 / T
    D = torch.load("reports/logs-2026-07-17/poly_bank_fixed_run3.pt", weights_only=False)
    rec = D[(3, T)]
    L = float(rec["L"]); N = rec["x"].shape[1]
    out = {"T": T, "grid": {}}
    rng = np.random.default_rng(5)
    N_GRID = [(0, 60), (10, 60), (50, 60), (200, 40), (800, 25)]   # (n_steps, attempts/bin)
    for n_steps, per_bin in N_GRID:
        t0 = time.time()
        res = {b: {"W": [], "ds": []} for b in BINS}
        need = {b: per_bin for b in BINS}
        seed_numba(4000 + n_steps)
        guard = 0
        while any(v > 0 for v in need.values()) and guard < 20000:
            guard += 1
            fr_j = guard % 16
            i = int(rng.integers(N)); j = int(rng.integers(N - 1))
            if j >= i:
                j += 1
            x0 = rec["x"][fr_j].astype(np.float64)
            s0 = rec["sigs"][fr_j].astype(np.float64)
            ds = abs(float(s0[i] - s0[j]))
            bb = next((b for b in BINS if b[0] <= ds < b[1]), None)
            if bb is None or need[bb] <= 0:
                continue
            need[bb] -= 1
            x = x0.copy(); sig = s0.copy()
            W = ncmc_work(x, sig, L, beta, i, j, n_steps, 1, STEP)
            res[bb]["W"].append(W); res[bb]["ds"].append(ds)
        out["grid"][n_steps] = {str(b): {k: np.array(v) for k, v in r.items()}
                                for b, r in res.items()}
        line = [f"n={n_steps:>4}"]
        for b in BINS:
            W = np.array(res[b]["W"])
            if len(W):
                acc = np.minimum(1.0, np.exp(np.clip(-beta * W, -700, 700)))
                line.append(f"{b}: acc={acc.mean():.3f} (Wmed {np.median(W):+.2f}, n={len(W)})")
        print(f"[{time.time()-t0:6.0f}s] " + " | ".join(line), flush=True)
        torch.save(out, f"reports/logs-2026-07-17/poly_ncmc_probe_T{T}.pt")
    print("NCMC PROBE DONE", flush=True)
