"""PROBE: does volume-conserving pair exchange (continuous Kawasaki analog) hold the
composition shape, unlike the unconstrained semi-grand u-sweep (which deflates mean sigma
0.998 -> 0.82)?

Move: pick pair (i,j); v = sigma^3; v_i += eps, v_j -= eps (total particle volume conserved);
MH accept with exp(-beta dU + dlogprior), prior in v-space p_v(v) prop sigma^-5
(P(sigma) prop sigma^-3 times dsigma/dv = 1/(3 sigma^2)).
NOTE: unlike swap_sweep (sigma_ij symmetric under exchange -> pair(i,j) term cancels),
this move changes the (i,j) pair energy, so row_e(i)+row_e(j) double-counts it -- subtract
the pair term change once.

Frozen positions (sharpest shape-drift test). Clean bank frame, T=0.085.
"""
import sys
import numpy as np
import torch
from numba import njit
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.poly.model import row_e, pair_v, sigma_ij, total_U, seed_numba
from liquid_coupling_flow.poly.model import SIG_MIN, SIG_MAX

SMIN3, SMAX3 = SIG_MIN ** 3, SIG_MAX ** 3


@njit(cache=True)
def _pair_e(x, i, j, si, sj, L):
    dx = x[i, 0] - x[j, 0]; dy = x[i, 1] - x[j, 1]; dz = x[i, 2] - x[j, 2]
    dx -= L * np.round(dx / L); dy -= L * np.round(dy / L); dz -= L * np.round(dz / L)
    return pair_v(dx * dx + dy * dy + dz * dz, sigma_ij(si, sj))


@njit(cache=True)
def vex_sweep(x, sig, L, beta, dv, n_try):
    n = x.shape[0]
    acc = 0
    for _ in range(n_try):
        i = np.random.randint(n); j = np.random.randint(n)
        if i == j:
            j = (j + 1) % n
        si, sj = sig[i], sig[j]
        eps = dv * (2.0 * np.random.random() - 1.0)
        vi2 = si ** 3 + eps; vj2 = sj ** 3 - eps
        if vi2 <= SMIN3 or vi2 >= SMAX3 or vj2 <= SMIN3 or vj2 >= SMAX3:
            continue
        si2 = vi2 ** (1.0 / 3.0); sj2 = vj2 ** (1.0 / 3.0)
        e0 = (row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
              + row_e(x, sig, j, x[j, 0], x[j, 1], x[j, 2], L)
              - _pair_e(x, i, j, si, sj, L))
        sig[i] = si2; sig[j] = sj2
        e1 = (row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
              + row_e(x, sig, j, x[j, 0], x[j, 1], x[j, 2], L)
              - _pair_e(x, i, j, si2, sj2, L))
        dlogprior = -5.0 * (np.log(si2) + np.log(sj2) - np.log(si) - np.log(sj))
        if np.log(np.random.random() + 1e-300) < -beta * (e1 - e0) + dlogprior:
            acc += 1
        else:
            sig[i] = si; sig[j] = sj
    return acc, n_try


if __name__ == "__main__":
    D = torch.load("reports/logs-2026-07-17/poly_bank_fixed_run1.pt", weights_only=False)
    rec = D[(1, 0.085)]
    L = float(rec["L"])
    x = rec["x"][0].astype(np.float64).copy()
    sig = rec["sigs"][0].astype(np.float64).copy()
    N = len(sig); beta = 1.0 / 0.085
    sig0 = sig.copy()
    qs = [5, 25, 50, 75, 95]
    print(f"start: U/N={total_U(x,sig,L)/N:+.4f} mean(sig)={sig.mean():.4f} sum(v)={np.sum(sig**3):.3f}")
    print(f"       sig quantiles {qs}: {np.round(np.percentile(sig,qs),3)}")
    seed_numba(123)
    mob = 0.0
    for it in range(8):
        sig_before = sig.copy()
        acc, att = vex_sweep(x, sig, L, beta, 0.15, N * 50)
        mob = np.abs(sig - sig_before).mean()
        print(f"after {(it+1)*50} sw/p: U/N={total_U(x,sig,L)/N:+.4f} mean(sig)={sig.mean():.4f} "
              f"sum(v)={np.sum(sig**3):.3f} acc={acc/att:.3f} mean|dsig|/50sw={mob:.4f}")
    print(f"end quantiles: {np.round(np.percentile(sig,qs),3)}")
    print(f"start->end per-particle |dsig|: mean {np.abs(sig-sig0).mean():.4f} "
          f"p95 {np.percentile(np.abs(sig-sig0),95):.4f} (identity churn: >0 means real exchange)")
