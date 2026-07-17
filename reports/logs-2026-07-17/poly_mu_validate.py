"""Fixed-mu validation: freeze mu, run LONG mixed MC (disp+swap+u_mu), measure the stationary
composition with ~10x the per-iteration statistics of the calibration loop. Decides
converged-within-noise (the iterative loop's frac_sup floor ~8-15% is per-iteration sampling
noise on a 32-bin sup metric, not mu bias).
Usage: poly_mu_validate.py --T 0.2 [--mu path] [--sweeps 120000] [--run 1] [--out path]
"""
import argparse, sys, time
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.poly.model import disp_sweep, swap_sweep, total_U, seed_numba
from liquid_coupling_flow.poly.semigrand import u_sweep_mu, u_of_sigma, SIG_MIN, SIG_MAX

p = argparse.ArgumentParser()
p.add_argument("--T", type=float, required=True)
p.add_argument("--mu", type=str, default=None)
p.add_argument("--sweeps", type=int, default=120000)
p.add_argument("--run", type=int, default=1)
p.add_argument("--delta", type=float, default=0.15)
p.add_argument("--out", type=str, default=None)
a = p.parse_args()
mu_path = a.mu or f"liquid_coupling_flow/artifacts/poly_mu_T{a.T}.pt"
M = torch.load(mu_path, weights_only=False)
mu = np.asarray(M["mu"], dtype=np.float64)
bank = torch.load(f"reports/logs-2026-07-17/poly_bank_fixed_run{a.run}.pt", weights_only=False)
rec = bank[(a.run, a.T)]
x = rec["x"][0].astype(np.float64).copy()
sig = rec["sigs"][0].astype(np.float64).copy()
u = u_of_sigma(sig)
L = float(rec["L"]); N = len(sig); beta = 1.0 / a.T

aa, bb = SIG_MIN ** -2, SIG_MAX ** -2
edges = np.linspace(SIG_MIN, SIG_MAX, 33)
F = lambda s: (aa - s ** -2) / (aa - bb)
p_target = F(edges[1:]) - F(edges[:-1])

seed_numba(777)
hist = np.zeros(32)
n_samp = 0
BURN = a.sweeps // 6
t0 = time.time()
for sw in range(1, a.sweeps + 1):
    disp_sweep(x, sig, L, beta, 0.12)
    swap_sweep(x, sig, L, beta, N)
    u_sweep_mu(x, sig, u, L, beta, a.delta, N, mu)
    if sw > BURN and sw % 20 == 0:
        h, _ = np.histogram(sig, bins=edges)
        hist += h; n_samp += N
p_meas = hist / max(hist.sum(), 1)
rel = np.abs(p_meas - p_target) / p_target
print(f"T={a.T} fixed-mu validation: {a.sweeps} sweeps, {n_samp} sigma-samples "
      f"({time.time()-t0:.0f}s)")
print(f"  frac_sup = {rel.max():.4f}  (bin {int(rel.argmax())}, target mass {p_target[rel.argmax()]:.4f})")
print(f"  L1 = {np.abs(p_meas-p_target).sum():.4f}   mean_sigma = {sig.mean():.4f} (target ~0.998)")
print(f"  U/N = {total_U(x, sig, L)/N:+.4f}")
print(f"  VERDICT: {'CONVERGED (<5%)' if rel.max() < 0.05 else 'NOT converged'}")
out = a.out or f"reports/logs-2026-07-17/poly_mu_validate_T{a.T}.pt"
torch.save({"p_meas": p_meas, "p_target": p_target, "rel": rel, "mu": mu, "T": a.T,
            "sweeps": a.sweeps, "mean_sigma": sig.mean(), "U_N": total_U(x, sig, L)/N}, out)
print(f"saved -> {out}")
