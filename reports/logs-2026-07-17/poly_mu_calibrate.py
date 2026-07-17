"""poly Task P2a: Boltzmann-inversion calibration of mu(sigma) for the semi-grand u-kernel.

Per docs/superpowers/specs/2026-07-17-joint-sigma-x-continuous-flow-amendment.md PHASE 2 KICKOFF:
without mu, the semi-grand u-sweep DEFLATES the composition (energetic tilt beats the sigma^-3
prior). mu(sigma) is a NBINS_MU-bin lookup, iteratively updated so the stationary per-particle
sigma-marginal under mixed {disp + swap + u_sweep_mu} dynamics matches the canonical composition
P(sigma) ~ sigma^-3 (mu_bin_target_mass in liquid_coupling_flow/poly/semigrand.py, exact via the
CDF, not sampled). Swaps are mu-neutral (permutations only permute existing sigma values, they
never change the composition) so they stay enabled throughout calibration purely to keep local
mixing fast; only u_sweep_mu moves the composition.

Update rule (Boltzmann inversion): mu_bin += eta * T * log(P_target_bin / P_measured_bin), then
mean-centered against the P_target-weighted mean (mu is defined up to an additive constant --
must be pinned down every iteration or it drifts, see semigrand.py's u_sweep_mu accept ratio,
which only ever sees DIFFERENCES mu(sig_new)-mu(sig_old)).

Usage:
    python reports/logs-2026-07-17/poly_mu_calibrate.py --T 0.2
    python reports/logs-2026-07-17/poly_mu_calibrate.py --T 0.1 \
        --mu_init liquid_coupling_flow/artifacts/poly_mu_T0.2.pt
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/ssd/GridTransformer")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from liquid_coupling_flow.poly.model import disp_sweep, swap_sweep, total_U, seed_numba
from liquid_coupling_flow.poly.semigrand import (u_of_sigma, u_sweep_mu, mu_bin_edges,
                                                  mu_bin_target_mass, NBINS_MU)

DISP_STEP = 0.12          # matches reports/logs-2026-07-17/poly_bank.py's ladder convention
DELTA_U_DEFAULT = 0.15
FLOOR = 1e-4
GATE_FRAC = 0.03           # convergence gate: sup relative-to-target bin error < 3%
GATE_STREAK = 3            # ... sustained this many consecutive iterations


def calibrate_mu(x, sig, u, L, beta, T, iters, sweeps_per_iter, eta, mu_init=None,
                  nbins=NBINS_MU, delta_u=DELTA_U_DEFAULT, floor=FLOOR, gate_frac=GATE_FRAC,
                  gate_streak=GATE_STREAK, disp_step=DISP_STEP, save_path=None, save_extra=None,
                  verbose=True):
    """Run the Boltzmann-inversion mu(sigma) calibration loop in place on (x, sig, u) (mutated).
    Returns a dict with the final mu, per-iteration history, and the (possibly early-stopped)
    convergence iteration. If save_path is given, {mu, history, T, ...save_extra} is checkpointed
    to save_path after EVERY iteration (never only at the end)."""
    n = x.shape[0]
    edges = mu_bin_edges(nbins)
    target = mu_bin_target_mass(nbins)
    mu = np.zeros(nbins) if mu_init is None else np.array(mu_init, dtype=np.float64).copy()
    assert mu.shape == (nbins,), mu.shape

    history = []
    streak = 0
    converged_at = None
    for it in range(iters):
        half = sweeps_per_iter // 2
        sig_samples = []
        u_acc_tot = u_att_tot = 0
        swap_acc_tot = swap_att_tot = 0
        for sw in range(sweeps_per_iter):
            disp_sweep(x, sig, L, beta, disp_step)
            sa, st = swap_sweep(x, sig, L, beta, n)
            swap_acc_tot += sa; swap_att_tot += st
            ua, ut = u_sweep_mu(x, sig, u, L, beta, delta_u, n, mu)
            u_acc_tot += ua; u_att_tot += ut
            if sw >= half:                                  # 2nd half = post-burn-in sampling
                sig_samples.append(sig.copy())
        sig_samples = np.concatenate(sig_samples)

        meas, _ = np.histogram(sig_samples, bins=edges)
        meas = meas / meas.sum()
        abs_err = meas - target
        sup_abs = float(np.max(np.abs(abs_err)))
        l1 = float(np.sum(np.abs(abs_err)))
        frac_err = np.abs(abs_err) / np.maximum(target, floor)
        frac_sup = float(np.max(frac_err))

        # Boltzmann-inversion update (uses the CURRENT mu's measured composition), then mean-center
        mu = mu + eta * T * np.log(target / np.maximum(meas, floor))
        mu = mu - np.sum(mu * target)                       # pin the additive constant

        mean_sig = float(sig.mean())
        u_n = float(total_U(x, sig, L) / n)
        u_acc = u_acc_tot / max(u_att_tot, 1)
        swap_acc = swap_acc_tot / max(swap_att_tot, 1)

        rec = dict(iter=it, sup_abs=sup_abs, l1=l1, frac_sup=frac_sup, mean_sigma=mean_sig,
                   U_N=u_n, u_acc=u_acc, swap_acc=swap_acc)
        history.append(rec)
        if verbose:
            print(f"iter {it:3d}: sup_abs={sup_abs:.4f} L1={l1:.4f} frac_sup={frac_sup*100:6.2f}% "
                  f"mean_sig={mean_sig:.4f} U/N={u_n:+.4f} u_acc={u_acc:.3f} swap_acc={swap_acc:.3f}",
                  flush=True)

        if save_path is not None:
            payload = {"mu": mu.copy(), "history": history, "T": T, "converged_at": converged_at}
            if save_extra:
                payload.update(save_extra)
            torch.save(payload, save_path)

        if frac_sup < gate_frac:
            streak += 1
        else:
            streak = 0
        if streak >= gate_streak and converged_at is None:
            converged_at = it
            if verbose:
                print(f"CONVERGED at iter {it} (frac_sup < {gate_frac*100:.0f}% for "
                      f"{gate_streak} consecutive iterations)", flush=True)
            if save_path is not None:
                payload = {"mu": mu.copy(), "history": history, "T": T,
                           "converged_at": converged_at}
                if save_extra:
                    payload.update(save_extra)
                torch.save(payload, save_path)
            break

    return dict(mu=mu, history=history, converged_at=converged_at, x=x, sig=sig, u=u)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--sweeps_per_iter", type=int, default=3000)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--mu_init", type=str, default=None)
    ap.add_argument("--frames", type=str,
                     default="reports/logs-2026-07-17/poly_bank_fixed_run1.pt")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", type=int, default=1, help="bank run index (key = (run, T))")
    args = ap.parse_args()

    frames_path = Path(args.frames)
    assert frames_path.exists(), f"--frames {frames_path} does not exist"
    bank = torch.load(frames_path, weights_only=False)
    key = (args.run, args.T)
    assert key in bank, f"key {key} not in {frames_path} (keys: {[k for k in bank if isinstance(k, tuple)]})"
    rec = bank[key]
    x_all, sig_all, L = rec["x"], rec["sigs"], rec["L"]
    n_bank = x_all.shape[1]
    assert n_bank == args.n, f"--n {args.n} != bank frame N {n_bank} (bank frames are fixed-N)"

    x = np.ascontiguousarray(x_all[0], dtype=np.float64)
    sig = np.ascontiguousarray(sig_all[0], dtype=np.float64)
    u = u_of_sigma(sig)
    beta = 1.0 / args.T

    mu_init = None
    if args.mu_init:
        prev = torch.load(args.mu_init, weights_only=False)
        mu_init = prev["mu"]
        print(f"warm-starting mu from {args.mu_init} (T={prev.get('T')})", flush=True)

    seed_numba(args.seed)

    out_path = Path(args.out) if args.out else REPO / f"liquid_coupling_flow/artifacts/poly_mu_T{args.T}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"calibrating mu(sigma) at T={args.T} (beta={beta:.4f}), N={args.n}, "
          f"iters={args.iters}, sweeps_per_iter={args.sweeps_per_iter}, eta={args.eta} "
          f"-> {out_path}", flush=True)
    t0 = time.time()
    result = calibrate_mu(x, sig, u, L, beta, args.T, args.iters, args.sweeps_per_iter, args.eta,
                           mu_init=mu_init, save_path=str(out_path),
                           save_extra={"args": vars(args)})
    print(f"done in {time.time()-t0:.0f}s; converged_at={result['converged_at']}; "
          f"saved to {out_path}", flush=True)


if __name__ == "__main__":
    _main()
