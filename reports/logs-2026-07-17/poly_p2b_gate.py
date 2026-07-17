"""poly Task P2b: kernel-mix relaxation + sigma-mobility GATE (HARNESS ONLY -- build+smoke; the
REAL run is deferred to the controller until a converged mu(sigma) artifact exists for the target T).

Per docs/superpowers/specs/2026-07-17-joint-sigma-x-continuous-flow-amendment.md PHASE 2 KICKOFF,
P2b ordering note: "P2b = THE GATE: sigma-mobility and tau_alpha with kernel mixes {disp+swap} vs
{disp+u(mu)} vs {disp+swap+u(mu)} at bracket temps incl. below arrest -- does gradual-sigma extend
relaxation where discrete swap arrests?"

THE MEASUREMENT: from a single CLEAN bank frame (poly_bank_fixed_run{run}.pt, frame j paired with
sigs[j] -- see poly_bank.py/poly_rebank.py; NEVER the stale poly_tau_curves banks), run long MC
independently under each of four KERNEL MIXES (fresh copy of the start config + a fresh seed_numba
call per mix, so mixes never share RNG state):
  MIX A "swap" (classical reference): per sweep = disp_sweep + swap_sweep(N tries).
  MIX B "usig": per sweep = disp_sweep + u_sweep_mu(N tries, delta, mu).
  MIX C "both": per sweep = disp_sweep + swap_sweep(N) + u_sweep_mu(N, delta, mu).
  MIX D "disp" (lower control): disp_sweep only.
kernels are liquid_coupling_flow/poly/model.py's disp_sweep/swap_sweep/total_U (energy/positions
unchanged) and liquid_coupling_flow/poly/semigrand.py's u_sweep_mu (P2a's mu-calibrated semi-grand
move) -- no kernel code is modified here.

NOTE on MIX C and u/sig consistency: classical swap_sweep permutes the `sig` array without touching
`u`, so after a swap the invariant "sig[i] == sigma_of_u(u[i])" that u_sweep_mu's docstring describes
can be momentarily broken at the swapped sites. This is the SAME combination (disp+swap+u_sweep_mu
in one loop) already used by poly_mu_calibrate.py's P2a calibration loop, which documents swaps as
"mu-neutral" (permutations only reshuffle existing sigma values among sites; they never change the
system-wide composition) and keeps them enabled purely to speed up local mixing. This harness mirrors
that established pattern verbatim for MIX C -- it is not a new design decision, and is out of scope
to change here.

Observables, recorded on a log-time grid t = 1, 2, 4, 8, ... (doubling) up to --budget sweeps (the
grid always ends exactly at --budget even if not a power of two):
  1. STRUCTURAL RELAXATION: Q_self(t) = fraction of particles within a=0.3 of their start position
     (min-image; exactly poly_bank.q_self / poly_bank.A_OV) -> tau_alpha = first t with Q < 1/e (or
     -1 if never reached within budget).
  2. SIGMA DECORRELATION: C_sig(t) = Pearson corr of per-particle sigma(t) vs sigma(0) (same array
     SLOT i across time, matching poly_bank's frame bookkeeping convention) -> tau_sig = first t with
     C < 1/e (or -1). For MIX A this measures swap-driven sigma exchange between sites; for MIX
     B/C it is the continuous u-channel (+ swap, for C).
  3. COMPOSITION GUARD: at every recorded t, sigma-histogram sup-norm relative error vs the exact
     analytic P(sigma) bin masses (liquid_coupling_flow/poly/semigrand.py's NBINS_MU=32-bin
     mu_bin_edges/mu_bin_target_mass, same convention + FLOOR as poly_mu_calibrate.py's frac_sup)
     + mean(sigma) + U/N. If frac_sup > 0.10 at any recorded t in MIX B/C, print a WARNING ("mu
     insufficient at this T") but KEEP RUNNING -- the data remains informative even if the mu
     artifact hasn't fully converged (this is the harness-only task; the real run waits for a
     converged mu).
  4. ACCEPTANCE RATES per kernel per WINDOW (i.e. only sweeps since the previous recorded t, not
     cumulative from t=0 -- so late-time acceptance collapse is visible rather than averaged away).

SWEEP-COST FAIRNESS NOTE (IMPORTANT): MIX C does roughly 2x the kernel work of MIX A/B per sweep
(disp + swap + u, vs disp + one of {swap, u}) -- a sweep-indexed tau comparison alone is not an
apples-to-apples wall-clock comparison. Every record carries a wall-clock stamp (time.time() - t0
for that mix), so BOTH a sweep-based tau and a wall-clock-based tau are tracked and printed in the
summary line for every mix.

INCREMENTAL SAVE (checkpoint-incrementally directive): the full time series (Q_self, C_sig, frac_sup,
mean_sigma, U_N, per-kernel window acceptance, wall-clock) for every mix is saved to --out after
EVERY recorded time point, not just at the end (record-simulation-data directive: full series are
kept, never just the tau summary scalars).

Usage:
    python reports/logs-2026-07-17/poly_p2b_gate.py --T 0.2 --budget 30000 \
        --mu liquid_coupling_flow/artifacts/poly_mu_T0.2.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/ssd/GridTransformer")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from liquid_coupling_flow.poly.model import (disp_sweep, swap_sweep, total_U,  # noqa: E402
                                              seed_numba)
from liquid_coupling_flow.poly.semigrand import (u_of_sigma, u_sweep_mu, mu_bin_edges,  # noqa: E402
                                                  mu_bin_target_mass, NBINS_MU)

STEP = 0.12                 # disp step, matches poly_bank.py / poly_mu_calibrate.py's DISP_STEP
A_OV = 0.3                  # q_self overlap radius, matches poly_bank.py's A_OV exactly
FLOOR = 1e-4                # frac_sup relative-error floor, matches poly_mu_calibrate.py's FLOOR
COMP_WARN_FRAC = 0.10       # composition-guard warning threshold (spec: > 0.10)
E_INV = float(np.exp(-1.0))
ALL_MIXES = ("swap", "usig", "both", "disp")
USES_SWAP = {"swap": True, "usig": False, "both": True, "disp": False}
USES_U = {"swap": False, "usig": True, "both": True, "disp": False}


def q_self(x, x0, L, a_ov=A_OV):
    d = x - x0
    d -= L * np.round(d / L)
    return float((np.sqrt((d ** 2).sum(1)) < a_ov).mean())


def c_sig(sig, sig0):
    """Pearson corr of per-slot sigma(t) vs sigma(0). Degenerate (zero-variance) arrays -> 1.0
    (cannot have decorrelated if nothing has ever moved, e.g. the very first record before any
    swap/u-move has landed)."""
    if np.std(sig) < 1e-12 or np.std(sig0) < 1e-12:
        return 1.0
    return float(np.corrcoef(sig, sig0)[0, 1])


def log_time_grid(budget):
    """1, 2, 4, 8, ... (doubling) strictly below budget, then budget itself appended (so the grid
    always ends exactly at --budget even when budget is not a power of two)."""
    ts = []
    t = 1
    while t < budget:
        ts.append(t)
        t *= 2
    ts.append(budget)
    return ts


def comp_guard(sig, edges, target):
    meas_hist, _ = np.histogram(sig, bins=edges)
    meas = meas_hist / meas_hist.sum()
    frac_err = np.abs(meas - target) / np.maximum(target, FLOOR)
    return float(np.max(frac_err))


def run_mix(mix, x0, sig0, L, beta, T, budget, delta, mu, seed, edges, target, out_data, out_path):
    """Run one kernel mix from a fresh copy of (x0, sig0), independent RNG (seed_numba(seed)).
    Mutates out_data[mix] and torch.saves out_data after every recorded time point."""
    assert mix in ALL_MIXES, mix
    use_swap, use_u = USES_SWAP[mix], USES_U[mix]
    if use_u:
        assert mu is not None, f"mix={mix} needs --mu"

    x = x0.copy()
    sig = sig0.copy()
    u = u_of_sigma(sig)          # kept byte-consistent with sig by u_sweep_mu whenever mix uses it
    seed_numba(seed)
    x_start = x.copy()
    sig_start = sig.copy()
    N = x.shape[0]

    grid = log_time_grid(budget)
    rec = {"mix": mix, "T": T, "seed": seed, "t": [], "Q_self": [], "C_sig": [], "wall": [],
           "frac_sup": [], "mean_sigma": [], "U_N": [],
           "disp_acc": [], "swap_acc": [], "u_acc": [],
           "tau_alpha": -1, "tau_sig": -1, "tau_alpha_wall": None, "tau_sig_wall": None,
           "comp_warned": False, "max_frac_sup": 0.0}

    disp_acc_w = disp_att_w = 0
    swap_acc_w = swap_att_w = 0
    u_acc_w = u_att_w = 0
    t_done = 0
    t0 = time.time()
    print(f"  [{mix}] start: use_swap={use_swap} use_u={use_u} budget={budget} "
          f"n_records={len(grid)}", flush=True)
    for t_target in grid:
        while t_done < t_target:
            da = disp_sweep(x, sig, L, beta, STEP)
            disp_acc_w += da; disp_att_w += N
            if use_swap:
                sa, st = swap_sweep(x, sig, L, beta, N)
                swap_acc_w += sa; swap_att_w += st
            if use_u:
                ua, ut = u_sweep_mu(x, sig, u, L, beta, delta, N, mu)
                u_acc_w += ua; u_att_w += ut
            t_done += 1

        q = q_self(x, x_start, L)
        c = c_sig(sig, sig_start)
        fs = comp_guard(sig, edges, target)
        mean_sig = float(sig.mean())
        u_n = float(total_U(x, sig, L) / N)
        wall = time.time() - t0

        disp_rate = disp_acc_w / max(disp_att_w, 1)
        swap_rate = (swap_acc_w / max(swap_att_w, 1)) if use_swap else float("nan")
        u_rate = (u_acc_w / max(u_att_w, 1)) if use_u else float("nan")
        disp_acc_w = disp_att_w = swap_acc_w = swap_att_w = u_acc_w = u_att_w = 0

        rec["t"].append(t_done); rec["Q_self"].append(q); rec["C_sig"].append(c)
        rec["wall"].append(wall); rec["frac_sup"].append(fs)
        rec["mean_sigma"].append(mean_sig); rec["U_N"].append(u_n)
        rec["disp_acc"].append(disp_rate); rec["swap_acc"].append(swap_rate)
        rec["u_acc"].append(u_rate)
        rec["max_frac_sup"] = max(rec["max_frac_sup"], fs)

        if rec["tau_alpha"] < 0 and q < E_INV:
            rec["tau_alpha"] = t_done
            rec["tau_alpha_wall"] = wall
        if rec["tau_sig"] < 0 and c < E_INV:
            rec["tau_sig"] = t_done
            rec["tau_sig_wall"] = wall

        if use_u and fs > COMP_WARN_FRAC:
            rec["comp_warned"] = True
            print(f"  WARNING [{mix}] mu insufficient at T={T}: t={t_done} frac_sup={fs:.3f} "
                  f"(> {COMP_WARN_FRAC}) -- continuing (data still informative)", flush=True)

        print(f"  [{mix}] t={t_done:8d} Q_self={q:.4f} C_sig={c:+.4f} frac_sup={fs:.4f} "
              f"mean_sig={mean_sig:.4f} U/N={u_n:+.4f} disp_acc={disp_rate:.3f} "
              f"swap_acc={swap_rate:.3f} u_acc={u_rate:.3f} wall={wall:.1f}s", flush=True)

        out_data[mix] = rec
        torch.save(out_data, out_path)

    return rec


def _print_summary(rec):
    mix = rec["mix"]
    ta, ta_w = rec["tau_alpha"], rec["tau_alpha_wall"]
    ts, ts_w = rec["tau_sig"], rec["tau_sig_wall"]
    ta_s = f"{ta} sw ({ta_w:.1f}s)" if ta > 0 else "-1 (not reached)"
    ts_s = f"{ts} sw ({ts_w:.1f}s)" if ts > 0 else "-1 (not reached)"
    last = -1
    disp_a = rec["disp_acc"][last]; swap_a = rec["swap_acc"][last]; u_a = rec["u_acc"][last]
    print(f"[SUMMARY {mix}] tau_alpha={ta_s} tau_sig={ts_s} "
          f"last-window acc: disp={disp_a:.3f} swap={swap_a:.3f} u={u_a:.3f} "
          f"comp_guard={'WARN' if rec['comp_warned'] else 'ok'} "
          f"(max_frac_sup={rec['max_frac_sup']:.3f})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--mixes", type=str, default="swap,usig,both,disp")
    p.add_argument("--budget", type=int, default=2_000_000)
    p.add_argument("--run", type=int, default=3, help="bank run index for the start config")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.15, help="u-move proposal std")
    p.add_argument("--mu", type=str, default=None,
                    help="path to a poly_mu_calibrate.py artifact; REQUIRED if --mixes includes "
                         "usig or both")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()

    mixes = [m.strip() for m in a.mixes.split(",") if m.strip()]
    for m in mixes:
        assert m in ALL_MIXES, f"unknown mix {m!r} (valid: {ALL_MIXES})"
    needs_mu = any(USES_U[m] for m in mixes)

    mu = None
    if a.mu is not None:
        mu_ck = torch.load(a.mu, weights_only=False)
        assert abs(float(mu_ck["T"]) - a.T) < 1e-9, (
            f"--mu artifact T={mu_ck['T']} != --T {a.T} (mu is T-specific)")
        mu = np.asarray(mu_ck["mu"], dtype=np.float64)
        assert mu.shape == (NBINS_MU,), mu.shape
    elif needs_mu:
        raise AssertionError(f"--mu is REQUIRED when --mixes includes usig/both (got {mixes})")

    out_path = Path(a.out) if a.out else REPO / f"reports/logs-2026-07-17/poly_p2b_T{a.T}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bank_path = REPO / f"reports/logs-2026-07-17/poly_bank_fixed_run{a.run}.pt"
    assert bank_path.exists(), f"NO FIXED BANK at {bank_path}"
    bank = torch.load(bank_path, weights_only=False)
    key = (a.run, a.T)
    assert key in bank, f"key {key} not in {bank_path} (keys: {[k for k in bank if isinstance(k, tuple)]})"
    brec = bank[key]
    assert "sigs" in brec, "fixed bank must carry per-frame sigs"
    x0 = np.ascontiguousarray(brec["x"][a.frame], dtype=np.float64)
    sig0 = np.ascontiguousarray(brec["sigs"][a.frame], dtype=np.float64)
    L = float(brec["L"])
    beta = 1.0 / a.T
    N = x0.shape[0]

    edges = mu_bin_edges(NBINS_MU)
    target = mu_bin_target_mass(NBINS_MU)

    out_data = {"T": a.T, "args": vars(a), "N": N, "L": L, "run": a.run, "frame": a.frame}
    torch.save(out_data, out_path)

    print(f"poly P2b GATE: T={a.T} mixes={mixes} budget={a.budget} run={a.run} frame={a.frame} "
          f"delta={a.delta} mu={a.mu} N={N} L={L:.4f} -> {out_path}", flush=True)

    t0_all = time.time()
    for mi, mix in enumerate(mixes):
        seed_i = a.seed + 1000 * (mi + 1)      # distinct RNG stream per mix; fresh seed_numba call
        print(f"--- MIX {mix} (seed={seed_i}) ---", flush=True)
        rec = run_mix(mix, x0, sig0, L, beta, a.T, a.budget, a.delta, mu, seed_i, edges, target,
                       out_data, out_path)
        _print_summary(rec)

    print(f"P2b GATE DONE ({time.time() - t0_all:.0f}s) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
