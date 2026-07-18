"""poly NOVELTY-1: NCMC diameter-swap as a PRODUCTION CHAINED kernel, plus a below-arrest
tau-measurement harness. CANONICAL ensemble throughout -- no mu anywhere (contrast with
poly_p2b_gate.py's mu-calibrated semi-grand mixes; this file never touches semigrand.py).

Builds on TWO prior probes in this same directory, reused (not re-derived):
  - poly_ncmc_probe.py (v1): ncmc_work -- whole-system propagation, per-attempt measurement from
    independent equilibrium starts, no chaining. Its docstring gives the base exactness argument
    (palindromic protocol / self-inverse endpoint swap / symmetric selection => composite
    acceptance min(1, exp(-beta*W)) is exact, Nilmeier et al. PNAS 2011).
  - poly_ncmc_v2.py (v2): ncmc_work_local -- restricts propagation to a local candidate set
    S(x) = {m : min(d(m,i), d(m,j)) < r_loc}, frozen per lambda-step, with a reflecting exit-
    rejection boundary (exact, same composite-acceptance argument, selection ratio 1 within S).
    v2's measured saturation: local-only W plateaus with r_loc, so production here defaults to
    r_loc=3.0 (not v2's largest probed value) -- see reports/logs-2026-07-17/poly_ncmc_v2_T0.085.out
    for the measured plateau; this file imports v2's kernel verbatim, it does not re-derive it.

THIS FILE adds exactly two new ingredients on top of v1/v2:

(1) WINDOW-RESTRICTED PAIR SELECTION (still exactly symmetric, no Hastings ratio):
    v1/v2 select the candidate pair (i, j) UNIFORMLY over ALL pairs. Production instead restricts
    candidates to |sigma_i - sigma_j| in [WINDOW_LO, WINDOW_HI] = [0.1, 0.9] (closed interval), to
    concentrate attempts on the mid-|dsigma| bins where v1/v2 measured the largest acceptance
    payoff. This is STILL exactly symmetric:
      - |sigma_i - sigma_j| is invariant under swapping the values held by i and j (|a-b|==|b-a|),
        so the CANDIDATE pair (i,j) itself is in the window before the swap iff it is after.
      - For any third particle k, swapping sigma_i<->sigma_j exchanges which of
        {|si_old-sk|, |sj_old-sk|} sits on pair (i,k) vs (j,k) -- it permutes which pair carries
        which value, it does not change the multiset of two values. So the number of window-
        qualifying pairs contributed by {(i,k),(j,k)} TOGETHER is identical before and after, for
        every k.
      - Every pair (k,l) not touching i or j is completely untouched.
      - Summing all three cases: the TOTAL qualifying-pair count is EXACTLY invariant under ANY
        completed sigma_i<->sigma_j endpoint swap (not just window-restricted ones) -- this is
        verified directly by test_window_count_invariant_under_endpoint_swap in
        liquid_coupling_flow/tests/test_poly_ncmc.py.
      - Consequently, selecting UNIFORMLY among the (count-invariant) qualifying pairs has forward
        selection probability 1/cnt and reverse selection probability 1/cnt -- ratio 1, no extra
        Hastings factor. The v1/v2 composite acceptance min(1, exp(-beta*W)) carries over UNCHANGED.

(2) CHAINED APPLICATION with restore-on-reject: v1/v2 only ever MEASURED W from independent
    equilibrium starts; the caller never applied the move, so there was nothing to undo. Production
    must actually accept/reject and CONTINUE the Markov chain from whichever state results. On
    ACCEPT, the post-switch (x, sig) produced by ncmc_work_local (which mutates in place) IS the new
    state -- nothing further to do. On REJECT, (x, sig) must be restored EXACTLY (bitwise) to the
    pre-attempt snapshot, because ncmc_work_local has no built-in undo. `ncmc_attempt` below is the
    snapshot/restore wrapper that ALL production usage goes through; this is the #1 bug risk here
    and is covered by an explicit direct test (test_restore_on_reject_and_apply_on_accept).

Kernel mixes (per sweep):
  MIX A "swap"       : disp_sweep + swap_sweep(N)                         (classical reference)
  MIX B "swap+ncmc"  : disp_sweep + swap_sweep(N), PLUS one window-selected NCMC diameter-swap
                        attempt every --ncmc_every sweeps (local propagation, r_loc/n_steps/
                        sweeps_per_step per --args), accepted/rejected and CHAINED.
  MIX C "disp"        : disp_sweep only                                    (lower control)

Observables mirror poly_p2b_gate.py's conventions (log-time grid, Q_self -> tau_alpha, C_sig ->
tau_sig, incremental per-record save, independent RNG per mix), PLUS the novel NCMC channel:
cumulative NCMC attempt/accept counts, the accepted-|dsigma| list (histogram material), and
cumulative accepted-|dsigma| transport. WALL-CLOCK FAIRNESS: MIX B does strictly more work per
sweep than A/C (an extra NCMC attempt every ncmc_every sweeps, each attempt costing
n_steps*sweeps_per_step*|S| local-move-attempts) -- a sweep-indexed comparison across mixes is NOT
apples-to-apples wall-clock. Every record therefore carries time.perf_counter() elapsed alongside
the sweep index (both "t" and "wall" arrays, same length, same index -- i.e. every recorded point is
simultaneously sweep-indexed and wall-clock-indexed), so downstream analysis can re-grid any mix
onto a common wall-clock axis by interpolation.

Usage:
    python reports/logs-2026-07-17/poly_ncmc_chain.py --T 0.085 --budget 600000 \
        --mixes swap,swap+ncmc,disp --ncmc_every 200 --n_steps 800 --r_loc 3.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numba import njit

REPO = Path("/mnt/ssd/GridTransformer")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
LOGDIR = REPO / "reports/logs-2026-07-17"
if str(LOGDIR) not in sys.path:
    sys.path.insert(0, str(LOGDIR))

from liquid_coupling_flow.poly.model import (disp_sweep, swap_sweep, total_U,  # noqa: E402
                                              seed_numba)
# v2's local-propagation kernel, reused verbatim (NOT re-derived, NOT re-run -- import only, its
# __main__ probe may still be running under a separate process; do not touch that process/output).
from poly_ncmc_v2 import ncmc_work_local  # noqa: E402

STEP = 0.12                 # disp step, matches poly_bank.py / poly_p2b_gate.py's DISP_STEP
A_OV = 0.3                  # q_self overlap radius, matches poly_bank.py's A_OV exactly
E_INV = float(np.exp(-1.0))
WINDOW_LO = 0.1              # |dsigma| pair-selection window (closed interval), per NOVELTY-1 spec
WINDOW_HI = 0.9

ALL_MIXES = ("swap", "swap+ncmc", "disp")
USES_SWAP = {"swap": True, "swap+ncmc": True, "disp": False}
USES_NCMC = {"swap": False, "swap+ncmc": True, "disp": False}


# ---------------------------------------------------------------------------
# Window-restricted pair selection (numba, shares the single seed_numba(...) RNG stream with
# every other kernel in the chain -- disp_sweep/swap_sweep/local_sweep all draw from the same
# numba np.random state, so one seed_numba call controls the entire chain deterministically).
# ---------------------------------------------------------------------------
@njit(cache=True)
def count_window_pairs(sig, lo, hi):
    """Count particle pairs (i<j) with |sigma_i - sigma_j| in [lo, hi] (closed interval). Standalone
    utility used directly by the count-invariance test; also the first pass of pick_window_pair."""
    n = sig.shape[0]
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            ds = abs(sig[i] - sig[j])
            if lo <= ds <= hi:
                cnt += 1
    return cnt


@njit(cache=True)
def pick_window_pair(sig, lo, hi):
    """Count qualifying pairs, then draw ONE index uniformly among them (numba RNG). Returns
    (i, j, cnt); returns (-1, -1, 0) if no pair currently qualifies (caller skips the attempt)."""
    n = sig.shape[0]
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            ds = abs(sig[i] - sig[j])
            if lo <= ds <= hi:
                cnt += 1
    if cnt == 0:
        return -1, -1, 0
    pick = np.random.randint(cnt)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            ds = abs(sig[i] - sig[j])
            if lo <= ds <= hi:
                if k == pick:
                    return i, j, cnt
                k += 1
    return -1, -1, cnt  # unreachable


@njit(cache=True)
def ncmc_attempt(x, sig, L, beta, r_loc, n_steps, sweeps_per_step, step, lo, hi):
    """ONE production chained NCMC diameter-swap attempt.
      1. select a pair (i,j) uniformly among |sigma_i-sigma_j| in [lo,hi] (module docstring has the
         count-invariance / no-extra-Hastings-factor argument for this window restriction);
      2. snapshot (x, sig) for restore-on-reject;
      3. drive sigma_i,sigma_j to their swapped values over `n_steps` lambda-steps using v2's LOCAL
         propagation (ncmc_work_local: build_local_set + local_sweep, r_loc-restricted), after every
         lambda switch. n_steps<=0 reproduces the classical instantaneous swap exactly (v2's
         schedule=[1.0] special case: no propagation, deterministic vertical work, x untouched).
      4. accept the whole composite move with min(1, exp(-beta*W)). On accept, keep the
         already-mutated (x, sig) (interpolation fully completed to the swapped endpoint). On
         reject, restore BOTH arrays bitwise to the step-2 snapshot -- ncmc_work_local has no
         built-in undo, so this wrapper is the only place restore-on-reject happens.
    Returns (attempted, accepted, i, j, ds, W, cnt). attempted=False (cnt==0) means no pair
    currently satisfies the window -- a true no-op, caller just counts it and moves on.
    """
    i, j, cnt = pick_window_pair(sig, lo, hi)
    if cnt == 0:
        return False, False, -1, -1, 0.0, 0.0, 0
    ds = abs(sig[i] - sig[j])

    x_snap = x.copy()
    sig_snap = sig.copy()

    if n_steps <= 0:
        schedule = np.empty(1, dtype=np.float64)
        schedule[0] = 1.0
    else:
        schedule = np.linspace(1.0 / n_steps, 1.0, n_steps)
    n_sched = schedule.shape[0]
    dW_step = np.zeros(n_sched)
    nS_step = np.zeros(n_sched)
    W = ncmc_work_local(x, sig, L, beta, i, j, schedule, sweeps_per_step, r_loc, step,
                         dW_step, nS_step)

    acc_prob = min(1.0, np.exp(min(700.0, max(-700.0, -beta * W))))
    accept = np.random.random() < acc_prob
    if not accept:
        x[:, :] = x_snap
        sig[:] = sig_snap
    return True, accept, i, j, ds, W, cnt


# ---------------------------------------------------------------------------
# Observables (mirror poly_p2b_gate.py's q_self / c_sig / log_time_grid conventions exactly).
# ---------------------------------------------------------------------------
def q_self(x, x0, L, a_ov=A_OV):
    d = x - x0
    d -= L * np.round(d / L)
    return float((np.sqrt((d ** 2).sum(1)) < a_ov).mean())


def c_sig(sig, sig0):
    """Pearson corr of per-slot sigma(t) vs sigma(0). Degenerate (zero-variance) arrays -> 1.0."""
    if np.std(sig) < 1e-12 or np.std(sig0) < 1e-12:
        return 1.0
    return float(np.corrcoef(sig, sig0)[0, 1])


def log_time_grid(budget):
    """1, 2, 4, 8, ... (doubling) strictly below budget, then budget itself appended."""
    ts = []
    t = 1
    while t < budget:
        ts.append(t)
        t *= 2
    ts.append(budget)
    return ts


# ---------------------------------------------------------------------------
# Per-mix run.
# ---------------------------------------------------------------------------
def run_mix(mix, x0, sig0, L, beta, T, budget, ncmc_every, n_steps, r_loc, sweeps_per_step, seed,
            out_data=None, out_path=None):
    """Run one kernel mix from a fresh copy of (x0, sig0), independent RNG (seed_numba(seed)).
    If out_data/out_path are given, mutates out_data[mix] and torch.saves out_data after every
    recorded time point (checkpoint-incrementally directive)."""
    assert mix in ALL_MIXES, mix
    use_swap = USES_SWAP[mix]
    use_ncmc = USES_NCMC[mix]

    x = x0.copy()
    sig = sig0.copy()
    seed_numba(seed)
    x_start = x.copy()
    sig_start = sig.copy()
    N = x.shape[0]

    grid = log_time_grid(budget)
    rec = {
        "mix": mix, "T": T, "seed": seed, "ncmc_every": ncmc_every, "n_steps": n_steps,
        "r_loc": r_loc, "sweeps_per_step": sweeps_per_step, "window": (WINDOW_LO, WINDOW_HI),
        "t": [], "wall": [], "Q_self": [], "C_sig": [], "U_N": [],
        "disp_acc": [], "swap_acc": [],
        "ncmc_attempts_window": [], "ncmc_accepted_window": [],
        "ncmc_attempts_cum": [], "ncmc_accepted_cum": [], "ncmc_transport_cum": [],
        "ncmc_accepted_ds": [],           # growing list of accepted |dsigma| (histogram material)
        "tau_alpha": -1, "tau_sig": -1, "tau_alpha_wall": None, "tau_sig_wall": None,
    }

    disp_acc_w = disp_att_w = 0
    swap_acc_w = swap_att_w = 0
    ncmc_att_w = ncmc_acc_w = 0
    ncmc_att_cum = ncmc_acc_cum = 0
    ncmc_transport_cum = 0.0
    accepted_ds_all = []

    t_done = 0
    t0 = time.perf_counter()
    print(f"  [{mix}] start: use_swap={use_swap} use_ncmc={use_ncmc} budget={budget} "
          f"n_records={len(grid)} ncmc_every={ncmc_every} n_steps={n_steps} r_loc={r_loc} "
          f"sweeps_per_step={sweeps_per_step} window=[{WINDOW_LO},{WINDOW_HI}]", flush=True)

    for t_target in grid:
        while t_done < t_target:
            da = disp_sweep(x, sig, L, beta, STEP)
            disp_acc_w += da; disp_att_w += N
            if use_swap:
                sa, st = swap_sweep(x, sig, L, beta, N)
                swap_acc_w += sa; swap_att_w += st
            t_done += 1
            if use_ncmc and t_done % ncmc_every == 0:
                attempted, accepted, i, j, ds, W, cnt = ncmc_attempt(
                    x, sig, L, beta, r_loc, n_steps, sweeps_per_step, STEP, WINDOW_LO, WINDOW_HI)
                if attempted:
                    ncmc_att_w += 1; ncmc_att_cum += 1
                    if accepted:
                        ncmc_acc_w += 1; ncmc_acc_cum += 1
                        ncmc_transport_cum += ds
                        accepted_ds_all.append(ds)

        q = q_self(x, x_start, L)
        c = c_sig(sig, sig_start)
        u_n = float(total_U(x, sig, L) / N)
        wall = time.perf_counter() - t0

        disp_rate = disp_acc_w / max(disp_att_w, 1)
        swap_rate = (swap_acc_w / max(swap_att_w, 1)) if use_swap else float("nan")
        ncmc_rate_window = (ncmc_acc_w / max(ncmc_att_w, 1)) if use_ncmc else float("nan")
        ncmc_att_w_saved, ncmc_acc_w_saved = ncmc_att_w, ncmc_acc_w
        disp_acc_w = disp_att_w = swap_acc_w = swap_att_w = 0
        ncmc_att_w = ncmc_acc_w = 0

        rec["t"].append(t_done); rec["wall"].append(wall)
        rec["Q_self"].append(q); rec["C_sig"].append(c); rec["U_N"].append(u_n)
        rec["disp_acc"].append(disp_rate); rec["swap_acc"].append(swap_rate)
        rec["ncmc_attempts_window"].append(ncmc_att_w_saved)
        rec["ncmc_accepted_window"].append(ncmc_acc_w_saved)
        rec["ncmc_attempts_cum"].append(ncmc_att_cum)
        rec["ncmc_accepted_cum"].append(ncmc_acc_cum)
        rec["ncmc_transport_cum"].append(ncmc_transport_cum)
        rec["ncmc_accepted_ds"] = list(accepted_ds_all)   # full growing list, snapshotted each save

        if rec["tau_alpha"] < 0 and q < E_INV:
            rec["tau_alpha"] = t_done; rec["tau_alpha_wall"] = wall
        if rec["tau_sig"] < 0 and c < E_INV:
            rec["tau_sig"] = t_done; rec["tau_sig_wall"] = wall

        print(f"  [{mix}] t={t_done:8d} wall={wall:7.1f}s Q_self={q:.4f} C_sig={c:+.4f} "
              f"U/N={u_n:+.4f} disp_acc={disp_rate:.3f} swap_acc={swap_rate:.3f} "
              f"ncmc(window)={ncmc_acc_w_saved}/{ncmc_att_w_saved} "
              f"ncmc(cum)={ncmc_acc_cum}/{ncmc_att_cum} rate={ncmc_rate_window:.3f} "
              f"transport_cum={ncmc_transport_cum:.3f}", flush=True)

        if out_data is not None and out_path is not None:
            out_data[mix] = rec
            torch.save(out_data, out_path)

    return rec


def _print_summary(rec):
    mix = rec["mix"]
    ta, ta_w = rec["tau_alpha"], rec["tau_alpha_wall"]
    ts, ts_w = rec["tau_sig"], rec["tau_sig_wall"]
    ta_s = f"{ta} sw ({ta_w:.1f}s)" if ta > 0 else "-1 (not reached)"
    ts_s = f"{ts} sw ({ts_w:.1f}s)" if ts > 0 else "-1 (not reached)"
    att = rec["ncmc_attempts_cum"][-1] if rec["ncmc_attempts_cum"] else 0
    acc = rec["ncmc_accepted_cum"][-1] if rec["ncmc_accepted_cum"] else 0
    transport = rec["ncmc_transport_cum"][-1] if rec["ncmc_transport_cum"] else 0.0
    wall_total = rec["wall"][-1] if rec["wall"] else 0.0
    print(f"[SUMMARY {mix}] tau_alpha={ta_s} tau_sig={ts_s} "
          f"ncmc: {acc}/{att} accepted (rate={acc / max(att, 1):.3f}) "
          f"transport_cum={transport:.3f} wall_total={wall_total:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--budget", type=int, default=600_000, help="sweeps")
    p.add_argument("--run", type=int, default=3, help="bank run index for the start config")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--ncmc_every", type=int, default=200)
    p.add_argument("--n_steps", type=int, default=800)
    p.add_argument("--r_loc", type=float, default=3.0)
    p.add_argument("--sweeps_per_step", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mixes", type=str, default="swap,swap+ncmc,disp")
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()

    mixes = [m.strip() for m in a.mixes.split(",") if m.strip()]
    for m in mixes:
        assert m in ALL_MIXES, f"unknown mix {m!r} (valid: {ALL_MIXES})"

    out_path = Path(a.out) if a.out else REPO / f"reports/logs-2026-07-17/poly_ncmc_chain_T{a.T}.pt"
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

    out_data = {"T": a.T, "args": vars(a), "N": N, "L": L, "run": a.run, "frame": a.frame}
    torch.save(out_data, out_path)

    print(f"poly NCMC CHAIN: T={a.T} mixes={mixes} budget={a.budget} run={a.run} frame={a.frame} "
          f"ncmc_every={a.ncmc_every} n_steps={a.n_steps} r_loc={a.r_loc} "
          f"sweeps_per_step={a.sweeps_per_step} N={N} L={L:.4f} window=[{WINDOW_LO},{WINDOW_HI}] "
          f"-> {out_path}", flush=True)

    t0_all = time.perf_counter()
    for mi, mix in enumerate(mixes):
        seed_i = a.seed + 1000 * (mi + 1)      # distinct RNG stream per mix; fresh seed_numba call
        print(f"--- MIX {mix} (seed={seed_i}) ---", flush=True)
        rec = run_mix(mix, x0, sig0, L, beta, a.T, a.budget, a.ncmc_every, a.n_steps, a.r_loc,
                       a.sweeps_per_step, seed_i, out_data, out_path)
        _print_summary(rec)

    print(f"NCMC CHAIN DONE ({time.perf_counter() - t0_all:.0f}s) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
