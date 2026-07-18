"""FLOW-ESCORTED NCMC propagator (poly ML item 2): the sigma(t)-conditioned SwapBlockFlow
(liquid_coupling_flow/poly/swap_flow.py) wired INSIDE the NCMC diameter-swap protocol
(poly_ncmc_v2.ncmc_work_local) as an EXTRA, periodic, fixed-lambda detailed-balanced relaxation
kernel -- not as a one-shot vertical proposal (that line was killed in phase 1, poly_gate_verdict_
phase1.md: 25-50 kT local free energy for dissimilar swaps, an oracle-flat thermodynamic wall).

PHYSICS CONTRACT (why this preserves NCMC exactness):
NCMC's acceptance min(1, e^{-beta W}) is exact (Crooks / Nilmeier et al. PNAS 2011) as long as EVERY
per-lambda-step propagation kernel is detailed-balanced w.r.t. the INSTANTANEOUS pi_lambda (the
lambda-schedule itself only needs to be a palindromic sequence of DB kernels interleaved with
INSTANTANEOUS, deterministic lambda switches -- the switches contribute the work W, the kernels
contribute nothing to W by construction). poly_ncmc_v2's local_sweep (single-particle Metropolis at
fixed lambda) is one such kernel. A flow-MH move is ANOTHER: propose a new BLOCK configuration x' via
SwapBlockFlow.propose using an IDENTITY sigma-path -- sig_start == sig_end == the block's CURRENT
lambda-interpolated per-particle sigma (so `labels_at_t` is CONSTANT along the flow's own internal
integration variable t in [0,1]; only the flow's structure-aware DISPLACEMENT field is exercised, no
species conditioning changes mid-move) -- and accept with the standard flow-MH ratio
    A = min(1, exp(-beta * dU_lambda + logq_rev - logq_fwd))
where dU_lambda is evaluated at the CURRENT (fixed) interpolated sigmas (block_dU, poly_gate_
acceptance.py's validated double-count-corrected energy, reused not reimplemented) and logq_fwd/
logq_rev are SwapBlockFlow.propose/.logq_of on that SAME identity path (per swap_flow.py's MH USAGE
docstring section: forward = logq_of(x_new, x_old, sig, sig, env...), reverse = logq_of(x_old, x_new,
sig, sig, env...) -- with sig_start==sig_end the "reversed path" is literally the same path, so there
is exactly one sigma argument pair to worry about, no path-reversal bookkeeping). This ratio is a
plain fixed-Hamiltonian Metropolis-Hastings acceptance at pi_lambda -- textbook DB, independent of how
elaborate the proposal kernel is. Instantaneous lambda-switches (and their work bookkeeping) are
UNCHANGED by any of this: the escort move never straddles a lambda switch, it only ever fires
*between* switches, at a lambda that is momentarily frozen for the whole move.

LEDGER SEPARATION (the CRITICAL bookkeeping point): W (returned by both ncmc_work_local and this
module's ncmc_work_local_escorted) accumulates ONLY the instantaneous lambda-switch energy deltas
(e1 - e0 evaluated at the two consecutive sigma assignments, same positions) -- exactly as in
poly_ncmc_v2.py, untouched. The flow-MH move's own dU_lambda is a SEPARATE, self-contained
accept/reject decision at a FIXED lambda: whether it fires or not, and whether it is accepted or
rejected, changes NOTHING about W's accumulation, because a DB-at-fixed-pi_lambda kernel contributes
ZERO net work to a nonequilibrium work functional by construction (same reason local_sweep's own
accepted/rejected single-particle moves never appear in W either -- W is a property of the
lambda-SCHEDULE, not of what happens to relax the system at a fixed lambda). Concretely: dW_step/
nS_step (the per-lambda-step work-profile arrays, unchanged shape/meaning from poly_ncmc_v2) record
only the switch deltas; the escort's own dU/acceptance are recorded in SEPARATE arrays
(flow_dU/flow_accepted) purely for diagnostic/probe reporting, never summed into W.

BLOCK DEFINITION: pair-centered k=8 (poly_escort_pairs.select_block_by_pair -- k nearest particles by
min(dist-to-i, dist-to-j), min-image; i, j are always the two closest members, self-distance 0, so
they are always IN the block for ANY separation -- this is the SAME definition the training-pair
generator uses, so the escort sees exactly the distribution it was trained on). env = all non-block
particles, centered on the block's OWN (pre-move) centroid and min-image wrapped -- the SAME
centered/uncentered mapping convention as poly_gate_swap.py's select_block_and_pair (block selection
in a frame centered on the block's own centroid; propose's base is per-particle x_old, so no
additional recentering is needed between the forward and reverse MH legs, per poly_gate_swap.py's
docstring derivation). SwapBlockFlow's own `_prep_env` truncates the passed env to the nearest m_env
(64) automatically (env is centered on the block, so "nearest to the block" == "nearest to the
origin" in that frame) -- callers here pass the FULL non-block remainder, exactly like poly_gate_
swap.py does, and rely on that truncation.

REJECT-RESTORE: the block's absolute positions x[idx] are copied BEFORE the flow move; on rejection,
x[idx] is reassigned from that exact copy (bitwise restore, not a re-derived value) -- see
_flow_mh_move below.

Usage:
  poly_escort_ncmc.py --T 0.085 --ckpt liquid_coupling_flow/artifacts/poly_escortflow_T0.085_k8_best.pt
      [--r_loc 3.0] [--per_bin 40] [--escort_every 100] [--out ...]
Probes 3 configs at T=0.085, r_loc=3.0, bins [(0.1,.2),(.2,.3),(.3,.45),(.45,.9)], per_bin=40:
  n=200 escorted (escort fires at steps 100, 200) vs n=200 plain vs n=800 plain.
Reports W_med per bin per config + wall-clock per attempt (GPU flow calls included).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/ssd/GridTransformer")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reports/logs-2026-07-17"))

from liquid_coupling_flow.poly.model import seed_numba  # noqa: E402
from liquid_coupling_flow.poly.swap_flow import SwapBlockFlow  # noqa: E402
from poly_ncmc_v2 import build_local_set, local_sweep, ncmc_work_local, _pair_env_e  # noqa: E402  (IMPORT ONLY, never modify)
from poly_gate_acceptance import block_dU  # noqa: E402  (REUSE, validated)
from poly_escort_pairs import select_pair, select_block_by_pair, DS_BINS  # noqa: E402  (mine, reused)

STEP = 0.12


# --------------------------------------------------------------------------------------------------
# the flow-MH move (fixed-lambda DB kernel, identity sigma path)
# --------------------------------------------------------------------------------------------------
def _center_and_wrap(arr, cen, L):
    c = arr - cen
    return c - L * np.round(c / L)


def _flow_mh_move(x, sig, L, beta, idx, flow, gen, rng):
    """One flow-MH proposal+accept/reject on block `idx`, at the block's CURRENT (fixed) sigma
    assignment (identity path: sig_start == sig_end == sig[idx], the CURRENT lambda-interpolated
    values). Mutates x[idx] IN PLACE on acceptance; leaves it exactly as found (bitwise, from an
    explicit pre-move copy) on rejection. Returns (accepted: bool, dU: float, logq_fwd: float,
    logq_rev: float)."""
    N = x.shape[0]
    mask = np.zeros(N, dtype=bool)
    mask[idx] = True

    x_blk_abs = x[idx].copy()                       # exact pre-move snapshot (restore target)
    sig_blk = sig[idx].copy()                        # identity path: same array both ends
    env_x_abs = x[~mask].copy()
    env_sig = sig[~mask].copy()

    cen = x_blk_abs.mean(0)
    x_old_c = _center_and_wrap(x_blk_abs, cen, L)
    env_x_c = _center_and_wrap(env_x_abs, cen, L)

    x_new_c, logq_fwd = flow.propose(x_old_c, sig_blk, sig_blk, env_x_c, env_sig, gen)
    logq_rev = flow.logq_of(x_old_c, x_new_c, sig_blk, sig_blk, env_x_c, env_sig)

    x_new_c64 = x_new_c.astype(np.float64)
    e_old = block_dU(x_old_c, sig_blk, env_x_c, env_sig, L)
    e_new = block_dU(x_new_c64, sig_blk, env_x_c, env_sig, L)
    dU = e_new - e_old

    A = float(min(1.0, np.exp(np.clip(-beta * dU + logq_rev - logq_fwd, -700, 700))))
    accepted = bool(rng.random() < A)

    if accepted:
        x_new_abs = np.mod(cen + x_new_c64, L)
        x[idx] = x_new_abs
    else:
        x[idx] = x_blk_abs                            # bitwise restore of the exact pre-move copy

    return accepted, float(dU), float(logq_fwd), float(logq_rev)


# --------------------------------------------------------------------------------------------------
# escorted NCMC driver -- replicates ncmc_work_local's loop (python-level: the flow call cannot be
# numba-jitted) using the SAME imported building blocks (build_local_set, local_sweep). W bookkeeping
# is BYTE-IDENTICAL to ncmc_work_local's when escort_every effectively never fires (see module
# docstring's LEDGER SEPARATION section, and test 3 in test_poly_escort.py).
# --------------------------------------------------------------------------------------------------
def ncmc_work_local_escorted(x, sig, L, beta, i, j, schedule, sweeps_per_step, r_loc, step,
                              dW_step, nS_step, flow, gen, rng, k_block, escort_every):
    """Same contract as poly_ncmc_v2.ncmc_work_local, PLUS: every `escort_every` lambda-steps (if
    flow is not None and escort_every < inf), fires one _flow_mh_move on the pair-centered k_block
    block. Returns (W, flow_calls, flow_accepts, flow_dU_list)."""
    si0 = sig[i]
    sj0 = sig[j]
    n_steps = schedule.shape[0]
    flow_dU_list = []
    flow_calls = 0
    flow_accepts = 0

    if n_steps == 1 and schedule[0] == 1.0:
        # matches ncmc_work_local's own n_steps==1 special case exactly (no propagation possible,
        # so no escort call either -- deterministic vertical work).
        e0 = _pair_env_e(x, sig, i, j, L)
        sig[i] = sj0
        sig[j] = si0
        e1 = _pair_env_e(x, sig, i, j, L)
        dW_step[0] = e1 - e0
        nS_step[0] = 0.0
        return e1 - e0, 0, 0, flow_dU_list

    N = x.shape[0]
    S = np.empty(N, dtype=np.int64)
    W = 0.0
    for k in range(n_steps):
        lam = schedule[k]
        e0 = _pair_env_e(x, sig, i, j, L)
        sig[i] = (1.0 - lam) * si0 + lam * sj0
        sig[j] = (1.0 - lam) * sj0 + lam * si0
        e1 = _pair_env_e(x, sig, i, j, L)
        dW = e1 - e0
        W += dW
        dW_step[k] = dW
        nS = build_local_set(x, i, j, r_loc, L, S)
        nS_step[k] = nS
        for _ in range(sweeps_per_step):
            local_sweep(x, sig, L, beta, i, j, r_loc, step, S, nS)

        if (flow is not None and escort_every is not None and escort_every > 0
                and (k + 1) % escort_every == 0):
            idx = select_block_by_pair(x, i, j, k_block, L)
            accepted, dU, _, _ = _flow_mh_move(x, sig, L, beta, idx, flow, gen, rng)
            flow_calls += 1
            flow_accepts += int(accepted)
            flow_dU_list.append(dU)

    return W, flow_calls, flow_accepts, flow_dU_list


# --------------------------------------------------------------------------------------------------
# checkpoint / flow loading
# --------------------------------------------------------------------------------------------------
def load_flow(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    ar = ck["args"]
    flow = SwapBlockFlow(k_max=ar["k_max"], m_env=ar["m_env"], hidden_nf=ar["hidden_nf"],
                          n_layers=ar["n_layers"], n_sig_bins=ar.get("n_sig_bins", 8),
                          base_w=ar.get("base_w", 0.35)).to(device)
    flow.load_state_dict(ck["state_dict"])
    flow.eval()
    return flow, ck


# --------------------------------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------------------------------
PROBE_BINS = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.9)]


def run_config(name, n_steps, escorted, T, k_block, r_loc, per_bin, flow, gen, rng, seed_base,
               x_frames, sigs_frames, L, device):
    beta = 1.0 / T
    schedule = (np.linspace(1.0 / n_steps, 1.0, n_steps) if n_steps > 0 else np.array([1.0]))
    escort_every = 100 if escorted else None
    results = {b: [] for b in PROBE_BINS}
    flow_acc_all = []
    need = {b: per_bin for b in PROBE_BINS}
    attempt = 0
    t0 = time.time()
    guard = 0
    while any(v > 0 for v in need.values()) and guard < 200000:
        guard += 1
        f_idx = int(rng.integers(x_frames.shape[0]))
        xf = x_frames[f_idx].astype(np.float64)
        sf = sigs_frames[f_idx].astype(np.float64)
        target_bin_di = attempt % len(PROBE_BINS)
        lo, hi = PROBE_BINS[target_bin_di]
        full_bin_idx = DS_BINS.index((lo, hi))
        i, j, ds, hit = select_pair(sf, xf.shape[0], rng, full_bin_idx)
        bb = PROBE_BINS[target_bin_di] if (lo <= ds < hi or (target_bin_di == len(PROBE_BINS) - 1 and ds <= hi)) else None
        if bb is None or need.get(bb, 0) <= 0:
            attempt += 1
            continue
        need[bb] -= 1
        attempt += 1

        x = xf.copy()
        sig = sf.copy()
        seed_numba(seed_base + guard)
        dW = np.zeros(len(schedule))
        nS = np.zeros(len(schedule))
        t_attempt0 = time.time()
        W, fcalls, faccepts, fdU = ncmc_work_local_escorted(
            x, sig, L, beta, i, j, schedule, 1, r_loc, STEP, dW, nS, flow, gen, rng, k_block,
            escort_every)
        t_attempt = time.time() - t_attempt0
        results[bb].append({"W": W, "ds": ds, "wall_s": t_attempt, "flow_calls": fcalls,
                             "flow_accepts": faccepts, "flow_dU": fdU})
        if fcalls > 0:
            flow_acc_all.append(faccepts / fcalls)

    elapsed = time.time() - t0
    print(f"[{name}] n={n_steps} escorted={escorted} done in {elapsed:.0f}s", flush=True)
    return {"name": name, "n_steps": n_steps, "escorted": escorted, "results": results,
            "elapsed_s": elapsed, "flow_acc_mean": float(np.mean(flow_acc_all)) if flow_acc_all else None}


def _print_summary(cfgs):
    print("\n=== ESCORT PROBE SUMMARY ===", flush=True)
    header = f"{'bin':>12} | " + " | ".join(f"{c['name']:>16}" for c in cfgs)
    print(header, flush=True)
    for b in PROBE_BINS:
        row = [f"{b}"]
        for c in cfgs:
            Ws = [r["W"] for r in c["results"][b]]
            walls = [r["wall_s"] for r in c["results"][b]]
            if Ws:
                row.append(f"Wmed={np.median(Ws):+7.2f} n={len(Ws):3d} t={np.mean(walls)*1000:6.0f}ms")
            else:
                row.append("no data")
        print(f"{b} | " + " | ".join(row[1:]), flush=True)
    for c in cfgs:
        if c["flow_acc_mean"] is not None:
            print(f"[{c['name']}] flow-MH mean acceptance = {c['flow_acc_mean']:.4f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=float, default=0.085)
    p.add_argument("--ckpt", type=str,
                    default="liquid_coupling_flow/artifacts/poly_escortflow_T0.085_k8_best.pt")
    p.add_argument("--run", type=int, default=3, help="held-out bank run")
    p.add_argument("--k_block", type=int, default=8)
    p.add_argument("--r_loc", type=float, default=3.0)
    p.add_argument("--per_bin", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="reports/logs-2026-07-17/poly_escort_ncmc_probe.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    D = torch.load(REPO / f"reports/logs-2026-07-17/poly_bank_fixed_run{a.run}.pt", weights_only=False)
    rec = D[(a.run, a.T)]
    L = float(rec["L"])
    x_frames = np.asarray(rec["x"], dtype=np.float64)
    sigs_frames = np.asarray(rec["sigs"], dtype=np.float64)

    ckpt_path = REPO / a.ckpt
    flow, ck = load_flow(ckpt_path, a.device)
    print(f"loaded escort flow {ckpt_path} (step {ck.get('step')}, fm_loss {ck.get('fm_loss'):.4f})",
          flush=True)

    seed_numba(a.seed)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)     # CPU generator (cuda generator crashes randn)

    print(f"poly ESCORT NCMC probe: T={a.T} r_loc={a.r_loc} per_bin={a.per_bin} run={a.run} "
          f"device={a.device}", flush=True)

    configs = [
        ("n200_escorted", 200, True),
        ("n200_plain", 200, False),
        ("n800_plain", 800, False),
    ]
    cfgs = []
    for cfg_i, (name, n_steps, escorted) in enumerate(configs):
        cfg = run_config(name, n_steps, escorted, a.T, a.k_block, a.r_loc, a.per_bin,
                          flow if escorted else None, gen, rng, a.seed + 1000 * (cfg_i + 1),
                          x_frames, sigs_frames, L, a.device)
        cfgs.append(cfg)
        torch.save({"configs": cfgs, "T": a.T, "r_loc": a.r_loc, "per_bin": a.per_bin,
                    "ckpt": str(ckpt_path)}, a.out)

    _print_summary(cfgs)
    torch.save({"configs": cfgs, "T": a.T, "r_loc": a.r_loc, "per_bin": a.per_bin,
                "ckpt": str(ckpt_path)}, a.out)
    print(f"ESCORT NCMC PROBE DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
