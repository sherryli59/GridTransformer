"""THE acceptance-vs-|Delta sigma| GATE for SwapBlockFlow (poly Task J3b, PHASE 1 of
docs/superpowers/specs/2026-07-17-joint-sigma-x-continuous-flow-amendment.md's DECISION 2026-07-17).

THE MEASURED PROPOSAL (per attempt, exact MH, PHASE-1 canonical fixed-composition ensemble):
  1. pick a held-out (run, T) FIXED bank frame (poly_bank_fixed_run{run}.pt -- frame j paired with
     sigs[j]; the STALE poly_tau_curves_run*.pt banks are NEVER used, see poly_rebank.py's docstring
     for why) + a random seed particle -> k-NN block (min-image, centroid-centered on the ORIGINAL
     block centroid -- identical selection convention to poly_swap_pairs.make_swap_pair /
     poly_gate_acceptance.select_block).
  2. pick a pair (a,b) of local block indices STRATIFIED over |Delta sigma| bins (round-robin target
     bin, rejection-sampled with a retry cap -- `select_block_and_pair` below mirrors
     poly_swap_pairs.make_swap_pair's pair-selection algorithm VERBATIM, including the fixed
     top-bin-inclusive condition at that module's HEAD; not imported because make_swap_pair also runs
     the block relax this gate does not need -- the flow proposes positions instead). Stratification
     is for attempt COVERAGE only; all analysis below bins by the REALIZED |Delta sigma|.
  3. sig_start = block sigmas pre-swap; sig_end = sig_start with (a,b) exchanged (the classical swap
     endpoint). Flow proposal x_new, logq_fwd = flow.propose_batch(x_old, sig_start, sig_end, env_x,
     env_sig, gen) -- attempts are assembled into groups of --batch and pushed through the flow
     together (a serial per-attempt pilot measured 4.4 s/attempt; batching targets >10 attempts/s).
  4. reverse MH leg, per swap_flow.py's module docstring MH USAGE section (reverse path = swapped ->
     old): logq_rev = flow.logq_of_batch(x_old, x_center=x_new, sig_start=sig_end, sig_end=sig_start,
     env_x, env_sig).
  5. dU_total = U(x_new, sig_end) - U(x_old, sig_start), via poly_gate_acceptance.block_dU (REUSED,
     not reimplemented) -- both positions AND sigma assignment change between the two evaluations.
     In --smoke this is cross-checked against poly_gate_acceptance.full_system_dU_check (also reused)
     to a scale-aware 1e-6 tolerance (see that module for the float64-summation-order rationale).
  6. A = min(1, exp(-beta*dU_total + logq_rev - logq_fwd)); accepted = Bernoulli(A).

BASELINE (same pair, same frame, no flow): classical direct swap -- FROZEN positions (x_old), sigma
swapped (sig_end) -- dU_cl = U(x_old, sig_end) - U(x_old, sig_start), A_cl = min(1, exp(-beta*dU_cl)).
This is the curve the flow must beat (measured in the amendment's DECISION section: acc_cl ~0.44 for
|dsig|<0.1, ~0.026 for 0.1-0.2, ~0.0000 (0/524) beyond 0.2).

Every attempt's (ds_realized, dU, dU_cl, logq_fwd, logq_rev, accepted, accepted_cl) is stored in full
(no means), incrementally saved per (T,k) cell every ~400 attempts.
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
from poly_gate_acceptance import block_dU, full_system_dU_check  # noqa: E402  (REUSE, see module docstring)

BANK_TMPL = "reports/logs-2026-07-17/poly_bank_fixed_run{run}.pt"
DS_BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.9)]
MAX_RETRIES = 200


# --------------------------------------------------------------------------------------------------
# block + pair selection (mirrors poly_swap_pairs.make_swap_pair's algorithm VERBATIM up through the
# swap; stops before the block relax -- the gate's "new" positions come from the flow, not from MC)
# --------------------------------------------------------------------------------------------------
def select_block_and_pair(x, sig, L, k, rng, target_bin):
    """x[N,3], sig[N] (a single held-out frame + its per-frame sigmas) -> a swap-endpoint attempt dict:
    idx, x_old (centered), env_x (centered), env_sig, sig_start, sig_end, pair (a,b), ds (realized
    |sigma_a - sigma_b|), retry_hit. Block selection = min-image k-NN of a random seed particle,
    centered on the block's own (pre-swap) centroid -- identical to poly_swap_pairs.make_swap_pair /
    poly_gate_acceptance.select_block. Pair selection = rejection-sample local indices (a,b) until
    |sigma_a - sigma_b| lands in DS_BINS[target_bin] (capped at MAX_RETRIES, falls back to the
    closest-matching candidate seen so the caller's attempt budget always terminates) -- SAME
    MAX_RETRIES/DS_BINS and the SAME fixed top-bin-inclusive condition as poly_swap_pairs.py's HEAD
    (review 96263ab caught the original OR-clause dropping `lo` for the top bin)."""
    n = x.shape[0]
    s0 = rng.integers(n)
    d = x - x[s0]
    d -= L * np.round(d / L)
    idx = np.argsort((d ** 2).sum(1))[:k].astype(np.int64)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    x_old = x[idx].copy()
    sig_old = sig[idx].copy()

    lo, hi = DS_BINS[target_bin]
    a, b = 0, 1
    best_d, best_pair = -1.0, (0, 1)
    hit = False
    for _ in range(MAX_RETRIES):
        a = int(rng.integers(k))
        b = int(rng.integers(k - 1))
        if b >= a:
            b += 1
        ds = abs(float(sig_old[a] - sig_old[b]))
        if ds > best_d:
            best_d, best_pair = ds, (a, b)
        if lo <= ds and (ds < hi or (target_bin == len(DS_BINS) - 1 and ds <= hi)):
            hit = True
            break
    if not hit:
        a, b = best_pair

    sig_end = sig_old.copy()
    sig_end[a], sig_end[b] = sig_end[b], sig_end[a]
    ds_realized = abs(float(sig_old[a] - sig_old[b]))

    cen = x_old.mean(0)

    def cent(arr):
        c = arr - cen
        return c - L * np.round(c / L)

    return {
        "idx": idx, "env_x": cent(x[~mask]), "env_sig": sig[~mask].copy(),
        "x_old": cent(x_old), "sig_start": sig_old, "sig_end": sig_end,
        "pair": (a, b), "ds": ds_realized, "retry_hit": hit,
    }


def _bin_of(ds_val):
    for bi, (lo, hi) in enumerate(DS_BINS):
        if lo <= ds_val < hi or (bi == len(DS_BINS) - 1 and ds_val <= hi):
            return bi
    return len(DS_BINS) - 1


# --------------------------------------------------------------------------------------------------
# bank / flow loading
# --------------------------------------------------------------------------------------------------
def load_bank(run, T):
    path = REPO / BANK_TMPL.format(run=run)
    assert path.exists(), f"NO FIXED BANK at {path}"
    D = torch.load(path, weights_only=False)
    key = (run, T)
    assert key in D, f"fixed bank run{run} has no T={T} yet ({path.name})"
    rec = D[key]
    assert "sigs" in rec, "fixed bank must carry per-frame sigs"
    return rec


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
# one (T,k) cell
# --------------------------------------------------------------------------------------------------
def run_cell(T, k, run, ckpt_tmpl, n_attempts, batch, device, rng, gen, smoke, out):
    rec = load_bank(run, T)
    x_frames = np.asarray(rec["x"], dtype=np.float64)          # [n_frames, N, 3]
    sigs_frames = np.asarray(rec["sigs"], dtype=np.float64)    # [n_frames, N]
    L = float(rec["L"])
    beta = 1.0 / T

    ckpt_path = Path(str(ckpt_tmpl).format(T=T, k=k))
    assert ckpt_path.exists(), f"NO CKPT at {ckpt_path}"
    flow, ck = load_flow(ckpt_path, device)
    print(f"T={T} k={k}: loaded flow {ckpt_path} (step {ck.get('step')}, "
          f"fm_loss {ck.get('fm_loss'):.4f})", flush=True)

    cell = {"T": T, "k": k, "run": run, "L": L, "beta": beta, "n": 0,
            "ds": [], "dU": [], "dU_cl": [], "logq_fwd": [], "logq_rev": [],
            "accepted": [], "accepted_cl": [], "dU_check_max_abs_diff": None}

    dcheck = []
    attempt_idx = 0
    n_done = 0
    t0 = time.time()
    while n_done < n_attempts:
        B = min(batch, n_attempts - n_done)

        x_old_b, sig_start_b, sig_end_b, env_x_b, env_sig_b, ds_list = [], [], [], [], [], []
        for _ in range(B):
            j = int(rng.integers(x_frames.shape[0]))
            xf = x_frames[j]
            sf = sigs_frames[j]
            target_bin = attempt_idx % len(DS_BINS)
            rp = select_block_and_pair(xf, sf, L, k, rng, target_bin)
            x_old_b.append(rp["x_old"]); sig_start_b.append(rp["sig_start"])
            sig_end_b.append(rp["sig_end"]); env_x_b.append(rp["env_x"])
            env_sig_b.append(rp["env_sig"]); ds_list.append(rp["ds"])
            attempt_idx += 1

        x_old_b = np.stack(x_old_b); sig_start_b = np.stack(sig_start_b)
        sig_end_b = np.stack(sig_end_b); env_x_b = np.stack(env_x_b); env_sig_b = np.stack(env_sig_b)

        x_new_b, logq_fwd_b = flow.propose_batch(x_old_b, sig_start_b, sig_end_b, env_x_b, env_sig_b, gen)
        # reverse leg: path swapped -> old (sig_start/sig_end SWAPPED relative to the forward call)
        logq_rev_b = flow.logq_of_batch(x_old_b, x_new_b, sig_end_b, sig_start_b, env_x_b, env_sig_b)

        for i in range(B):
            x_new_i = x_new_b[i].astype(np.float64)
            e_old = block_dU(x_old_b[i], sig_start_b[i], env_x_b[i], env_sig_b[i], L)
            e_new_flow = block_dU(x_new_i, sig_end_b[i], env_x_b[i], env_sig_b[i], L)
            e_new_cl = block_dU(x_old_b[i], sig_end_b[i], env_x_b[i], env_sig_b[i], L)
            dU = e_new_flow - e_old
            dU_cl = e_new_cl - e_old

            A = float(min(1.0, np.exp(-beta * dU + logq_rev_b[i] - logq_fwd_b[i])))
            accepted = bool(rng.random() < A)
            A_cl = float(min(1.0, np.exp(-beta * dU_cl)))
            accepted_cl = bool(rng.random() < A_cl)

            cell["ds"].append(ds_list[i]); cell["dU"].append(dU); cell["dU_cl"].append(dU_cl)
            cell["logq_fwd"].append(float(logq_fwd_b[i])); cell["logq_rev"].append(float(logq_rev_b[i]))
            cell["accepted"].append(accepted); cell["accepted_cl"].append(accepted_cl)

            if smoke:
                dU_check = full_system_dU_check(x_old_b[i], sig_start_b[i], x_new_i, sig_end_b[i],
                                                 env_x_b[i], env_sig_b[i], L)
                diff = abs(dU_check - dU)
                tol = 1e-6 * max(1.0, abs(dU), abs(dU_check))
                dcheck.append(diff)
                assert diff < tol, (
                    f"double-count check FAILED at attempt {n_done + i}: block_dU_dU={dU:.10f} "
                    f"full_total_U_delta={dU_check:.10f} diff={diff:.3e} tol={tol:.3e}")
                assert np.isfinite(logq_fwd_b[i]) and np.isfinite(logq_rev_b[i]), (
                    f"non-finite logq at attempt {n_done + i}: fwd={logq_fwd_b[i]} rev={logq_rev_b[i]}")

        n_done += B
        cell["n"] = n_done
        if n_done % 400 < B or n_done == n_attempts:
            _save_cell(out, T, k, cell)
            print(f"  T={T} k={k}: {n_done}/{n_attempts} attempts "
                  f"({n_done / max(time.time() - t0, 1e-9):.2f} attempts/s)", flush=True)

    if smoke and dcheck:
        cell["dU_check_max_abs_diff"] = float(max(dcheck))
        print(f"T={T} k={k}: double-count check PASS, max|diff|={cell['dU_check_max_abs_diff']:.3e} "
              f"over {len(dcheck)} attempts", flush=True)

    _save_cell(out, T, k, cell)
    _print_cell(cell)
    return cell


def _save_cell(out_path, T, k, cell):
    out = torch.load(out_path, weights_only=False) if Path(out_path).exists() else {}
    out[(T, k)] = cell
    torch.save(out, out_path)


def _print_cell(cell):
    T, k, n = cell["T"], cell["k"], cell["n"]
    ds_arr = np.asarray(cell["ds"])
    acc = np.asarray(cell["accepted"], dtype=bool)
    acc_cl = np.asarray(cell["accepted_cl"], dtype=bool)
    print(f"T={T} k={k} n={n} overall acc_flow={acc.mean():.4f} acc_cl={acc_cl.mean():.4f}", flush=True)
    print(f"  {'ds_bin':>12} | {'n':>6} | {'acc_flow':>9} {'acc_cl':>9}", flush=True)
    for bi, (lo, hi) in enumerate(DS_BINS):
        m = np.array([_bin_of(v) == bi for v in ds_arr])
        cnt = int(m.sum())
        af = float(acc[m].mean()) if cnt > 0 else float("nan")
        ac = float(acc_cl[m].mean()) if cnt > 0 else float("nan")
        print(f"  {lo:5.2f}-{hi:<5.2f} | {cnt:6d} | {af:9.4f} {ac:9.4f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=str, default="0.085", help="comma list of temperatures")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--attempts", type=int, default=2000)
    p.add_argument("--ckpt_tmpl", type=str,
                    default="liquid_coupling_flow/artifacts/poly_swapflow_T{T}_k{k}_best.pt")
    p.add_argument("--run", type=int, default=3, help="held-out bank run")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--smoke", type=int, default=None,
                    help="if set, overrides --attempts and runs the double-count/finite-logq checks")
    p.add_argument("--out", type=str, default="reports/logs-2026-07-17/poly_gate_swap.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    T_list = [float(t) for t in a.T.split(",")]
    n_attempts = a.smoke if a.smoke is not None else a.attempts
    smoke = a.smoke is not None

    seed_numba(a.seed)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)     # CPU generator (cuda generator crashes randn)

    print(f"poly GATE swap-endpoint acceptance-vs-|dsig|: T={T_list} k={a.k} run={a.run} "
          f"attempts={n_attempts} batch={a.batch} smoke={smoke} device={a.device} out={a.out}",
          flush=True)

    t0 = time.time()
    for T in T_list:
        run_cell(T, a.k, a.run, a.ckpt_tmpl, n_attempts, a.batch, a.device, rng, gen, smoke, a.out)
    print(f"GATE SWAP DONE ({time.time() - t0:.0f}s) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
