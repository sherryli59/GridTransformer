"""GATE (Task 5, plan docs/superpowers/plans/2026-07-16-joint-sigma-x-block-gate.md lines 568-595):
measure exact-MH acceptance of the learned joint (sigma-permutation + flow-relaxed-position) block
proposal vs block size k, at temperatures bracketing the swap-arrest of the NBC polydisperse glass,
against three baselines. Held-out configs only (bank run != training runs; default run=3).

THE MEASURED PROPOSAL (per attempt, exact MH):
  1. pick a held-out (run, T) bank frame + a random seed particle; block = k-NN of that particle
     (min-image, centroid-centered) -- identical selection/centering to poly_block_data.make_block_pair.
  2. draw an internal sigma-permutation pi = composition of 2 random transpositions of the block's k
     sigma-labels (retried until pi != identity; k=2 is degenerate under "2 transpositions" -- see
     two_transposition_perm -- so k=2 uses a single transposition instead, matching the plain-swap
     baseline it anchors).
  3. propose new block positions via the flow: x_new, logq_fwd = flow.propose(x_old, bin(sig∘pi), env_x,
     bin(env_sig), gen).
  4. reverse leg: logq_rev = flow.logq_of(x_old, x_center=x_new, bin(sig_old), env_x, bin(env_sig)).
     DERIVATION (see block_flow.py's propose): the flow's base distribution is z = x_center + N(0,
     BASE_W^2), added PER PARTICLE to the x_center array -- x_center is not a scalar centroid, it is the
     full [k,3] array of per-mover base points. propose(x_old, ...) uses base=x_old (per particle).
     Therefore the reverse leg -- "if we were standing at x_new and proposed back toward x_old with the
     ORIGINAL labels" -- has base=x_new, i.e. x_center=x_new, taken AS IS. Because env_x never moves
     during a block MH step (the env is frozen, only the block is proposed), x_old, x_new and env_x are
     already expressed in one shared centered coordinate frame (centered once, on the ORIGINAL block
     centroid, by select_block) for the whole attempt -- no recentering/shift is needed between the
     forward and reverse legs. This differs from a naive reading of "x_center = the centroid of x_new";
     that reading only applies if propose recentered its base at a single scalar centroid, which it does
     not (it is per-particle).
  5. dU_total = U(x_new, sig∘pi) - U(x_old, sig_old), computed as row_e sums over BLOCK particles only
     against the FULL (block+env) system, MINUS the block-block energy counted once via total_U on the
     block-only sub-system (double-count correction -- see block_dU). In --smoke this is cross-checked
     against a full-system total_U delta (env-env term cancels exactly) to < 1e-6.
  6. A = min(1, exp(-beta*dU_total + logq_rev - logq_fwd)); accepted = Bernoulli(A). The uniform
     2-transposition proposal is symmetric (pi and pi^-1 equally likely) so its factor cancels and is
     never included.
  7. every attempt's (dU, logq_fwd, logq_rev, accepted) is stored in full (no means), incrementally saved
     per (T,k) cell.

BASELINES (same run, same held-out frames):
  (a) plain pair-swap (k=2 anchor): classic single-swap Metropolis on two random FULL-system particles,
      no flow -- reported alongside k=2 cells only.
  (b) x-only flow (identity permutation): same measured-proposal machinery with pi=identity (sigma
      unchanged) -- isolates the position-flow channel's own acceptance from the sigma-permutation's
      contribution.
  (c) sigma-permutation + block-local MC relax (RELAX_SW sweeps, poly_block_data._block_relax): a
      non-learned MTM-style comparator. Its "acceptance" at the outer level is NOT well-defined (RELAX_SW
      sweeps are themselves an equilibrating MC, not a single proposal to accept/reject) -- only its raw
      dU distribution is reported, clearly labeled as context.
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

from liquid_coupling_flow.poly.model import row_e, total_U, seed_numba  # noqa: E402
from liquid_coupling_flow.poly.block_flow import PolyBlockFlow, sigma_to_bin  # noqa: E402
from poly_block_data import _block_relax, RELAX_SW  # noqa: E402  (needs sys.path insert above)

BANK_TMPL = "reports/logs-2026-07-17/poly_tau_curves_run{run}.pt"


# --------------------------------------------------------------------------------------------------
# block selection (identical k-NN + centroid-centering convention to poly_block_data.make_block_pair)
# --------------------------------------------------------------------------------------------------
def select_block(x, sig, L, k, rng):
    """x[N,3], sig[N], -> idx[k], x_blk_centered[k,3], sig_blk[k], env_x_centered[N-k,3], env_sig[N-k].
    Centered on the RAW block centroid (cen = x[idx].mean(0)), min-image wrapped -- same as
    make_block_pair so the flow (trained on make_block_pair pairs) sees the same coordinate convention."""
    n = x.shape[0]
    s0 = rng.integers(n)
    d = x - x[s0]
    d -= L * np.round(d / L)
    idx = np.argsort((d ** 2).sum(1))[:k].astype(np.int64)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    cen = x[idx].mean(0)

    def cent(a):
        b = a - cen
        return b - L * np.round(b / L)

    return idx, cent(x[idx]), sig[idx].copy(), cent(x[~mask]), sig[~mask].copy()


def two_transposition_perm(k, rng):
    """pi = composition of 2 random transpositions of {0..k-1}, retried until pi != identity.
    k=2 is degenerate: the only transposition is swap(0,1), and composing it with itself twice is
    ALWAYS identity (infinite retry) -- so k=2 uses a single transposition instead (this is exactly
    the plain pair-swap baseline (a) it anchors)."""
    if k == 2:
        return np.array([1, 0])
    while True:
        perm = np.arange(k)
        for _ in range(2):
            i, j = rng.choice(k, size=2, replace=False)
            perm[i], perm[j] = perm[j], perm[i]
        if not np.all(perm == np.arange(k)):
            return perm


def block_dU(x_blk, sig_blk, env_x, env_sig, L):
    """Energy of the block's own interactions (block-block ONCE + block-env FULLY) via the double-count
    correction: S = sum_{i in block} row_e(i; full system) double-counts block-block pairs (each pair
    seen from both endpoints) and single-counts block-env pairs; E_bb = total_U(block-only) recovers the
    block-block term counted once; S - E_bb = (2*E_bb + E_be) - E_bb = E_bb + E_be, each pair once."""
    full_x = np.concatenate([x_blk, env_x], axis=0).astype(np.float64)
    full_sig = np.concatenate([sig_blk, env_sig], axis=0).astype(np.float64)
    k = x_blk.shape[0]
    S = 0.0
    for i in range(k):
        S += row_e(full_x, full_sig, i, full_x[i, 0], full_x[i, 1], full_x[i, 2], L)
    E_bb = total_U(x_blk.astype(np.float64), sig_blk.astype(np.float64), L)
    return S - E_bb


def full_system_dU_check(x_old_blk, sig_old_blk, x_new_blk, sig_new_blk, env_x, env_sig, L):
    """Independent cross-check: full-system total_U delta (env-env term cancels exactly since env is
    frozen across the move) vs block_dU's incremental method. Used only in --smoke."""
    full_x_old = np.concatenate([x_old_blk, env_x], axis=0).astype(np.float64)
    full_sig_old = np.concatenate([sig_old_blk, env_sig], axis=0).astype(np.float64)
    full_x_new = np.concatenate([x_new_blk, env_x], axis=0).astype(np.float64)
    full_sig_new = np.concatenate([sig_new_blk, env_sig], axis=0).astype(np.float64)
    return total_U(full_x_new, full_sig_new, L) - total_U(full_x_old, full_sig_old, L)


def swap_trial(x, sig, L, beta, i, j, rng):
    """Single independent plain-swap Metropolis trial on FULL-system indices i,j (baseline (a)); always
    reverts sig afterward so repeated attempts on the same frame are independent draws, not a chain --
    matching the measured proposal's "fresh draw per attempt" convention."""
    e0 = (row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
          + row_e(x, sig, j, x[j, 0], x[j, 1], x[j, 2], L))
    si = sig[i]
    sig[i] = sig[j]
    sig[j] = si
    e1 = (row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
          + row_e(x, sig, j, x[j, 0], x[j, 1], x[j, 2], L))
    dU = e1 - e0
    A = min(1.0, np.exp(-beta * dU))
    accepted = rng.random() < A
    sj = sig[i]
    sig[i] = sig[j]
    sig[j] = sj
    return dU, accepted


def load_flow(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    ar = ck["args"]
    flow = PolyBlockFlow(k_max=ar["k_max"], m_env=ar["m_env"], hidden_nf=ar["hidden_nf"],
                          n_layers=ar["n_layers"]).to(device)
    flow.load_state_dict(ck["state_dict"])
    flow.eval()
    return flow, ck


def run_cell(T, k, run, ckpt_tmpl, n_attempts, device, rng, gen, smoke, out):
    bank_path = REPO / BANK_TMPL.format(run=run)
    if not bank_path.exists():
        print(f"T={T} k={k}: NO BANK at {bank_path}, skipping", flush=True)
        return None
    bank = torch.load(bank_path, weights_only=False)
    key = (run, T)
    if key not in bank:
        print(f"T={T} k={k}: T not yet in run-{run} bank ({bank_path.name}), skipping", flush=True)
        return None
    rec = bank[key]
    x_frames = np.asarray(rec["x"], dtype=np.float64)         # [n_frames, N, 3]
    sig_full = np.asarray(rec["sig"], dtype=np.float64)       # [N]
    L = float(rec["L"])
    beta = 1.0 / T

    ckpt_path = Path(str(ckpt_tmpl).format(T=T, k=k))
    have_flow = ckpt_path.exists()
    if have_flow:
        flow, ck = load_flow(ckpt_path, device)
        print(f"T={T} k={k}: loaded flow {ckpt_path} (step {ck.get('step')}, "
              f"fm_loss {ck.get('fm_loss'):.4f})", flush=True)
    else:
        flow = None
        print(f"T={T} k={k}: NO CKPT at {ckpt_path} -- skipping flow-dependent measurements "
              f"(joint + x-only), running baselines (a)/(c) only", flush=True)

    cell = {"T": T, "k": k, "run": run, "L": L, "beta": beta, "n": 0,
            "dU": [], "logq_fwd": [], "logq_rev": [], "accepted": [],
            "dU_b": [], "logq_fwd_b": [], "logq_rev_b": [], "accepted_b": [],
            "dU_c": [],
            "dU_a": [], "accepted_a": [] if k == 2 else None,
            "dU_check_max_abs_diff": None}

    dcheck = []
    for m in range(n_attempts):
        f_idx = int(rng.integers(x_frames.shape[0]))
        x = x_frames[f_idx]
        idx, x_old_blk, sig_old_blk, env_x, env_sig = select_block(x, sig_full, L, k, rng)

        # --- baseline (a): plain pair-swap, FULL-system indices, k==2 cell only ---
        if k == 2:
            i, j = rng.choice(x.shape[0], size=2, replace=False)
            dU_a, acc_a = swap_trial(x.copy(), sig_full.copy(), L, beta, int(i), int(j), rng)
            cell["dU_a"].append(dU_a)
            cell["accepted_a"].append(acc_a)

        # --- baseline (c): sigma-perm + block-local MC relax (raw dU, no acceptance) ---
        perm_c = two_transposition_perm(k, rng)
        sig_new_c = sig_old_blk[perm_c]
        x_full_c = np.concatenate([x_old_blk, env_x], axis=0).copy()
        sig_full_c = np.concatenate([sig_new_c, env_sig], axis=0).copy()
        block_idx = np.arange(k, dtype=np.int64)
        seed_numba(int(rng.integers(1 << 30)))
        _block_relax(x_full_c, sig_full_c, L, beta, block_idx, RELAX_SW, 0.1)
        dU_c = block_dU(x_full_c[block_idx], sig_new_c, env_x, env_sig, L) - \
            block_dU(x_old_blk, sig_old_blk, env_x, env_sig, L)
        cell["dU_c"].append(dU_c)

        if flow is not None:
            # --- measured joint proposal: sigma-perm (2 transpositions) + flow position propose ---
            perm = two_transposition_perm(k, rng)
            sig_new_blk = sig_old_blk[perm]
            x_new_blk, logq_fwd = flow.propose(x_old_blk, sigma_to_bin(sig_new_blk), env_x,
                                                sigma_to_bin(env_sig), gen)
            logq_rev = flow.logq_of(x_old_blk, x_new_blk, sigma_to_bin(sig_old_blk), env_x,
                                     sigma_to_bin(env_sig))
            dU_total = block_dU(x_new_blk.astype(np.float64), sig_new_blk, env_x, env_sig, L) - \
                block_dU(x_old_blk, sig_old_blk, env_x, env_sig, L)
            A = float(min(1.0, np.exp(-beta * dU_total + logq_rev - logq_fwd)))
            accepted = bool(rng.random() < A)
            cell["dU"].append(dU_total)
            cell["logq_fwd"].append(logq_fwd)
            cell["logq_rev"].append(logq_rev)
            cell["accepted"].append(accepted)

            if smoke:
                dU_check = full_system_dU_check(x_old_blk, sig_old_blk, x_new_blk.astype(np.float64),
                                                 sig_new_blk, env_x, env_sig, L)
                diff = abs(dU_check - dU_total)
                # Tolerance is scale-aware: block_dU and the full-system total_U delta are the SAME
                # pairwise terms summed in different orders (S sums via k row_e calls of ~N terms each,
                # full_system_dU_check sums via two independent O(N^2) total_U passes), so any mismatch
                # is float64 summation-order rounding, not formula error -- and its ABSOLUTE size scales
                # with the magnitude of the (possibly huge, near-singular-clash) energies being
                # differenced, not with a fixed constant. An undertrained smoke flow (300 FM steps) can
                # propose genuine hard-core overlaps where v(r) ~ (sigma_ij/r)^12 blows up to dU ~ 1e8,
                # at which scale float64's ~1e-16 relative epsilon alone gives an absolute noise floor of
                # ~1e-8 per term summed over ~2400 terms -- so 1e-6 absolute is only a meaningful bar at
                # O(1-100) energies. Interpret "agreement to 1e-6" as that literal absolute band at normal
                # scales, scaled up by max(1, |dU_total|) at clash scales -- still a far tighter RELATIVE
                # bar (~1e-14, observed) than a naive fixed-1e-6 absolute check would enforce.
                tol = 1e-6 * max(1.0, abs(dU_total), abs(dU_check))
                dcheck.append(diff)
                assert diff < tol, (
                    f"double-count check FAILED at attempt {m}: block_dU={dU_total:.10f} "
                    f"full_total_U_delta={dU_check:.10f} diff={diff:.3e} tol={tol:.3e}")
                assert np.isfinite(logq_fwd) and np.isfinite(logq_rev), (
                    f"non-finite logq at attempt {m}: fwd={logq_fwd} rev={logq_rev}")

            # --- baseline (b): x-only flow, identity permutation ---
            x_new_b, logq_fwd_b = flow.propose(x_old_blk, sigma_to_bin(sig_old_blk), env_x,
                                                sigma_to_bin(env_sig), gen)
            logq_rev_b = flow.logq_of(x_old_blk, x_new_b, sigma_to_bin(sig_old_blk), env_x,
                                       sigma_to_bin(env_sig))
            dU_b = block_dU(x_new_b.astype(np.float64), sig_old_blk, env_x, env_sig, L) - \
                block_dU(x_old_blk, sig_old_blk, env_x, env_sig, L)
            A_b = float(min(1.0, np.exp(-beta * dU_b + logq_rev_b - logq_fwd_b)))
            accepted_b = bool(rng.random() < A_b)
            cell["dU_b"].append(dU_b)
            cell["logq_fwd_b"].append(logq_fwd_b)
            cell["logq_rev_b"].append(logq_rev_b)
            cell["accepted_b"].append(accepted_b)

        cell["n"] = m + 1
        if (m + 1) % max(1, n_attempts // 5) == 0 or (m + 1) == n_attempts:
            _save_cell(out, T, k, cell)

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
    if cell["accepted"]:
        acc = float(np.mean(cell["accepted"]))
        dU_med = float(np.median(cell["dU"]))
        gap_med = float(np.median(np.asarray(cell["logq_rev"]) - np.asarray(cell["logq_fwd"])))
    else:
        acc, dU_med, gap_med = float("nan"), float("nan"), float("nan")
    print(f"T={T} k={k} acc={acc:.4f} n={n} dU_med={dU_med:+.2f} logq_gap_med={gap_med:+.2f}",
          flush=True)
    if cell["accepted_b"]:
        acc_b = float(np.mean(cell["accepted_b"]))
        print(f"  baseline(b) x-only (identity perm): acc_b={acc_b:.4f}", flush=True)
    if cell["dU_c"]:
        print(f"  baseline(c) sigma-perm+RELAX_SW (dU only, acceptance not well-defined): "
              f"dU_c_med={np.median(cell['dU_c']):+.2f}", flush=True)
    if cell["accepted_a"]:
        acc_a = float(np.mean(cell["accepted_a"]))
        print(f"  baseline(a) plain pair-swap (k=2 anchor): acc_a={acc_a:.4f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=str, default="0.2", help="comma list of temperatures")
    p.add_argument("--k", type=str, default="2,4,8,16,24", help="comma list of block sizes")
    p.add_argument("--attempts", type=int, default=2000)
    p.add_argument("--ckpt_tmpl", type=str,
                    default="liquid_coupling_flow/artifacts/poly_blockflow_T{T}_k{k}_best.pt")
    p.add_argument("--run", type=int, default=3, help="held-out bank run")
    p.add_argument("--smoke", type=int, default=None,
                    help="if set, overrides --attempts and runs the double-count/finite-logq checks")
    p.add_argument("--out", type=str, default="reports/logs-2026-07-17/poly_gate_acceptance.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    T_list = [float(t) for t in a.T.split(",")]
    k_list = [int(x) for x in a.k.split(",")]
    n_attempts = a.smoke if a.smoke is not None else a.attempts
    smoke = a.smoke is not None

    seed_numba(a.seed)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)

    print(f"poly GATE acceptance: T={T_list} k={k_list} run={a.run} attempts={n_attempts} "
          f"smoke={smoke} device={a.device} out={a.out}", flush=True)

    t0 = time.time()
    for T in T_list:
        for k in k_list:
            run_cell(T, k, a.run, a.ckpt_tmpl, n_attempts, a.device, rng, gen, smoke, a.out)
    print(f"GATE ACCEPTANCE DONE ({time.time()-t0:.0f}s) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
