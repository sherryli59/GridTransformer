"""Training-pair generator for the FLOW-ESCORTED NCMC propagator (poly ML item 2, spec: escort the
sigma(t)-conditioned SwapBlockFlow -- liquid_coupling_flow/poly/swap_flow.py -- into the NCMC protocol
(reports/logs-2026-07-17/poly_ncmc_v2.py's ncmc_work_local) as a fixed-lambda detailed-balanced
relaxation kernel).

WHY A NEW PAIR DISTRIBUTION (not poly_swap_pairs.py's): phase-1 killed SwapBlockFlow as a ONE-SHOT
kernel spanning sig_start (pre-swap) -> sig_end (post-swap) in a single flow leap -- the vertical
free-energy cost (25-50 kT for dissimilar pairs, poly_gate_verdict_phase1.md) is a THERMODYNAMIC wall,
not a learning one. Inside NCMC, no single kernel application ever needs to cross that gap: the
lambda-schedule already does the crossing in small increments, and between switches ordinary local
Metropolis sweeps (ncmc_work_local's local_sweep) relax the block. The escort's job is narrower and
much easier -- accelerate exactly THAT per-lambda-step relaxation, at a FIXED, already-small-|dsigma|
interpolated diameter assignment. So training data must show the flow examples of that regime:
"a block that already contains two mid-relaxation, non-lattice, sigma(lambda) diameters -- get it
closer to its own local equilibrium a bit faster than plain local MC would."

PROCEDURE (per pair):
  1. Draw a fixed-bank frame (poly_bank_fixed_run{1,2}.pt -- see poly_rebank.py for why the ORIGINAL
     poly_tau_curves banks are unusable: they pair frame x[j] with a STALE end-of-rung sigma).
  2. Select a swap pair (i, j) by the SAME convention as poly_ncmc_v2.py's __main__ driver: draw
     FULLY RANDOM (i, j) over the whole system (uniform, no spatial constraint -- this is the
     established NCMC selection-symmetry convention, see that module's docstring), retried
     (capped) until |sigma_i - sigma_j| lands in a target DS_BINS window (round-robin across
     pairs for attempt coverage; falls back to the closest candidate seen so generation always
     terminates, exactly mirroring poly_swap_pairs.py's / poly_gate_swap.py's retry-cap pattern).
  3. Run poly_ncmc_v2's PHYSICAL PROTOCOL up to a RANDOMLY CHOSEN checkpoint step (replicated here,
     not imported as one opaque numba call, so a checkpoint can be taken mid-schedule; building
     blocks build_local_set/local_sweep ARE imported verbatim from poly_ncmc_v2 -- never
     reimplemented, per the DISJOINT-FILES contract: import, never modify that module): for each
     lambda step k < checkpoint_step, interpolate sigma_i/sigma_j, rebuild the local propagation
     set S once, run sweeps_per_step local sweeps. This produces a REALISTIC "we are mid-protocol,
     env has already relaxed some" state -- not an artificial fresh jump.
  4. Define the k=8 PAIR-CENTERED block: the k nearest particles by min(dist-to-i, dist-to-j)
     (min-image). This is well-defined for ANY (i,j) separation (unlike a single-seed k-NN block,
     which could exclude one of i/j if they end up far apart) and guarantees i, j THEMSELVES are
     always the two closest members (self-distance 0) -- so they are always IN the block, matching
     the escort's deployment role (poly_escort_ncmc.py applies the flow-MH move to exactly this
     block definition).
  5. Snapshot x_before = block positions at the checkpoint (BEFORE any escort-training relaxation);
     sig_start = sig_end = the block's CURRENT (checkpoint-time, lambda-interpolated) per-particle
     sigma -- an IDENTITY PATH by construction (only i, j's sigma is time-dependent and it is
     ALREADY interpolated to this checkpoint's value; nothing further changes it in steps 5-6). env
     (all non-block particles) is snapshotted ONCE, at the same checkpoint moment, and held fixed
     for the rest of this pair's construction -- matching SwapBlockFlow's frozen-env contract (the
     flow never moves env; env_x/env_sig are pure conditioning).
  6. Run `escort_sweeps` (default 100) MORE local sweeps (SAME build_local_set/local_sweep kernel,
     same r_loc, sigma frozen -- no further lambda switching) to get x_after. Sweeps are NOT
     restricted to the k=8 block (they use the same local-region kernel real deployment uses
     between escort calls) -- env particles inside the local set S may drift a little during these
     100 sweeps even though the SNAPSHOT env_x/env_sig (step 5) does not track that drift. This is
     an intentional, deployment-matched approximation: in real NCMC, the escort's own flow-MH move
     ALSO only ever sees a frozen env snapshot while the surrounding protocol's ordinary local
     sweeps keep evolving it -- training on the true block-conditional-on-env relaxation target
     (not an artificially env-frozen relax) is the more faithful choice, and env drift over 100
     sweeps at r_loc=3.0 is small relative to the block's own accommodation motion.
  7. Pair dict = poly_train_swapflow.py's EXACT expected format (x_old, x_new, sig_start, sig_end,
     env_x, env_sig [+ ds/L/beta, reused for provenance/plots only]), centered EXACTLY like
     poly_swap_pairs.py: center = x_before.mean(0), all four arrays (x_old, x_new, env_x) shifted by
     that center and min-image wrapped.

Usage:
  poly_escort_pairs.py --T 0.085 --k 8 --npairs 3000 [--runs 1,2] [--r_loc 3.0] [--n_steps 800]
                        [--escort_sweeps 100] [--step 0.12] [--out ...]
Output: poly_escortpairs_T{T}_k{k}.pt (list of pair dicts + a 'meta' entry), matching
poly_swap_pairs.py's save format so poly_train_swapflow.py needs zero changes.
"""
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
from poly_ncmc_v2 import build_local_set, local_sweep  # noqa: E402  (IMPORT ONLY, never modify)

DS_BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.9)]
MAX_RETRIES = 200


def _min_image_dist_all(x, ref_idx, L):
    """x[N,3], ref_idx -> dist[N] from every particle to x[ref_idx] (min-image). Plain numpy (called
    twice per pair, negligible cost) -- not a numba kernel, unlike poly_ncmc_v2's per-call _dist_pbc."""
    d = x - x[ref_idx]
    d -= L * np.round(d / L)
    return np.sqrt((d ** 2).sum(1))


def select_block_by_pair(x, i, j, k, L):
    """k nearest particles by min(dist-to-i, dist-to-j) (min-image) -- the PAIR-CENTERED block
    definition shared by pair generation here AND deployment (poly_escort_ncmc.py imports this
    directly, never reimplements it). i, j are always the two closest members (self-distance 0), so
    they are always IN the returned block for ANY (i, j) separation."""
    di = _min_image_dist_all(x, i, L)
    dj = _min_image_dist_all(x, j, L)
    keys = np.minimum(di, dj)
    return np.argsort(keys)[:k].astype(np.int64)


def select_pair(sig, N, rng, target_bin):
    """Fully-random (i, j) over the WHOLE system, retried (capped) until |sigma_i - sigma_j| lands in
    DS_BINS[target_bin] -- SAME selection convention as poly_ncmc_v2.py's __main__ driver (uniform,
    state-independent pair choice -- the NCMC selection-symmetry argument that module's exactness
    relies on), NOT a within-block pair like poly_swap_pairs.make_swap_pair. Falls back to the closest
    candidate seen after MAX_RETRIES (generation always terminates)."""
    lo, hi = DS_BINS[target_bin]
    best_d, best_pair = -1.0, (0, 1)
    for _ in range(MAX_RETRIES):
        ii = int(rng.integers(N))
        jj = int(rng.integers(N - 1))
        if jj >= ii:
            jj += 1
        ds = abs(float(sig[ii] - sig[jj]))
        if ds > best_d:
            best_d, best_pair = ds, (ii, jj)
        if lo <= ds and (ds < hi or (target_bin == len(DS_BINS) - 1 and ds <= hi)):
            return ii, jj, ds, True
    ii, jj = best_pair
    return ii, jj, abs(float(sig[ii] - sig[jj])), False


def make_escort_pair(x, sig, L, beta, k, seed, target_bin, n_steps, r_loc, step, escort_sweeps,
                      sweeps_per_step=1):
    """Returns a pair dict in poly_train_swapflow.py's exact format (see module docstring)."""
    rng = np.random.default_rng(seed)
    seed_numba(seed)
    N = x.shape[0]

    i, j, ds_target, hit = select_pair(sig, N, rng, target_bin)

    xw, sw = x.copy(), sig.copy()
    schedule = np.linspace(1.0 / n_steps, 1.0, n_steps)
    checkpoint_step = int(rng.integers(1, n_steps + 1))   # 1..n_steps inclusive

    si0, sj0 = sw[i], sw[j]
    S = np.empty(N, dtype=np.int64)
    for kk in range(checkpoint_step):
        lam = schedule[kk]
        sw[i] = (1.0 - lam) * si0 + lam * sj0
        sw[j] = (1.0 - lam) * sj0 + lam * si0
        nS = build_local_set(xw, i, j, r_loc, L, S)
        for _ in range(sweeps_per_step):
            local_sweep(xw, sw, L, beta, i, j, r_loc, step, S, nS)

    # pair-centered k=8 block: k nearest by min(dist-to-i, dist-to-j) -- i, j always included
    # (self-distance 0), well-defined for any i/j separation (unlike a single-seed k-NN block).
    idx = select_block_by_pair(xw, i, j, k, L)
    mask = np.zeros(N, dtype=bool)
    mask[idx] = True

    x_before = xw[idx].copy()
    sig_ckpt = sw[idx].copy()            # identity path: block's CURRENT (lambda-interpolated) sigma
    env_x_snap = xw[~mask].copy()
    env_sig_snap = sw[~mask].copy()

    nS2 = build_local_set(xw, i, j, r_loc, L, S)
    for _ in range(escort_sweeps):
        local_sweep(xw, sw, L, beta, i, j, r_loc, step, S, nS2)
    x_after = xw[idx].copy()

    cen = x_before.mean(0)

    def cent(arr):
        c = arr - cen
        return c - L * np.round(c / L)

    return {
        "idx": idx, "env_x": cent(env_x_snap), "env_sig": env_sig_snap,
        "x_old": cent(x_before), "x_new": cent(x_after),
        "sig_start": sig_ckpt.copy(), "sig_end": sig_ckpt.copy(),
        "pair_global": (i, j), "ds": ds_target, "retry_hit": hit,
        "checkpoint_step": checkpoint_step, "n_steps": n_steps,
        "lam_checkpoint": float(schedule[checkpoint_step - 1]),
        "L": L, "beta": beta,
    }


def _load_banks(runs, T):
    banks = {}
    for r in runs:
        D = torch.load(REPO / f"reports/logs-2026-07-17/poly_bank_fixed_run{r}.pt", weights_only=False)
        key = (r, T)
        assert key in D, f"fixed bank run{r} has no T={T} yet"
        rec = D[key]
        assert "sigs" in rec, "fixed bank must carry per-frame sigs"
        banks[r] = rec
    return banks


def _stats(pairs, L):
    rms, dmax = [], []
    for q in pairs:
        d = q["x_new"] - q["x_old"]
        d -= L * np.round(d / L)
        r = np.sqrt((d ** 2).sum(1))
        rms.append((r ** 2).mean())
        dmax.append(r.max())
    ds = np.array([q["ds"] for q in pairs])
    misses = sum(1 for q in pairs if not q["retry_hit"])
    return (float(np.sqrt(np.mean(rms))), float(np.percentile(dmax, 95)), float(ds.mean()), misses)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--npairs", type=int, default=3000)
    p.add_argument("--runs", type=str, default="1,2", help="training runs (run 3 held out for probe)")
    p.add_argument("--r_loc", type=float, default=3.0)
    p.add_argument("--n_steps", type=int, default=800)
    p.add_argument("--escort_sweeps", type=int, default=100)
    p.add_argument("--step", type=float, default=0.12)
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()
    runs = [int(r) for r in a.runs.split(",")]
    beta = 1.0 / a.T

    banks = _load_banks(runs, a.T)
    L = float(banks[runs[0]]["L"])

    out = a.out or f"reports/logs-2026-07-17/poly_escortpairs_T{a.T}_k{a.k}.pt"
    pairs = []
    t0 = time.time()
    for m in range(a.npairs):
        r = runs[m % len(runs)]
        rec = banks[r]
        j = m % rec["x"].shape[0]
        fr = rec["x"][j].astype(np.float64).copy()
        sig = rec["sigs"][j].astype(np.float64).copy()
        target_bin = m % len(DS_BINS)
        pair = make_escort_pair(fr, sig, float(rec["L"]), beta, a.k, 11000 + m, target_bin,
                                 a.n_steps, a.r_loc, a.step, a.escort_sweeps)
        pairs.append(pair)
        if (m + 1) % 200 == 0 or (m + 1) == a.npairs:
            torch.save({"pairs": pairs, "meta": vars(a)}, out)
            rx, p95x, mds, misses = _stats(pairs, L)
            print(f"{m+1}/{a.npairs} ({time.time()-t0:.0f}s) rms|dx|={rx:.3f} p95max|dx|={p95x:.3f} "
                  f"mean|ds|={mds:.3f} retry_miss={misses}", flush=True)
    torch.save({"pairs": pairs, "meta": vars(a)}, out)
    rx, p95x, mds, misses = _stats(pairs, L)
    print(f"ESCORT PAIRS DONE -> {out} ({len(pairs)}; rms|dx| {rx:.3f}, p95max|dx| {p95x:.3f}, "
          f"mean|ds| {mds:.3f}, retry_miss {misses}; {time.time()-t0:.0f}s)", flush=True)
