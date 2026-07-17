"""Swap-endpoint block-pair dataset for the sigma(t)-conditioned SwapBlockFlow (poly Task J3a, PHASE 1
of docs/superpowers/specs/2026-07-17-joint-sigma-x-continuous-flow-amendment.md's DECISION 2026-07-17).

From the FIXED banks (poly_bank_fixed_run{r}.pt, which pair x[j] WITH sigs[j] per frame -- see
poly_rebank.py for why the original poly_tau_curves_run*.pt banks are unusable) -- this script never
touches the original poly_tau_curves banks.

Per pair: k-NN block selection/centering identical to poly_block_data.make_block_pair (min-image
k-NN of a random seed particle, centered on the block's OLD centroid). Inside the block, pick a pair
(a,b) of LOCAL indices STRATIFIED UNIFORMLY over |Delta sigma| bins
    [(0,.1), (.1,.2), (.2,.3), (.3,.45), (.45,.9)]
(round-robin target bin across pairs; rejection-sample candidate (a,b) pairs uniformly until one lands
in the target bin, capped retries -- if the cap is hit, falls back to the closest-matching candidate
seen and logs a miss, so --npairs always terminates). Swap sigma_a <-> sigma_b EXACTLY (the classical
swap-MC endpoint), then run poly_block_data._block_relax (RELAX_SW sweeps, POSITIONS ONLY -- sigma is
fixed post-swap for the whole relax, matching SwapBlockFlow's PHASE-1 contract: sigma is a conditioning
signal end-to-end, never a continuous physical state, unlike joint_flow.py's semi-grand u channel).

Usage:
  poly_swap_pairs.py --T 0.085 --k 8 --npairs 3000 [--runs 1,2] [--relax_sw 400] [--calibrate] [--out ...]
--calibrate: instead of writing a dataset, generates 150 pairs PER BIN (750 total) and prints per-bin
transport (rms|dx|, p95 max|dx|, mean/observed |dsigma|, retry-cap miss count) to the console.
Differences are min-imaged (a naive x_new-x_old on centered coords shows ~L wrap artifacts), mirroring
poly_joint_pairs.py's stats() convention.
Output: poly_swappairs_T{T}_k{k}.pt (list of make_swap_pair dicts + a 'meta' entry).
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
from poly_block_data import _block_relax, RELAX_SW  # noqa: E402  (needs sys.path insert above)

DS_BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.9)]
MAX_RETRIES = 200


def make_swap_pair(x, sig, L, beta, k, seed, target_bin, relax_sw=RELAX_SW, step=0.1):
    """Mirrors poly_block_data.make_block_pair's block selection/centering EXACTLY, but instead of a
    random 2-transposition permutation of the WHOLE block, picks ONE pair (a,b) of local block indices
    whose |sigma_a - sigma_b| falls in DS_BINS[target_bin] (rejection-sampled, capped), swaps just that
    pair, then relaxes. Returns idx, env_x/env_sig (unchanged by the swap), x_old/x_new (centered),
    sig_start (pre-swap block sigmas, index-aligned with idx/x_old), sig_end (post-swap, aligned with
    x_new), pair (local (a,b)), ds (=|sigma_a-sigma_b|, the realized bin value), L, beta."""
    rng = np.random.default_rng(seed)
    seed_numba(seed)
    n = x.shape[0]
    s0 = rng.integers(n)
    d = x - x[s0]
    d -= L * np.round(d / L)
    idx = np.argsort((d ** 2).sum(1))[:k].astype(np.int64)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    xw, sw = x.copy(), sig.copy()
    x_old = xw[idx].copy()
    sig_old = sw[idx].copy()

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
        if lo <= ds < hi or (target_bin == len(DS_BINS) - 1 and ds <= hi):
            hit = True
            break
    if not hit:
        a, b = best_pair       # fallback: closest candidate seen (script always terminates)

    sw[idx[a]], sw[idx[b]] = sw[idx[b]], sw[idx[a]]
    ds_realized = abs(float(sig_old[a] - sig_old[b]))

    _block_relax(xw, sw, L, beta, idx, relax_sw, step)
    cen = x_old.mean(0)

    def cent(arr):
        c = arr - cen
        return c - L * np.round(c / L)

    return {
        "idx": idx, "env_x": cent(xw[~mask]), "env_sig": sw[~mask].copy(),
        "x_old": cent(x_old), "x_new": cent(xw[idx]),
        "sig_start": sig_old, "sig_end": sw[idx].copy(),
        "pair": (a, b), "ds": ds_realized, "retry_hit": hit,
        "L": L, "beta": beta,
    }


def _load_banks(runs, T):
    banks = {}
    for r in runs:
        D = torch.load(f"reports/logs-2026-07-17/poly_bank_fixed_run{r}.pt", weights_only=False)
        key = (r, T)
        assert key in D, f"fixed bank run{r} has no T={T} yet"
        rec = D[key]
        assert "sigs" in rec, "fixed bank must carry per-frame sigs"
        banks[r] = rec
    return banks


def _stats(pairs, L):
    dmax, rms = [], []
    for q in pairs:
        d = q["x_new"] - q["x_old"]
        d -= L * np.round(d / L)                          # min-image the DIFFERENCE
        r = np.sqrt((d ** 2).sum(1))
        rms.append((r ** 2).mean())
        dmax.append(r.max())
    ds = np.array([q["ds"] for q in pairs])
    misses = sum(1 for q in pairs if not q["retry_hit"])
    return (float(np.sqrt(np.mean(rms))), float(np.percentile(dmax, 95)),
            float(ds.mean()), float(ds.min()), float(ds.max()), misses)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--npairs", type=int, default=3000)
    p.add_argument("--runs", type=str, default="1,2", help="training runs (run 3 held out for gate)")
    p.add_argument("--relax_sw", type=int, default=RELAX_SW)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()
    runs = [int(r) for r in a.runs.split(",")]
    beta = 1.0 / a.T

    banks = _load_banks(runs, a.T)

    def build(m, seed0=9000):
        r = runs[m % len(runs)]
        rec = banks[r]
        j = m % rec["x"].shape[0]
        fr = rec["x"][j].astype(np.float64).copy()
        sig = rec["sigs"][j].astype(np.float64).copy()
        target_bin = m % len(DS_BINS)
        return make_swap_pair(fr, sig, float(rec["L"]), beta, a.k, seed0 + m, target_bin,
                               relax_sw=a.relax_sw), target_bin

    if a.calibrate:
        L = float(banks[runs[0]]["L"])
        N_PER_BIN = 150
        print(f"calibration at T={a.T} k={a.k} ({N_PER_BIN} pairs/bin, fixed banks, "
              f"relax_sw={a.relax_sw}, min-imaged diffs)")
        print(f"{'bin':>12} | {'rms|dx|':>8} {'p95max|dx|':>10} | "
              f"{'mean|ds|':>9} {'min|ds|':>7} {'max|ds|':>7} | {'retry_miss':>10}")
        m = 0
        for bi, (lo, hi) in enumerate(DS_BINS):
            ps = []
            for _ in range(N_PER_BIN):
                r = runs[m % len(runs)]
                rec = banks[r]
                j = m % rec["x"].shape[0]
                fr = rec["x"][j].astype(np.float64).copy()
                sig = rec["sigs"][j].astype(np.float64).copy()
                ps.append(make_swap_pair(fr, sig, float(rec["L"]), beta, a.k,
                                          70000 + 1000 * bi + m, bi, relax_sw=a.relax_sw))
                m += 1
            rx, p95x, mds, mnds, mxds, misses = _stats(ps, L)
            print(f"{lo:.2f}-{hi:.2f} | {rx:8.3f} {p95x:10.3f} | "
                  f"{mds:9.3f} {mnds:7.3f} {mxds:7.3f} | {misses:10d}", flush=True)
        sys.exit(0)

    out = a.out or f"reports/logs-2026-07-17/poly_swappairs_T{a.T}_k{a.k}.pt"
    pairs = []
    t0 = time.time()
    for m in range(a.npairs):
        pair, target_bin = build(m)
        pairs.append(pair)
        if (m + 1) % 500 == 0:
            torch.save({"pairs": pairs, "meta": vars(a)}, out)
            bin_counts = np.bincount([int(np.digitize(q["ds"], [b[0] for b in DS_BINS]) - 1)
                                       for q in pairs], minlength=len(DS_BINS))
            print(f"{m+1}/{a.npairs} ({time.time()-t0:.0f}s) bin_fill={bin_counts.tolist()}", flush=True)
    torch.save({"pairs": pairs, "meta": vars(a)}, out)
    L = float(banks[runs[0]]["L"])
    rx, p95x, mds, mnds, mxds, misses = _stats(pairs, L)
    print(f"PAIRS DONE -> {out} ({len(pairs)}; rms|dx| {rx:.3f}, p95max|dx| {p95x:.3f}, "
          f"mean|ds| {mds:.3f}, retry_miss {misses}; {time.time()-t0:.0f}s)", flush=True)
