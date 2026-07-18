"""Label generator for the LEARNED PAIR SELECTOR (NCMC diameter-swap screening kernel).

Context (see poly_ncmc_v2.py + .superpowers/sdd/progress.md 2026-07-18 NCMC-pivot entry):
acceptance of a targeted-local NCMC diameter swap is a TAIL phenomenon -- most
uniformly-drawn pairs in the |dsigma| in [0.1, 0.9] window are doomed (W very large,
acc ~0) and burn CPU on a protocol that was never going to accept. The selector's job
is to predict, CHEAPLY (features computable in O(1) neighbor lookups, no protocol run),
which pairs are worth attempting -- so the screened kernel spends its CPU budget on the
tail that actually pays off.

This script:
  1. draws pairs uniformly from the |dsigma| in [0.1, 0.9] window on CLEAN fixed bank
     frames from runs 1+2 (TRAIN split; run 3 is held out for evaluate_screening() in
     poly_selector.py -- NEVER touched here),
  2. computes cheap, deployment-computable FEATURES per pair (see pair_features below),
  3. labels each pair with W from ncmc_work_local (r_loc=3.0, n_steps=200 -- the "cheap
     label" recipe; more steps would be a better W estimate but blow the labeling
     budget for 12k samples),
  4. saves incrementally to poly_selector_dataset_T<T>.pt every 500 samples so a crash
     never loses more than 500 samples of a ~20-30 min run (checkpoint-incrementally
     directive).

ncmc_work_local is IMPORTED from poly_ncmc_v2.py, never modified, never reimplemented:
it is the exact labeling engine. The "vertical dU of the instant swap" feature, by
contrast, is NOT taken from ncmc_work_local's n_steps=0 special case (which would cost
a second numba dispatch + schedule-array allocation per sample) -- it is recomputed
locally via _pair_env_e/_pair_e, which are VERBATIM copies of the identical helper
functions already independently defined in both poly_ncmc_probe.py (v1) and
poly_ncmc_v2.py (v2) (i.e. duplicating this exact small energy-difference formula
per-script is the established convention in this file family, not a deviation from
it). A cross-check test (test_poly_selector.py) verifies this local computation agrees
with ncmc_work_local(schedule=[1.0], ...) bit-for-bit-close.
"""
import sys, time
import numpy as np
import torch
from numba import njit

sys.path.insert(0, "/mnt/ssd/GridTransformer")
sys.path.insert(0, "/mnt/ssd/GridTransformer/reports/logs-2026-07-17")
from liquid_coupling_flow.poly.model import seed_numba, EPS_NA, pair_v, sigma_ij, row_e  # noqa: E402
from poly_ncmc_v2 import ncmc_work_local  # noqa: E402  -- imported, never modified

DS_LO, DS_HI = 0.1, 0.9          # screened NCMC window: |dsigma| in [DS_LO, DS_HI)
R_LOC = 3.0                       # cheap-label local propagation radius
N_STEPS = 200                     # cheap-label schedule length
STEP = 0.12                       # displacement step (matches poly_ncmc_v2.STEP)
N_NN = 8                          # nearest-neighbor count per pair member
DATASET_PATH_FMT = "reports/logs-2026-07-17/poly_selector_dataset_T{T}.pt"

FEATURE_NAMES = (
    ["sig_lo", "sig_hi", "abs_dsig", "pair_dist", "vertical_dU", "pair_penetration"]
    + [f"d_lo_{k}" for k in range(N_NN)] + [f"s_lo_{k}" for k in range(N_NN)]
    + [f"d_hi_{k}" for k in range(N_NN)] + [f"s_hi_{k}" for k in range(N_NN)]
    + ["voronoi_lo", "voronoi_hi"]
)
N_FEATURES = len(FEATURE_NAMES)  # 40


# ---------------------------------------------------------------------------
# Energy helpers -- VERBATIM copies of poly_ncmc_v2.py's _pair_e/_pair_env_e
# (same formula independently defined in v1 and v2; see module docstring).
# ---------------------------------------------------------------------------
@njit(cache=True)
def _pair_e(x, i, j, si, sj, L):
    dx = x[i, 0] - x[j, 0]; dy = x[i, 1] - x[j, 1]; dz = x[i, 2] - x[j, 2]
    dx -= L * np.round(dx / L); dy -= L * np.round(dy / L); dz -= L * np.round(dz / L)
    return pair_v(dx * dx + dy * dy + dz * dz, sigma_ij(si, sj))


@njit(cache=True)
def _pair_env_e(x, sig, i, j, L):
    return (row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
            + row_e(x, sig, j, x[j, 0], x[j, 1], x[j, 2], L)
            - _pair_e(x, i, j, sig[i], sig[j], L))


@njit(cache=True)
def vertical_dU(x, sig, i, j, L):
    """Instant-swap vertical work e1-e0 (sig[i],sig[j] swapped). Restores sig
    afterwards -- non-mutating from the caller's perspective, x untouched."""
    e0 = _pair_env_e(x, sig, i, j, L)
    si = sig[i]; sj = sig[j]
    sig[i] = sj; sig[j] = si
    e1 = _pair_env_e(x, sig, i, j, L)
    sig[i] = si; sig[j] = sj
    return e1 - e0


def _sigma_ij_np(si, sj):
    """Vectorized mirror of liquid_coupling_flow.poly.model.sigma_ij (same formula)."""
    return 0.5 * (si + sj) * (1.0 - EPS_NA * np.abs(si - sj))


def _min_image(d, L):
    return d - L * np.round(d / L)


def pair_min_dist(x, L, i, j):
    d = _min_image(x[i] - x[j], L)
    return float(np.sqrt((d * d).sum()))


def nearest_neighbors(x, sig, L, idx, k=N_NN):
    """k nearest OTHER particles to idx (min-image), sorted ascending by distance.
    Returns (dist[k], sig[k])."""
    d = _min_image(x - x[idx], L)
    dist = np.sqrt((d * d).sum(axis=1))
    dist[idx] = np.inf
    part = np.argpartition(dist, k)[:k]
    order = part[np.argsort(dist[part])]
    return dist[order], sig[order]


def voronoi_proxy(d_k, s_k, s_self):
    """Mean neighbor distance minus mean sigma_ij contact estimate to those same
    neighbors -- a cheap local free-volume indicator."""
    sij_est = _sigma_ij_np(s_self, s_k)
    return float(np.mean(d_k) - np.mean(sij_est))


def canonical_order(sig, i, j):
    """Deterministic (i,j)-order-independent canonicalization: 'lo' = smaller sigma
    (tie-broken by smaller particle index). Guarantees pair_features(x,sig,L,i,j) ==
    pair_features(x,sig,L,j,i) EXACTLY (test 1 in test_poly_selector.py)."""
    if sig[i] < sig[j]:
        return i, j
    if sig[i] > sig[j]:
        return j, i
    return (i, j) if i < j else (j, i)


def pair_features(x, sig, L, i, j):
    """All-cheap, deployment-computable feature vector for the (i,j) swap candidate,
    evaluated at state (x, sig). Order-independent in (i,j) by construction (see
    canonical_order). Returns a float64 array of length N_FEATURES (== len(FEATURE_NAMES)).
    Deterministic and finite for any valid, distinct (i,j)."""
    lo, hi = canonical_order(sig, i, j)
    ds = abs(float(sig[i]) - float(sig[j]))
    pdist = pair_min_dist(x, L, i, j)
    dU = float(vertical_dU(x, sig, i, j, L))
    sij_contact = float(_sigma_ij_np(sig[lo], sig[hi]))
    penetration = pdist - sij_contact

    d_lo, s_lo = nearest_neighbors(x, sig, L, lo)
    d_hi, s_hi = nearest_neighbors(x, sig, L, hi)
    vp_lo = voronoi_proxy(d_lo, s_lo, sig[lo])
    vp_hi = voronoi_proxy(d_hi, s_hi, sig[hi])

    feat = np.concatenate([
        [sig[lo], sig[hi], ds, pdist, dU, penetration],
        d_lo, s_lo, d_hi, s_hi,
        [vp_lo, vp_hi],
    ]).astype(np.float64)
    assert feat.shape[0] == N_FEATURES
    return feat


def sample_pair_in_window(rng, sig, ds_lo=DS_LO, ds_hi=DS_HI, guard=100000):
    """Uniform pair (i,j) subject to |dsigma| in [ds_lo, ds_hi) via rejection sampling
    from uniform ordered-pair draws -- the accepted distribution is exactly uniform
    over qualifying pairs (matches the exactness-contract's 'pick pair uniformly in
    the window' clause; window symmetric under swap since it depends on |dsigma| only)."""
    N = sig.shape[0]
    for _ in range(guard):
        i = int(rng.integers(N))
        j = int(rng.integers(N - 1))
        if j >= i:
            j += 1
        ds = abs(float(sig[i] - sig[j]))
        if ds_lo <= ds < ds_hi:
            return i, j
    raise RuntimeError("sample_pair_in_window: guard exceeded, window too sparse")


def load_train_frames(T):
    """Concatenate all (run, frame) states from runs 1+2 (TRAIN split). run 3 is
    NEVER loaded here -- it is the held-out set for evaluate_screening()."""
    frames = []
    for run in (1, 2):
        D = torch.load(f"reports/logs-2026-07-17/poly_bank_fixed_run{run}.pt", weights_only=False)
        rec = D[(run, T)]
        L = float(rec["L"])
        for fr in range(rec["x"].shape[0]):
            frames.append({
                "run": run, "frame": fr, "L": L,
                "x": rec["x"][fr].astype(np.float64),
                "sig": rec["sigs"][fr].astype(np.float64),
            })
    return frames


def generate_dataset(T=0.085, n_samples=12000, save_every=500, seed=17, out_path=None):
    out_path = out_path or DATASET_PATH_FMT.format(T=T)
    frames = load_train_frames(T)
    n_frames = len(frames)
    N = frames[0]["x"].shape[0]
    print(f"[selector-data] {n_frames} TRAIN frames (runs 1+2), N={N}, T={T}, "
          f"window=[{DS_LO},{DS_HI}), r_loc={R_LOC}, n_steps={N_STEPS}", flush=True)

    rng = np.random.default_rng(seed)
    seed_numba(seed + 9000)
    schedule = np.linspace(1.0 / N_STEPS, 1.0, N_STEPS)

    feats = np.zeros((n_samples, N_FEATURES), dtype=np.float64)
    W = np.zeros(n_samples, dtype=np.float64)
    meta = np.zeros((n_samples, 5), dtype=np.float64)  # run, frame, i, j, ds
    cost = np.zeros(n_samples, dtype=np.float64)        # nS_step.sum() -- CPU units

    t0 = time.time()
    for k in range(n_samples):
        fr = frames[k % n_frames] if k < n_frames else frames[int(rng.integers(n_frames))]
        i, j = sample_pair_in_window(rng, fr["sig"])
        feats[k] = pair_features(fr["x"], fr["sig"], fr["L"], i, j)

        x = fr["x"].copy(); sig = fr["sig"].copy()
        dW_step = np.zeros(N_STEPS); nS_step = np.zeros(N_STEPS)
        w = ncmc_work_local(x, sig, fr["L"], 1.0 / T, i, j, schedule, 1, R_LOC, STEP,
                             dW_step, nS_step)
        W[k] = w
        cost[k] = nS_step.sum()
        meta[k] = [fr["run"], fr["frame"], i, j, feats[k, 2]]

        if (k + 1) % save_every == 0 or (k + 1) == n_samples:
            elapsed = time.time() - t0
            torch.save({
                "T": T, "feature_names": FEATURE_NAMES, "r_loc": R_LOC, "n_steps": N_STEPS,
                "ds_window": (DS_LO, DS_HI), "step": STEP, "seed": seed,
                "n_done": k + 1,
                "features": feats[:k + 1].copy(), "W": W[:k + 1].copy(),
                "cost_smu": cost[:k + 1].copy(), "meta": meta[:k + 1].copy(),
                "meta_cols": ["run", "frame", "i", "j", "abs_dsig"],
            }, out_path)
            print(f"[{elapsed:7.1f}s] saved {k + 1}/{n_samples} "
                  f"(W med={np.median(W[:k+1]):+.2f}, {(k+1)/elapsed:.2f} samp/s)", flush=True)

    print(f"[selector-data] DONE {n_samples} samples in {time.time()-t0:.0f}s -> {out_path}",
          flush=True)
    return out_path


if __name__ == "__main__":
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.085
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
    generate_dataset(T=T, n_samples=n_samples)
