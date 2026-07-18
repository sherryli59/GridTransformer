"""NCMC diameter-swap kernel v2: TARGETED LOCAL propagation.

v1 (poly_ncmc_probe.py) propagates the WHOLE system with full displacement sweeps
after each lambda-switch; its results show W falling monotonically with n_steps but
decaying sublinearly, because ~90% of a full N=300 sweep's moves are far from the
swapping pair -- dead weight in the CPU budget. v2 restricts propagation to a LOCAL
selectable set

    S(x) = {m : min(d(m,i), d(m,j)) < r_loc}

built from min-image distances to the swap pair's CURRENT positions (i, j are always
trivially members of S -- distance to themselves is 0 < r_loc).

Exactness argument (selection ratio == 1, so plain Metropolis min(1,exp(-b*dU)) is
correct for each sub-move, and the composite NCMC acceptance min(1,exp(-b*W)) is
exact by the same palindromic-protocol / Crooks argument as v1 [Nilmeier et al.,
PNAS 2011]):

  - S is built ONCE per lambda-step, from the state x entering that step (recomputing
    S at every single attempt would also be exact, just O(N) per attempt instead of
    O(N) per lambda-step -- this file uses the cheaper per-lambda-step refresh, as
    explicitly permitted by the design: "recompute S every attempt OR per
    lambda-step -- per lambda-step is fine and cheaper").
  - Within that lambda-step, `sweeps_per_step` local sweeps are performed; each local
    sweep is |S| single-particle Metropolis displacement attempts. Every attempt
    picks m ~ Uniform(S), where S is the FROZEN INDEX LIST built at step-entry --
    particles never enter or leave the CANDIDATE POOL mid-step, even though pair or
    member particles may physically move during the step. This keeps the forward
    selection probability 1/|S| for every attempt in the block.
  - The continuous membership test used for the exit-rejection ("would the reverse
    move still be able to pick m from S?") is evaluated against the LIVE current
    positions of i and j at the moment of the attempt -- i.e. if an EARLIER attempt
    in this same lambda-step moved i or j (because m==i or m==j was drawn and
    accepted), later membership tests correctly use i/j's new position, matching what
    S(x) means at the true current state x. As long as the proposed move keeps m
    inside S, the reverse chain's selection probability for m is identical (1/|S|,
    same frozen list, same size) -- the selection-ratio term in the Metropolis ratio
    is exactly 1, and plain min(1, exp(-b*dU)) is correct.
  - When the proposed move would carry m outside S, the reverse selection probability
    for m from S would be 0 (m would no longer satisfy the membership test), so we
    REJECT UNCONDITIONALLY (this is the reflecting boundary verified by test 1 -- no
    net probability flux is lost, positions inside S remain uniformly distributed,
    they simply cannot cross out of S within this lambda-step).
  - i and j are ALWAYS trivially members of S (self-distance 0 < r_loc for any
    r_loc > 0), for ANY position they move to -- so the exit-rejection can NEVER fire
    for m==i or m==j. Moving the swap pair itself is always accepted/rejected on
    energy alone, exactly like every other candidate in S.

Per-lambda-step work dW(lambda) = e1-e0 (same bookkeeping as v1's total W) is
additionally saved per-attempt (unlike v1, which only kept the total W) so the work
profile along the schedule can be inspected post hoc -- this exposes WHERE the work
concentrates and is meant to feed a future dissipation-equalized (non-uniform)
schedule.
"""
import sys, time
import numpy as np
import torch
from numba import njit
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.poly.model import row_e, pair_v, sigma_ij, seed_numba

BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.9)]
STEP = 0.12


@njit(cache=True)
def _pair_e(x, i, j, si, sj, L):
    dx = x[i, 0] - x[j, 0]; dy = x[i, 1] - x[j, 1]; dz = x[i, 2] - x[j, 2]
    dx -= L * np.round(dx / L); dy -= L * np.round(dy / L); dz -= L * np.round(dz / L)
    return pair_v(dx * dx + dy * dy + dz * dz, sigma_ij(si, sj))


@njit(cache=True)
def _pair_env_e(x, sig, i, j, L):
    """Energy of everything touching i or j (pair counted once). Identical to v1."""
    return (row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
            + row_e(x, sig, j, x[j, 0], x[j, 1], x[j, 2], L)
            - _pair_e(x, i, j, sig[i], sig[j], L))


@njit(cache=True)
def _dist_pbc(x, m, k, L):
    dx = x[m, 0] - x[k, 0]; dy = x[m, 1] - x[k, 1]; dz = x[m, 2] - x[k, 2]
    dx -= L * np.round(dx / L); dy -= L * np.round(dy / L); dz -= L * np.round(dz / L)
    return np.sqrt(dx * dx + dy * dy + dz * dz)


@njit(cache=True)
def build_local_set(x, i, j, r_loc, L, buf):
    """Fill buf[:cnt] with indices m s.t. min(d(m,i), d(m,j)) < r_loc, evaluated at
    the CURRENT state x. i and j are always included (self-distance 0). buf must have
    length >= x.shape[0]. Returns cnt = |S|."""
    n = x.shape[0]
    cnt = 0
    for m in range(n):
        dmi = _dist_pbc(x, m, i, L)
        dmj = _dist_pbc(x, m, j, L)
        if dmi < r_loc or dmj < r_loc:
            buf[cnt] = m
            cnt += 1
    return cnt


@njit(cache=True)
def local_sweep(x, sig, L, beta, i, j, r_loc, step, S, nS):
    """One local sweep = nS single-particle Metropolis displacement attempts, each
    picking m uniformly from the FROZEN index list S[:nS] (built by build_local_set
    once at the start of the current lambda-step; NOT rebuilt inside this function).
    See module docstring for the exactness / selection-ratio argument. Returns
    (acc, nS) -- attempts == nS by construction (one full pass over S)."""
    acc = 0
    for _ in range(nS):
        idx = np.random.randint(nS)
        m = S[idx]
        xn0 = (x[m, 0] + step * np.random.randn()) % L
        xn1 = (x[m, 1] + step * np.random.randn()) % L
        xn2 = (x[m, 2] + step * np.random.randn()) % L
        # Exit-rejection: would the reverse move still find m in S(x'), the state
        # AFTER this proposal? S(x') anchors membership on i's and j's positions IN
        # x' -- if m==i (or m==j), that anchor IS the just-proposed new position
        # itself, so the self-distance is trivially 0 (i, j can never be exit-
        # rejected). Otherwise the anchor is the LIVE current position of i/j
        # (possibly already moved earlier this lambda-step if m==i or m==j was drawn
        # on a prior attempt).
        if m == i:
            di_new = 0.0
        else:
            dgx = xn0 - x[i, 0]; dgy = xn1 - x[i, 1]; dgz = xn2 - x[i, 2]
            dgx -= L * np.round(dgx / L); dgy -= L * np.round(dgy / L); dgz -= L * np.round(dgz / L)
            di_new = np.sqrt(dgx * dgx + dgy * dgy + dgz * dgz)
        if m == j:
            dj_new = 0.0
        else:
            dgx = xn0 - x[j, 0]; dgy = xn1 - x[j, 1]; dgz = xn2 - x[j, 2]
            dgx -= L * np.round(dgx / L); dgy -= L * np.round(dgy / L); dgz -= L * np.round(dgz / L)
            dj_new = np.sqrt(dgx * dgx + dgy * dgy + dgz * dgz)
        if di_new >= r_loc and dj_new >= r_loc:
            continue  # would leave S -> reverse selection prob 0 -> reject unconditionally
        e0 = row_e(x, sig, m, x[m, 0], x[m, 1], x[m, 2], L)
        e1 = row_e(x, sig, m, xn0, xn1, xn2, L)
        if np.random.random() < np.exp(-beta * (e1 - e0)):
            x[m, 0] = xn0; x[m, 1] = xn1; x[m, 2] = xn2
            acc += 1
    return acc, nS


@njit(cache=True)
def ncmc_work_local(x, sig, L, beta, i, j, schedule, sweeps_per_step, r_loc, step,
                     dW_step, nS_step):
    """Drive sigma_i,sigma_j to swapped values along `schedule` (increasing lambda
    values, schedule[-1] must be 1.0); after each lambda-switch, propagate with
    `sweeps_per_step` LOCAL sweeps restricted to a set S rebuilt ONCE at the start of
    that lambda-step (see module docstring). Returns W = accumulated switching work.

    len(schedule)==1 and schedule[0]==1.0 reproduces the classical instantaneous swap
    exactly (matches v1's n_steps==0 special case: no propagation, deterministic
    vertical work).

    dW_step (len == len(schedule)) and nS_step (len == len(schedule)) are filled with
    this trajectory's per-lambda-step work increment and |S|; the caller averages
    dW_step across many independent trajectories to obtain the dW(lambda) work
    profile, and n_steps*mean(nS_step)*sweeps_per_step gives the CPU cost of this
    trajectory in single-move-attempt units (comparable to v1's n_steps*300)."""
    si0 = sig[i]; sj0 = sig[j]
    n_steps = schedule.shape[0]
    if n_steps == 1 and schedule[0] == 1.0:
        e0 = _pair_env_e(x, sig, i, j, L)
        sig[i] = sj0; sig[j] = si0
        e1 = _pair_env_e(x, sig, i, j, L)
        dW_step[0] = e1 - e0
        nS_step[0] = 0.0
        return e1 - e0
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
    return W


if __name__ == "__main__":
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.085
    beta = 1.0 / T
    D = torch.load("reports/logs-2026-07-17/poly_bank_fixed_run3.pt", weights_only=False)
    rec = D[(3, T)]
    L = float(rec["L"]); N = rec["x"].shape[1]
    out = {"T": T, "L": L, "N": N, "grid": {}}
    rng = np.random.default_rng(5)

    # (r_loc, n_steps) grid. n=6000 only probed at r_loc=2.0 (spec: 12 attempts/bin,
    # hardcoded -- the local-propagation cost there is high enough that adaptive
    # sizing from the first cell would otherwise round it down to near-uselessness).
    GRID = [(2.0, 200), (2.0, 800), (2.0, 2400), (2.0, 6000),
            (3.0, 200), (3.0, 800), (3.0, 2400)]
    PER_BIN_OVERRIDE = {(2.0, 6000): 12}
    FIRST_PER_BIN = 60          # calibration cell: real data, not thrown away
    TIME_BUDGET_S = 1500.0      # ~25 min single-core wall-clock target for the whole grid

    cost_per_smu = None         # seconds per single-move-attempt, measured from the first cell
    mean_S_seen = {}            # r_loc -> measured mean|S| (informs later cells' per_bin)
    t_start = time.time()
    n_left = len(GRID)

    for gi, (r_loc, n_steps) in enumerate(GRID):
        key = (r_loc, n_steps)
        if key in PER_BIN_OVERRIDE:
            per_bin = PER_BIN_OVERRIDE[key]
        elif cost_per_smu is None:
            per_bin = FIRST_PER_BIN
        else:
            elapsed_so_far = time.time() - t_start
            cell_budget = max(20.0, (TIME_BUDGET_S - elapsed_so_far) / max(n_left, 1))
            est_S = mean_S_seen.get(r_loc, N)
            per_attempt_cost = max(n_steps, 1) * est_S * cost_per_smu
            per_bin = max(5, int(cell_budget / (len(BINS) * per_attempt_cost)))
        n_left -= 1

        t0 = time.time()
        res = {b: {"W": [], "ds": []} for b in BINS}
        dW_all = {b: [] for b in BINS}
        nS_all = {b: [] for b in BINS}
        need = {b: per_bin for b in BINS}
        seed_numba(4000 + n_steps + int(r_loc * 10))
        schedule = (np.linspace(1.0 / n_steps, 1.0, n_steps) if n_steps > 0
                    else np.array([1.0]))
        guard = 0
        total_smu = 0.0
        n_calls = 0
        while any(v > 0 for v in need.values()) and guard < 20000:
            guard += 1
            fr_j = guard % 16
            i = int(rng.integers(N)); j = int(rng.integers(N - 1))
            if j >= i:
                j += 1
            x0 = rec["x"][fr_j].astype(np.float64)
            s0 = rec["sigs"][fr_j].astype(np.float64)
            ds = abs(float(s0[i] - s0[j]))
            bb = next((b for b in BINS if b[0] <= ds < b[1]), None)
            if bb is None or need[bb] <= 0:
                continue
            need[bb] -= 1
            x = x0.copy(); sig = s0.copy()
            dW_step = np.zeros(len(schedule)); nS_step = np.zeros(len(schedule))
            W = ncmc_work_local(x, sig, L, beta, i, j, schedule, 1, r_loc, STEP,
                                 dW_step, nS_step)
            res[bb]["W"].append(W); res[bb]["ds"].append(ds)
            dW_all[bb].append(dW_step); nS_all[bb].append(nS_step)
            total_smu += nS_step.sum()
            n_calls += 1
        elapsed = time.time() - t0

        mean_S_this = (total_smu / max(n_calls, 1) / max(n_steps, 1)) if n_steps > 0 else N
        if cost_per_smu is None and n_steps > 0 and total_smu > 0:
            cost_per_smu = elapsed / total_smu
        mean_S_seen[r_loc] = mean_S_this
        cpu_smu_per_attempt = n_steps * mean_S_this * 1  # sweeps_per_step=1

        out["grid"][key] = {
            str(b): {
                "W": np.array(r["W"]), "ds": np.array(r["ds"]),
                "dW_profile": np.array(dW_all[b]) if len(dW_all[b]) else np.zeros((0, len(schedule))),
                "nS_profile": np.array(nS_all[b]) if len(nS_all[b]) else np.zeros((0, len(schedule))),
            } for b, r in res.items()
        }
        out["grid"][key]["_meta"] = {
            "r_loc": r_loc, "n_steps": n_steps, "per_bin": per_bin,
            "mean_S": mean_S_this, "cpu_smu_per_attempt": cpu_smu_per_attempt,
            "elapsed_s": elapsed,
        }

        line = [f"r={r_loc:.1f} n={n_steps:>4} (|S|~{mean_S_this:5.1f} cpu/att={cpu_smu_per_attempt:8.0f} "
                f"vs v1 {n_steps * 300:>9})"]
        for b in BINS:
            W = np.array(res[b]["W"])
            if len(W):
                acc = np.minimum(1.0, np.exp(np.clip(-beta * W, -700, 700)))
                line.append(f"{b}: acc={acc.mean():.3f} (Wmed {np.median(W):+.2f}, n={len(W)})")
        print(f"[{elapsed:6.0f}s] " + " | ".join(line), flush=True)
        torch.save(out, f"reports/logs-2026-07-17/poly_ncmc_v2_T{T}.pt")

    print(f"NCMC V2 PROBE DONE [{time.time()-t_start:.0f}s total]", flush=True)
