"""Domain-decomposed (PARALLEL) NCMC demo at N=600 -- the scalability anchor.

Everything upstream of this file (poly_ncmc_v2.py's build_local_set/local_sweep, its
exactness argument for the single-pair LOCAL NCMC kernel) is imported verbatim, not
modified, not re-derived. This file adds exactly one new ingredient: running TWO
independent single-pair NCMC diameter-swap moves SIMULTANEOUSLY, in disjoint spatial
domains, with independent (per-domain) Metropolis acceptance -- and proving that doing
so is EXACT (not an approximation) because the two moves' influence regions never
touch, so the composite two-move transition kernel factorizes into the product of two
single-move kernels, each already known-exact from v2.

PHYSICS / EXACTNESS CONTRACT
-----------------------------
Two NCMC pair-morphs are exactly independent iff their INFLUENCE REGIONS never
interact. A move's influence region = its FROZEN mobile set S (built once at move
start: S = {m : min(d(m,i), d(m,j)) < r_loc}, r_loc=1.5) UNIONED with a one-cutoff-
radius halo (XC*sig_max ~ 2.0125, the largest possible pair-interaction range for this
model: sij <= SIG_MAX=1.61 only when si=sj so the nonadditivity term vanishes, cutoff =
XC*sij <= 1.25*1.61 = 2.0125) -- because v2's local_sweep computes each candidate's
energy via row_e, a full O(N) sum over ALL particles; only those within the cutoff
contribute (pair_v returns exactly 0.0 beyond XC*sij). So a particle p in domain 1's
mobile set can only "feel" a particle q if d(p,q) < XC*sig_max; if no such (p,q) pair
ever exists during either move, the two moves' energy bookkeeping (and hence W, and
hence the accept/reject decision and the resulting trajectory) are BITWISE IDENTICAL
whether they run serially on separate copies or interleaved step-by-step on one shared
config. That bitwise-identity is the load-bearing empirical proof in Part A below --
not a derivation from first principles, but a direct measurement that the coupling
term is exactly zero.

Domain construction (STATE-INDEPENDENT, so pair selection stays exactly symmetric --
same argument as v1/v2's uniform pair selection: the domains themselves must not
depend on which pair ends up selected, or selection probabilities would need an extra
Hastings correction):
    u ~ Uniform(box)^3                                  (random offset, drawn BEFORE
                                                          looking at any configuration)
    D1 = ball(u, R_DOM)
    D2 = ball(u + (L/2, L/2, L/2) min-imaged, R_DOM)     (antipodal offset)
For N=600, L = N^(1/3) = 8.434 (this repo's fixed rho=1 convention, poly_bank.py).
Center separation = sqrt(3)*L/2 = 7.303. With r_loc=1.5 and R_DOM=1.5, the worst-case
influence reach of a single move from its domain's OFFSET u is r_loc + XC*sig_max =
1.5 + 2.0125 = 3.5125; two such reaches (3.5125 each, 7.025 total) fit under the 7.303
center separation -- with only ~0.28 headroom, hence the additional per-round EMPIRICAL
disjointness check below (measured, not just this a-priori estimate) and the frozen-set
drift bound.

Within each domain: qualifying pairs = both particles inside ball(center, R_DOM-0.2)
(the 0.2 margin keeps qualifying PAIRS -- not their full mobile-set reach -- comfortably
inside the domain), |dsig| in [0.1,0.9] (v1/v2/production's mid-window, avoids the
degenerate ds~0 no-op and the empirically-dead ds>0.9 tail), pair distance < 2.0.
Selected UNIFORMLY among qualifying pairs. If a domain finds none, it IDLES that round
(no move attempted there; counted, not retried).

Because "select uniformly among qualifying pairs" is a STATE-DEPENDENT selection
mechanism (the qualifying set changes composition as the move changes positions/sigma),
plain min(1, exp(-beta*W)) is NOT by itself detailed-balanced for the whole compound
move (draw-pair + NCMC-drive); the standard Metropolis-Hastings correction is the ratio
of forward/reverse SELECTION probabilities: forward picks this pair with probability
1/n_qual(x_start); the reverse move (same domain, same NCMC schedule run backwards from
the end state) would need to re-select the SAME pair among whatever qualifies at
x_end, probability 1/n_qual(x_end). So the acceptance is
    min(1, exp(-beta*W) * n_qual(x_start)/n_qual(x_end))
n_qual is cheap (a handful of local particles per domain -- see FIRST_PER_BIN-scale
occupancy below) so this is essentially free to recompute post-move.

FROZEN mobile set for the WHOLE move (not rebuilt per lambda-step, unlike v2's
production default): build_local_set(x, i, j, r_loc, L, buf) is called ONCE, before the
first lambda-step, and the resulting S is reused for all 400 lambda-steps via
local_sweep(x, sig, L, beta, i, j, r_loc, step, S, nS) -- v2's own docstring already
establishes that ANY fixed refresh interval for S (per-attempt, per-lambda-step) is
exact via the same selection-ratio-1 / exit-rejection argument; freezing S for the
ENTIRE move is just the next point on that same spectrum (cheaper, and here also
REQUIRED to keep the disjointness guarantee for the whole move -- if S were rebuilt
mid-move, e.g. because i or j drifted near the domain boundary, membership could in
principle grow to include a particle just inside the other domain's halo, so freezing
S at move-start and taking the a-priori radius budget above at FULL (rather than
allowing it to be exceeded via mid-move growth) is what makes the disjointness
statement decidable ahead of time). The one thing a frozen S does NOT prevent is member
particles physically DRIFTING (via ordinary accepted local_sweep displacement moves)
away from where they were when S was built -- eating into the a-priori radius budget's
~0.28 headroom. We therefore (a) fold a 0.3 slack into the required minimum PRE-MOVE
separation between the two domains' mobile-set members (measured directly, empirically,
each round, BEFORE either move starts: min distance between ANY S1 particle and ANY S2
particle must exceed XC*sig_max + 0.3 = 2.3125), and (b) assert POST HOC, every round,
the actual physical safety criterion: (pre-move min separation) - drift_A - drift_B >
XC*sig_max, i.e. that however far each side's mobile-set particles wandered during
their own 400-step move, the two sets never got close enough to actually interact.

CORRECTION vs the original task spec (measured, not assumed): the literal instruction
"assert the measured max single-particle drift < 0.3" turns out to be EMPIRICALLY FALSE
at the real N_STEPS=400 -- i and j are always trivially in S and free to random-walk
unboundedly (v2's docstring), and over 400 lambda-steps x ~1 selection/member/sweep
their accumulated displacement routinely reaches 0.3-0.5+ (a ~400-step random walk at
step=0.12 has RMS displacement of the right order). A blanket "drift < 0.3" assert was
verified to FAIL in the first smoke round (drift 0.519) while the actual bitwise-
equality test (the thing that matters) still held exactly (dW1=dW2=0.0, positions
bitwise identical) -- because the true margin is sep-at-start minus BOTH sides' drift,
which in that same round was 3.648-0.225-0.519=2.90, still comfortably above the 2.0125
interaction cutoff. So the drift-based safety check implemented below is the corrected,
physically-meaningful one ("does the pre-move separation survive both sides' measured
drift"), not the flat single-particle bound originally specified; DRIFT_BOUND is kept
only as an informational reporting threshold, not a hard gate.

Each domain's move = the standard v2 protocol: schedule = linspace(1/400, 1, 400)
(n_steps=400), sweeps_per_step=1, r_loc=1.5, step=0.12 (matches v2's STEP), applied via
one new small helper `ncmc_one_step` (a single lambda-step: bump sigma_i,sigma_j to the
next lambda value, run ONE local_sweep against the frozen (S, nS)) called n_steps times
in a driver loop -- functionally identical to v2's ncmc_work_local for
sweeps_per_step=1, except S is threaded through unchanged across steps instead of
rebuilt inside the loop. Accept/reject is min(1, exp(-beta*W) * count_ratio), applied
INDEPENDENTLY per domain.

TESTING METHODOLOGY NOTE (why per-step RNG reseeding is used, and why it doesn't beg
the question): numba's np.random is a SINGLE global stream per process. If domain 1's
and domain 2's local_sweep calls were interleaved within one process, they would draw
from that one shared stream in different orders depending on interleaving -- a pure RNG-
bookkeeping artifact that would masquerade as "domains disagree between serial and
interleaved" even with zero physical coupling. To eliminate that confound, EVERY atomic
lambda-step (both here and in the serial-copy runs used for comparison) reseeds numba's
stream to a value that is a deterministic function of (round, domain_id, step_index)
ONLY -- not of what happened before it or which order it's called in. This makes each
domain's random draws for step k IDENTICAL regardless of execution order; the ONLY
remaining way serial and interleaved results could differ is real energy coupling
through the shared position array (row_e's all-pairs sum). That is exactly the thing
Part A measures.
"""
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import torch
from numba import njit

REPO = "/mnt/ssd/GridTransformer"
LOGDIR = REPO + "/reports/logs-2026-07-17"
for p in (REPO, LOGDIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from liquid_coupling_flow.poly.model import total_U, seed_numba, XC, SIG_MAX  # noqa: E402
# v2's local-propagation kernel, imported verbatim -- NOT modified, NOT re-derived.
from poly_ncmc_v2 import _pair_env_e, build_local_set, local_sweep  # noqa: E402

try:
    os.nice(19)
except Exception as e:  # pragma: no cover - best-effort niceness only
    print(f"[warn] os.nice(19) failed: {e}", flush=True)

# ---------------------------------------------------------------------------
# Constants (see module docstring for the derivation of each).
# ---------------------------------------------------------------------------
R_DOM = 1.5
QUAL_MARGIN = 0.2
R_QUAL = R_DOM - QUAL_MARGIN            # 1.3, qualifying-pair membership radius
DS_LO, DS_HI = 0.1, 0.9
DIST_MAX = 2.0
R_LOC = 1.5                             # mobile-set radius (matches R_DOM per spec)
N_STEPS = 400
SWEEPS_PER_STEP = 1                     # local_sweep called once per lambda-step
STEP = 0.12                             # displacement step, matches v2's STEP
XC_SIGMAX = XC * SIG_MAX                # 2.0125, max possible pair-interaction reach
SLACK = 0.3
MIN_SEP_REQUIRED = XC_SIGMAX + SLACK    # 2.3125
DRIFT_BOUND = 0.3

DATA_PATH = f"{LOGDIR}/poly_tau_curves_run10.pt"
DATA_KEY = (10, 0.058)
OUT_PT = f"{LOGDIR}/poly_ncmc_parallel_demo.pt"


# ---------------------------------------------------------------------------
# Domain geometry / qualifying-pair bookkeeping (pure numpy -- cheap, few particles).
# ---------------------------------------------------------------------------
def min_image(d, L):
    return d - L * np.round(d / L)


def displacement_norm(a, b, L):
    """||min_image(a - b)|| along the last axis."""
    d = min_image(a - b, L)
    return np.sqrt((d ** 2).sum(-1))


def indices_in_ball(x, center, L, R):
    d = min_image(x - center, L)
    r2 = (d ** 2).sum(1)
    return np.where(r2 < R * R)[0]


def qualifying_pairs(x, sig, center, L, R_qual=R_QUAL, ds_lo=DS_LO, ds_hi=DS_HI,
                      dist_max=DIST_MAX):
    """Both particles inside ball(center, R_qual), |dsig| in [ds_lo, ds_hi], pair
    distance < dist_max. Returns a list of (i, j) index tuples."""
    idx = indices_in_ball(x, center, L, R_qual)
    pairs = []
    for a in range(len(idx)):
        ia = int(idx[a])
        for b in range(a + 1, len(idx)):
            ib = int(idx[b])
            ds = abs(float(sig[ia] - sig[ib]))
            if ds < ds_lo or ds > ds_hi:
                continue
            d = min_image(x[ia] - x[ib], L)
            dist = float(np.sqrt((d ** 2).sum()))
            if dist < dist_max:
                pairs.append((ia, ib))
    return pairs


def make_domains(offset_rng, L):
    u = offset_rng.random(3) * L
    v = (u + L / 2.0) % L
    return u, v


def min_dist_between_sets(x, L, idx_a, idx_b):
    xa = x[idx_a]; xb = x[idx_b]
    d = min_image(xa[:, None, :] - xb[None, :, :], L)
    r = np.sqrt((d ** 2).sum(-1))
    return float(r.min())


# ---------------------------------------------------------------------------
# Numba kernel: one atomic lambda-step against a FROZEN (S, nS) -- identical to one
# iteration of v2's ncmc_work_local body at sweeps_per_step=1, except S is threaded in
# rather than rebuilt (see module docstring, "frozen mobile set" paragraph).
# ---------------------------------------------------------------------------
@njit(cache=True)
def ncmc_one_step(x, sig, L, beta, i, j, si0, sj0, lam, r_loc, step, S, nS):
    e0 = _pair_env_e(x, sig, i, j, L)
    sig[i] = (1.0 - lam) * si0 + lam * sj0
    sig[j] = (1.0 - lam) * sj0 + lam * si0
    e1 = _pair_env_e(x, sig, i, j, L)
    local_sweep(x, sig, L, beta, i, j, r_loc, step, S, nS)
    return e1 - e0


def step_seed(base, domain_id, k):
    """Deterministic function of (round/base, domain, lambda-step) ONLY -- see the
    module docstring's TESTING METHODOLOGY NOTE for why."""
    return int((base * 1_000_003 + domain_id * 97_003 + k) % 2_147_483_647)


def drive_full_move(x, sig, L, beta, i, j, S, nS, schedule, base_seed, domain_id):
    """Run the whole n_steps-step NCMC protocol against the frozen (S, nS), mutating
    x, sig in place. Returns (W, dW_step)."""
    si0 = float(sig[i]); sj0 = float(sig[j])
    W = 0.0
    dW = np.zeros(len(schedule))
    for k in range(len(schedule)):
        lam = schedule[k]
        seed_numba(step_seed(base_seed, domain_id, k))
        w = ncmc_one_step(x, sig, L, beta, i, j, si0, sj0, lam, R_LOC, STEP, S, nS)
        W += w
        dW[k] = w
    return W, dW


def drive_interleaved(x, sig, L, beta, iA, jA, SA, nSA, iB, jB, SB, nSB, schedule,
                       base_seed):
    """Alternate ONE lambda-step of protocol A with ONE lambda-step of protocol B, on
    a single SHARED (x, sig). Domain ids are fixed (A=1, B=2) so step_seed matches the
    serial-copy runs exactly."""
    siA0 = float(sig[iA]); sjA0 = float(sig[jA])
    siB0 = float(sig[iB]); sjB0 = float(sig[jB])
    WA = WB = 0.0
    dWA = np.zeros(len(schedule)); dWB = np.zeros(len(schedule))
    for k in range(len(schedule)):
        lam = schedule[k]
        seed_numba(step_seed(base_seed, 1, k))
        wa = ncmc_one_step(x, sig, L, beta, iA, jA, siA0, sjA0, lam, R_LOC, STEP, SA, nSA)
        WA += wa; dWA[k] = wa
        seed_numba(step_seed(base_seed, 2, k))
        wb = ncmc_one_step(x, sig, L, beta, iB, jB, siB0, sjB0, lam, R_LOC, STEP, SB, nSB)
        WB += wb; dWB[k] = wb
    return WA, WB, dWA, dWB


# ---------------------------------------------------------------------------
# Part A: exactness / independence verification.
# ---------------------------------------------------------------------------
def build_domain(x, sig, L, center, pick_rng):
    pairs = qualifying_pairs(x, sig, center, L)
    if not pairs:
        return None
    i, j = pairs[int(pick_rng.integers(len(pairs)))]
    N = x.shape[0]
    buf = np.empty(N, dtype=np.int64)
    nS = build_local_set(x, i, j, R_LOC, L, buf)
    S = buf[:nS].copy()
    return dict(i=i, j=j, S=S, nS=nS, n_qual=len(pairs))


def part_a(x0, sig0, L, beta, n_rounds=25, seed=20260718):
    schedule = np.linspace(1.0 / N_STEPS, 1.0, N_STEPS)
    offset_rng = np.random.default_rng(seed)
    results = []
    idle_rounds = 0
    tried_rounds = 0
    max_dW = 0.0
    max_pos_diff = 0.0
    max_drift_seen = 0.0
    min_sep_seen = np.inf
    min_residual_margin = np.inf
    all_bitwise = True

    while len(results) < n_rounds:
        tried_rounds += 1
        u, v = make_domains(offset_rng, L)
        pick = np.random.default_rng(seed * 7919 + tried_rounds)
        domA = build_domain(x0, sig0, L, u, pick)
        domB = build_domain(x0, sig0, L, v, pick)
        if domA is None or domB is None:
            idle_rounds += 1
            continue

        SA, SB = domA["S"], domB["S"]
        overlap = np.intersect1d(SA, SB, assume_unique=False)
        assert overlap.size == 0, (
            f"round {tried_rounds}: mobile sets OVERLAP ({overlap}) -- domain "
            f"construction violated, cannot proceed")
        sep = min_dist_between_sets(x0, L, SA, SB)
        min_sep_seen = min(min_sep_seen, sep)
        assert sep > MIN_SEP_REQUIRED, (
            f"round {tried_rounds}: disjointness precondition FAILED, "
            f"min_sep={sep:.4f} <= required {MIN_SEP_REQUIRED:.4f}")

        base_seed = seed * 131 + tried_rounds
        iA, jA, nSA = domA["i"], domA["j"], domA["nS"]
        iB, jB, nSB = domB["i"], domB["j"], domB["nS"]

        # (i) SERIAL on separate full-config copies.
        xA = x0.copy(); sigA = sig0.copy()
        WA_s, _ = drive_full_move(xA, sigA, L, beta, iA, jA, SA, nSA, schedule, base_seed, 1)
        xB = x0.copy(); sigB = sig0.copy()
        WB_s, _ = drive_full_move(xB, sigB, L, beta, iB, jB, SB, nSB, schedule, base_seed, 2)

        # (ii) INTERLEAVED on one shared config.
        xI = x0.copy(); sigI = sig0.copy()
        WA_i, WB_i, _, _ = drive_interleaved(xI, sigI, L, beta, iA, jA, SA, nSA,
                                              iB, jB, SB, nSB, schedule, base_seed)

        dW1 = abs(WA_s - WA_i); dW2 = abs(WB_s - WB_i)
        max_dW = max(max_dW, dW1, dW2)
        posA_diff = float(np.max(np.abs(xA[SA] - xI[SA]))) if nSA else 0.0
        posB_diff = float(np.max(np.abs(xB[SB] - xI[SB]))) if nSB else 0.0
        max_pos_diff = max(max_pos_diff, posA_diff, posB_diff)
        bitwise = bool(
            np.array_equal(xA[SA], xI[SA]) and np.array_equal(xB[SB], xI[SB])
            and sigA[iA] == sigI[iA] and sigA[jA] == sigI[jA]
            and sigB[iB] == sigI[iB] and sigB[jB] == sigI[jB]
        )
        all_bitwise = all_bitwise and bitwise

        driftA = float(displacement_norm(xI[SA], x0[SA], L).max()) if nSA else 0.0
        driftB = float(displacement_norm(xI[SB], x0[SB], L).max()) if nSB else 0.0
        max_drift_seen = max(max_drift_seen, driftA, driftB)

        assert dW1 < 1e-10 and dW2 < 1e-10, (
            f"round {tried_rounds}: W MISMATCH serial-vs-interleaved "
            f"dW1={dW1:.3e} dW2={dW2:.3e}")

        # Corrected post-hoc safety criterion (see module docstring's CORRECTION note):
        # the true margin is the pre-move separation minus BOTH sides' measured drift,
        # not a flat single-particle drift bound.
        residual_margin = sep - driftA - driftB - XC_SIGMAX
        assert residual_margin > 0.0, (
            f"round {tried_rounds}: post-hoc safety criterion FAILED -- "
            f"sep={sep:.4f} driftA={driftA:.4f} driftB={driftB:.4f} "
            f"residual_margin={residual_margin:.4f} (this WOULD invalidate the "
            f"bitwise-equality result above; investigate immediately, do not ignore)")

        min_residual_margin = min(min_residual_margin, residual_margin)

        results.append(dict(
            round=tried_rounds, u=u.copy(), v=v.copy(), iA=iA, jA=jA, iB=iB, jB=jB,
            nSA=nSA, nSB=nSB, n_qual_A=domA["n_qual"], n_qual_B=domB["n_qual"],
            WA_serial=WA_s, WB_serial=WB_s, WA_interleaved=WA_i, WB_interleaved=WB_i,
            dW1=dW1, dW2=dW2, pos_diff_A=posA_diff, pos_diff_B=posB_diff,
            bitwise=bitwise, min_sep=sep, driftA=driftA, driftB=driftB,
            residual_margin=residual_margin,
        ))
        print(f"[partA] round {tried_rounds:>3} OK: dW1={dW1:.2e} dW2={dW2:.2e} "
              f"posdiff=({posA_diff:.2e},{posB_diff:.2e}) bitwise={bitwise} "
              f"sep={sep:.3f} drift=({driftA:.3f},{driftB:.3f}) "
              f"residual_margin={residual_margin:.3f} "
              f"|SA|={nSA} |SB|={nSB}", flush=True)

    # NOTE: no blanket "max drift < DRIFT_BOUND" gate here -- see module docstring's
    # CORRECTION paragraph. The real safety criterion (residual_margin > 0, i.e.
    # pre-move separation survives BOTH sides' measured drift) was already asserted,
    # per round, above; every round that reached this point already satisfied it.
    summary = dict(
        n_rounds=len(results), idle_rounds=idle_rounds, tried_rounds=tried_rounds,
        max_dW=max_dW, max_pos_diff=max_pos_diff, all_bitwise=all_bitwise,
        max_drift_seen=max_drift_seen, min_sep_seen=min_sep_seen,
        min_residual_margin=min_residual_margin,
    )
    return results, summary


# ---------------------------------------------------------------------------
# Part B: throughput (2-worker multiprocessing vs. serial), production accept/reject +
# apply, chained across rounds.
# ---------------------------------------------------------------------------
def _domain_worker(args):
    """ONE domain's full move: select a qualifying pair, drive the frozen-S NCMC
    protocol, decide accept/reject (min(1, exp(-beta W) * count_ratio)), restore on
    reject. Mutates x, sig IN PLACE (safe: multiprocessing pickles a private copy of
    whatever it's given; a direct in-process call mutates the caller's own array,
    which is the desired chaining behaviour there). Always returns the FINAL (post-
    decision, i.e. post-restore-if-rejected) state of the touched indices, so the
    caller can apply it uniformly regardless of calling convention."""
    (x, sig, L, beta, center, base_seed, domain_id, pick_seed) = args
    pairs = qualifying_pairs(x, sig, center, L)
    if not pairs:
        return dict(idle=True, domain_id=domain_id)
    pick = np.random.default_rng(pick_seed)
    i, j = pairs[int(pick.integers(len(pairs)))]
    n_qual_start = len(pairs)
    N = x.shape[0]
    buf = np.empty(N, dtype=np.int64)
    nS = build_local_set(x, i, j, R_LOC, L, buf)
    S = buf[:nS].copy()
    x_snap = x[S].copy()
    si0 = float(sig[i]); sj0 = float(sig[j])

    schedule = np.linspace(1.0 / N_STEPS, 1.0, N_STEPS)
    t0 = time.perf_counter()
    W, _ = drive_full_move(x, sig, L, beta, i, j, S, nS, schedule, base_seed, domain_id)
    elapsed = time.perf_counter() - t0

    n_qual_end = len(qualifying_pairs(x, sig, center, L))
    count_ratio = (n_qual_start / n_qual_end) if n_qual_end > 0 else 0.0
    acc_prob = 0.0 if n_qual_end == 0 else min(
        1.0, float(np.exp(np.clip(-beta * W, -700.0, 700.0))) * count_ratio)
    accept_rng = np.random.default_rng(pick_seed + 999_983)
    accept = bool(accept_rng.random() < acc_prob)
    drift = float(displacement_norm(x[S], x_snap, L).max()) if nS else 0.0

    if not accept:
        x[S] = x_snap
        sig[i] = si0; sig[j] = sj0

    return dict(idle=False, domain_id=domain_id, i=i, j=j, S=S, W=W,
                n_qual_start=n_qual_start, n_qual_end=n_qual_end,
                count_ratio=count_ratio, acc_prob=acc_prob, accept=accept,
                pos_final=x[S].copy(), si_final=float(sig[i]), sj_final=float(sig[j]),
                elapsed=elapsed, drift=drift)


def _apply(x, sig, res):
    if res["idle"]:
        return
    x[res["S"]] = res["pos_final"]
    sig[res["i"]] = res["si_final"]
    sig[res["j"]] = res["sj_final"]


def part_b(x0, sig0, L, beta, n_rounds=60, seed=20260718_1):
    offset_rng = np.random.default_rng(seed)
    round_specs = [make_domains(offset_rng, L) for _ in range(n_rounds)]

    # ---- Parallel: 2-worker multiprocessing pool, persistent across rounds. ----
    x_par = x0.copy(); sig_par = sig0.copy()
    pool = mp.Pool(processes=2)
    # Warm-up call (absorbs first-call numba JIT compile / process spin-up out of the
    # timed region) -- discarded, does not touch x_par/sig_par.
    pool.map(_domain_worker, [
        (x_par.copy(), sig_par.copy(), L, beta, round_specs[0][0], 0, 1, 1),
        (x_par.copy(), sig_par.copy(), L, beta, round_specs[0][1], 0, 2, 2),
    ])

    par_records = []
    par_idle = 0
    t0 = time.perf_counter()
    for r, (u, v) in enumerate(round_specs):
        args = [
            (x_par.copy(), sig_par.copy(), L, beta, u, seed * 131 + r, 1, seed * 911 + 2 * r),
            (x_par.copy(), sig_par.copy(), L, beta, v, seed * 131 + r, 2, seed * 911 + 2 * r + 1),
        ]
        resA, resB = pool.map(_domain_worker, args)
        if not resA["idle"] and not resB["idle"]:
            overlap = np.intersect1d(resA["S"], resB["S"])
            assert overlap.size == 0, (
                f"round {r}: PARALLEL apply would clobber -- mobile sets overlap "
                f"({overlap}); the antipodal domain-construction guarantee was "
                f"violated, do not silently apply")
        for res in (resA, resB):
            if res["idle"]:
                par_idle += 1
            else:
                _apply(x_par, sig_par, res)
        par_records.append((resA, resB))
    wall_par = time.perf_counter() - t0
    pool.close(); pool.join()

    # ---- Serial: same 120 moves, one process, chained sequentially. ----
    x_ser = x0.copy(); sig_ser = sig0.copy()
    _domain_worker((x_ser.copy(), sig_ser.copy(), L, beta, round_specs[0][0], 0, 1, 1))  # warmup
    ser_records = []
    ser_idle = 0
    t0 = time.perf_counter()
    for r, (u, v) in enumerate(round_specs):
        for domain_id, center in ((1, u), (2, v)):
            pick_seed = seed * 911 + 2 * r + (0 if domain_id == 1 else 1)
            res = _domain_worker((x_ser, sig_ser, L, beta, center, seed * 131 + r,
                                   domain_id, pick_seed))
            if res["idle"]:
                ser_idle += 1
            else:
                _apply(x_ser, sig_ser, res)
            ser_records.append(res)
    wall_ser = time.perf_counter() - t0

    def stats(records_flat, wall, n_workers):
        moved = [r for r in records_flat if not r["idle"]]
        accepted = [r for r in moved if r["accept"]]
        n_moves = len(moved)
        n_acc = len(accepted)
        max_drift = max((r["drift"] for r in moved), default=0.0)
        mean_worker_elapsed = float(np.mean([r["elapsed"] for r in moved])) if moved else 0.0
        return dict(
            wall=wall, n_moves=n_moves, n_accepted=n_acc,
            accept_rate=(n_acc / n_moves if n_moves else 0.0),
            wall_per_move=(wall / n_moves if n_moves else float("nan")),
            wall_per_accept=(wall / n_acc if n_acc else float("nan")),
            mean_worker_elapsed=mean_worker_elapsed, max_drift=max_drift,
            n_workers=n_workers,
        )

    par_flat = [r for pair in par_records for r in pair]
    ser_stats = stats(ser_records, wall_ser, 1)
    par_stats = stats(par_flat, wall_par, 2)

    return dict(
        round_specs=round_specs, ser_records=ser_records, par_records=par_records,
        ser_idle=ser_idle, par_idle=par_idle, ser_stats=ser_stats, par_stats=par_stats,
    )


# ---------------------------------------------------------------------------
# Part C: scaling arithmetic / projection.
# ---------------------------------------------------------------------------
def part_c(ser_stats, N=600):
    d = 2.0 * (R_LOC + XC_SIGMAX)  # minimum non-interacting center-to-center spacing
    T_move = ser_stats["mean_worker_elapsed"]
    accept_rate = ser_stats["accept_rate"]

    def domains_fit(n):
        L_ = n ** (1.0 / 3.0)
        return max(1, int(L_ // d)) ** 3

    def exch_per_hour(n):
        return domains_fit(n) * (3600.0 / T_move) * accept_rate if T_move > 0 else 0.0

    lines = [
        f"[partC] measured per-move wall (single-domain, serial baseline): "
        f"{T_move * 1000:.1f} ms/move ({1.0 / T_move if T_move > 0 else 0:.1f} moves/s), "
        f"accept_rate={accept_rate:.3f}",
        f"[partC] non-interacting domain spacing d = 2*(r_loc + XC*sig_max) = "
        f"2*({R_LOC}+{XC_SIGMAX:.4f}) = {d:.3f}; this demo's N={N} box (L={N ** (1.0 / 3.0):.3f}) "
        f"fits only the 2 domains built via the antipodal-offset trick (simple-cubic "
        f"tiling gives floor(L/d)^3 = {domains_fit(N)})",
        f"[partC] simple-cubic domain-packing projection: N=1e3 (L={1000 ** (1 / 3):.2f}) "
        f"-> floor(L/d)^3 = {domains_fit(1000)} simultaneous domains; "
        f"N=1e4 (L={10000 ** (1 / 3):.2f}) -> {domains_fit(10000)} simultaneous domains "
        f"(ARITHMETIC EXTRAPOLATION -- only M=2 was built/tested here)",
        f"[partC] projected accepted-exchanges/wall-hour at 1 CPU-core-per-domain: "
        f"N=600: {exch_per_hour(600):.0f}/hr ({domains_fit(600)} domain); "
        f"N=1e3: {exch_per_hour(1000):.0f}/hr ({domains_fit(1000)} domain); "
        f"N=1e4: {exch_per_hour(10000):.0f}/hr ({domains_fit(10000)} domains, "
        f"needs {domains_fit(10000)} cores to realize in wall-time, not just CPU-time)",
        f"[partC] TAKEAWAY: per-exchange CPU cost is flat in N (local moves only see "
        f"~|S| particles); PARALLEL WALL-CLOCK throughput scales ~ domains_fit(N) ~ "
        f"O(N) once L exceeds ~2 domain-spacings (N gtrsim {int(2 * d) ** 3}), whereas a "
        f"single glassy trajectory's tau_alpha cannot be parallelized at all -- this is "
        f"the scalability lever the campaign deliverable needs.",
    ]
    for ln in lines:
        print(ln, flush=True)
    return dict(d=d, T_move=T_move, accept_rate=accept_rate,
                domains_fit_N=domains_fit(N), domains_fit_1e3=domains_fit(1000),
                domains_fit_1e4=domains_fit(10000),
                exch_per_hour_N=exch_per_hour(N), exch_per_hour_1e3=exch_per_hour(1000),
                exch_per_hour_1e4=exch_per_hour(10000))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t_start = time.time()
    D = torch.load(DATA_PATH, weights_only=False)
    rec = D[DATA_KEY]
    L = float(rec["L"])
    T = float(rec["T"])
    beta = 1.0 / T
    # The last banked frame is the ONE consistent (x, sig) pairing for this
    # (un-rebanked, run10) file -- see progress.md's CRITICAL DATA BUG note: swap_sweep
    # permutes sig between captured frames but only the end-of-rung sig was saved, so
    # only the LAST frame's x is guaranteed to pair correctly with the saved 'sig'.
    x0 = np.asarray(rec["x"][-1], dtype=np.float64).copy()
    sig0 = np.asarray(rec["sig"], dtype=np.float64).copy()
    N = x0.shape[0]
    u_n = total_U(x0, sig0, L) / N
    print(f"Loaded N={N} L={L:.4f} T={T} U/N={u_n:+.4f}", flush=True)
    assert 0.15 <= u_n <= 0.35, f"U/N={u_n:.4f} outside sanity band [0.15,0.35]"

    print("=" * 70, flush=True)
    print("PART A: exactness / independence verification (25 rounds)", flush=True)
    print("=" * 70, flush=True)
    a_results, a_summary = part_a(x0, sig0, L, beta, n_rounds=25)
    print(f"[partA] DONE: {a_summary}", flush=True)

    print("=" * 70, flush=True)
    print("PART B: throughput (2-worker multiprocessing vs serial, 60 rounds)", flush=True)
    print("=" * 70, flush=True)
    b_out = part_b(x0, sig0, L, beta, n_rounds=60)
    ss, ps = b_out["ser_stats"], b_out["par_stats"]
    print(f"[partB] SERIAL : wall={ss['wall']:.2f}s moves={ss['n_moves']} "
          f"accepted={ss['n_accepted']} acc_rate={ss['accept_rate']:.3f} "
          f"wall/move={ss['wall_per_move'] * 1000:.2f}ms "
          f"wall/accept={ss['wall_per_accept']:.3f}s idle={b_out['ser_idle']} "
          f"max_drift={ss['max_drift']:.3f}", flush=True)
    print(f"[partB] PARALLEL(2w): wall={ps['wall']:.2f}s moves={ps['n_moves']} "
          f"accepted={ps['n_accepted']} acc_rate={ps['accept_rate']:.3f} "
          f"wall/move={ps['wall_per_move'] * 1000:.2f}ms "
          f"wall/accept={ps['wall_per_accept']:.3f}s idle={b_out['par_idle']} "
          f"max_drift={ps['max_drift']:.3f}", flush=True)
    speedup = ss["wall"] / ps["wall"] if ps["wall"] > 0 else float("nan")
    print(f"[partB] wall-clock speedup (parallel vs serial, 2 workers) = {speedup:.2f}x", flush=True)

    print("=" * 70, flush=True)
    print("PART C: scaling statement", flush=True)
    print("=" * 70, flush=True)
    c_out = part_c(ss, N=N)

    # See module docstring's CORRECTION note: single-particle drift routinely exceeds
    # the originally-specified 0.3 bound at real N_STEPS=400 (i, j random-walk freely),
    # so this is reported, not gated on. The actual load-bearing safety criterion
    # (residual_margin = pre-move-sep - driftA - driftB - XC_SIGMAX > 0) was asserted
    # per-round inside part_a already; part_a's min_residual_margin summarizes it here.
    global_max_drift = max(a_summary["max_drift_seen"], ss["max_drift"], ps["max_drift"])
    print(f"[final] global max measured single-particle drift across all rounds = "
          f"{global_max_drift:.4f} (informational only, NOT gated -- see docstring "
          f"CORRECTION); Part A's min residual safety margin = "
          f"{a_summary['min_residual_margin']:.4f} (must be > 0, WAS asserted per-round)",
          flush=True)
    assert a_summary["min_residual_margin"] > 0.0

    out = dict(
        N=N, L=L, T=T, beta=beta, U_N=u_n,
        params=dict(R_DOM=R_DOM, R_QUAL=R_QUAL, DS_LO=DS_LO, DS_HI=DS_HI,
                    DIST_MAX=DIST_MAX, R_LOC=R_LOC, N_STEPS=N_STEPS,
                    SWEEPS_PER_STEP=SWEEPS_PER_STEP, STEP=STEP, XC_SIGMAX=XC_SIGMAX,
                    SLACK=SLACK, MIN_SEP_REQUIRED=MIN_SEP_REQUIRED,
                    DRIFT_BOUND=DRIFT_BOUND),
        part_a_results=a_results, part_a_summary=a_summary,
        part_b_ser_records=b_out["ser_records"], part_b_par_records=b_out["par_records"],
        part_b_ser_stats=ss, part_b_par_stats=ps,
        part_b_ser_idle=b_out["ser_idle"], part_b_par_idle=b_out["par_idle"],
        part_b_speedup=speedup,
        part_c=c_out,
        global_max_drift=global_max_drift,
        elapsed_total_s=time.time() - t_start,
    )
    torch.save(out, OUT_PT)
    print(f"Saved {OUT_PT}", flush=True)
    print(f"TOTAL ELAPSED {time.time() - t_start:.1f}s", flush=True)
    print("PARALLEL NCMC DEMO DONE", flush=True)
