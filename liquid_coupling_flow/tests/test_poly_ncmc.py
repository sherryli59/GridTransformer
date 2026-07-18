import sys
import numpy as np
import pytest

sys.path.insert(0, "/mnt/ssd/GridTransformer/reports/logs-2026-07-17")
from poly_ncmc_v2 import build_local_set, local_sweep, ncmc_work_local, _dist_pbc  # noqa: E402
from poly_ncmc_probe import ncmc_work  # noqa: E402  (v1, for the matched-limit consistency test)
from poly_ncmc_chain import (count_window_pairs, pick_window_pair, ncmc_attempt,  # noqa: E402
                              run_mix, WINDOW_LO, WINDOW_HI)
from liquid_coupling_flow.poly.model import draw_sigmas, seed_numba  # noqa: E402


def _safe_positions(rng, n, L, min_d=0.3):
    for _ in range(10_000):
        x = rng.random((n, 3)) * L
        d = x[:, None, :] - x[None, :, :]
        d -= L * np.round(d / L)
        r = np.sqrt((d ** 2).sum(-1)) + np.eye(n) * 1e9
        if r.min() >= min_d:
            return x
    raise RuntimeError("could not draw safe positions")


def _small_system(seed=0, n=40, rho=0.9):
    rng = np.random.default_rng(seed)
    L = (n / rho) ** (1.0 / 3.0)
    x = _safe_positions(rng, n, L, min_d=0.25)
    sig = draw_sigmas(n, seed=seed + 1)
    return x, sig, L


# ---------------------------------------------------------------------------
# Test 1: beta=0 exactness of the local kernel -- positions inside S stay uniform,
# reflecting boundary => no pileup, no drift of |S| occupancy.
#
# Both sub-tests below deliberately EXCLUDE i and j from the candidate list (build S
# normally, then drop indices i, j from it). This isolates the thing being tested --
# whether a MOBILE particle's motion near the r_loc boundary is a correctly
# reflecting DB chain -- from the (separately, exactly covered by test 4 and the
# trivial-self-membership argument in the module docstring) fact that i and j
# themselves are always trivially in S and can drift freely. Without pinning i,j,
# a long run lets the pair random-walk far enough that the FROZEN candidate list
# (built once, at t=0) legitimately goes stale relative to i/j's live position --
# that is expected/intentional per-lambda-step behaviour (S refreshes next step),
# not a boundary bug, so testing it here would conflate two different things.
# ---------------------------------------------------------------------------
def test_local_sweep_beta0_uniform_no_boundary_pileup():
    seed_numba(11)
    n = 60
    rng = np.random.default_rng(1)
    L = (n / 0.5) ** (1.0 / 3.0)
    x = _safe_positions(rng, n, L, min_d=0.25)
    sig = draw_sigmas(n, seed=2)
    i, j = 0, 1
    r_loc = 1.6
    assert r_loc < L / 2

    N = x.shape[0]
    Sfull = np.empty(N, dtype=np.int64)
    nSfull = build_local_set(x, i, j, r_loc, L, Sfull)
    mobiles = np.array([m for m in Sfull[:nSfull] if m != i and m != j], dtype=np.int64)
    assert len(mobiles) >= 5
    nS = len(mobiles)

    dists_before = np.array([min(_dist_pbc(x, m, i, L), _dist_pbc(x, m, j, L)) for m in mobiles])
    for _ in range(600):
        local_sweep(x, sig, L, 0.0, i, j, r_loc, 0.25, mobiles, nS)
    dists_after = np.array([min(_dist_pbc(x, m, i, L), _dist_pbc(x, m, j, L)) for m in mobiles])

    # i, j are pinned (excluded from the candidate list) here, so "current position of
    # i/j" == "block-start position of i/j" throughout -- the < r_loc invariant is now
    # an EXACT guarantee of a correct exit-rejection, not merely statistical.
    assert np.all(dists_after < r_loc + 1e-9)

    # Distribution check: at beta=0 the accepted sub-chain is a symmetric random walk
    # reflected at the r_loc boundary -- no systematic pileup at the boundary shell.
    edges = np.linspace(0.0, r_loc, 5)
    h0, _ = np.histogram(dists_before, bins=edges, density=True)
    h1, _ = np.histogram(dists_after, bins=edges, density=True)
    assert np.max(np.abs(h0 - h1)) < 0.6 * (h0.max() + 1e-9)


def test_local_sweep_beta0_histogram_matches_uniform_ball_shape():
    """A more direct density check: with beta=0 and a reflecting boundary, the
    stationary distribution of mobile particles' distance to the (pinned) pair
    inside a ball of radius r_loc should be volume-weighted (density ~ r^2 in 3D),
    not piled up at the boundary. Run many independent short chains from the same
    start and check the outer-shell occupancy fraction sits in the volume-weighted
    ballpark instead of collapsing to ~1 (pileup) or ~0 (leak/over-reflection)."""
    seed_numba(12)
    n = 60
    rng = np.random.default_rng(3)
    L = (n / 0.4) ** (1.0 / 3.0)
    x0 = _safe_positions(rng, n, L, min_d=0.2)
    sig = draw_sigmas(n, seed=4)
    i, j = 0, 1
    r_loc = 1.5
    N = x0.shape[0]
    Sfull = np.empty(N, dtype=np.int64)
    nSfull = build_local_set(x0, i, j, r_loc, L, Sfull)
    mobiles = np.array([m for m in Sfull[:nSfull] if m != i and m != j], dtype=np.int64)
    assert len(mobiles) >= 5
    nS = len(mobiles)
    # Volume-weighted expectation for the outer 20% radial shell of a single sphere
    # (r in [0.8 r_loc, r_loc]): (1 - 0.8^3) ~= 0.488 of the ball's volume. The true
    # region is a union of two spheres (one per pair member) so the exact number
    # differs, but it should land well inside a broad band, not at the 0/1 extremes.
    outer_shell_frac = []
    for trial in range(80):
        xt = x0.copy()
        for _ in range(30):
            local_sweep(xt, sig, L, 0.0, i, j, r_loc, 0.35, mobiles, nS)
        d = np.array([min(_dist_pbc(xt, m, i, L), _dist_pbc(xt, m, j, L)) for m in mobiles])
        assert np.all(d < r_loc + 1e-9)
        outer_shell_frac.append(np.mean(d > 0.8 * r_loc))
    frac = np.mean(outer_shell_frac)
    assert 0.05 < frac < 0.9


# ---------------------------------------------------------------------------
# Test 2: consistency with v1 in the matched limit r_loc > L/2 (S == all particles)
# ---------------------------------------------------------------------------
def test_matches_v1_when_r_loc_covers_whole_box():
    n = 40
    rng = np.random.default_rng(7)
    rho = 0.85
    L = (n / rho) ** (1.0 / 3.0)
    x0 = _safe_positions(rng, n, L, min_d=0.3)
    sig0 = draw_sigmas(n, seed=8)
    i, j = 3, 17
    r_loc = L  # > L/2 => S(x) == all particles for any x in the box
    n_steps = 50
    schedule = np.linspace(1.0 / n_steps, 1.0, n_steps)
    beta = 1.0 / 0.3

    Ws_local, Ws_v1 = [], []
    REPEATS = 40
    for rep in range(REPEATS):
        seed_numba(100 + rep)
        x = x0.copy(); sig = sig0.copy()
        dW = np.zeros(n_steps); nS = np.zeros(n_steps)
        Ws_local.append(ncmc_work_local(x, sig, L, beta, i, j, schedule, 1, r_loc, 0.15, dW, nS))

    for rep in range(REPEATS):
        seed_numba(200 + rep)
        x = x0.copy(); sig = sig0.copy()
        Ws_v1.append(ncmc_work(x, sig, L, beta, i, j, n_steps, 1, 0.15))

    Ws_local = np.array(Ws_local); Ws_v1 = np.array(Ws_v1)
    m1, m2 = Ws_local.mean(), Ws_v1.mean()
    sem = np.sqrt(Ws_local.std(ddof=1) ** 2 / REPEATS + Ws_v1.std(ddof=1) ** 2 / REPEATS)
    assert abs(m1 - m2) < 2 * sem


# ---------------------------------------------------------------------------
# Test 3: work bookkeeping -- schedule=[1.0] reproduces the classical vertical swap
# exactly (deterministic, no RNG involved in this branch).
# ---------------------------------------------------------------------------
def test_zero_step_schedule_reproduces_classical_swap_exactly():
    n = 30
    rng = np.random.default_rng(9)
    L = (n / 0.8) ** (1.0 / 3.0)
    x0 = _safe_positions(rng, n, L, min_d=0.3)
    sig0 = draw_sigmas(n, seed=10)
    i, j = 2, 11
    beta = 1.0 / 0.2

    seed_numba(0)
    x = x0.copy(); sig = sig0.copy()
    dW = np.zeros(1); nS = np.zeros(1)
    schedule = np.array([1.0])
    W_local = ncmc_work_local(x, sig, L, beta, i, j, schedule, 1, 2.0, 0.15, dW, nS)

    seed_numba(0)
    x_ref = x0.copy(); sig_ref = sig0.copy()
    W_v1 = ncmc_work(x_ref, sig_ref, L, beta, i, j, 0, 1, 0.15)

    assert abs(W_local - W_v1) < 1e-10
    assert dW[0] == pytest.approx(W_local, abs=1e-10)
    assert nS[0] == 0.0  # no propagation happened -- pure vertical swap
    # positions must be untouched (only sigma flips), same as v1's n_steps==0 branch
    assert np.allclose(x, x0)
    expect_sig = sig0.copy()
    expect_sig[i], expect_sig[j] = sig0[j], sig0[i]
    assert np.array_equal(sig, expect_sig)
    assert np.array_equal(sig, sig_ref)  # matches v1's result exactly, not just W


# ---------------------------------------------------------------------------
# Test 4: exit-rejection unit test -- a particle proposed to move outside r_loc of
# BOTH pair members is always rejected.
# ---------------------------------------------------------------------------
def test_exit_rejection_always_rejects_out_of_range_proposal():
    # Construct a tiny hand-built system: i, j far apart from a lone mobile particle
    # m sitting just inside the r_loc boundary, with a huge proposal step so the
    # single displacement attempt is essentially guaranteed to try to leave S. Run
    # many attempts and require m NEVER ends up outside S (i.e. every excursion
    # attempt was rejected -- since if even one leaked through, its post-move
    # distance would exceed r_loc).
    L = 50.0
    x = np.array([
        [5.0, 5.0, 5.0],    # i
        [40.0, 5.0, 5.0],   # j -- far away, irrelevant to m's boundary test (must be
                             # a distinct position: identical i==j positions divide by
                             # zero in pair_v's 1/r^2 term, a model edge case unrelated
                             # to the kernel under test)
        [6.2, 5.0, 5.0],    # m: distance 1.2 from i, just inside r_loc=1.5
    ])
    sig = np.array([1.0, 1.0, 1.0])
    i, j = 0, 1
    r_loc = 1.5
    N = x.shape[0]
    # Candidate list = {m} ONLY (i, j pinned/excluded): isolates the exit-rejection
    # rule for a genuinely mobile non-pair particle -- if i or j were also selectable
    # here they'd be free to random-walk away with huge_step (never exit-rejected,
    # per the trivial self-membership rule), which would drag the anchor and falsely
    # look like m escaping S. That's the same confound test 1 avoids.
    S = np.array([2], dtype=np.int64)
    nS = 1

    seed_numba(555)
    huge_step = 20.0  # gaussian std of 20 in a box that only allows r_loc=1.5 -> almost
                       # every proposal for m lands outside S
    for _ in range(500):
        local_sweep(x, sig, L, 1.0, i, j, r_loc, huge_step, S, nS)
        d = min(_dist_pbc(x, 2, i, L), _dist_pbc(x, 2, j, L))
        assert d < r_loc, "particle m escaped S -- exit-rejection failed to fire"


# ===========================================================================
# NOVELTY-1: production chained kernel (poly_ncmc_chain.py) tests.
# ===========================================================================

# ---------------------------------------------------------------------------
# Test 1: window-selection symmetry -- the qualifying-pair count under the |dsigma| in
# [WINDOW_LO, WINDOW_HI] restriction is IDENTICAL before vs after a completed sigma_i<->sigma_j
# endpoint swap (any completed swap, not just window-restricted ones -- see poly_ncmc_chain.py's
# module docstring for the argument: swapping the values held by i,j only PERMUTES which pair
# carries which |dsigma| value, for every third particle k, so the total count can't change).
# ---------------------------------------------------------------------------
def test_window_count_invariant_under_endpoint_swap():
    n = 80
    rng = np.random.default_rng(21)
    L = (n / 0.85) ** (1.0 / 3.0)
    x = _safe_positions(rng, n, L, min_d=0.25)
    sig = draw_sigmas(n, seed=22)

    seed_numba(9001)
    # pick_window_pair uses the numba RNG stream (same one seed_numba controls everywhere else
    # in the chain); draw a real in-window pair, exactly like the production attempt would.
    i, j, cnt_before_via_pick = pick_window_pair(sig, WINDOW_LO, WINDOW_HI)
    assert cnt_before_via_pick > 0, "test system should have >=1 qualifying pair in [0.1, 0.9]"

    cnt_before = count_window_pairs(sig, WINDOW_LO, WINDOW_HI)
    assert cnt_before == cnt_before_via_pick

    # apply a FULLY COMPLETED swap of the chosen pair's sigma values (this is exactly what an
    # accepted NCMC move leaves behind: sig[i],sig[j] fully exchanged, interpolation at lambda=1).
    si0, sj0 = sig[i], sig[j]
    sig[i], sig[j] = sj0, si0

    cnt_after = count_window_pairs(sig, WINDOW_LO, WINDOW_HI)
    assert cnt_after == cnt_before, (
        f"qualifying-pair count changed under a completed endpoint swap: {cnt_before} -> {cnt_after} "
        "-- this breaks the no-Hastings-factor exactness argument")

    # also check a handful of OTHER (non-window-selected) pairs, to confirm the invariance is
    # general and not an artifact of this one pair.
    for trial_seed in range(30, 40):
        rng2 = np.random.default_rng(trial_seed)
        sig2 = draw_sigmas(n, seed=trial_seed)
        p, q = rng2.integers(n), rng2.integers(n - 1)
        if q >= p:
            q += 1
        c_before = count_window_pairs(sig2, WINDOW_LO, WINDOW_HI)
        sig2[p], sig2[q] = sig2[q], sig2[p]
        c_after = count_window_pairs(sig2, WINDOW_LO, WINDOW_HI)
        assert c_after == c_before


# ---------------------------------------------------------------------------
# Test 2: chained-application correctness. THE #1 BUG RISK -- v1/v2's kernels mutate x, sig in
# place with no undo; ncmc_attempt (poly_ncmc_chain.py) is the only place restore-on-reject
# happens, and this test verifies it directly and explicitly, alongside the accept-path's
# "interpolation fully completed" guarantee. Uses a seed search (not a hardcoded lucky seed) so
# it exercises BOTH branches deterministically-discovered from a real system, rather than relying
# on contrived beta=0/beta=inf edge cases that would dodge the actual accept/reject machinery.
# ---------------------------------------------------------------------------
def test_restore_on_reject_and_apply_on_accept():
    x0, sig0, L = _small_system(seed=77, n=60, rho=0.9)
    beta = 1.0 / 0.3   # moderate T: neither always-accept nor always-reject
    r_loc, n_steps, sweeps_per_step, step = 2.0, 30, 1, 0.15

    found_accept = found_reject = False
    for trial in range(150):
        x = x0.copy()
        sig = sig0.copy()
        seed_numba(trial)
        attempted, accepted, i, j, ds, W, cnt = ncmc_attempt(
            x, sig, L, beta, r_loc, n_steps, sweeps_per_step, step, WINDOW_LO, WINDOW_HI)
        if not attempted:
            continue

        if accepted and not found_accept:
            found_accept = True
            si0_orig, sj0_orig = sig0[i], sig0[j]
            # sig array is the EXACT swap of the pair's ORIGINAL sigmas -- interpolation fully
            # completed (lambda reached exactly 1.0), not left partway.
            assert abs(sig[i] - sj0_orig) < 1e-12
            assert abs(sig[j] - si0_orig) < 1e-12
            # every OTHER particle's sigma is untouched (only i,j's sigma channel moves).
            others = np.array([k for k in range(60) if k not in (i, j)])
            assert np.array_equal(sig[others], sig0[others])
            # x has evolved (local propagation ran and moved at least something).
            assert not np.array_equal(x, x0), "x did not evolve on an accepted NCMC move"

        if (not accepted) and not found_reject:
            found_reject = True
            # restore-on-reject: BOTH arrays bitwise identical to the pre-attempt snapshot.
            assert np.array_equal(x, x0), "x not exactly restored on NCMC reject"
            assert np.array_equal(sig, sig0), "sig not exactly restored on NCMC reject"

        if found_accept and found_reject:
            break

    assert found_accept, "no accepted NCMC attempt found in 150 trials -- cannot test accept path"
    assert found_reject, "no rejected NCMC attempt found in 150 trials -- cannot test restore-on-reject"


# ---------------------------------------------------------------------------
# Test 3: composite-chain smoke -- 2000 sweeps of MIX B ("swap+ncmc") at T=0.2 runs end to end
# through run_mix, energies stay finite, and at least one NCMC attempt actually occurred.
# ---------------------------------------------------------------------------
def test_composite_chain_smoke_swap_ncmc(tmp_path):
    x0, sig0, L = _small_system(seed=123, n=80, rho=0.85)
    T = 0.2
    beta = 1.0 / T
    out_path = tmp_path / "smoke.pt"
    rec = run_mix(
        mix="swap+ncmc", x0=x0, sig0=sig0, L=L, beta=beta, T=T, budget=2000,
        ncmc_every=50, n_steps=20, r_loc=1.5, sweeps_per_step=1, seed=1,
        out_data={}, out_path=out_path,
    )
    assert np.all(np.isfinite(rec["U_N"]))
    assert np.all(np.isfinite(rec["Q_self"]))
    assert np.all(np.isfinite(rec["C_sig"]))
    assert rec["ncmc_attempts_cum"][-1] >= 1, "no NCMC attempt occurred in 2000 sweeps @ ncmc_every=50"
    assert out_path.exists()  # incremental save happened
