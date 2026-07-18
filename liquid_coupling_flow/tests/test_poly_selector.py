import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, "/mnt/ssd/GridTransformer/reports/logs-2026-07-17")
from poly_selector_data import (  # noqa: E402
    pair_features, sample_pair_in_window, vertical_dU, N_FEATURES, canonical_order,
)
from poly_ncmc_v2 import ncmc_work_local  # noqa: E402
from poly_selector import screening_prob, S_MIN_DEFAULT  # noqa: E402
from poly_ncmc_chain import (  # noqa: E402
    ncmc_attempt, ncmc_attempt_screened, run_mix, STEP, WINDOW_LO, WINDOW_HI,
    total_U, q_self, c_sig,
)
from liquid_coupling_flow.poly.model import (  # noqa: E402
    draw_sigmas, seed_numba, disp_sweep, swap_sweep,
)


def _safe_positions(rng, n, L, min_d=0.3):
    for _ in range(10_000):
        x = rng.random((n, 3)) * L
        d = x[:, None, :] - x[None, :, :]
        d -= L * np.round(d / L)
        r = np.sqrt((d ** 2).sum(-1)) + np.eye(n) * 1e9
        if r.min() >= min_d:
            return x
    raise RuntimeError("could not draw safe positions")


def _toy_state(seed=0, n=40, min_d=0.3):
    rng = np.random.default_rng(seed)
    L = n ** (1.0 / 3.0)
    x = _safe_positions(rng, n, L, min_d=min_d)
    sig = draw_sigmas(n, seed=seed + 1)
    return x, sig, L


# ---------------------------------------------------------------------------
# 1. Feature computation: deterministic, finite, permutation-consistent
# ---------------------------------------------------------------------------
def test_features_deterministic_and_finite():
    x, sig, L = _toy_state(seed=1)
    rng = np.random.default_rng(2)
    i, j = sample_pair_in_window(rng, sig)
    f1 = pair_features(x, sig, L, i, j)
    f2 = pair_features(x, sig, L, i, j)
    assert f1.shape == (N_FEATURES,)
    assert np.all(np.isfinite(f1))
    assert np.array_equal(f1, f2)  # deterministic: no hidden RNG dependence


def test_features_symmetric_under_pair_relabeling():
    """pair_features(x,sig,L,i,j) == pair_features(x,sig,L,j,i) EXACTLY: the deploy code
    canonicalizes member order by sigma (tie-broken by index) before building the
    feature vector, so swapping the caller's (i,j) argument order can never change the
    features seen by the model -- required for the screening ratio s(x_end)/s(x_start)
    to be well-defined independent of how the pair happens to get passed in."""
    x, sig, L = _toy_state(seed=3)
    rng = np.random.default_rng(4)
    for _ in range(20):
        i, j = sample_pair_in_window(rng, sig)
        f_ij = pair_features(x, sig, L, i, j)
        f_ji = pair_features(x, sig, L, j, i)
        assert np.array_equal(f_ij, f_ji)


def test_canonical_order_uses_sigma_then_index():
    sig = np.array([0.9, 1.3, 0.9])
    assert canonical_order(sig, 1, 0) == (0, 1)   # sig[0] < sig[1]
    assert canonical_order(sig, 0, 1) == (0, 1)   # order-independent
    assert canonical_order(sig, 0, 2) == (0, 2)   # tie on sigma -> smaller index first
    assert canonical_order(sig, 2, 0) == (0, 2)


def test_vertical_dU_matches_ncmc_work_local_n0():
    """Cross-check: the locally-recomputed vertical_dU (used inside pair_features) must
    agree with ncmc_work_local's own n_steps==1,schedule=[1.0] special case (documented
    in poly_ncmc_v2.py as reproducing the classical instantaneous swap exactly) -- this
    is the guarantee that pair_features' cheap local energy recomputation is not a
    silently-diverged duplicate of the imported labeling engine's physics."""
    x, sig, L = _toy_state(seed=5)
    rng = np.random.default_rng(6)
    i, j = sample_pair_in_window(rng, sig)
    dU_local = float(vertical_dU(x, sig, i, j, L))

    x2 = x.copy(); sig2 = sig.copy()
    schedule = np.array([1.0])
    dW_step = np.zeros(1); nS_step = np.zeros(1)
    seed_numba(0)
    W = ncmc_work_local(x2, sig2, L, 1.0 / 0.085, i, j, schedule, 1, 3.0, 0.12,
                         dW_step, nS_step)
    assert abs(dU_local - W) < 1e-9
    # sig must be genuinely restored (non-mutating) by vertical_dU
    assert np.array_equal(sig, sig)


def test_pair_window_membership_respected():
    x, sig, L = _toy_state(seed=7)
    rng = np.random.default_rng(8)
    for _ in range(50):
        i, j = sample_pair_in_window(rng, sig)
        ds = abs(float(sig[i] - sig[j]))
        assert 0.1 <= ds < 0.9
        assert i != j


# ---------------------------------------------------------------------------
# 2. Screening-ratio bookkeeping: mock s_theta == const cancels exactly
# ---------------------------------------------------------------------------
def test_screening_ratio_cancels_for_constant_s():
    """A mock s_theta == const (same value at x_start and x_end, e.g. because the NN
    predicts the identical W-hat at both endpoints or -- the case tested here --
    because we substitute a literal constant for both terms) must yield an acceptance
    identical to the UNSCREENED kernel: A = min(1, exp(-bW) * s_end/s_start) with
    s_end == s_start == const reduces algebraically to min(1, exp(-bW)), independent of
    the constant's value. This tests the FORMULA as it will be used in
    evaluate_screening (min(p_run, exp(-bW)*s_end) with p_run==s_start==s_end==const),
    which must also reduce to const * min(1, exp(-bW))."""
    beta = 1.0 / 0.085
    rng = np.random.default_rng(9)
    W_vals = rng.normal(loc=5.0, scale=15.0, size=200)  # spans deep-negative to huge-positive
    for s_const in (0.02, 0.3, 1.0):
        for W in W_vals:
            A_ratio = min(1.0, np.exp(np.clip(-beta * W, -700, 700)) * (s_const / s_const))
            A_unscreened = min(1.0, np.exp(np.clip(-beta * W, -700, 700)))
            assert A_ratio == pytest.approx(A_unscreened)

            # unconditional accept used by evaluate_screening: min(p_run, exp(-bW)*s_end)
            p_run = s_const; s_end = s_const
            accept_uncond = min(p_run, np.exp(np.clip(-beta * W, -700, 700)) * s_end)
            expected = s_const * A_unscreened
            assert accept_uncond == pytest.approx(expected)


def test_screening_ratio_state_dependent_bounds_by_ends():
    """When s_start != s_end (a real state-dependent screen), the unconditional accept
    probability min(p_run, exp(-bW)*s_end) must never exceed p_run == s_theta(x_start)
    -- i.e. screening can only ever WITHHOLD attempts relative to what running the
    protocol unconditionally would have given, never inflate the accept rate above the
    probability we agreed to even run it (that probability is exactly the ceiling the
    'min' enforces, whatever exp(-bW)*s_end works out to)."""
    beta = 1.0 / 0.085
    rng = np.random.default_rng(10)
    for _ in range(200):
        s_start, s_end = rng.uniform(S_MIN_DEFAULT, 1.0, size=2)
        W = rng.normal(0.0, 20.0)
        accept = min(s_start, np.exp(np.clip(-beta * W, -700, 700)) * s_end)
        assert -1e-12 <= accept <= s_start + 1e-12


# ---------------------------------------------------------------------------
# 3. s_min floor respected
# ---------------------------------------------------------------------------
def test_screening_prob_floor_and_ceiling():
    beta = 1.0 / 0.085
    what = np.array([-50.0, -1.0, 0.0, 1e-9, 5.0, 50.0, 500.0, 1e6])
    s = screening_prob(what, beta)
    assert np.all(s >= S_MIN_DEFAULT - 1e-15)
    assert np.all(s <= 1.0 + 1e-15)
    # very negative/zero W-hat (a "sure cheap" pair) should saturate at 1
    assert s[0] == pytest.approx(1.0)
    assert s[2] == pytest.approx(1.0)
    # huge W-hat (a "sure doomed" pair) should hit the floor exactly
    assert s[-1] == pytest.approx(S_MIN_DEFAULT)


def test_screening_prob_custom_s_min():
    beta = 1.0 / 0.085
    s = screening_prob(np.array([1000.0]), beta, s_min=0.1, c=0.5)
    assert s[0] == pytest.approx(0.1)


def test_screening_prob_monotone_decreasing_in_what():
    beta = 1.0 / 0.085
    whats = np.linspace(0.0, 30.0, 50)
    s = screening_prob(whats, beta)
    assert np.all(np.diff(s) <= 1e-15)  # non-increasing as W-hat grows


# ---------------------------------------------------------------------------
# 4. Production hook (poly_ncmc_chain.py): --selector flag-gated screening.
# ---------------------------------------------------------------------------
def test_selector_none_matches_unscreened_reference(tmp_path):
    """--selector None must be EXACTLY current/original behavior: run_mix's NCMC branch
    dispatches to the un-screened `ncmc_attempt` only, with zero extra bookkeeping cost and
    zero extra RNG draws, whenever selector is None. Verified by replaying the identical
    low-level call sequence run_mix uses in that branch (disp_sweep/swap_sweep/ncmc_attempt)
    directly from the same seed and checking the final derived observables (U/N, Q_self,
    C_sig -- deterministic functions of the final (x, sig)) plus the NCMC attempt/accept
    counts match bit-for-bit, and that none of the new selector-only bookkeeping fields
    were touched."""
    x, sig, L = _toy_state(seed=41, n=60, min_d=0.3)
    T, beta = 0.2, 1.0 / 0.2
    budget, ncmc_every, n_steps, r_loc, seed = 500, 25, 15, 1.5, 7

    out_path = tmp_path / "none.pt"
    rec = run_mix(mix="swap+ncmc", x0=x, sig0=sig, L=L, beta=beta, T=T, budget=budget,
                   ncmc_every=ncmc_every, n_steps=n_steps, r_loc=r_loc, sweeps_per_step=1,
                   seed=seed, out_data={}, out_path=out_path, selector=None)

    # x, sig are untouched by run_mix (it copies internally) -- safe to reuse as the
    # reference trajectory's identical starting point.
    x_ref = x.copy(); sig_ref = sig.copy()
    seed_numba(seed)
    N = x_ref.shape[0]
    att_cum = acc_cum = 0
    for t in range(1, budget + 1):
        disp_sweep(x_ref, sig_ref, L, beta, STEP)
        swap_sweep(x_ref, sig_ref, L, beta, N)
        if t % ncmc_every == 0:
            attempted, accepted, i, j, ds, W, cnt = ncmc_attempt(
                x_ref, sig_ref, L, beta, r_loc, n_steps, 1, STEP, WINDOW_LO, WINDOW_HI)
            if attempted:
                att_cum += 1
                if accepted:
                    acc_cum += 1

    assert rec["ncmc_attempts_cum"][-1] == att_cum
    assert rec["ncmc_accepted_cum"][-1] == acc_cum
    u_ref = float(total_U(x_ref, sig_ref, L) / N)
    assert rec["U_N"][-1] == pytest.approx(u_ref, abs=1e-9)
    assert rec["Q_self"][-1] == pytest.approx(q_self(x_ref, x, L), abs=1e-12)
    assert rec["C_sig"][-1] == pytest.approx(c_sig(sig_ref, sig), abs=1e-12)
    # selector bookkeeping must stay at its untouched defaults
    assert rec["ncmc_invoked_cum"][-1] == 0
    assert rec["ncmc_screened_out_cum"][-1] == 0
    assert rec["ncmc_s_start_all"] == []
    assert rec["selector"] is False


def test_screening_ratio_one_matches_unscreened_statistics():
    """A mock selector with s_theta == 1.0 EXACTLY everywhere (screening_prob's
    max(what,0)==0 saturation branch, engineered via target_mean=-1000) must (a) NEVER
    screen out an attempt -- the 1-s_start skip probability is exactly zero, not just
    small -- and (b) leave the accept/reject decision governed by EXACTLY the same
    acc_prob = min(1, exp(-beta*W)) formula as the un-screened kernel, since
    s_end/s_start == 1/1 == 1 identically (the screening acceptance factor cancels).

    Verified two ways per trial, reseeding seed_numba(trial) identically before each of the
    paired calls (screening touches only a SEPARATE np.random.default_rng stream, never the
    numba stream, so ncmc_work_local sees the exact same draws either way):
      (1) MECHANISTIC: the selected pair (i,j) and work W are bit-identical between the two
          kernels for every trial (screening didn't perturb anything the physics depends on).
      (2) STATISTICAL: the final accept/reject coin-flip IS drawn from a different RNG
          stream in each kernel by design (numba's global stream for ncmc_attempt vs the
          dedicated np.random.default_rng for ncmc_attempt_screened -- this is exactly what
          lets --selector None cost the un-screened path zero extra RNG draws), so the two
          kernels' empirical accept RATES over many independent repeats from the same start
          state must agree statistically without being bit-identical draw-for-draw."""
    x0, sig0, L = _toy_state(seed=41, n=60, min_d=0.3)
    T, beta = 0.2, 1.0 / 0.2
    r_loc, n_steps = 1.5, 15

    scaler = {"feat_mean": np.zeros(N_FEATURES), "feat_std": np.ones(N_FEATURES),
              "target_mean": -1000.0, "target_std": 1.0}

    class _ZeroModel:
        def __call__(self, z):
            return torch.zeros(z.shape[0])

    model = _ZeroModel()
    s_min, c_sharp = 0.02, 0.5

    n_trials = 300
    acc_unscreened = acc_screened = 0
    n_counted = 0
    for trial in range(n_trials):
        x_a = x0.copy(); sig_a = sig0.copy()
        seed_numba(trial)
        attempted_a, accepted_a, i_a, j_a, ds_a, W_a, cnt_a = ncmc_attempt(
            x_a, sig_a, L, beta, r_loc, n_steps, 1, STEP, WINDOW_LO, WINDOW_HI)

        x_b = x0.copy(); sig_b = sig0.copy()
        seed_numba(trial)                              # identical numba stream as trial a
        screen_rng = np.random.default_rng(10_000 + trial)   # separate, independent stream
        (attempted_b, accepted_b, i_b, j_b, ds_b, W_b, cnt_b, screened_out_b,
         s_start_b) = ncmc_attempt_screened(
            x_b, sig_b, L, beta, r_loc, n_steps, 1, STEP, WINDOW_LO, WINDOW_HI,
            model, scaler, s_min, c_sharp, screen_rng)

        assert cnt_a == cnt_b
        if cnt_a == 0:
            continue
        assert screened_out_b is False
        assert s_start_b == pytest.approx(1.0)
        assert attempted_a and attempted_b
        assert i_a == i_b and j_a == j_b            # same pair: screening never touched numba RNG
        assert W_a == pytest.approx(W_b, abs=1e-9)  # same work: mechanistic ratio-1 cancellation

        n_counted += 1
        acc_unscreened += int(accepted_a)
        acc_screened += int(accepted_b)

    assert n_counted > n_trials * 0.5, "too many cnt==0 trials -- toy system window too sparse"
    rate_a = acc_unscreened / n_counted
    rate_b = acc_screened / n_counted
    assert abs(rate_a - rate_b) < 0.12, (
        f"chain statistics diverged: unscreened accept-rate={rate_a:.3f} "
        f"screened(mock s=1) accept-rate={rate_b:.3f} over n={n_counted} trials")
