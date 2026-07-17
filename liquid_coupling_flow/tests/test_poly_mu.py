import sys

import numpy as np

sys.path.insert(0, "/mnt/ssd/GridTransformer/reports/logs-2026-07-17")
from poly_mu_calibrate import calibrate_mu  # noqa: E402

from liquid_coupling_flow.poly.model import SIG_MIN, SIG_MAX, draw_sigmas, seed_numba, disp_sweep, \
    swap_sweep  # noqa: E402
from liquid_coupling_flow.poly.semigrand import (u_of_sigma, u_sweep, u_sweep_mu, mu_of_sigma,
                                                  mu_bin_edges, mu_bin_centers,
                                                  mu_bin_target_mass, NBINS_MU)  # noqa: E402

EQ_SWEEPS = 1000  # see test_poly_blockdata._equilibrated_positions for why this is required


def _equilibrated_positions(rng, sig, n, L, beta, step=0.12, n_sweeps=EQ_SWEEPS):
    """Same pre-equilibration as test_poly_semigrand.py/test_poly_blockdata.py: raw uniform draws
    at this model's density/sigma range give catastrophic (~1e7/particle) r^-12 energies."""
    x = rng.random((n, 3)) * L
    for _ in range(n_sweeps):
        disp_sweep(x, sig, L, beta, step)
    return x


# ---------------------------------------------------------------- (1) mu_of_sigma interpolation --

def test_mu_of_sigma_matches_at_bin_centers():
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(NBINS_MU)
    centers = mu_bin_centers()
    for i in range(NBINS_MU):
        got = mu_of_sigma(centers[i], mu)
        assert abs(got - mu[i]) < 1e-10, (i, got, mu[i])


def test_mu_of_sigma_linear_at_midpoints():
    rng = np.random.default_rng(1)
    mu = rng.standard_normal(NBINS_MU)
    centers = mu_bin_centers()
    for i in range(NBINS_MU - 1):
        mid = 0.5 * (centers[i] + centers[i + 1])
        got = mu_of_sigma(mid, mu)
        expect = 0.5 * (mu[i] + mu[i + 1])
        assert abs(got - expect) < 1e-10, (i, got, expect)


def test_mu_of_sigma_clamps_outside_first_last_center():
    rng = np.random.default_rng(2)
    mu = rng.standard_normal(NBINS_MU)
    centers = mu_bin_centers()
    # sigma below the first bin center (including SIG_MIN exactly) clamps to mu[0]
    assert abs(mu_of_sigma(SIG_MIN, mu) - mu[0]) < 1e-10
    assert abs(mu_of_sigma(centers[0] - 0.01, mu) - mu[0]) < 1e-10
    # sigma above the last bin center (including SIG_MAX exactly) clamps to mu[-1]
    assert abs(mu_of_sigma(SIG_MAX, mu) - mu[-1]) < 1e-10
    assert abs(mu_of_sigma(centers[-1] + 0.01, mu) - mu[-1]) < 1e-10


# ---------------------------------------------------------------- (2) mu=0 regression -----------

def test_u_sweep_mu_zero_matches_u_sweep_std_normal_at_beta0():
    """mu=zeros(NBINS_MU) must reduce u_sweep_mu to exactly u_sweep's stationary distribution:
    at beta=0 the walk targets phi(u) alone (std normal), same moment gates as J1's
    test_semigrand_u_walk_matches_std_normal_at_beta0 (this is the crux sign gate: a mu-term sign
    error would settle onto a shifted stationary distribution even with mu identically zero, if
    e.g. the mu delta were computed with the wrong sign convention baked in)."""
    seed_numba(42)
    rng = np.random.default_rng(42)
    n = 512
    L = n ** (1.0 / 3.0)
    sig = draw_sigmas(n, seed=7)
    x = _equilibrated_positions(rng, sig, n, L, beta=1.0 / 0.3)
    u = np.zeros(n)
    mu_zero = np.zeros(NBINS_MU)

    n_try = n
    for _ in range(200):
        u_sweep_mu(x, sig, u, L, 0.0, 1.0, n_try, mu_zero)
    samples = []
    for sweep in range(500):
        u_sweep_mu(x, sig, u, L, 0.0, 1.0, n_try, mu_zero)
        if sweep % 5 == 0:
            samples.append(u.copy())
    samp = np.concatenate(samples)

    mean = samp.mean()
    var = samp.var()
    skew = np.mean((samp - mean) ** 3) / var ** 1.5
    assert abs(mean) < 0.05, mean
    assert abs(var - 1.0) < 0.05, var
    assert abs(skew) < 0.15, skew


def test_u_sweep_mu_zero_matches_u_sweep_bitwise_under_same_rng_draws():
    """Stronger regression than the statistical gate above: with mu=zeros, u_sweep_mu's accept
    ratio is byte-identical to u_sweep's (the added beta*(mu_new-mu_old) term is exactly 0.0), so
    under the SAME RNG stream the two kernels must produce identical trajectories."""
    n = 64
    L = n ** (1.0 / 3.0)
    beta = 1.0 / 0.2
    seed_numba(9)
    rng = np.random.default_rng(9)
    sig_base = draw_sigmas(n, seed=10)
    x_base = _equilibrated_positions(rng, sig_base, n, L, beta)

    x1, sig1 = x_base.copy(), sig_base.copy()
    u1 = u_of_sigma(sig1)
    seed_numba(123)
    for _ in range(20):
        u_sweep(x1, sig1, u1, L, beta, 0.4, n)

    x2, sig2 = x_base.copy(), sig_base.copy()
    u2 = u_of_sigma(sig2)
    mu_zero = np.zeros(NBINS_MU)
    seed_numba(123)
    for _ in range(20):
        u_sweep_mu(x2, sig2, u2, L, beta, 0.4, n, mu_zero)

    assert np.array_equal(u1, u2)
    assert np.array_equal(sig1, sig2)


# ---------------------------------------------------------------- (3) analytic 2-particle --------

def _quad_bin_probs(mu_arr, centers, beta, edges, n_grid=20000):
    """Independent reference (shares no code with mu_of_sigma's njit path beyond the linear-interp
    MATH, reimplemented via np.interp which clamps outside its domain identically to mu_of_sigma):
    numeric quadrature of p(sigma) ~ sigma^-3 * exp(beta*mu(sigma)) integrated over each bin."""
    grid = np.linspace(SIG_MIN, SIG_MAX, n_grid)
    dens = grid ** -3 * np.exp(beta * np.interp(grid, centers, mu_arr))
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    Z = cdf[-1]
    cdf /= Z
    cdf_edges = np.interp(edges, grid, cdf)
    return np.diff(cdf_edges)


def test_analytic_two_particle_semigrand_matches_quadrature():
    """Two particles far enough apart that row_e is IDENTICALLY 0 regardless of sigma (dU=0 at
    every proposed move -- verified via cutoff distance below), at beta=2 with a known nonzero
    mu(sigma)=c*sigma (c=0.5, discretized onto the NBINS_MU lookup): the stationary sigma-marginal
    of pure u_sweep_mu moves must match p(sigma) ~ P(sigma)*exp(beta*mu(sigma)) exactly (up to
    binning/sampling noise), independently verified by 1D quadrature (no shared code path with
    mu_of_sigma's njit interpolation beyond the linear-interp+clamp MATH)."""
    n = 2
    L = 6.0                                   # max cutoff = XC*SIG_MAX ~= 1.25*1.61 = 2.01 << L/2=3
    x = np.array([[0.0, 0.0, 0.0], [L / 2.0, 0.0, 0.0]])
    beta = 2.0
    c = 0.5
    centers = mu_bin_centers()
    mu_arr = c * centers

    seed_numba(77)
    rng = np.random.default_rng(77)
    sig = draw_sigmas(n, seed=88)
    u = u_of_sigma(sig)

    n_sweeps_burn = 2000
    n_sweeps_samp = 20000
    delta_u = 0.6
    for _ in range(n_sweeps_burn):
        u_sweep_mu(x, sig, u, L, beta, delta_u, n, mu_arr)
    sig_samples = []
    for _ in range(n_sweeps_samp):
        u_sweep_mu(x, sig, u, L, beta, delta_u, n, mu_arr)
        sig_samples.append(sig.copy())
    sig_samples = np.concatenate(sig_samples)

    # dU really is 0 for every possible sigma combo at this separation (sanity on the test setup)
    from liquid_coupling_flow.poly.model import row_e
    assert row_e(x, np.array([SIG_MAX, SIG_MAX]), 0, x[0, 0], x[0, 1], x[0, 2], L) == 0.0

    n_bins = 20
    edges = np.linspace(SIG_MIN, SIG_MAX, n_bins + 1)
    meas, _ = np.histogram(sig_samples, bins=edges)
    meas = meas / meas.sum()
    target = _quad_bin_probs(mu_arr, centers, beta, edges)

    sup_err = np.max(np.abs(meas - target))
    assert sup_err < 0.03, (sup_err, meas, target)
    # sign check (code-independent of mu_of_sigma/u_sweep_mu): a positive c*sigma tilt must pull
    # the sampled mean sigma ABOVE the untilted prior's quadrature mean (P(sigma)~sigma^-3 alone)
    grid = np.linspace(SIG_MIN, SIG_MAX, 20000)
    prior_dens = grid ** -3
    prior_mean = np.trapz(grid * prior_dens, grid) / np.trapz(prior_dens, grid)
    assert sig_samples.mean() > prior_mean + 0.02, (sig_samples.mean(), prior_mean)


# ---------------------------------------------------------------- (4) calibration smoke ----------

def test_calibration_reduces_composition_error_3x_vs_mu_zero():
    """On a small pre-equilibrated warm system (N=64, T=0.3), 15 calibration iterations must cut
    the sup-norm composition error (measured vs P(sigma), same target as poly_mu_calibrate.py's
    own gate) to <= 1/3 of the mu=0 steady-state error measured over the SAME total sweep budget."""
    n = 64
    L = n ** (1.0 / 3.0)
    T = 0.3
    beta = 1.0 / T
    iters = 15
    sweeps_per_iter = 300

    seed_numba(21)
    rng = np.random.default_rng(21)
    sig0 = draw_sigmas(n, seed=22)
    x0 = _equilibrated_positions(rng, sig0, n, L, beta)
    u0 = u_of_sigma(sig0)

    edges = mu_bin_edges()
    target = mu_bin_target_mass()

    # --- mu=0 steady state, same total sweep budget as the calibration run below ---
    x_b, sig_b, u_b = x0.copy(), sig0.copy(), u0.copy()
    mu_zero = np.zeros(NBINS_MU)
    total_sw = iters * sweeps_per_iter
    half = total_sw // 2
    sig_samples = []
    for sw in range(total_sw):
        disp_sweep(x_b, sig_b, L, beta, 0.12)
        swap_sweep(x_b, sig_b, L, beta, n)
        u_sweep_mu(x_b, sig_b, u_b, L, beta, 0.15, n, mu_zero)
        if sw >= half:
            sig_samples.append(sig_b.copy())
    meas0, _ = np.histogram(np.concatenate(sig_samples), bins=edges)
    meas0 = meas0 / meas0.sum()
    baseline_sup = np.max(np.abs(meas0 - target))
    assert baseline_sup > 0.01, "mu=0 baseline should show a real deflation error (sanity)"

    # --- calibration run from mu=0, identical total sweep budget ---
    x_c, sig_c, u_c = x0.copy(), sig0.copy(), u0.copy()
    result = calibrate_mu(x_c, sig_c, u_c, L, beta, T, iters, sweeps_per_iter, eta=0.5,
                           verbose=False)
    final_sup = result["history"][-1]["sup_abs"]

    assert final_sup <= baseline_sup / 3.0, (baseline_sup, final_sup)
