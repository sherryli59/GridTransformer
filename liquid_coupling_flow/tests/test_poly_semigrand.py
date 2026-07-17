import numpy as np
from liquid_coupling_flow.poly.model import (SIG_MIN, SIG_MAX, draw_sigmas, seed_numba, total_U,
                                              disp_sweep)
from liquid_coupling_flow.poly.semigrand import (u_of_sigma, sigma_of_u, sigma_of_u_arr, u_sweep,
                                                  joint_block_relax, make_joint_block_pair)

EQ_SWEEPS = 1000  # see test_poly_blockdata._equilibrated_positions for why this is required


def _equilibrated_positions(rng, sig, n, L, beta, step=0.12, n_sweeps=EQ_SWEEPS):
    """Same pre-equilibration as test_poly_blockdata.py: raw uniform draws at this model's
    density/sigma range give catastrophic (~1e7/particle) r^-12 energies -- never test on them."""
    x = rng.random((n, 3)) * L
    for _ in range(n_sweeps):
        disp_sweep(x, sig, L, beta, step)
    return x


# ---------------------------------------------------------------- (a) u<->sigma roundtrip -----

def test_u_sigma_roundtrip_and_monotone():
    sig_grid = np.linspace(SIG_MIN + 1e-6, SIG_MAX - 1e-6, 5001)
    u = u_of_sigma(sig_grid)
    sig_back = sigma_of_u_arr(u)
    assert np.max(np.abs(sig_back - sig_grid)) < 1e-10
    assert np.all(np.diff(u) > 0)                       # F (hence u_of_sigma) is monotone in sigma


def test_u_extremes_finite_and_in_bounds():
    for uu in (-8.0, 8.0, 0.0):
        s = sigma_of_u(uu)
        assert np.isfinite(s)
        assert SIG_MIN < s < SIG_MAX


# ---------------------------------------------------------------- (b) semi-grand exactness -----

def test_semigrand_u_walk_matches_std_normal_at_beta0():
    """At beta=0, exp(-beta*dU)=1 regardless of dU (as long as it's finite -- hence the
    pre-equilibrated, non-overlapping positions), so u_sweep reduces to a symmetric-proposal
    Metropolis walk targeting phi(u) alone. This is the crux gate on the accept-ratio SIGN: a
    sign error here settles onto the WRONG stationary distribution instead of std normal."""
    seed_numba(42)
    rng = np.random.default_rng(42)
    n = 512
    L = n ** (1.0 / 3.0)
    sig = draw_sigmas(n, seed=7)
    x = _equilibrated_positions(rng, sig, n, L, beta=1.0 / 0.3)
    u = np.zeros(n)

    n_try = n
    for _ in range(200):                                  # burn-in
        u_sweep(x, sig, u, L, 0.0, 1.0, n_try)
    samples = []
    for sweep in range(500):
        u_sweep(x, sig, u, L, 0.0, 1.0, n_try)
        if sweep % 5 == 0:
            samples.append(u.copy())
    samp = np.concatenate(samples)

    mean = samp.mean()
    var = samp.var()
    skew = np.mean((samp - mean) ** 3) / var ** 1.5
    assert abs(mean) < 0.05, mean
    assert abs(var - 1.0) < 0.05, var
    assert abs(skew) < 0.15, skew


# ---------------------------------------------------------------- (c) energy consistency -------

def test_sig_stays_consistent_with_u_after_moves():
    seed_numba(1)
    rng = np.random.default_rng(1)
    n = 200
    L = n ** (1.0 / 3.0)
    sig = draw_sigmas(n, seed=2)
    x = _equilibrated_positions(rng, sig, n, L, beta=1.0 / 0.15)
    u = u_of_sigma(sig)
    beta = 1.0 / 0.15

    for _ in range(50):
        u_sweep(x, sig, u, L, beta, 0.5, n)

    u_reconstructed_sig = sigma_of_u_arr(u)
    assert np.max(np.abs(u_reconstructed_sig - sig)) < 1e-10
    e_maintained = total_U(x, sig, L)
    e_recomputed = total_U(x, u_reconstructed_sig, L)
    assert abs(e_maintained - e_recomputed) < 1e-10


# ---------------------------------------------------------------- (d) joint_block_relax --------

def test_joint_block_relax_env_frozen_and_moves_accept():
    seed_numba(3)
    rng = np.random.default_rng(3)
    n = 128
    L = n ** (1.0 / 3.0)
    beta = 1.0 / 0.2
    sig = draw_sigmas(n, seed=4)
    x = _equilibrated_positions(rng, sig, n, L, beta=1.0 / 0.12)
    assert total_U(x, sig, L) / n < 5.0                   # sane liquid state before the block test

    u = u_of_sigma(sig)
    k = 8
    d = x - x[10]
    d -= L * np.round(d / L)
    idx = np.argsort((d ** 2).sum(1))[:k].astype(np.int64)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True

    x_before = x.copy()
    sig_before = sig.copy()
    u_before = u.copy()

    joint_block_relax(x, sig, u, L, beta, idx, 300, 0.1, 0.5)

    # env rows byte-identical
    assert np.array_equal(x[~mask], x_before[~mask])
    assert np.array_equal(sig[~mask], sig_before[~mask])
    assert np.array_equal(u[~mask], u_before[~mask])
    # block sig stays in bounds
    assert np.all(sig[idx] > SIG_MIN) and np.all(sig[idx] < SIG_MAX)
    # sig/u never desynced by the block moves either
    assert np.max(np.abs(sigma_of_u_arr(u[idx]) - sig[idx])) < 1e-9
    # both channels actually moved (acceptance > 0)
    assert np.abs(x[idx] - x_before[idx]).max() > 0
    assert np.abs(sig[idx] - sig_before[idx]).max() > 0
    assert not np.array_equal(u[idx], u_before[idx])


# ---------------------------------------------------------------- (e) make_joint_block_pair ----

def test_make_joint_block_pair_shapes_and_transport():
    seed_numba(5)
    rng = np.random.default_rng(5)
    n = 128
    L = n ** (1.0 / 3.0)
    beta = 1.0 / 0.085
    sig = draw_sigmas(n, seed=6)
    x = _equilibrated_positions(rng, sig, n, L, beta=1.0 / 0.12)
    u = u_of_sigma(sig)

    k = 8
    p = make_joint_block_pair(x, sig, u, L, beta, k=k, seed=11)

    for key in ("idx", "env_x", "env_sig", "env_u", "x_old", "sig_old", "u_old",
                "x_new", "sig_new", "u_new", "L", "beta"):
        assert key in p
    assert p["idx"].shape == (k,)
    assert p["x_old"].shape == (k, 3) and p["x_new"].shape == (k, 3)
    assert p["sig_old"].shape == (k,) and p["sig_new"].shape == (k,)
    assert p["u_old"].shape == (k,) and p["u_new"].shape == (k,)
    assert p["env_x"].shape == (n - k, 3)
    assert p["env_sig"].shape == (n - k,) and p["env_u"].shape == (n - k,)

    # centering preserves internal block geometry: pairwise min-image distances among the centered
    # x_old rows must equal those of the raw (uncentered) block. (A literal "centroid at origin"
    # check is NOT always true post-cent(): a compact k-NN block whose RAW coordinates straddle the
    # periodic boundary gets individual rows folded by cent()'s minimum-image wrap, shifting the
    # mean away from 0 -- verified this is not a bug introduced here: the mirrored original
    # reports/logs-2026-07-17/poly_block_data.py::make_block_pair has the identical artifact on
    # this exact fixture (measured centroid [0.63, 0, 1.26] at L=5.04).)
    def _pairwise_mind(a):
        d = a[:, None, :] - a[None, :, :]
        d -= L * np.round(d / L)
        return np.sqrt((d ** 2).sum(-1))
    raw_block = x[p["idx"]]
    assert np.allclose(_pairwise_mind(p["x_old"]), _pairwise_mind(raw_block), atol=1e-9)

    # sig/u consistency carried through the pair
    assert np.max(np.abs(sigma_of_u_arr(p["u_old"]) - p["sig_old"])) < 1e-9
    assert np.max(np.abs(sigma_of_u_arr(p["u_new"]) - p["sig_new"])) < 1e-9

    # transport sanity: short but nonzero on both channels
    rms_dx = np.sqrt(np.mean(np.sum((p["x_new"] - p["x_old"]) ** 2, axis=1)))
    mean_dsig = np.mean(np.abs(p["sig_new"] - p["sig_old"]))
    assert 0 < rms_dx < 0.8
    assert mean_dsig > 0
