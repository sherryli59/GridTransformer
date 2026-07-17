"""Semi-grand u-space substrate for the joint (sigma,x) continuous flow (poly Task J1).

Per docs/superpowers/specs/2026-07-17-joint-sigma-x-continuous-flow-amendment.md: each particle
carries a continuous diameter-code u in addition to its diameter sigma, related by
    F(sigma) = (a - sigma^-2)/(a-b),  a = SIG_MIN^-2, b = SIG_MAX^-2
    sigma(u) = (a - Phi(u)*(a-b))^-0.5,  Phi = standard normal CDF
so that u is std-normal distributed exactly when sigma ~ sigma^-3 on [SIG_MIN, SIG_MAX] -- the
canonical composition prior of poly/model.py's draw_sigmas. The semi-grand target is
    pi(x, u) ~ exp(-beta*U(x, sigma(u))) * prod_i phi(u_i)
so single-site u-Metropolis moves with a symmetric Gaussian proposal accept with
    min(1, exp(-beta*dU) * phi(u')/phi(u))     [phi = std normal pdf]
Energy kernels (row_e, total_U, ...) are untouched; sigma always enters through the sig array,
which every kernel here keeps byte-consistent with u (sig[i] is written/rolled back in lockstep
with every proposed/accepted/rejected u[i] move -- the arrays must never desync).

u_of_sigma (scipy.special.ndtri, Phi^-1) is SETUP-TIME ONLY -- used once to seed u from an
existing draw_sigmas() array. Hot loops (u_sweep, joint_block_relax) only ever call the numba
sigma_of_u (math.erf), never u_of_sigma.
"""
import sys
import math
from pathlib import Path

import numpy as np
from numba import njit
from scipy.special import ndtri

REPO = Path("/mnt/ssd/GridTransformer")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from liquid_coupling_flow.poly.model import row_e, seed_numba, SIG_MIN, SIG_MAX  # noqa: E402

A_CONST = SIG_MIN ** -2
B_CONST = SIG_MAX ** -2
PHI_CLIP = 1e-12
STEP_X_DEFAULT = 0.1      # matches reports/logs-2026-07-17/poly_block_data.py _block_relax
DELTA_U_DEFAULT = 0.5     # single-site u proposal std (u-space is O(1) std-normal scale)
RELAX_SW_DEFAULT = 400    # matches poly_block_data.RELAX_SW
NBINS_MU = 32              # poly Task P2a: mu(sigma) lookup resolution


def u_of_sigma(sig):
    """sigma -> u = Phi^-1(F(sigma)). Setup-time (scipy), NOT called from any numba hot loop."""
    sig = np.asarray(sig, dtype=np.float64)
    F = (A_CONST - sig ** -2) / (A_CONST - B_CONST)
    F = np.clip(F, PHI_CLIP, 1.0 - PHI_CLIP)
    return ndtri(F)


@njit(cache=True, fastmath=True)
def sigma_of_u(u):
    """u -> sigma = (a - Phi(u)*(a-b))^-0.5, Phi via math.erf (numba-safe). Clipped so
    u = +/-8 (or any float) always lands strictly inside (SIG_MIN, SIG_MAX), never NaN/out-of-range."""
    phi = 0.5 * (1.0 + math.erf(u * 0.7071067811865476))
    if phi < PHI_CLIP:
        phi = PHI_CLIP
    elif phi > 1.0 - PHI_CLIP:
        phi = 1.0 - PHI_CLIP
    return (A_CONST - phi * (A_CONST - B_CONST)) ** -0.5


def sigma_of_u_arr(u):
    """Non-jit convenience wrapper (loops the jitted scalar kernel) -- avoids a second, divergable
    formula implementation for numpy-array call sites (tests, pair-generation setup)."""
    u = np.atleast_1d(np.asarray(u, dtype=np.float64))
    out = np.empty_like(u)
    for i in range(u.shape[0]):
        out[i] = sigma_of_u(u[i])
    return out


@njit(cache=True, fastmath=True)
def u_sweep(x, sig, u, L, beta, delta, n_try):
    """Single-site semi-grand MH moves: propose u'_i = u_i + delta*randn(), sig'_i = sigma_of_u(u'_i),
    accept with min(1, exp(-beta*dU) * phi(u')/phi(u)) where dU is particle i's row energy change.
    x is untouched; sig is kept byte-consistent with u at every step (written speculatively before
    the energy evaluation, rolled back on reject)."""
    n = x.shape[0]
    acc = 0
    for _ in range(n_try):
        i = np.random.randint(n)
        u_old = u[i]
        sig_old = sig[i]
        u_new = u_old + delta * np.random.randn()
        sig_new = sigma_of_u(u_new)
        e0 = row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
        sig[i] = sig_new
        e1 = row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
        dU = e1 - e0
        log_a = -beta * dU + 0.5 * (u_old * u_old - u_new * u_new)   # log[phi(u')/phi(u)] term
        if log_a >= 0.0 or np.random.random() < np.exp(log_a):
            u[i] = u_new
            acc += 1
        else:
            sig[i] = sig_old
    return acc, n_try


# ---------------------------------------------------------------- (P2a) mu(sigma) calibration --
# poly Task P2a (docs/superpowers/specs/2026-07-17-joint-sigma-x-continuous-flow-amendment.md,
# PHASE 2 KICKOFF): mu(sigma) is a per-sigma chemical-potential-like bias added to the semi-grand
# u-move accept ratio to CANCEL the energetic tilt that otherwise deflates the composition away
# from the canonical prior P(sigma) ~ sigma^-3 (measured: mean sigma 0.998->0.82 at mu=0). mu is
# represented as a NBINS_MU-bin lookup over sigma in [SIG_MIN, SIG_MAX] with linear interpolation
# between BIN CENTERS (not edges) -- edges/centers/target-mass helpers below are plain numpy
# (setup/calibration-time only, never called from a numba hot loop); mu_of_sigma and u_sweep_mu
# are the numba hot-loop pieces.

def mu_bin_edges(nbins=NBINS_MU):
    """NBINS_MU+1 bin edges spanning [SIG_MIN, SIG_MAX] (uniform width in sigma)."""
    return np.linspace(SIG_MIN, SIG_MAX, nbins + 1)


def mu_bin_centers(nbins=NBINS_MU):
    """Bin-center sigma values -- the abscissas mu_of_sigma's njit lookup interpolates between."""
    edges = mu_bin_edges(nbins)
    return 0.5 * (edges[:-1] + edges[1:])


def sigma_cdf(sig):
    """F(sigma) = (a - sigma^-2)/(a-b): the CDF of the canonical prior P(sigma) ~ sigma^-3 on
    [SIG_MIN, SIG_MAX] (same F as u_of_sigma's, exposed standalone -- no Phi^-1 -- so calibration
    code can integrate P(sigma) ANALYTICALLY over a bin via F(hi) - F(lo), not by quadrature)."""
    sig = np.asarray(sig, dtype=np.float64)
    return (A_CONST - sig ** -2) / (A_CONST - B_CONST)


def mu_bin_target_mass(nbins=NBINS_MU):
    """Exact P(sigma)~sigma^-3 probability mass falling in each of the NBINS_MU bins, via the CDF
    (F(hi) - F(lo) per bin) -- the calibration target composition."""
    F = sigma_cdf(mu_bin_edges(nbins))
    return F[1:] - F[:-1]


@njit(cache=True, fastmath=True)
def mu_of_sigma(sig, mu):
    """Linear interpolation of the mu(sigma) lookup `mu` (length-NBINS_MU array of values AT the
    mu_bin_centers() abscissas) evaluated at an arbitrary sigma. sigma at or beyond the first/last
    bin CENTER is clamped to that end value (constant extrapolation, no linear blow-up outside the
    lookup's support)."""
    nb = mu.shape[0]
    width = (SIG_MAX - SIG_MIN) / nb
    pos = (sig - SIG_MIN) / width - 0.5   # fractional bin-center index
    if pos <= 0.0:
        return mu[0]
    if pos >= nb - 1:
        return mu[nb - 1]
    i0 = int(pos)
    frac = pos - i0
    return mu[i0] * (1.0 - frac) + mu[i0 + 1] * frac


@njit(cache=True, fastmath=True)
def u_sweep_mu(x, sig, u, L, beta, delta, n_try, mu):
    """Same single-site semi-grand MH move as u_sweep, but the accept log-ratio also gains
    beta*(mu(sig_new) - mu(sig_old)) -- the calibrated chemical-potential tilt that (once mu is
    correctly calibrated) cancels the energetic pull away from the canonical composition P(sigma).
    mu=zeros(NBINS_MU) reduces exactly to u_sweep (regression-tested). sig/u are kept byte-
    consistent exactly as u_sweep does (speculative write, rollback on reject)."""
    n = x.shape[0]
    acc = 0
    for _ in range(n_try):
        i = np.random.randint(n)
        u_old = u[i]
        sig_old = sig[i]
        u_new = u_old + delta * np.random.randn()
        sig_new = sigma_of_u(u_new)
        e0 = row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
        sig[i] = sig_new
        e1 = row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
        dU = e1 - e0
        dmu = mu_of_sigma(sig_new, mu) - mu_of_sigma(sig_old, mu)
        log_a = -beta * dU + 0.5 * (u_old * u_old - u_new * u_new) + beta * dmu
        if log_a >= 0.0 or np.random.random() < np.exp(log_a):
            u[i] = u_new
            acc += 1
        else:
            sig[i] = sig_old
    return acc, n_try


@njit(cache=True, fastmath=True)
def joint_block_relax(x, sig, u, L, beta, idx, n_sweeps, step, delta):
    """Block-local JOINT MC with the environment frozen: each sweep does idx.shape[0] displacement
    moves (mirrors poly_block_data._block_relax exactly) THEN idx.shape[0] single-site u-moves
    (same MH rule as u_sweep), both restricted to block particles (idx). Produces the short,
    unimodal, continuous (x,u) transport used as training pairs."""
    k = idx.shape[0]
    for _ in range(n_sweeps):
        for t in range(k):
            i = idx[np.random.randint(k)]
            xn0 = (x[i, 0] + step * np.random.randn()) % L
            xn1 = (x[i, 1] + step * np.random.randn()) % L
            xn2 = (x[i, 2] + step * np.random.randn()) % L
            e0 = row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
            e1 = row_e(x, sig, i, xn0, xn1, xn2, L)
            if np.random.random() < np.exp(-beta * (e1 - e0)):
                x[i, 0] = xn0; x[i, 1] = xn1; x[i, 2] = xn2
        for t in range(k):
            i = idx[np.random.randint(k)]
            u_old = u[i]
            sig_old = sig[i]
            u_new = u_old + delta * np.random.randn()
            sig_new = sigma_of_u(u_new)
            e0 = row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
            sig[i] = sig_new
            e1 = row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
            dU = e1 - e0
            log_a = -beta * dU + 0.5 * (u_old * u_old - u_new * u_new)
            if log_a >= 0.0 or np.random.random() < np.exp(log_a):
                u[i] = u_new
            else:
                sig[i] = sig_old


def make_joint_block_pair(x, sig, u, L, beta, k, seed, relax_sw=RELAX_SW_DEFAULT,
                           step=STEP_X_DEFAULT, delta=DELTA_U_DEFAULT):
    """Mirrors reports/logs-2026-07-17/poly_block_data.py::make_block_pair's block-selection and
    centering conventions EXACTLY (k-NN block of a random seed particle, min-image, centering
    about the OLD block centroid), but with NO permutation step -- joint_block_relax supplies the
    continuous (x,u) transport directly."""
    rng = np.random.default_rng(seed)
    seed_numba(seed)
    n = x.shape[0]
    s0 = rng.integers(n)
    d = x - x[s0]
    d -= L * np.round(d / L)
    idx = np.argsort((d ** 2).sum(1))[:k].astype(np.int64)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    xw, sw, uw = x.copy(), sig.copy(), u.copy()
    x_old = xw[idx].copy()
    sig_old = sw[idx].copy()
    u_old = uw[idx].copy()
    joint_block_relax(xw, sw, uw, L, beta, idx, relax_sw, step, delta)
    cen = x_old.mean(0)

    def cent(a):
        b = a - cen
        return b - L * np.round(b / L)

    return {
        "idx": idx,
        "env_x": cent(xw[~mask]), "env_sig": sw[~mask], "env_u": uw[~mask],
        "x_old": cent(x_old), "sig_old": sig_old, "u_old": u_old,
        "x_new": cent(xw[idx]), "sig_new": sw[idx].copy(), "u_new": uw[idx].copy(),
        "L": L, "beta": beta,
    }
