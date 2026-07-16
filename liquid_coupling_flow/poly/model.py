"""NBC-2017 continuously polydisperse soft spheres (numba, float64, canonical-in-sigma).
v(r) = (sij/r)^12 + C0 + C2 (r/sij)^2 + C4 (r/sij)^4 for r < XC*sij (C2-smooth at cutoff);
nonadditive sij = 0.5(si+sj)(1 - 0.2|si-sj|); P(sigma) ~ sigma^-3 on [0.725, 1.61] (<sigma>=1)."""
import numpy as np
from numba import njit

XC = 1.25
C0 = -28.0 / XC ** 12
C2 = 48.0 / XC ** 14
C4 = -21.0 / XC ** 16
EPS_NA = 0.2
SIG_MIN, SIG_MAX = 0.725, 1.61


def draw_sigmas(n, seed):
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    a, b = SIG_MIN ** -2, SIG_MAX ** -2                 # inverse CDF of A s^-3
    return 1.0 / np.sqrt(a - u * (a - b))


@njit(cache=True, fastmath=True)
def seed_numba(seed):
    np.random.seed(seed)


@njit(cache=True, fastmath=True)
def sigma_ij(si, sj):
    return 0.5 * (si + sj) * (1.0 - EPS_NA * abs(si - sj))


@njit(cache=True, fastmath=True)
def pair_v(r2, sij):
    rc = XC * sij
    if r2 >= rc * rc:
        return 0.0
    inv = sij * sij / r2
    x2 = r2 / (sij * sij)
    return inv ** 6 + C0 + C2 * x2 + C4 * x2 * x2


@njit(cache=True, fastmath=True)
def row_e(x, sig, i, xi0, xi1, xi2, L):
    e = 0.0
    for j in range(x.shape[0]):
        if j == i:
            continue
        dx = x[j, 0] - xi0; dy = x[j, 1] - xi1; dz = x[j, 2] - xi2
        dx -= L * np.round(dx / L); dy -= L * np.round(dy / L); dz -= L * np.round(dz / L)
        e += pair_v(dx * dx + dy * dy + dz * dz, sigma_ij(sig[i], sig[j]))
    return e


@njit(cache=True, fastmath=True)
def total_U(x, sig, L):
    u = 0.0
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            dx = x[j, 0] - x[i, 0]; dy = x[j, 1] - x[i, 1]; dz = x[j, 2] - x[i, 2]
            dx -= L * np.round(dx / L); dy -= L * np.round(dy / L); dz -= L * np.round(dz / L)
            u += pair_v(dx * dx + dy * dy + dz * dz, sigma_ij(sig[i], sig[j]))
    return u


@njit(cache=True, fastmath=True)
def disp_sweep(x, sig, L, beta, step):
    n = x.shape[0]
    acc = 0
    for _ in range(n):
        i = np.random.randint(n)
        xn0 = (x[i, 0] + step * np.random.randn()) % L
        xn1 = (x[i, 1] + step * np.random.randn()) % L
        xn2 = (x[i, 2] + step * np.random.randn()) % L
        e0 = row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
        e1 = row_e(x, sig, i, xn0, xn1, xn2, L)
        if np.random.random() < np.exp(-beta * (e1 - e0)):
            x[i, 0] = xn0; x[i, 1] = xn1; x[i, 2] = xn2
            acc += 1
    return acc


@njit(cache=True, fastmath=True)
def swap_sweep(x, sig, L, beta, n_try):
    n = x.shape[0]
    acc = 0
    for _ in range(n_try):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            j = (j + 1) % n
        e0 = (row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
              + row_e(x, sig, j, x[j, 0], x[j, 1], x[j, 2], L))
        si = sig[i]
        sig[i] = sig[j]; sig[j] = si
        e1 = (row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
              + row_e(x, sig, j, x[j, 0], x[j, 1], x[j, 2], L))
        if np.random.random() < np.exp(-beta * (e1 - e0)):
            acc += 1
        else:
            sj = sig[i]
            sig[i] = sig[j]; sig[j] = sj
    return acc, n_try
