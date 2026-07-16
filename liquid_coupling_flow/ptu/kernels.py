"""numba energy/MC kernels with per-rung interaction tables. Partner class decides the table:
j < n -> mobile-mobile (mm), j >= n -> mobile-pinned (mp). Shifted LJ, rc = 2.5*sigma_pair.
All float64. Displacement move = BCY convention: dx = l*nhat, l ~ U[0, step], hard wall |x|<R."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def row_e(allx, alls, i, x0, x1, x2, n, n_tot, sig_mm, eps_mm, sig_mp, eps_mp):
    e = 0.0
    si = alls[i]
    for j in range(n_tot):
        if j == i:
            continue
        dx = allx[j, 0] - x0; dy = allx[j, 1] - x1; dz = allx[j, 2] - x2
        r2 = dx * dx + dy * dy + dz * dz
        if j < n:
            s = sig_mm[si, alls[j]]; ep = eps_mm[si, alls[j]]
        else:
            s = sig_mp[si, alls[j]]; ep = eps_mp[si, alls[j]]
        rc = 2.5 * s
        if r2 >= rc * rc:
            continue
        sr6 = (s * s / r2) ** 3
        sc6 = (1.0 / 2.5) ** 6
        e += 4.0 * ep * ((sr6 * sr6 - sr6) - (sc6 * sc6 - sc6))
    return e


@njit(cache=True, fastmath=True)
def mobile_U(allx, alls, n, n_tot, sig_mm, eps_mm, sig_mp, eps_mp):
    u = 0.0
    for i in range(n):
        si = alls[i]
        for j in range(i + 1, n):                       # mm once
            dx = allx[j, 0] - allx[i, 0]; dy = allx[j, 1] - allx[i, 1]; dz = allx[j, 2] - allx[i, 2]
            r2 = dx * dx + dy * dy + dz * dz
            s = sig_mm[si, alls[j]]; ep = eps_mm[si, alls[j]]
            rc = 2.5 * s
            if r2 < rc * rc:
                sr6 = (s * s / r2) ** 3
                sc6 = (1.0 / 2.5) ** 6
                u += 4.0 * ep * ((sr6 * sr6 - sr6) - (sc6 * sc6 - sc6))
        for j in range(n, n_tot):                       # mp fully
            dx = allx[j, 0] - allx[i, 0]; dy = allx[j, 1] - allx[i, 1]; dz = allx[j, 2] - allx[i, 2]
            r2 = dx * dx + dy * dy + dz * dz
            s = sig_mp[si, alls[j]]; ep = eps_mp[si, alls[j]]
            rc = 2.5 * s
            if r2 < rc * rc:
                sr6 = (s * s / r2) ** 3
                sc6 = (1.0 / 2.5) ** 6
                u += 4.0 * ep * ((sr6 * sr6 - sr6) - (sc6 * sc6 - sc6))
    return u


@njit(cache=True, fastmath=True)
def disp_sweep(allx, alls, n, n_tot, beta, R, step, sig_mm, eps_mm, sig_mp, eps_mp):
    acc = 0
    for i in range(n):
        l = step * np.random.random()
        v0 = np.random.randn(); v1 = np.random.randn(); v2 = np.random.randn()
        vn = (v0 * v0 + v1 * v1 + v2 * v2) ** 0.5
        xn0 = allx[i, 0] + l * v0 / vn
        xn1 = allx[i, 1] + l * v1 / vn
        xn2 = allx[i, 2] + l * v2 / vn
        if xn0 * xn0 + xn1 * xn1 + xn2 * xn2 >= R * R:
            continue
        e0 = row_e(allx, alls, i, allx[i, 0], allx[i, 1], allx[i, 2], n, n_tot,
                   sig_mm, eps_mm, sig_mp, eps_mp)
        e1 = row_e(allx, alls, i, xn0, xn1, xn2, n, n_tot,
                   sig_mm, eps_mm, sig_mp, eps_mp)
        if np.random.random() < np.exp(-beta * (e1 - e0)):
            allx[i, 0] = xn0; allx[i, 1] = xn1; allx[i, 2] = xn2
            acc += 1
    return acc
