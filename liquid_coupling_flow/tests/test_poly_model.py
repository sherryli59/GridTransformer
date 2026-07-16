import numpy as np
import pytest
from liquid_coupling_flow.poly.model import (XC, C0, C2, C4, draw_sigmas, pair_v, sigma_ij,
                                             row_e, total_U, disp_sweep, swap_sweep, seed_numba)
from liquid_coupling_flow.poly.torch_ref import total_U_torch


def test_cutoff_smoothness_analytic():
    """c0,c2,c4 must make v, v', v'' vanish at r = XC*sij (the NBC C2-smooth cutoff)."""
    sij = 1.0
    rc = XC * sij
    h = 1e-6
    def v(r):
        return pair_v(r * r, sij)
    assert abs(v(rc - 1e-9)) < 1e-9
    d1 = (v(rc - h) - v(rc - 2 * h)) / h
    d2 = (v(rc - h) - 2 * v(rc - 2 * h) + v(rc - 3 * h)) / h ** 2
    assert abs(d1) < 1e-4 and abs(d2) < 1e-1


def test_sigma_distribution_mean_one():
    s = draw_sigmas(200_000, seed=1)
    assert s.min() >= 0.725 - 1e-12 and s.max() <= 1.61 + 1e-12
    assert abs(s.mean() - 1.0) < 2e-3


def test_nonadditivity():
    assert abs(sigma_ij(1.0, 1.0) - 1.0) < 1e-12
    si, sj = 0.8, 1.4
    assert abs(sigma_ij(si, sj) - 0.5 * (si + sj) * (1 - 0.2 * abs(si - sj))) < 1e-12


def test_total_U_matches_torch_reference():
    seed_numba(0)
    rng = np.random.default_rng(134)
    n = 128
    L = n ** (1.0 / 3.0) / 1.0                       # rho = 1
    x = rng.random((n, 3)) * L
    sig = draw_sigmas(n, seed=2)
    u_nb = total_U(x, sig, L)
    u_th = total_U_torch(x, sig, L)
    assert abs(u_nb - u_th) / n < 1e-8


def test_row_e_consistent_with_total_U():
    seed_numba(0)
    rng = np.random.default_rng(216)
    n = 64
    L = n ** (1.0 / 3.0)
    x = rng.random((n, 3)) * L
    sig = draw_sigmas(n, seed=4)
    i = 7
    u0 = total_U(x, sig, L)
    xi_new = (x[i] + 0.1) % L
    du_row = (row_e(x, sig, i, xi_new[0], xi_new[1], xi_new[2], L)
              - row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L))
    x2 = x.copy(); x2[i] = xi_new
    du_full = total_U(x2, sig, L) - u0
    assert abs(du_row - du_full) < 1e-9


def test_sweeps_run_and_preserve_multiset():
    seed_numba(0)
    rng = np.random.default_rng(5)
    n = 64
    L = n ** (1.0 / 3.0)
    x = rng.random((n, 3)) * L
    sig = draw_sigmas(n, seed=6)
    ms = np.sort(sig.copy())
    acc = disp_sweep(x, sig, L, 1.0 / 0.3, 0.15)
    assert 0 < acc < n
    a, t = swap_sweep(x, sig, L, 1.0 / 0.3, 100)
    assert t == 100 and 0 <= a <= t
    assert np.allclose(np.sort(sig), ms)             # pair swaps preserve the diameter multiset
    assert np.all((x >= 0) & (x < L))
