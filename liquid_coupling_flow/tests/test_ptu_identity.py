import numpy as np
import pytest
from liquid_coupling_flow.ptu.tables import build_tables_u
from liquid_coupling_flow.ptu.kernels import mobile_U, seed_numba
from liquid_coupling_flow.ptu.identity import identity_sweep
from liquid_coupling_flow.tests.test_ptu_kernels import cavity  # reuse fixture


def test_identity_swap_free_when_species_blind(cavity):
    """With fully species-blind tables (mm==mp==blind), dU==0 for every label swap -> acc == att."""
    allx, alls, n, n_tot, *_ = cavity
    sig_mm, eps_mm, _, _ = build_tables_u(0.0)
    tabs = (sig_mm, eps_mm, sig_mm, eps_mm)
    seed_numba(0)
    a = alls.copy()
    acc, att = identity_sweep(allx, a, n, n_tot, 2.0, 200, *tabs)
    assert att > 0 and acc == att


def test_identity_swap_dead_at_physical(cavity):
    """At u=1 (physical KA) plain label swaps are known exact-dead (measured: ~0 acceptance)."""
    allx, alls, n, n_tot, *_ = cavity
    tabs = build_tables_u(1.0)
    seed_numba(0)
    a = alls.copy()
    acc, att = identity_sweep(allx, a, n, n_tot, 2.0, 500, *tabs)
    assert att > 0 and acc / att < 0.02


def test_identity_swap_preserves_counts(cavity):
    allx, alls, n, n_tot, *_ = cavity
    tabs = build_tables_u(0.5)
    seed_numba(1)
    a = alls.copy()
    nb_before = int((a[:n] == 1).sum())
    identity_sweep(allx, a, n, n_tot, 2.0, 300, *tabs)
    assert int((a[:n] == 1).sum()) == nb_before
    assert np.all(a[n:] == alls[n:])            # pinned labels untouched
