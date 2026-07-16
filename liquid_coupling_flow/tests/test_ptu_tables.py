import numpy as np
from liquid_coupling_flow.ptu.tables import build_tables_u, build_tables_lam
from liquid_coupling_flow.ka_energy import SIGMA, EPS

KA_SIG = np.array(SIGMA, dtype=np.float64)
KA_EPS = np.array(EPS, dtype=np.float64)


def test_u1_is_physical_ka():
    sig_mm, eps_mm, sig_mp, eps_mp = build_tables_u(1.0)
    assert np.allclose(sig_mm, KA_SIG) and np.allclose(eps_mm, KA_EPS)
    assert np.allclose(sig_mp, KA_SIG) and np.allclose(eps_mp, KA_EPS)


def test_u0_is_species_blind():
    sig_mm, eps_mm, _, _ = build_tables_u(0.0, sig_bar=0.95, eps_bar=1.0)
    # all entries identical -> species labels carry zero energy information
    assert np.allclose(sig_mm, 0.95) and np.allclose(eps_mm, 1.0)


def test_u0_mp_is_halfway():
    # pinned partner keeps physical character: interpolation strength (1+u)/2 -> 0.5 at u=0
    _, _, sig_mp, eps_mp = build_tables_u(0.0, sig_bar=0.95, eps_bar=1.0)
    assert np.allclose(sig_mp, 0.5 * KA_SIG + 0.5 * 0.95)
    assert np.allclose(eps_mp, 0.5 * KA_EPS + 0.5 * 1.0)


def test_lam_arm_matches_bcy_convention():
    lam = 0.9
    sig_mm, eps_mm, sig_mp, eps_mp = build_tables_lam(lam)
    assert np.allclose(sig_mm, lam * KA_SIG)
    assert np.allclose(sig_mp, 0.5 * (1 + lam) * KA_SIG)
    assert np.allclose(eps_mm, KA_EPS) and np.allclose(eps_mp, KA_EPS)
