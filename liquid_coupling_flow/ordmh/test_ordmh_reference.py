"""Task 1 tests --- the independent ground-truth reference.

Two kinds of check:
  * exact unit tests of the machinery (grid reduction; MC samples uniform when
    the energy is switched off) --- these are true RED->GREEN units;
  * statistical cross-validation (Gate R): the displacement-MC mean energy must
    agree with grid quadrature, which shares NO code with it.  This is run with
    heavier settings in ordmh_reference.py's __main__; here it runs at modest
    settings so the suite stays fast.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from ordmh_energy import wca_energy
from ordmh_reference import (
    weighted_boltzmann_mean,
    grid_quadrature,
    displacement_mc,
    observables_from_configs,
)


# -- exact unit test: the Boltzmann-weighted reduction, incl. inf handling ----
def test_weighted_reduction_exact():
    U = np.array([0.0, 1.0, 2.0, np.inf])
    beta = 1.0
    w = np.exp(-beta * U)  # last is 0
    expect = (0 * w[0] + 1 * w[1] + 2 * w[2]) / (w[0] + w[1] + w[2])
    got = weighted_boltzmann_mean(U, beta)
    assert np.isclose(got, expect, atol=1e-12)
    # inf must contribute 0 (not nan): dropping it explicitly gives same answer
    got_finite = weighted_boltzmann_mean(U[:3], beta)
    assert np.isclose(got, got_finite, atol=1e-12)


# -- MC machinery: with the energy switched off, stationary dist is uniform ---
def test_mc_uniform_when_energy_off():
    L = 2.4494897
    zero_sp = lambda pos, k, L: np.zeros(pos.shape[0])  # noqa: E731
    out = displacement_mc(N=4, L=L, beta=1.0, n_chains=256, n_equil=200,
                          n_collect=200, thin=2, step=0.3, seed=1,
                          sp_energy_fn=zero_sp)
    assert out["acc_rate"] > 0.999  # every move accepted when dU==0
    coords = out["configs"].reshape(-1)
    # uniform on [0,L): mean L/2, and both halves equally populated
    assert abs(coords.mean() - L / 2) < 0.02 * L
    frac_low = np.mean(coords < L / 2)
    assert abs(frac_low - 0.5) < 0.02


# -- N=2 fast oracle: MC <U> agrees with a high-res 2D quadrature ------------
def test_gate_R_N2_fast():
    L, beta = 2.0, 1.0
    grid = grid_quadrature(N=2, L=L, beta=beta, n_grid=400)
    mc = displacement_mc(N=2, L=L, beta=beta, n_chains=256, n_equil=400,
                         n_collect=400, thin=2, step=0.25, seed=2)
    obs = observables_from_configs(mc["configs"], L=L, beta=beta)
    diff = abs(obs["mean_U"] - grid["mean_U"])
    tol = 5 * obs["mean_U_err"] + grid["mean_U_err"]
    assert diff < tol, (obs["mean_U"], grid["mean_U"], diff, tol)


# -- N=3 headline Gate R (modest settings here; heavier in __main__) ----------
def test_gate_R_N3():
    L, beta = 2.4494897427831781, 1.0
    grid = grid_quadrature(N=3, L=L, beta=beta, n_grid=48)
    mc = displacement_mc(N=3, L=L, beta=beta, n_chains=384, n_equil=500,
                         n_collect=400, thin=3, step=0.22, seed=3)
    obs = observables_from_configs(mc["configs"], L=L, beta=beta)
    diff = abs(obs["mean_U"] - grid["mean_U"])
    tol = 5 * obs["mean_U_err"] + grid["mean_U_err"]
    assert diff < tol, (obs["mean_U"], grid["mean_U"], diff, tol)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{'TASK 1 TESTS PASS' if failed == 0 else f'TASK 1 TESTS FAIL ({failed})'}")
    return failed


if __name__ == "__main__":
    sys.exit(_run_all())
