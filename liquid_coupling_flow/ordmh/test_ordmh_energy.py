"""Task 0 --- Energy verification (the FIRST gate).

A wrong energy silently corrupts every downstream check, so we verify the
pure-numpy WCA energy under test (``ordmh_energy.wca_energy``) against THREE
independent oracles before anything else uses it:

  1. Analytic point values of the WCA pair potential (hand-computed).
  2. An independent numpy re-implementation that sums over *explicit* periodic
     images (a double loop, no round()-based minimum-image), machine-precision
     agreement on random configs.  This catches a wrong min-image convention.
  3. The repo's existing torch ``lj_energy`` (different library, different
     author) evaluated in WCA mode --- a cross-library cross-check.

Independence note: oracle #2 lives entirely in THIS file and shares no code
with ordmh_energy; oracle #3 is torch, not numpy.  Only if all three agree do
we trust the energy.

Runnable as ``python test_ordmh_energy.py`` (prints PASS/FAIL) and under pytest.
"""
import os
import sys
import itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from ordmh_energy import wca_pair_energy, wca_energy, R_C  # module under test


# ---------------------------------------------------------------------------
# Oracle #2: independent explicit-periodic-image energy (no minimum-image round)
# ---------------------------------------------------------------------------
def _wca_energy_allimages(pos, L, eps=1.0, sigma=1.0, n_img=2):
    """Total WCA energy summing over explicit periodic images.

    Deliberately written differently from the module under test: an explicit
    double loop over i<j and over image shifts in {-n_img..n_img}^2, NO
    round()-based minimum image.  When L/2 >= r_c only one image of each pair
    is ever within the cutoff, so this must equal the minimum-image energy.
    """
    pos = np.asarray(pos, float)
    N = pos.shape[0]
    rc = 2.0 ** (1.0 / 6.0) * sigma
    shifts = [np.array([kx, ky]) * L
              for kx in range(-n_img, n_img + 1)
              for ky in range(-n_img, n_img + 1)]
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d0 = pos[i] - pos[j]
            for s in shifts:
                d = d0 + s
                r = np.hypot(d[0], d[1])
                if r < rc:
                    sr6 = (sigma / r) ** 6
                    total += 4.0 * eps * (sr6 * sr6 - sr6) + eps
    return total


def _rng():
    return np.random.default_rng(12345)


# ---------------------------------------------------------------------------
# 1. Analytic pair-potential values (hand-computed, eps=sigma=1)
# ---------------------------------------------------------------------------
def test_pair_potential_analytic_values():
    # u(sigma) : LJ term is 4*(1-1)=0, WCA adds +eps -> 1
    assert np.isclose(wca_pair_energy(1.0), 1.0, atol=1e-12)
    # u(2^(1/6) sigma) : LJ minimum -eps, WCA shift +eps -> exactly 0 (continuity)
    assert np.isclose(wca_pair_energy(R_C), 0.0, atol=1e-9)
    # u(0.9): hand value 4*((1/0.9)^12 - (1/0.9)^6) + 1
    sr6 = (1.0 / 0.9) ** 6
    expect = 4.0 * (sr6 * sr6 - sr6) + 1.0
    assert np.isclose(wca_pair_energy(0.9), expect, atol=1e-10)
    # purely repulsive: zero at and beyond the cutoff
    assert wca_pair_energy(R_C * 1.0000001) == 0.0
    assert wca_pair_energy(1.5) == 0.0
    # monotone decreasing on (0, r_c)
    rs = np.linspace(0.85, R_C - 1e-3, 50)
    u = wca_pair_energy(rs)
    assert np.all(np.diff(u) < 0)


def test_two_particle_energy_matches_pair():
    # Two particles in a box: total energy == single pair potential at their
    # minimum-image separation.
    L = 3.0
    for r in [0.9, 1.0, 1.05, 1.2]:
        pos = np.array([[0.0, 0.0], [r, 0.0]])
        assert np.isclose(wca_energy(pos, L), wca_pair_energy(min(r, L - r)),
                          atol=1e-12)


# ---------------------------------------------------------------------------
# 2. numpy min-image vs numpy explicit-images, machine precision, random configs
# ---------------------------------------------------------------------------
def test_minimage_vs_allimages_random():
    rng = _rng()
    L = 2.449489742783178  # sqrt(6): the N=3, rho*=0.5 box; L/2 > r_c
    for _ in range(5):
        N = rng.integers(3, 7)
        pos = rng.uniform(0.0, L, size=(N, 2))
        u_mi = wca_energy(pos, L)
        u_ai = _wca_energy_allimages(pos, L)
        assert np.isclose(u_mi, u_ai, rtol=0, atol=1e-11), (u_mi, u_ai)


def test_batched_matches_per_config():
    rng = _rng()
    L = 3.1
    pos = rng.uniform(0.0, L, size=(17, 5, 2))
    U_batch = wca_energy(pos, L)
    assert U_batch.shape == (17,)
    for b in range(17):
        assert np.isclose(U_batch[b], wca_energy(pos[b], L), atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Cross-library oracle: repo torch lj_energy in WCA mode
# ---------------------------------------------------------------------------
def test_vs_repo_torch_lj_energy():
    import torch
    from liquid_coupling_flow.energy import lj_energy
    rng = _rng()
    L = 2.449489742783178
    rc = R_C
    for _ in range(5):
        N = int(rng.integers(3, 7))
        pos = rng.uniform(0.0, L, size=(N, 2))
        u_np = wca_energy(pos, L)
        x = torch.tensor(pos[None], dtype=torch.float64)
        u_torch = lj_energy(x, L, sigma=1.0, eps=1.0, cutoff=rc, shift=True).item()
        assert np.isclose(u_np, u_torch, rtol=1e-9, atol=1e-9), (u_np, u_torch)


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
    print(f"\n{'GATE 0 PASS' if failed == 0 else f'GATE 0 FAIL ({failed} failed)'}")
    return failed


if __name__ == "__main__":
    sys.exit(_run_all())
