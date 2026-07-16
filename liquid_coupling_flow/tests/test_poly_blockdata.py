import numpy as np
import torch
import sys
sys.path.insert(0, "/mnt/ssd/GridTransformer/reports/logs-2026-07-17")
from poly_block_data import make_block_pair
from liquid_coupling_flow.poly.model import draw_sigmas, seed_numba, total_U, disp_sweep

EQ_SWEEPS = 1000  # see _equilibrated_positions docstring


def _equilibrated_positions(rng, sig, n, L, beta, step=0.12, n_sweeps=EQ_SWEEPS):
    """A plain `rng.random((n, 3)) * L` draw at rho=1 with this model's sigma range (up to 1.61)
    is not just "a near-touching pair" (r^-12 ~ 1e14) but catastrophic system-wide: at n=128 under
    this module's fixed seeds (seed_numba(0), rng seed 0, draw_sigmas seed 1), U/N = 5.67e7.
    That's fine for exactness tests (row_e vs total_U agree regardless of scale), but
    make_block_pair's block-local relax MC then legitimately blows the permuted block apart
    escaping that overlap, so disp comes out at ~3.95 > 1.5 every run under the fixed seeds --
    deterministic, not flaky. The liquid_coupling_flow/tests/test_poly_model.py `_safe_positions`
    min-distance regeneration guard does NOT fix this here: with sigma up to 1.61, min_d=0.3
    (that file's own n=128 call-site value) still leaves U/N = 2.03e5 (measured), and the guard's
    own docstring notes higher min_d is combinatorial in n -- min_d=0.6 found no hit in 200k
    tries at n=128 (ran >120s, aborted). The actual fix is what poly_bank.py does to every config
    before it is used downstream: run disp_sweep (local Metropolis, unconditional accept/reject,
    no assertions involved) from the raw random draw. n_sweeps=1000 at beta=1/0.12 brings this
    fixture from U/N=5.67e7 to U/N=0.508 (measured, deterministic under the fixed seeds), a
    normal liquid-state energy, after which the block-local relax moves particles by a fraction
    of a diameter as intended (disp=0.352 for this fixture, seed=3)."""
    x = rng.random((n, 3)) * L
    for _ in range(n_sweeps):
        disp_sweep(x, sig, L, beta, step)
    return x


def test_block_pair_structure_and_invariants():
    seed_numba(0)
    rng = np.random.default_rng(0)
    n = 128
    L = n ** (1.0 / 3.0)
    beta = 1.0 / 0.12
    sig = draw_sigmas(n, seed=1)
    x = _equilibrated_positions(rng, sig, n, L, beta)
    assert total_U(x, sig, L) / n < 5.0                                   # sane liquid state
    p = make_block_pair(x, sig, L, beta=beta, k=8, seed=3)
    assert p["x_old"].shape == (8, 3) and p["x_new"].shape == (8, 3)
    assert np.allclose(np.sort(p["sig_new"]), np.sort(p["sig_old"]))     # internal perm only
    assert not np.allclose(p["sig_new"], p["sig_old"])                   # a real permutation
    assert p["env_x"].shape[0] == n - 8
    # env untouched, block moved but bounded (short transport)
    disp = np.abs(p["x_new"] - p["x_old"]).max()
    assert 0 < disp < 1.5
