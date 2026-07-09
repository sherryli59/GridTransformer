"""Task 2 tests --- the deliberately-mediocre, fully-tractable toy proposal.

The proposal is NOT the learned generator.  Its only jobs: (i) a closed-form,
correctly-normalised ordered density q~(x_vec); (ii) full support everywhere pi
has support; (iii) genuine dependence on the ordering (so the Task-4 ordering-
mismatch diagnostic is non-vacuous).  Quality is irrelevant --- it is bad on
purpose.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from ordmh_toy_proposal import ToyOrderedProposal


# -- the load-bearing round-trip: sampler log_q == evaluator log_q on same tuple
def test_roundtrip_logq():
    L = float(np.sqrt(6.0))
    for N in (3, 5):
        prop = ToyOrderedProposal(N=N, L=L, seed=7)
        x, logq_vec = prop.sample_ordered(2000)
        assert x.shape == (2000, N, 2)
        assert logq_vec.shape == (2000,)
        logq_eval = prop.log_q_ordered(x)
        assert np.allclose(logq_vec, logq_eval, atol=1e-9), \
            np.abs(logq_vec - logq_eval).max()


# -- the conditional q(x_j | x_<j) must integrate to 1 (wrapped-normal + mixture)
def test_conditional_normalized():
    L = 2.4494897427831781
    prop = ToyOrderedProposal(N=5, L=L, seed=1)
    rng = np.random.default_rng(0)
    ng = 400
    ax = (np.arange(ng) + 0.5) * (L / ng)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    xj = np.stack([X.ravel(), Y.ravel()], axis=1)      # (ng^2, 2)
    cell = (L / ng) ** 2
    for m in (0, 1, 3):                                  # 0,1,3 already-placed
        placed = rng.uniform(0, L, size=(m, 2))
        logp = prop.conditional_logpdf(xj, placed)
        integral = np.exp(logp).sum() * cell
        assert abs(integral - 1.0) < 5e-3, (m, integral)


# -- full support: density bounded below by the uniform component everywhere
def test_full_support_lower_bound():
    L = float(np.sqrt(6.0))
    N = 5
    prop = ToyOrderedProposal(N=N, L=L, w_u=0.5, seed=3)
    rng = np.random.default_rng(9)
    x = rng.uniform(0, L, size=(500, N, 2))             # arbitrary configs
    logq = prop.log_q_ordered(x)
    assert np.all(np.isfinite(logq))
    # each step's density >= w_u / A  ->  total log_q >= N*log(w_u/A)
    floor = N * np.log(prop.w_u / (L * L))
    assert np.all(logq >= floor - 1e-9)


# -- genuine order dependence: permuting a config changes log_q (Task 4 needs this)
def test_order_dependence():
    L = float(np.sqrt(6.0))
    N = 5
    prop = ToyOrderedProposal(N=N, L=L, seed=2)
    rng = np.random.default_rng(5)
    cfg = rng.uniform(0, L, size=(N, 2))
    perms = np.stack([rng.permutation(N) for _ in range(200)])
    variants = np.stack([cfg[p] for p in perms])         # (200, N, 2)
    logq = prop.log_q_ordered(variants)
    assert logq.std() > 1e-3, logq.std()                 # non-trivial spread


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
    print(f"\n{'TASK 2 TESTS PASS' if failed == 0 else f'TASK 2 TESTS FAIL ({failed})'}")
    return failed


if __name__ == "__main__":
    sys.exit(_run_all())
