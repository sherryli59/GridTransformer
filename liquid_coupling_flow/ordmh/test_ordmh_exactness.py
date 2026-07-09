"""Task 3 --- the CORE independent exactness check (Gate E).

Does ordered-space independence-MH targeting pi~ ∝ e^{-beta U} actually sample
the physical Boltzmann distribution?  We check the MH-corrected sampler --- run
with the DELIBERATELY-MEDIOCRE toy proposal --- against the independent Task-1
ground truth (grid quadrature + displacement MC).

The discriminating point (test_mh_correction_is_load_bearing): the RAW proposal
draws give the WRONG <U>; only after the MH accept/reject do we recover the
grid-quadrature <U>.  That proves exactness comes from the MH construction, not
from a good generator.
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
from ordmh_reference import grid_quadrature, radial_distribution, weighted_boltzmann_mean
from ordmh_toy_proposal import ToyOrderedProposal
from ordmh_sampler import OrderedMH

L3 = float(np.sqrt(6.0))   # N=3, rho*=0.5
BETA = 1.0


def _hist_L1_KS(a, b, nbins=50):
    """L1 (total-variation-ish) and KS distance between two samples on shared bins."""
    lo = min(a.min(), b.min())
    hi = max(np.quantile(a, 0.999), np.quantile(b, 0.999))
    edges = np.linspace(lo, hi, nbins + 1)
    pa, _ = np.histogram(a, bins=edges, density=False)
    pb, _ = np.histogram(b, bins=edges, density=False)
    pa = pa / pa.sum(); pb = pb / pb.sum()
    l1 = 0.5 * np.abs(pa - pb).sum()          # total variation
    ks = np.abs(np.cumsum(pa) - np.cumsum(pb)).max()
    return l1, ks


# -- Gate E core: sampler <U> matches the BEDROCK grid quadrature -------------
def test_exactness_meanU_matches_grid_N3():
    grid = grid_quadrature(N=3, L=L3, beta=BETA, n_grid=48)
    prop = ToyOrderedProposal(N=3, L=L3, seed=11)
    smp = OrderedMH(N=3, L=L3, beta=BETA, proposal=prop, cache_check_every=1)
    out = smp.run(n_chains=2000, n_steps=1500, n_equil=400, thin=2, seed=11)
    diff = abs(out["mean_U"] - grid["mean_U"])
    tol = 5 * out["mean_U_err"] + grid["mean_U_err"]
    assert diff < tol, (out["mean_U"], grid["mean_U"], diff, tol, out["acc_rate"])


# -- the MH correction is load-bearing: raw proposal is WRONG, MH fixes it ----
def test_mh_correction_is_load_bearing():
    grid = grid_quadrature(N=3, L=L3, beta=BETA, n_grid=48)
    prop = ToyOrderedProposal(N=3, L=L3, seed=12)
    # raw proposal draws (NO MH): E_q[U] must differ from Boltzmann <U>
    x_raw, _ = prop.sample_ordered(200000)
    U_raw = wca_energy(x_raw, L3)
    meanU_raw = float(U_raw[np.isfinite(U_raw)].mean())
    # MH-corrected sampler: must match grid
    smp = OrderedMH(N=3, L=L3, beta=BETA, proposal=prop)
    out = smp.run(n_chains=2000, n_steps=1500, n_equil=400, thin=2, seed=12)
    assert abs(meanU_raw - grid["mean_U"]) > 20 * grid["mean_U_err"], \
        ("raw proposal unexpectedly close to Boltzmann", meanU_raw, grid["mean_U"])
    assert abs(out["mean_U"] - grid["mean_U"]) < 5 * out["mean_U_err"] + grid["mean_U_err"]


# -- P(U) distribution agreement (stronger than the mean) --------------------
def test_exactness_PU_matches_reference_N3():
    from ordmh_reference import displacement_mc
    mc = displacement_mc(N=3, L=L3, beta=BETA, n_chains=512, n_equil=600,
                         n_collect=800, thin=3, step=0.22, seed=13)
    U_ref = wca_energy(mc["configs"], L3)
    prop = ToyOrderedProposal(N=3, L=L3, seed=13)
    smp = OrderedMH(N=3, L=L3, beta=BETA, proposal=prop)
    out = smp.run(n_chains=2000, n_steps=1500, n_equil=400, thin=2, seed=13)
    U_smp = wca_energy(out["configs"], L3)
    l1, ks = _hist_L1_KS(U_ref, U_smp)
    assert l1 < 0.03, (l1, ks)   # total-variation distance between P(U)s


# -- g(r) agreement ----------------------------------------------------------
def test_exactness_gr_matches_reference_N3():
    from ordmh_reference import displacement_mc
    mc = displacement_mc(N=3, L=L3, beta=BETA, n_chains=512, n_equil=600,
                         n_collect=800, thin=3, step=0.22, seed=14)
    c_ref, g_ref = radial_distribution(mc["configs"], L3, nbins=40)
    prop = ToyOrderedProposal(N=3, L=L3, seed=14)
    smp = OrderedMH(N=3, L=L3, beta=BETA, proposal=prop)
    out = smp.run(n_chains=2000, n_steps=1500, n_equil=400, thin=2, seed=14)
    c_s, g_s = radial_distribution(out["configs"], L3, nbins=40)
    # compare where g_ref has structure (r in [0.9, L/2])
    mask = c_ref > 0.9
    assert np.max(np.abs(g_ref[mask] - g_s[mask])) < 0.15, \
        np.max(np.abs(g_ref[mask] - g_s[mask]))


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
    print(f"\n{'GATE E PASS' if failed == 0 else f'GATE E FAIL ({failed})'}")
    return failed


if __name__ == "__main__":
    sys.exit(_run_all())
