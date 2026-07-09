import torch
from liquid_coupling_flow.ka_pin import pin_mask, masked_mala, masked_swap, masked_block_relabel
from liquid_coupling_flow.ka_energy import ka_energy, ka_forces
from liquid_coupling_flow.ipl44.ipl_swap_smc import uniform_weight_fn

def _setup(B=4, N=64, dev="cpu", seed=0):
    torch.manual_seed(seed)
    L = (N / 1.2) ** 0.5
    x = torch.rand(B, N, 2, device=dev) * L
    s = (torch.rand(B, N, device=dev) < 0.37).long()
    return x, s, L

def test_pin_mask_count_and_frozen_fixed():
    x, s, L = _setup()
    mobile = pin_mask(4, 64, c=0.25, device="cpu", generator=torch.Generator().manual_seed(1))
    assert mobile.dtype == torch.bool and mobile.shape == (4, 64)
    # exactly ceil(c*N)=16 pinned per row
    assert int((~mobile[0]).sum()) == 16
    U = ka_energy(x, s, L)
    efn = lambda a, b: ka_energy(a, b, L); ffn = lambda a, b: ka_forces(a, b, L)
    x2, U2, acc = masked_mala(x, s, U, mobile, beta=2.0, L=L, dt=0.01, energy_fn=efn, force_fn=ffn)
    # frozen particles are bit-for-bit unchanged
    assert torch.equal(x2[~mobile], x[~mobile])
    # energy recomputed from x2 equals the tracked U2 (accept bookkeeping is exact)
    assert torch.allclose(ka_energy(x2, s, L), U2, atol=1e-4)
    assert 0.0 <= acc <= 1.0

def test_masked_mala_multi_step_frozen_invariance():
    """Regression test: frozen particles stay bit-for-bit fixed across repeated masked_mala calls."""
    x0, s, L = _setup()
    mobile = pin_mask(4, 64, c=0.25, device="cpu", generator=torch.Generator().manual_seed(1))
    U = ka_energy(x0, s, L)
    efn = lambda a, b: ka_energy(a, b, L); ffn = lambda a, b: ka_forces(a, b, L)

    # Run 20 iterations of masked_mala, threading x and U through
    x, U_curr = x0.clone(), U.clone()
    for step in range(20):
        x, U_curr, acc = masked_mala(x, s, U_curr, mobile, beta=2.0, L=L, dt=0.01, energy_fn=efn, force_fn=ffn)
        # Acceptance rate must be in [0, 1] every step
        assert 0.0 <= acc <= 1.0, f"Step {step}: acceptance rate {acc} out of bounds"

    # After 20 steps, frozen particles must be unchanged from initial
    assert torch.equal(x[~mobile], x0[~mobile]), "Frozen particles changed after multi-step loop"

def test_masked_species_moves_keep_frozen_and_composition():
    x, s, L = _setup(seed=2)
    mobile = pin_mask(4, 64, 0.25, "cpu", generator=torch.Generator().manual_seed(3))
    efn = lambda a, b: ka_energy(a, b, L)
    U = ka_energy(x, s, L)
    s0 = s.clone()
    s1, U1, acc = masked_swap(x, s, U, mobile, 2.0, efn, uniform_weight_fn)
    assert torch.equal(s1[~mobile], s0[~mobile])                 # frozen species untouched
    assert torch.equal(s1.sum(1), s0.sum(1))                     # swap conserves count
    assert torch.allclose(ka_energy(x, s1, L), U1, atol=1e-4)
    # block-relabel with a constant table (uncertain everywhere) -> exact, frozen fixed, count preserved
    table_fn = lambda xx: torch.full((xx.shape[0], xx.shape[1]), 0.4)
    s2, U2, acc2 = masked_block_relabel(x, s0, U, mobile, 2.0, efn, table_fn, k=4)
    assert torch.equal(s2[~mobile], s0[~mobile])
    assert torch.equal(s2.sum(1), s0.sum(1))
    assert torch.allclose(ka_energy(x, s2, L), U2, atol=1e-4)

def test_masked_swap_exact_with_nonuniform_weight():
    """Detailed-balance check for masked_swap with a non-uniform, s-independent weight_fn and frozen
    same-species particles present. Density 0.2 (not the KA-liquid 1.2) is used deliberately: with only
    4 particles, density 1.2 puts random configs deep in core-overlap (U ~ 1e7), which makes ratio_theory
    underflow to exactly 0 (math domain error on log) regardless of kernel correctness -- an artifact of
    the tiny particle count, not of the property under test. At density 0.2 the energy gap is O(1) so both
    states mix, and the test is decisive: the pre-fix unmasked-normalizer bug reproducibly gives
    |log(ratio_emp) - log(ratio_theory)| ~= 1.39 on this exact seed (mask-independent constant selection
    bias, verified by hand), while the fix gives ~= 0.007."""
    import math
    from collections import Counter
    torch.manual_seed(6)
    L = (4 / 0.2) ** 0.5
    x = torch.rand(1, 4, 2) * L
    mobile = torch.tensor([[True, True, False, False]])          # 1 mobile-A(0), 1 mobile-B(1); frozen A(2),B(3)
    w = torch.tensor([[0.8, 0.3, 0.6, 0.2]])                     # non-uniform, s-independent B-affinity
    weight_fn = lambda xx, ss: w.expand(xx.shape[0], -1).clone()
    efn = lambda a, b: ka_energy(a, b, L); beta = 1.5
    s = torch.tensor([[0, 1, 0, 1]]); U = efn(x, s); cnt = Counter()
    for _ in range(20000):
        s, U, _ = masked_swap(x, s, U, mobile, beta, efn, weight_fn)
        cnt[tuple(s[0].tolist())] += 1
    ratio_theory = math.exp(-beta * float(efn(x, torch.tensor([[1,0,0,1]])) - efn(x, torch.tensor([[0,1,0,1]]))))
    ratio_emp = cnt[(1,0,0,1)] / cnt[(0,1,0,1)]
    assert abs(math.log(ratio_emp) - math.log(ratio_theory)) < 0.1

def test_block_relabel_guards_k_gt_mobile():
    import pytest
    x, s, L = _setup(B=2, N=16, seed=4)
    mobile = torch.zeros(2, 16, dtype=torch.bool); mobile[:, :3] = True   # only 3 mobile
    efn = lambda a, b: ka_energy(a, b, L)
    table_fn = lambda xx: torch.full((xx.shape[0], xx.shape[1]), 0.4)
    with pytest.raises(AssertionError):
        masked_block_relabel(x, s, efn(x, s), mobile, 2.0, efn, table_fn, k=4)

from liquid_coupling_flow.ka_pin_overlap import cell_occupancy, q_rand, overlap_Q

def test_overlap_self_is_one_and_qrand():
    assert abs(q_rand(0.3, 1.2) - 0.108) < 1e-9
    x, s, L = _setup(seed=5)
    occ = cell_occupancy(x, L, a_c=0.3)
    excluded = torch.zeros_like(occ)
    # Q(ref, ref) over all cells = 1 exactly
    assert abs(overlap_Q(occ, occ, excluded) - 1.0) < 1e-9
    # occupancy is species-blind: relabelling s must not change occ
    occ2 = cell_occupancy(x, L, a_c=0.3)
    assert torch.equal(occ, occ2)
    # a fully-decorrelated config has Q near q_rand (loose bound, statistical)
    torch.manual_seed(9); xr = torch.rand_like(x) * L
    q = overlap_Q(cell_occupancy(xr, L), occ, excluded)
    assert 0.0 <= q <= 0.4

from liquid_coupling_flow.ka_pin import constrained_run

def test_constrained_run_two_arms_frozen_fixed():
    x, s, L = _setup(B=8, N=100, seed=7)
    mobile = pin_mask(8, 100, 0.2, "cpu", generator=torch.Generator().manual_seed(8))
    torch.manual_seed(1); scr = torch.rand_like(x) * L
    ref = constrained_run(x, s, mobile, T=0.8, L=L, n_iter=30, dt=0.01, record_every=5, arm="ref")
    scb = constrained_run(x, s, mobile, T=0.8, L=L, n_iter=30, dt=0.01, record_every=5,
                          arm="scramble", scramble_x=scr)
    # frozen positions never move in either arm
    assert torch.equal(ref["x_final"][~mobile], x[~mobile])
    assert torch.equal(scb["x_final"][~mobile], x[~mobile])
    # Q(t) recorded; ref arm starts at 1 (mobile at reference), scramble arm starts below 1
    assert ref["Q"][0] > 0.95 and scb["Q"][0] < ref["Q"][0]
    assert len(ref["t"]) == len(ref["Q"]) == len(ref["U"])

def test_overlap_perchain_matches_batch_mean():
    from liquid_coupling_flow.ka_pin_overlap import overlap_Q_perchain
    x, s, L = _setup(B=6, N=64, seed=11)
    occ = cell_occupancy(x, L); occ2 = cell_occupancy(torch.rand_like(x) * L, L)
    excl = torch.zeros_like(occ)
    pc = overlap_Q_perchain(occ2, occ, excl)
    assert pc.shape == (6,)
    assert abs(float(pc.mean()) - overlap_Q(occ2, occ, excl)) < 1e-6

def test_constrained_run_records_qchain_and_guards_arm():
    import pytest
    x, s, L = _setup(B=4, N=100, seed=12)
    mobile = pin_mask(4, 100, 0.2, "cpu", generator=torch.Generator().manual_seed(2))
    r = constrained_run(x, s, mobile, T=0.8, L=L, n_iter=10, dt=0.01, record_every=5, arm="ref")
    assert len(r["Q_chain"]) == len(r["Q"]) and len(r["Q_chain"][0]) == 4    # per-chain, B=4
    with pytest.raises(AssertionError):
        constrained_run(x, s, mobile, T=0.8, L=L, n_iter=5, arm="bogus")

import numpy as np
from liquid_coupling_flow.ka_pin_extract import stretched_exp_fit, xi_threshold

def test_stretched_exp_recovers_plateau():
    t = np.arange(0, 200, 5.0)
    Q = 0.35 + 0.6 * np.exp(-(t / 30.0) ** 0.7)
    fit = stretched_exp_fit(t, Q)
    assert fit["ok"] and abs(fit["Qinf"] - 0.35) < 0.02

def test_xi_threshold_and_error_propagation():
    lvals = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    excess = np.array([0.50, 0.35, 0.22, 0.12, 0.05])           # Qinf - Qrand, monotone decreasing
    Qinf = excess + 0.108
    out = xi_threshold(lvals, Qinf, Qinf_err=np.full(5, 0.01), thr=0.2, Qrand=0.108, dQ_tol=0.02)
    assert out["kind"] == "point" and abs(out["xi"] - 2.6) < 0.05   # 0.22->0.12 crosses 0.2 across l=2.5..3.0
    # near-flat curve AT THE CROSSING inflates dxi (small slope -> large horizontal error).
    # This curve also crosses 0.2 (last excess 0.19 < 0.2) but with a shallow final slope.
    flat = np.array([0.30, 0.26, 0.23, 0.21, 0.19]) + 0.108
    o2 = xi_threshold(lvals, flat, np.full(5, 0.01), thr=0.2, Qrand=0.108, dQ_tol=0.02)
    assert o2["kind"] == "point" and o2["dxi"] > out["dxi"]

def test_xi_threshold_exact_touch_last_point():
    lvals = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    excess = np.array([0.50, 0.40, 0.30, 0.25, 0.20])       # last point touches thr=0.2 exactly
    out = xi_threshold(lvals, excess + 0.108, np.full(5, 0.01), thr=0.2, Qrand=0.108, dQ_tol=0.02)
    assert out["kind"] == "point" and abs(out["xi"] - 3.5) < 1e-9

def test_xi_threshold_no_crossing_and_range():
    lvals = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    # never drops to thr -> none
    o_none = xi_threshold(lvals, np.array([0.5,0.5,0.5,0.5,0.5]) + 0.108, np.full(5,0.01), thr=0.2)
    assert o_none["kind"] == "none" and o_none["xi"] is None
    # non-monotone: crosses thr=0.2 twice (down then up) -> range
    o_rng = xi_threshold(lvals, np.array([0.30,0.15,0.10,0.15,0.30]) + 0.108, np.full(5,0.01), thr=0.2)
    assert o_rng["kind"] == "range" and isinstance(o_rng["xi"], tuple) and o_rng["xi"][0] < o_rng["xi"][1]

def test_stretched_exp_fit_graceful_on_short_input():
    out = stretched_exp_fit(np.array([]), np.array([]))
    assert out["ok"] is False
    out2 = stretched_exp_fit(np.array([0.,1.,2.]), np.array([1.,0.8,0.6]))   # 3 < 4 params
    assert out2["ok"] is False

from liquid_coupling_flow.ka_pin_gates import g_conv, g_corr, g_stick

def _run(qinf, tau, tmax, n=40, arm="ref"):
    t = np.linspace(0, tmax, n)
    sign = 1.0 if arm == "ref" else -1.0
    Q = qinf + sign * 0.4 * np.exp(-(t / tau) ** 0.8)
    return {"t": list(t), "Q": list(Q), "arm": arm}

def test_gconv_needs_agreement_and_run_length():
    good_ref = _run(0.30, 20, 200, arm="ref"); good_scr = _run(0.30, 20, 200, arm="scramble")
    r = g_conv(good_ref, good_scr, tol=0.02)
    assert r["passed"] and r["run_ok"]
    # same plateaus but run too short (tmax < 3 tau) -> run_ok False -> gate fails
    short_ref = _run(0.30, 100, 150, arm="ref"); short_scr = _run(0.30, 100, 150, arm="scramble")
    assert not g_conv(short_ref, short_scr, tol=0.02)["passed"]
    # disagreeing plateaus -> fail
    assert not g_conv(_run(0.30, 20, 200, arm="ref"), _run(0.36, 20, 200, arm="scramble"), tol=0.02)["passed"]

def test_gstick_and_gcorr():
    a = _run(0.30, 20, 200, arm="ref"); b = _run(0.30, 20, 200, arm="scramble"); c = _run(0.30, 20, 400, arm="ref")
    assert g_stick(a, b, c, tol=0.02)["passed"]
    assert g_corr(_run(0.30, 20, 200, arm="ref"), _run(0.31, 20, 200, arm="ref"), tol=0.02)["passed"]

def test_gconv_fails_short_scramble_with_asymmetric_tau():
    # reference well-converged (150/5=30 >> 3); scramble under-converged (150/55=2.73 < 3) -> MUST fail run-length
    ref = _run(0.30, 5, 150, arm="ref")
    scr = _run(0.30, 55, 150, arm="scramble")
    r = g_conv(ref, scr, tol=0.02)
    assert not r["run_ok"], f"expected run_ok False (fitted tau_s={r['tau']:.1f}), got {r}"
    assert not r["passed"]

import os
from liquid_coupling_flow.ka_pin_refs import get_references

def test_get_references_highT_smoke(tmp_path):
    r = get_references(T=0.8, N=100, n_configs=8, device="cpu", out_dir=str(tmp_path))
    assert r["x"].shape == (8, 100, 2) and r["s"].shape == (100,)
    assert 0.30 < r["xB"] < 0.45                                    # ~0.37 nominal
    assert os.path.exists(os.path.join(str(tmp_path), "refs_T0.8_N100.pt"))
    # equilibrated high-T energy is well below the ideal-gas 0 and finite
    from liquid_coupling_flow.ka_energy import ka_energy
    u = float((ka_energy(r["x"], r["s"], r["L"]) / 100).median()); assert -4.0 < u < -1.0

def test_get_references_T05_pt2_branch(tmp_path):
    import os
    from liquid_coupling_flow.ka_pin_refs import get_references, PT2_N256
    if not os.path.exists(PT2_N256):
        import pytest; pytest.skip("PT2 dataset not present")
    r = get_references(T=0.5, N=256, n_configs=6, device="cpu", out_dir=str(tmp_path))
    assert r["x"].shape == (6, 256, 2) and r["s"].shape == (256,)
    from liquid_coupling_flow.ka_energy import ka_energy
    u = float((ka_energy(r["x"], r["s"], r["L"]) / 256).median())
    assert -3.4 < u < -3.0                                       # cold T=0.5 rung, ~ -3.23
    assert 0.30 < r["xB"] < 0.42

from liquid_coupling_flow.ka_pin_campaign import run_cell, aggregate

def test_run_cell_and_aggregate_smoke(tmp_path):
    from liquid_coupling_flow.ka_pin_refs import get_references
    refs = get_references(0.8, 100, 8, "cpu", str(tmp_path))
    cell = run_cell(T=0.8, c=0.16, refs=refs, n_iter=20, table_fn=None, device="cpu",
                    out_dir=str(tmp_path), n_real=8)
    assert set(["T", "c", "lc", "Qinf", "gconv", "xB"]).issubset(cell.keys())
    assert cell["lc"] > 0
    # aggregate a synthetic monotone set of cells into a xi at threshold 0.2
    cells = [{"T": 0.8, "lc": lc, "Qinf": q, "Qinf_err": 0.01}
             for lc, q in zip([1.5, 2.0, 2.5, 3.0], [0.55, 0.38, 0.22, 0.12])]
    agg = aggregate(cells, thresholds=(0.2,))
    assert 0.8 in {round(t, 3) for t in agg[0.2].keys()}
