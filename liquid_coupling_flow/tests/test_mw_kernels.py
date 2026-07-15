import math, torch, pytest
from liquid_coupling_flow.mw.mw_generator_v4 import MWGlobalAR
from liquid_coupling_flow.mw.mw_block_ar import MWPeriodicBlockAR
from liquid_coupling_flow.mw.mw_energy import mw_energy
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept
from liquid_coupling_flow.mw.mw_kernels import (
    recanonicalize, is_canonical, suffix_move, two_blob_move, draw_centers, conveyor_regions)

N, L = 27, 3.9


def q0_tiny():
    torch.manual_seed(0)
    return MWGlobalAR(d_model=32, n_layers=1, n_heads=2, n_mix=4, rail_k=4).eval()


def blk_tiny():
    torch.manual_seed(1)
    return MWPeriodicBlockAR(d_model=32, n_rbf=6).eval()


def test_recanonicalize_makes_canonical_and_preserves_canonical_logq0():
    q0 = q0_tiny()
    x = torch.rand(4, N, 3) * L
    xc = recanonicalize(x, L)
    assert is_canonical(xc, L).all()
    with torch.no_grad():   # canonical lift is permutation-invariant
        assert torch.allclose(q0.log_prob(x, L), q0.log_prob(xc, L), atol=1e-4)


def test_suffix_move_lambda0_accepts_all_canonical():
    q0 = q0_tiny()
    with torch.no_grad():
        x = recanonicalize(q0.sample(8, N, L, gen=torch.Generator().manual_seed(2))[0], L)
        U = mw_energy(x, L)
        x2, U2, st = suffix_move(x, U, q0, 8, 12, lam=0.0, beta=10.0, L=L,
                                 gen=torch.Generator().manual_seed(3))
    assert st["acc"] == pytest.approx(1.0 - st["reject_canon"], abs=1e-9)
    assert torch.allclose(mw_energy(x2, L), U2, atol=1e-3)


def test_suffix_reduction_equals_general_bridge_formula():
    """-lam*(beta*dU + lqf - lqr) must equal the full geometric-bridge ratio with
    canonical-lift q0 whenever current AND proposed states are canonical.

    NB (deviation from brief step-1 draft: m 9->1, batch 16->48): the brief drafted
    m=9 resampled slots on batch 16, but the deliberately-broad *untrained* q0_tiny
    (max per-axis displacement s*R = L/2 for N=27) scatters a 9-of-27 suffix resample
    across the whole box, so ZERO proposals stay canonical (measured 0/1024 for any
    m>=5) and the non-vacuousness precondition `ok.any()` can never fire. m=1 is the
    only regime with an appreciable canonical rate (~8%); the reduction property is
    independent of m, so this strengthens the test (the atol=1e-3 exactness guard now
    actually runs) rather than loosening it. On canonical proposals red==gen_form to
    ~1e-5; on non-canonical they diverge by O(15) nats (guard is meaningful)."""
    q0 = q0_tiny()
    lam, beta = 0.37, 5.0
    with torch.no_grad():
        x = recanonicalize(q0.sample(48, N, L, gen=torch.Generator().manual_seed(4))[0], L)
        m = 1
        lqr = q0.suffix_log_prob(x, m, L)
        xp, lqf = q0.sample_suffix(x, m, L, gen=torch.Generator().manual_seed(5))
        ok = is_canonical(xp, L)
        assert ok.any(), "need at least one canonical proposal for the check"
        U, Up = mw_energy(x, L), mw_energy(xp, L)
        red = -lam * (beta * (Up - U) + (lqf - lqr))
        gen_form = geometric_bridge_log_accept(
            log_q0_current=q0.log_prob(x, L), log_q0_proposed=q0.log_prob(xp, L),
            energy_current=U, energy_proposed=Up,
            log_r_reverse=lqr, log_r_forward=lqf, lam=lam, beta=beta)
        assert torch.allclose(red[ok], gen_form[ok], atol=1e-3)


def test_two_blob_reverse_check_and_state_update():
    q0, blk = q0_tiny(), blk_tiny()
    with torch.no_grad():
        x = recanonicalize(torch.rand(6, N, 3) * L, L)
        U = mw_energy(x, L)
        lq0 = q0.log_prob(x, L)
        x2, U2, lq02, st = two_blob_move(x, U, lq0, q0, blk, K=3, lam=0.2, beta=2.0,
                                         L=L, min_sep=1.5,
                                         gen=torch.Generator().manual_seed(6))
    assert torch.allclose(mw_energy(x2, L), U2, atol=1e-3)
    with torch.no_grad():
        assert torch.allclose(q0.log_prob(x2, L), lq02, atol=1e-4)
    assert 0.0 <= st["acc"] <= 1.0 and 0.0 <= st["reject_reverse"] <= 1.0


def test_conveyor_regions_partition_and_center_draw():
    pre, suf = conveyor_regions(N, L, torch.device("cpu"), m_frac=0.4)
    assert pre.shape[0] + suf.shape[0] == N and suf.shape[0] == round(0.4 * N)
    cA, cB = draw_centers(L, 1.0, torch.Generator().manual_seed(7),
                          torch.device("cpu"), regions=(pre, suf))
    assert cA.shape == (3,) and cB.shape == (3,)


@pytest.mark.slow
def test_two_blob_stationarity_hot():
    """pi_1-invariance smoke at beta=2 (hot, so acceptance is non-trivial): U/N of a
    reference single-site chain must be statistically unchanged after 300 two-blob moves."""
    from liquid_coupling_flow.mw.mw_energy import RHO_STAR
    from liquid_coupling_flow.mw.mw_reference import mc_run
    q0, blk = q0_tiny(), blk_tiny()
    Ls = (N / RHO_STAR) ** (1.0 / 3.0)
    ref = mc_run(N, Ls, 2.0, n_equil=1500, n_collect=1500, every=50, seed=8, B=8)
    x = recanonicalize(ref["cfgs"][-8:].clone(), Ls)
    U = mw_energy(x, Ls)
    gen = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for _ in range(300):
            x, U, _, _ = two_blob_move(x, U, None, q0, blk, K=3, lam=1.0, beta=2.0,
                                       L=Ls, min_sep=1.5, gen=gen)
    u_ref = (ref["U"] / N)
    z = abs(float(U.mean() / N) - float(u_ref.mean())) / max(float(u_ref.std()) / math.sqrt(8), 1e-6)
    assert z < 4.0, f"two-blob chain drifted: z={z:.1f}"
