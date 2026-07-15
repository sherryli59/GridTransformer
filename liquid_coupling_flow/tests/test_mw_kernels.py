import math, torch, pytest
from liquid_coupling_flow.mw.mw_generator_v4 import MWGlobalAR
from liquid_coupling_flow.mw.mw_energy import mw_energy
from liquid_coupling_flow.mw.mw_generator import wrap_pm
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept
from liquid_coupling_flow.mw.mw_kernels import (
    recanonicalize, is_canonical, suffix_move, two_blob_move, draw_centers, conveyor_regions)

N, L = 27, 3.9


def q0_tiny():
    torch.manual_seed(0)
    return MWGlobalAR(d_model=32, n_layers=1, n_heads=2, n_mix=4, rail_k=4).eval()


# --------------------------------------------------------------------------------------
# Two-blob rig (review fix 2026-07-15): the original tests drove two_blob_move with an
# untrained MWPeriodicBlockAR whose scattered proposals NEVER survived the reverse-check
# (measured 0 live / 0 accepted in 960 attempts) -- the acceptance path was tested by
# zero firing moves. This rig makes the path fire deterministically and with real power:
#
# - FixedMeanBlockProposal: independent wrapped normals around FIXED means (slot t = t-th
#   index of the sorted block), constant scale. State-independent constants => trivially a
#   valid context-only proposal for MH; exact density.
# - Geometry: 27 particles on a periodic 3x3x3 simple-cubic lattice, spacing 1.1 == the
#   mW pair minimum (L_RIG = 3.3 commensurate). Block = particles 0..3 on two adjacent
#   site PAIRS (blob A, blob B); context = the other 23 sites, frozen. The conditional
#   pi(x_block | x_ctx) is unimodal AT the vacant sites (measured mode offset ~0.003,
#   per-axis sigma ~0.073) and site escape costs ~+8 (e^-16 at beta=2): both the two-blob
#   chain and the reference provably sample the SAME basin (asserted via localization
#   guards, not assumed). NB spacing 1.3 was tried and REJECTED: its face-channel barrier
#   is soft and the reference wandered 2.1 from the means.
# - Centers: cA*/cB* = the two pair midpoints; blob sites are 0.55 from their center, all
#   other sites >= 1.23, so k-nearest selection is robust to the conveyor cell jitter
#   (4096 rows/region => cw = L_RIG/20, jitter <= 0.143) plus thermal motion; selection
#   tail events only cause reverse-check rejects (exactness unaffected).
# - SIGMA_RIG = 0.10 is deliberately ~1.4x wider than pi's 0.073: a mildly mismatched
#   proposal keeps the stationarity gate sensitive to q-term bugs. Mutation-tested
#   (probe 2026-07-15): honest z = 1.18/0.37/0.66 (seeds 9/21/33); swapping lq_r/lq_f
#   z = 15.5; flipping the energy delta z = 7.0 -- both bug classes fail the z<4 gate.
# - Chains start from one forced proposal draw, NOT the lattice minimum: the minimum is
#   the max-pi/q point, the stickiest state of an independence sampler (cold-start bias
#   z ~ 3 measured), while a q-draw start equilibrates within ~100 moves.
# --------------------------------------------------------------------------------------
L_RIG, BETA_RIG, K_RIG, SIGMA_RIG = 3.3, 2.0, 2, 0.10


class FixedMeanBlockProposal:
    """Test double for the block-proposal interface (sample_block / block_log_prob).

    Independent wrapped normals with fixed means mu[t] for the t-th particle of the
    sorted index block, fixed scale. The single-image Gaussian on the min-image
    displacement is exact here far beyond float precision: the nearest neglected image
    sits >= 20 sigma out for any reachable displacement (sigma=0.10, L=3.3)."""

    def __init__(self, mu, scale):
        self.mu, self.scale = mu, float(scale)

    def block_log_prob(self, x, idx, L):
        idx = torch.as_tensor(idx, dtype=torch.long, device=x.device).flatten().sort().values
        d = wrap_pm(x[:, idx] - self.mu[None], L)
        return (-0.5 * (d / self.scale) ** 2 - math.log(self.scale)
                - 0.5 * math.log(2 * math.pi)).sum((1, 2))

    def sample_block(self, x, idx, L, gen=None):
        idx = torch.as_tensor(idx, dtype=torch.long, device=x.device).flatten().sort().values
        out = x.clone()
        eps = torch.randn(x.shape[0], idx.numel(), 3, generator=gen)
        out[:, idx] = torch.remainder(self.mu[None] + self.scale * eps, L)
        return out, self.block_log_prob(out, idx, L)


def two_blob_rig():
    """Return (x0 [1,27,3], mu [4,3], regions) -- see the rig comment above."""
    g = torch.arange(3) * 1.1 + 0.55
    sites = torch.cartesian_prod(g, g, g).float()
    sA0 = torch.tensor([0.55, 0.55, 0.55]); sA1 = torch.tensor([1.65, 0.55, 0.55])
    sB0 = torch.tensor([1.65, 2.75, 2.75]); sB1 = torch.tensor([2.75, 2.75, 2.75])
    mu = torch.stack([sA0, sA1, sB0, sB1])
    is_blk = torch.zeros(27, dtype=torch.bool)
    for s in (sA0, sA1, sB0, sB1):
        is_blk |= (sites - s).norm(dim=-1) < 1e-6
    assert int(is_blk.sum()) == 4
    x0 = torch.cat([mu, sites[~is_blk]])[None]
    cA, cB = (sA0 + sA1) / 2, (sB0 + sB1) / 2
    assert wrap_pm(cA - cB, L_RIG).norm() > 1.5
    regions = (cA[None].repeat(4096, 1), cB[None].repeat(4096, 1))
    return x0, mu, regions


def test_recanonicalize_makes_canonical_and_preserves_canonical_logq0():
    q0 = q0_tiny()
    x = torch.rand(4, N, 3) * L
    xc = recanonicalize(x, L)
    assert is_canonical(xc, L).all()
    with torch.no_grad():   # canonical lift is permutation-invariant
        assert torch.allclose(q0.log_prob(x, L), q0.log_prob(xc, L), atol=1e-4)


def test_suffix_move_lambda0_accepts_all_canonical():
    """At lam=0 every canonical proposal is accepted (la == 0 > log U(0,1)), so per-move
    acc == 1 - reject_canon exactly, and the energy ledger stays honest across a chain.
    m in 1..3 (review fix: the drafted 8..12 gave a 0% canonical-proposal rate for the
    untrained q0, so the accept path never fired); a 20-move loop is used because m is
    redrawn per move and only m=1 has a large (~8%) canonical rate."""
    q0 = q0_tiny()
    with torch.no_grad():
        x = recanonicalize(q0.sample(8, N, L, gen=torch.Generator().manual_seed(2))[0], L)
        U = mw_energy(x, L)
        gen = torch.Generator().manual_seed(3)
        tot_acc = 0.0
        for _ in range(20):
            x, U, st = suffix_move(x, U, q0, 1, 3, lam=0.0, beta=10.0, L=L, gen=gen)
            assert st["acc"] == pytest.approx(1.0 - st["reject_canon"], abs=1e-9)
            tot_acc += st["acc"]
    assert tot_acc > 0.0, "accept path never fired"
    assert torch.allclose(mw_energy(x, L), U, atol=1e-3)


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
    """The acceptance path must FIRE (review fix: with the untrained block AR it never
    did -- 0 live in 960 attempts -- so the state-update asserts held vacuously).
    lam=1 leg: live and accepted moves happen, the energy ledger tracks a fresh
    recompute, and the frozen context is never touched. lam=0.5 leg: the q0 branch
    runs and lq0 is maintained through accepts/rejects; the leg loops until an accept
    has fired so the lq0-update path is genuinely exercised."""
    q0 = q0_tiny()
    x0, mu, regions = two_blob_rig()
    blk = FixedMeanBlockProposal(mu, SIGMA_RIG)
    B = 6
    with torch.no_grad():
        x = x0.repeat(B, 1, 1)
        x, _ = blk.sample_block(x, torch.arange(4), L_RIG,
                                gen=torch.Generator().manual_seed(60))   # dispersed start
        U = mw_energy(x, L_RIG)
        gen = torch.Generator().manual_seed(6)
        n_live = n_acc = 0.0
        for _ in range(30):                                              # lam = 1 leg
            x, U, _, st = two_blob_move(x, U, None, None, blk, K=K_RIG, lam=1.0,
                                        beta=BETA_RIG, L=L_RIG, min_sep=1.0, gen=gen,
                                        regions=regions)
            assert 0.0 <= st["acc"] <= 1.0 and 0.0 <= st["reject_reverse"] <= 1.0
            n_live += st["live"] * B
            n_acc += st["acc"] * B
        assert n_live > 0, "reverse-check never passed: proposal path is dead"
        assert n_acc >= 1, "acceptance path never fired"
        assert torch.allclose(mw_energy(x, L_RIG), U, atol=1e-3)

        lq0 = q0.log_prob(x, L_RIG)                                      # lam = 0.5 leg
        n_acc05 = 0.0
        for i in range(80):
            x, U, lq0, st = two_blob_move(x, U, lq0, q0, blk, K=K_RIG, lam=0.5,
                                          beta=BETA_RIG, L=L_RIG, min_sep=1.0, gen=gen,
                                          regions=regions)
            n_acc05 += st["acc"] * B
            if n_acc05 >= 1 and i >= 4:
                break
        assert n_acc05 >= 1, "lam<1 acceptance path never fired"
        assert torch.allclose(mw_energy(x, L_RIG), U, atol=1e-3)
        assert torch.allclose(q0.log_prob(x, L_RIG), lq0, atol=1e-4)     # lq0 maintained
    assert torch.equal(x[:, 4:], x0.repeat(B, 1, 1)[:, 4:]), "frozen context was touched"


def test_conveyor_regions_partition_and_center_draw():
    pre, suf = conveyor_regions(N, L, torch.device("cpu"), m_frac=0.4)
    assert pre.shape[0] + suf.shape[0] == N and suf.shape[0] == round(0.4 * N)
    cA, cB = draw_centers(L, 1.0, torch.Generator().manual_seed(7),
                          torch.device("cpu"), regions=(pre, suf))
    assert cA.shape == (3,) and cB.shape == (3,)


@pytest.mark.slow
def test_two_blob_stationarity_hot():
    """REAL pi_1-invariance gate at beta=2 on the firing rig (review fix: the old
    version compared against an mc_run reference through a chain whose moves never
    fired). With the context frozen and the same 4 indices always re-selected, the
    two-blob chain's invariant law is pi(x_block | x_ctx) \\propto e^{-beta U}. The
    reference samples the SAME conditional by plain single-site random-walk MH over
    only those 4 particles. Gates: z<4 on the U means (both-side variances over 8
    independent walkers each), accepted > 50, and localization guards proving both
    chains stayed in the shared basin (so the comparison is apples-to-apples by
    assertion, not hope). Mutation-tested: swapped lq z=15.5, flipped dU z=7.0."""
    x0, mu, regions = two_blob_rig()
    blk = FixedMeanBlockProposal(mu, SIGMA_RIG)
    B = 8

    # --- two-blob chain: 500 moves from a dispersed (proposal-draw) start ---
    with torch.no_grad():
        x = x0.repeat(B, 1, 1)
        x, _ = blk.sample_block(x, torch.arange(4), L_RIG,
                                gen=torch.Generator().manual_seed(1009))
        U = mw_energy(x, L_RIG)
        gen = torch.Generator().manual_seed(9)
        n_acc = 0.0
        u_blob = []
        for i in range(500):
            x, U, _, st = two_blob_move(x, U, None, None, blk, K=K_RIG, lam=1.0,
                                        beta=BETA_RIG, L=L_RIG, min_sep=1.0, gen=gen,
                                        regions=regions)
            n_acc += st["acc"] * B
            if i >= 100:
                u_blob.append(U.clone())
        u_blob = torch.stack(u_blob)                                     # [T,B]
    assert n_acc > 50, f"only {n_acc:.0f} accepted moves: chain effectively frozen"
    assert torch.equal(x[:, 4:], x0.repeat(B, 1, 1)[:, 4:]), "frozen context was touched"
    d_blob = wrap_pm(x[:, :4] - mu[None], L_RIG).norm(dim=-1).max()
    assert float(d_blob) < 0.8, f"two-blob chain left the site basin ({float(d_blob):.2f})"

    # --- reference: single-site RW-MH over ONLY the 4 block particles, same target ---
    with torch.no_grad():
        xr = x0.repeat(B, 1, 1)
        Ur = mw_energy(xr, L_RIG)
        gen_r = torch.Generator().manual_seed(10)
        u_ref = []
        for s in range(3000):
            for i in range(4):
                prop = xr.clone()
                prop[:, i] = torch.remainder(
                    xr[:, i] + 0.1 * torch.randn(B, 3, generator=gen_r), L_RIG)
                Up = mw_energy(prop, L_RIG)
                a = (torch.rand(B, generator=gen_r).clamp_min(1e-38).log()
                     < -BETA_RIG * (Up - Ur))
                xr = torch.where(a[:, None, None], prop, xr)
                Ur = torch.where(a, Up, Ur)
            if s >= 1500 and s % 5 == 0:
                u_ref.append(Ur.clone())
        u_ref = torch.stack(u_ref)                                       # [T,B]
    d_ref = wrap_pm(xr[:, :4] - mu[None], L_RIG).norm(dim=-1).max()
    assert float(d_ref) < 0.8, f"reference chain left the site basin ({float(d_ref):.2f})"

    # --- z on means, both-side variances over per-walker means (8 independent each) ---
    mb, mr = u_blob.mean(0), u_ref.mean(0)
    sem = math.sqrt(float(mb.var(unbiased=True)) / B + float(mr.var(unbiased=True)) / B)
    z = abs(float(mb.mean() - mr.mean())) / max(sem, 1e-9)
    assert z < 4.0, (f"two-blob chain off the conditional: z={z:.1f} "
                     f"(blob {float(mb.mean()):.3f} vs ref {float(mr.mean()):.3f})")
