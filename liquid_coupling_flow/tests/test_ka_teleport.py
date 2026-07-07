"""Teleport-swap kernel tests (ka_teleport.py): two distant slots exchange occupants, each placed by A2's
frozen-cage conditional in the other's neighborhood. Exactness surface: non-overlapping-cage abort symmetric;
cage densities frozen (independent of both movers); MH invariances incl species-swap bookkeeping."""
import os, torch, pytest
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, ART

DEV = "cpu"
HB_CKPT = os.path.join(ART, "ka_heatbath_N100.pt")
REF = os.path.join(ART, "ka_reference_N100.pt")


def _env(B=4):
    if not (os.path.exists(HB_CKPT) and os.path.exists(REF)):
        pytest.skip("A2 ckpt or reference unavailable")
    from liquid_coupling_flow.ka_heatbath import _load
    sc, L, geo = _scaffold(100, DEV)
    ref = torch.load(REF, map_location=DEV, weights_only=False)
    pos, s = slot_order(ref["x"][:B].to(DEV), ref["s"].to(DEV).long(), geo, 100)
    HB = _load(torch.load(HB_CKPT, map_location=DEV, weights_only=False), DEV)
    return sc, L, geo, pos, s, HB


def test_overlap_abort_symmetric_and_fires():
    from liquid_coupling_flow.ka_teleport import cages_overlap
    sc, L, geo, pos, s, HB = _env()
    n_ctx = HB.n_ctx
    # adjacent slots must overlap; far-apart slots must not; symmetry always
    for a, b in [(5, 6), (5, 55), (10, 60), (3, 80)]:
        oab = cages_overlap(a, b, sc, L, n_ctx)
        oba = cages_overlap(b, a, sc, L, n_ctx)
        assert oab == oba, f"overlap({a},{b}) asymmetric"
    assert cages_overlap(5, 6, sc, L, n_ctx) is True
    far = [(a, b) for a in range(100) for b in range(a + 30, 100)
           if not cages_overlap(a, b, sc, L, n_ctx)]
    assert len(far) > 0, "no non-overlapping slot pairs found (kernel would always abort)"


def test_frozen_cages_independent_of_movers():
    """The DB keystone: both directions' densities are computed on cages that EXCLUDE both movers, so
    perturbing the movers' positions must not change any of the four densities."""
    from liquid_coupling_flow.ka_teleport import cages_overlap, pair_logq
    sc, L, geo, pos, s, HB = _env()
    a, b = next((i, j) for i in range(100) for j in range(i + 30, 100)
                if not cages_overlap(i, j, sc, L, HB.n_ctx))
    beta = torch.full((4,), 2.0)
    xq_a = pos[:, a] + 0.2; xq_b = pos[:, b] - 0.2
    lq1 = pair_logq(HB, pos, s, a, b, xq_a, xq_b, sc, L, beta)
    pos2 = pos.clone()
    pos2[:, a] = torch.rand(4, 2) * L; pos2[:, b] = torch.rand(4, 2) * L   # move ONLY the movers
    lq2 = pair_logq(HB, pos2, s, a, b, xq_a, xq_b, sc, L, beta)
    assert torch.allclose(lq1, lq2, atol=1e-5), f"cages not frozen: {(lq1-lq2).abs().max():.2e}"


def test_move_invariances_and_species_swap():
    from liquid_coupling_flow.ka_teleport import teleport_swap_move, cages_overlap
    sc, L, geo, pos, s, HB = _env()
    a, b = next((i, j) for i in range(100) for j in range(i + 30, 100)
                if not cages_overlap(i, j, sc, L, HB.n_ctx) and (s[0, i] != s[0, j]))
    torch.manual_seed(0)
    counts0 = s.sum(1).clone()
    pos2, s2, info = teleport_swap_move(HB, pos, s, a, b, sc, L, beta=2.0)
    assert torch.equal(s2.sum(1), counts0), "species counts must be preserved"
    mask = torch.ones(100, dtype=torch.bool); mask[a] = False; mask[b] = False
    assert torch.equal(pos2[:, mask], pos[:, mask]), "non-mover positions must be untouched"
    assert torch.equal(s2[:, mask], s[:, mask]), "non-mover species must be untouched"
    acc = info["accept_mask"]
    if acc.any():
        assert (s2[acc][:, a] == s[acc][:, b]).all() and (s2[acc][:, b] == s[acc][:, a]).all(), \
            "accepted unlike teleport must swap species labels at the two slots"
    assert 0.0 <= info["accept"] <= 1.0


def test_sweep_runs_finite():
    from liquid_coupling_flow.ka_teleport import teleport_sweep
    sc, L, geo, pos, s, HB = _env()
    torch.manual_seed(0)
    pos2, s2, info = teleport_sweep(HB, pos, s, sc, L, geo, beta=2.0, n_moves=10)
    assert torch.isfinite(pos2).all()
    assert 0.0 <= info["accept"] <= 1.0 and 0.0 <= info["abort_frac"] <= 1.0
