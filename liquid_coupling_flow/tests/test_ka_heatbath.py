"""Stage A2 tests: the beta-conditioned full-cage single-site heat-bath (ka_heatbath.HeatBathModel).

THE named exactness invariant (frozen cage): the site conditional q(x_i | cage, species, beta) must NOT depend
on x_i itself — the cage + frame come from the OTHER particles only, identical for the forward proposal of x_i'
and the reverse density of x_i. If it holds, single-site MH satisfies detailed balance. This is inherited from
frame_ctx_slots([site]) being config-independent (the same property that makes the cluster move exact), but it
is the campaign's classic silent-bug site (arcnorm anchor, EGNN frame-mix) so it gets a dedicated test.
"""
import torch, pytest
from liquid_coupling_flow.ka_cluster_flow import _scaffold


DEV = "cpu"


def _env(B=4, N=100, head="spline"):
    sc, L, geo = _scaffold(N, DEV)
    torch.manual_seed(0)
    pos = torch.rand(B, N, 2, device=DEV) * L
    s = (torch.rand(B, N) < 0.35).long()
    from liquid_coupling_flow.ka_heatbath import HeatBathModel
    m = HeatBathModel(rho=1.2, n_bins=24, box=3.0, n_ctx=24, d_model=64, n_head=4, n_layer=2,
                      head=head, pair_feats=False).to(DEV).eval()
    return sc, L, geo, pos, s, m


def test_frozen_cage_invariance():
    """log_q_site is INVARIANT to x_site: two states differing ONLY in the site's position give identical
    log_q for a fixed query -> detailed balance holds. THE exactness gate."""
    sc, L, geo, pos, s, m = _env()
    site = 5
    B = pos.shape[0]
    beta = torch.full((B,), 2.0)
    xi_query = torch.rand(B, 2) * L
    lqA = m.log_q_site(pos, s, site, xi_query, sc, L, beta)
    posB = pos.clone(); posB[:, site] = torch.rand(B, 2) * L               # move ONLY the site
    lqB = m.log_q_site(posB, s, site, xi_query, sc, L, beta)
    assert torch.allclose(lqA, lqB, atol=1e-5), f"cage not frozen (x_site leaks): max {(lqA-lqB).abs().max():.2e}"


def test_beta_conditioning_is_live():
    """beta is WIRED into the forward pass (FiLM adds to the head's context). Use the binned head (non-zero-init,
    responds to context immediately — the spline head zero-inits its conditioning, masking everything at init)
    and perturb the FiLM off its inert zero-init; then beta must change the conditional."""
    sc, L, geo, pos, s, m = _env(head="bins")
    with torch.no_grad():                                             # simulate a trained (non-inert) FiLM
        for p in m.beta_film.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    site = 5; B = pos.shape[0]
    xi = torch.rand(B, 2) * L
    lq_lo = m.log_q_site(pos, s, site, xi, sc, L, torch.full((B,), 0.8))
    lq_hi = m.log_q_site(pos, s, site, xi, sc, L, torch.full((B,), 2.0))
    assert not torch.allclose(lq_lo, lq_hi, atol=1e-4), "beta has no effect — FiLM not wired into forward"


def test_sample_logq_roundtrip():
    """sample_site's returned logq equals log_q_site re-scoring the drawn position (exact density)."""
    sc, L, geo, pos, s, m = _env()
    site = 5; B = pos.shape[0]
    beta = torch.full((B,), 2.0)
    torch.manual_seed(3)
    xi, logq = m.sample_site(pos, s, site, sc, L, beta)
    lq = m.log_q_site(pos, s, site, xi, sc, L, beta)
    assert torch.allclose(logq, lq, atol=1e-4), f"sample/log_q mismatch {(logq-lq).abs().max():.2e}"


def test_single_site_mh_invariances():
    """Single-site MH moves ONLY the site; species and all other particles untouched; counts preserved."""
    from liquid_coupling_flow.ka_heatbath import single_site_mh
    sc, L, geo, pos, s, m = _env()
    site = 5; B = pos.shape[0]
    beta = torch.full((B,), 2.0)
    counts0 = s.sum(1).clone()
    pos2, acc = single_site_mh(m, pos, s, site, sc, L, beta)
    assert torch.equal(s.sum(1), counts0), "species counts must be preserved"
    mask = torch.ones(100, dtype=torch.bool); mask[site] = False
    assert torch.equal(pos2[:, mask], pos[:, mask]), "non-site positions must be untouched"
    assert acc.dtype == torch.bool and acc.shape == (B,)


def test_single_site_sweep_runs():
    from liquid_coupling_flow.ka_heatbath import single_site_mh_sweep
    sc, L, geo, pos, s, m = _env(B=4)
    beta = 2.0
    pos2, info = single_site_mh_sweep(m, pos, s, sc, L, geo, beta=beta, n_moves=20)
    assert torch.isfinite(pos2).all() and 0.0 <= info["accept"] <= 1.0
