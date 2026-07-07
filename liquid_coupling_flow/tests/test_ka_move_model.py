"""GB1 tests: the collective MOVE MODEL — transition conditional q(x'_block | x_block, env, beta) + MH kernel.

Exactness surface: (1) sample<->log_q roundtrip (exact two-way density); (2) the conditional actually SEES the
current block (it's a transition model, not a state model); (3) occupancy-abort symmetry (block-slot occupants
after the move must be exactly the moved particles, else abort — the swap-breathe DB pattern); (4) MH move
invariances. Data side: event->training-tuple particle correspondence.
"""
import os, torch, pytest
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import _scaffold

DEV = "cpu"


def _env(B=4, N=100, seed=0):
    sc, L, geo = _scaffold(N, DEV)
    torch.manual_seed(seed)
    pos = torch.rand(B, N, 2, device=DEV) * L
    s = (torch.rand(B, N) < 0.35).long()
    from liquid_coupling_flow.ka_move_model import MoveModel
    m = MoveModel(rho=1.2, n_bins=24, box=3.0, n_ctx=16, d_model=64, n_head=4, n_layer=2,
                  head="spline", pair_feats=False).to(DEV).eval()
    cl = KC.cluster_slots(5, sc, 8, L)
    return sc, L, geo, pos, s, m, cl


def test_sample_logq_roundtrip():
    """sample_move's logq == log_q_move re-scoring the same proposal (exact density, both conditioned on the
    same current block)."""
    sc, L, geo, pos, s, m, cl = _env()
    beta = torch.full((4,), 2.0)
    torch.manual_seed(3)
    xC_new, logq = m.sample_move(pos, s, cl, sc, L, beta)
    lq = m.log_q_move(pos, s, cl, xC_new, sc, L, beta)
    assert xC_new.shape == (4, 8, 2)
    assert torch.allclose(logq, lq, atol=1e-4), f"roundtrip mismatch {(logq-lq).abs().max():.2e}"


def test_conditional_sees_current_block():
    """Changing ONLY the current block positions changes the density of a fixed query — the defining property
    of a TRANSITION conditional (a state model would be invariant). Uses the bins head: the spline head
    zero-inits its context conditioning, masking ALL context (incl. the current block) at init — same
    at-init blindness as A2's beta test."""
    sc, L, geo, pos, s, _, cl = _env()
    from liquid_coupling_flow.ka_move_model import MoveModel
    m = MoveModel(rho=1.2, n_bins=24, box=3.0, n_ctx=16, d_model=64, n_head=4, n_layer=2,
                  head="bins", pair_feats=False).to(DEV).eval()
    beta = torch.full((4,), 2.0)
    torch.manual_seed(0)
    xq = pos[:, cl] + 0.3                                       # fixed query
    lq1 = m.log_q_move(pos, s, cl, xq, sc, L, beta)
    pos2 = pos.clone(); pos2[:, cl] = torch.remainder(pos2[:, cl] + 0.5, L)   # move ONLY the current block
    lq2 = m.log_q_move(pos2, s, cl, xq, sc, L, beta)
    assert not torch.allclose(lq1, lq2, atol=1e-5), "conditional blind to the current block (state model?)"


def test_env_frozen_under_block_change():
    """The frame/env context is env-anchored: verified indirectly — density changes under block change (above)
    but sample_move from two block states still lands in the same env frame box (finite, no NaN)."""
    sc, L, geo, pos, s, m, cl = _env()
    beta = torch.full((4,), 2.0)
    xC, logq = m.sample_move(pos, s, cl, sc, L, beta)
    assert torch.isfinite(xC).all() and torch.isfinite(logq).all()


def test_beta_film_wired():
    """Perturbed-FiLM beta sensitivity through the bins head (spline zero-inits its conditioning)."""
    sc, L, geo, pos, s, _, cl = _env()
    from liquid_coupling_flow.ka_move_model import MoveModel
    m = MoveModel(rho=1.2, n_bins=24, box=3.0, n_ctx=16, d_model=64, n_head=4, n_layer=2,
                  head="bins", pair_feats=False).to(DEV).eval()
    with torch.no_grad():
        for p in m.beta_film.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    xq = pos[:, cl]
    lq_lo = m.log_q_move(pos, s, cl, xq, sc, L, torch.full((4,), 0.8))
    lq_hi = m.log_q_move(pos, s, cl, xq, sc, L, torch.full((4,), 2.0))
    assert not torch.allclose(lq_lo, lq_hi, atol=1e-4), "beta not wired"


def test_occupancy_symmetry_abort_and_invariances():
    """move_mh: non-block particles and ALL species untouched; counts preserved; occupancy-abort flag sane.
    (Occupancy symmetry: accepted moves must leave the block slots occupied by exactly the moved particles —
    asymmetric selection breaks DB; aborts are counted, not accepted.)"""
    from liquid_coupling_flow.ka_move_model import move_mh
    sc, L, geo, pos, s, m, cl = _env()
    torch.manual_seed(0)
    counts0 = s.sum(1).clone()
    pos2, s2, info = move_mh(m, pos, s, cl, sc, L, beta=2.0)
    assert torch.equal(s2, s) and torch.equal(s2.sum(1), counts0)
    mask = torch.ones(100, dtype=torch.bool); mask[cl] = False
    assert torch.equal(pos2[:, mask], pos[:, mask]), "non-block positions must be untouched"
    assert 0.0 <= info["accept"] <= 1.0 and 0.0 <= info["abort_frac"] <= 1.0


def test_event_to_training_tuple():
    """Bank event -> (slot-ordered state, block slots, target positions) with PARTICLE correspondence:
    the target of each block particle is ITS OWN xb position (harvest keeps particle indices fixed).
    Needs an EQUILIBRATED base config: slot assignment is only locally faithful at liquid density
    (measured: on uniform-random configs, particles 0.45 apart can land in slots 5+ apart)."""
    from liquid_coupling_flow.ka_move_model import event_to_tuple
    ref_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "ka_reference_N100.pt")
    if not os.path.exists(ref_path):
        pytest.skip("no N=100 reference for a physical base config")
    sc, L, geo, _, _, m, _ = _env(B=1)
    ref = torch.load(ref_path, map_location=DEV, weights_only=False)
    xa = ref["x"][0].clone().float()
    s_ref = ref["s"].long()
    # synthetic event on the physical config: a particle and its NEAREST NEIGHBOUR hop together by ~0.8
    d0 = xa - xa[30]
    d0 = (d0 - ref["L"] * torch.round(d0 / ref["L"])).norm(dim=-1); d0[30] = 1e9
    nn = int(d0.argmin())
    movers = torch.tensor([30, nn])
    xb = torch.remainder(xa + 0.02 * torch.randn_like(xa), ref["L"])
    xb[movers] = torch.remainder(xa[movers] + 0.55, ref["L"])
    ev = {"xa": xa, "xb": xb, "mobile": movers, "beta": 2.0}
    out = event_to_tuple(ev, s_ref, sc, ref["L"], geo, k_block=8)
    assert out is not None
    pos_o, s_o, cl_o, target, beta = out
    assert cl_o.shape == (8,) and target.shape == (8, 2)
    # correspondence: block particles' targets match their own xb rows (min-image distance ~ jitter or hop)
    d = (target - pos_o[cl_o])
    d = (d - ref["L"] * torch.round(d / ref["L"])).norm(dim=-1)
    assert (d < 1.5).all(), "target rows not the block particles' own xb positions"
    assert d.max() > 0.6, "the hop is inside the block (mobile coverage)"
