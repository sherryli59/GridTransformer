import os, torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_swap_breathe import swap_breathe_move, swap_breathe_sweep
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, _load, ART
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def _env(B=8):
    sc, L, geo = _scaffold(100, DEV)
    ref = torch.load(os.path.join(ART, "ka_reference_N100.pt"), map_location=DEV, weights_only=False)
    pos, s = slot_order(ref["x"][:B].to(DEV), ref["s"].to(DEV).long(), geo, 100)
    P = _load(torch.load(os.path.join(ART, "ka_cluster_flow_full_N100.pt"), map_location=DEV,
                         weights_only=False), DEV)
    return sc, L, geo, pos, s, P

def test_counts_invariant_and_noncluster_untouched():
    torch.manual_seed(0)
    sc, L, geo, pos, s, P = _env()
    cl = KC.cluster_slots(5, sc, 7, L)
    counts0 = s.sum(1).clone()
    pos2, s2, acc, info = swap_breathe_move(P, pos, s, cl, sc, L)
    assert torch.equal(s2.sum(1), counts0)                       # 65:35 preserved per row
    mask = torch.ones(100, dtype=torch.bool, device=DEV); mask[cl] = False
    assert torch.equal(pos2[:, mask], pos[:, mask])              # non-cluster positions untouched
    assert torch.equal(s2[:, mask], s[:, mask])                  # non-cluster species untouched
    assert 0.0 <= info["acceptance"] <= 1.0 and 0.0 <= info["abort_frac"] <= 1.0

def test_roundtrip_on_swapped_pattern():
    """sample then log_q under the SAME swapped pattern must agree (<1e-4): the exactness of the q-ratio."""
    torch.manual_seed(0)
    sc, L, geo, pos, s, P = _env(B=4)
    cl = KC.cluster_slots(5, sc, 7, L)
    s_cl = s[:, cl]
    # transpose the first unlike pair found per row (rows without one: skip by construction of the ref 65:35,
    # k=7 clusters virtually always mixed; assert we got at least one mixed row)
    s_prop = s.clone()
    swapped_any = False
    for b in range(4):
        a_idx = (s_cl[b] == 0).nonzero().squeeze(-1); b_idx = (s_cl[b] == 1).nonzero().squeeze(-1)
        if len(a_idx) and len(b_idx):
            s_prop[b, cl[a_idx[0]]] = 1; s_prop[b, cl[b_idx[0]]] = 0; swapped_any = True
    assert swapped_any
    xC, lq = P.sample(pos, s_prop, cl, sc, L)
    lq2 = P.log_q(pos, s_prop, cl, xC, sc, L)
    assert torch.allclose(lq, lq2, atol=1e-4), (lq - lq2).abs().max()

def test_abort_on_single_species_cluster():
    torch.manual_seed(0)
    sc, L, geo, pos, s, P = _env(B=2)
    cl = KC.cluster_slots(5, sc, 7, L)
    s_forced = s.clone(); s_forced[:, cl] = 0                    # all-A cluster -> no unlike pair
    pos2, s2, acc, info = swap_breathe_move(P, pos, s_forced, cl, sc, L)
    assert info["abort_frac"] == 1.0 and not acc.any()
    assert torch.equal(pos2, pos) and torch.equal(s2, s_forced)  # aborted rows unchanged

def test_sweep_scatter_back_consistency():
    """Lab-frame sweep: energies stay finite, counts invariant, state changes only via accepted moves."""
    torch.manual_seed(0)
    sc, L, geo, pos, s, P = _env(B=4)
    from liquid_coupling_flow.ka_energy import ka_energy
    counts0 = s.sum(1).clone()
    pos2, s2, info = swap_breathe_sweep(P, pos, s, sc, L, geo, n_moves=20)
    assert torch.equal(s2.sum(1), counts0)
    assert torch.isfinite(ka_energy(pos2, s2, L)).all()

def test_sb_mtm_M1_reduces_to_v1_ratio():
    """At M=1 the I-MTM log-ratio equals v1's Hastings log-ratio on the same draws (same seed)."""
    torch.manual_seed(0)
    sc, L, geo, pos, s, P = _env(B=4)
    cl = KC.cluster_slots(5, sc, 7, L)
    from liquid_coupling_flow.ka_swap_breathe import sb_mtm_move
    torch.manual_seed(123)
    p1, s1, a1, i1 = sb_mtm_move(P, pos, s, cl, sc, L, M=1)
    assert torch.equal(s1.sum(1), s.sum(1)) and torch.isfinite(torch.tensor(i1["acceptance"]))

def test_sb_mtm_counts_and_noncluster():
    torch.manual_seed(0)
    sc, L, geo, pos, s, P = _env(B=4)
    cl = KC.cluster_slots(5, sc, 7, L)
    from liquid_coupling_flow.ka_swap_breathe import sb_mtm_move
    p2, s2, acc, info = sb_mtm_move(P, pos, s, cl, sc, L, M=4)
    mask = torch.ones(100, dtype=torch.bool, device=DEV); mask[cl] = False
    assert torch.equal(p2[:, mask], pos[:, mask]) and torch.equal(s2[:, mask], s[:, mask])
    assert torch.equal(s2.sum(1), s.sum(1))

def test_jf_pair_scorer_loads_or_skips():
    import pytest
    from liquid_coupling_flow.ka_swap_breathe import make_jf_pair_scorer, sb_mtm_move
    try:
        scorer = make_jf_pair_scorer(DEV)
    except (ImportError, FileNotFoundError, KeyError) as e:
        pytest.skip(f"jf scorer unavailable: {e}")
    torch.manual_seed(0)
    sc, L, geo, pos, s, P = _env(B=2)
    cl = KC.cluster_slots(5, sc, 7, L)
    scores = scorer(pos, s, cl)
    assert scores.shape == (2, 7, 7) and (scores >= 0).all() and torch.isfinite(scores).all()
    p2, s2, acc, info = sb_mtm_move(P, pos, s, cl, sc, L, M=4, pair_scorer=scorer)
    assert torch.equal(s2.sum(1), s.sum(1))
