import torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_mtm import cluster_energy, mtm_move, assert_mtm_valid
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, _load, ART
from liquid_coupling_flow.ka_energy import ka_energy
import os
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def _env(B=4):
    sc, L, geo = _scaffold(100, DEV)
    ref = torch.load(os.path.join(ART, "ka_reference_N100.pt"), map_location=DEV, weights_only=False)
    pos, s = slot_order(ref["x"][:B].to(DEV), ref["s"].to(DEV).long(), geo, 100)
    cl = KC.cluster_slots(5, sc, 7, L)
    return sc, L, pos, s, cl

def test_cluster_energy_matches_full_difference():
    """U_clu(new)-U_clu(cur) must equal ka_energy(new)-ka_energy(cur) (rest-rest cancels)."""
    torch.manual_seed(0)
    sc, L, pos, s, cl = _env()
    xC_new = torch.remainder(pos[:, cl] + 0.3 * torch.randn_like(pos[:, cl]), L)
    pos_new = pos.clone(); pos_new[:, cl] = xC_new
    # ka_energy takes a single [N] species vector per call; post-slot_order `s` varies per batch row
    # (slot j holds whichever physical particle is nearest scaffold j, which differs config-to-config),
    # so each row must use its OWN species-per-slot mapping s[i], not s[0] broadcast over the batch.
    dU_full = torch.stack([ka_energy(pos_new[i:i + 1], s[i], L) - ka_energy(pos[i:i + 1], s[i], L)
                           for i in range(pos.shape[0])]).squeeze(-1)
    dU_clu = (cluster_energy(xC_new.unsqueeze(1), pos, cl, s, L)
              - cluster_energy(pos[:, cl].unsqueeze(1), pos, cl, s, L)).squeeze(1)
    assert torch.allclose(dU_full, dU_clu, atol=1e-3), (dU_full - dU_clu).abs().max()

def test_assert_mtm_valid_spline_pass_bins_fail():
    sc, L, pos, s, cl = _env()
    ck = torch.load(os.path.join(ART, "ka_cluster_flow_full_N100.pt"), map_location=DEV, weights_only=False)
    P = _load(ck, DEV)
    assert_mtm_valid(P, pos, s, cl, sc, L)                      # spline: passes
    ckb = torch.load(os.path.join(ART, "ka_cluster_flow_N100.pt"), map_location=DEV, weights_only=False)
    Pb = _load(ckb, DEV)
    import pytest
    with pytest.raises(AssertionError):
        assert_mtm_valid(Pb, pos, s, cl, sc, L)                 # bins: rejected

def test_mtm_move_semantics():
    torch.manual_seed(0)
    sc, L, pos, s, cl = _env()
    ck = torch.load(os.path.join(ART, "ka_cluster_flow_full_N100.pt"), map_location=DEV, weights_only=False)
    P = _load(ck, DEV)
    pos_new, moved, info = mtm_move(P, pos, s, cl, sc, L, M=8)
    assert pos_new.shape == pos.shape and torch.isfinite(pos_new).all()
    # non-cluster columns NEVER change
    mask = torch.ones(100, dtype=torch.bool, device=DEV); mask[cl] = False
    assert torch.equal(pos_new[:, mask], pos[:, mask])
    # unmoved chains keep their cluster exactly
    if (~moved).any():
        assert torch.equal(pos_new[~moved][:, cl], pos[~moved][:, cl])
    assert 0.0 <= info["move_prob"] <= 1.0 and info["w_ess"] >= 1.0

def test_mtm_move_chunked_equals_unchunked_shapes():
    torch.manual_seed(0)
    sc, L, pos, s, cl = _env(B=2)
    ck = torch.load(os.path.join(ART, "ka_cluster_flow_full_N100.pt"), map_location=DEV, weights_only=False)
    P = _load(ck, DEV)
    pos_new, moved, info = mtm_move(P, pos, s, cl, sc, L, M=8, chunk_rows=4)   # forces chunking path
    assert pos_new.shape == pos.shape and torch.isfinite(pos_new).all()
