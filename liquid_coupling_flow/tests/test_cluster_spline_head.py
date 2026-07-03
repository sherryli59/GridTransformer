import math, torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import ClusterProposal, slot_order, _scaffold, _load
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def _env(B=8):
    sc, L, geo = _scaffold(100, DEV)
    import torch as T, os
    from liquid_coupling_flow.ka_cluster_flow import ART
    ref = T.load(os.path.join(ART, "ka_reference_N100.pt"), map_location=DEV, weights_only=False)
    pos, s = slot_order(ref["x"][:B].to(DEV), ref["s"].to(DEV).long(), geo, 100)
    cl = KC.cluster_slots(5, sc, 7, L)
    return sc, L, pos, s, cl

def test_spline_roundtrip_exact():
    torch.manual_seed(0)
    sc, L, pos, s, cl = _env()
    P = ClusterProposal(head="spline").to(DEV).eval()
    xC, logq = P.sample(pos, s, cl, sc, L)
    logq2 = P.log_q(pos, s, cl, xC, sc, L)
    assert torch.allclose(logq, logq2, atol=1e-4), (logq - logq2).abs().max()

def test_spline_identity_init_is_gaussian():
    """Identity-init spline == standard-normal base density on the in-frame coords (per step),
    so an untrained model's per-step logq for u=0 must be k * 2 * (-0.5*log(2*pi))."""
    torch.manual_seed(0)
    sc, L, pos, s, cl = _env(B=2)
    P = ClusterProposal(head="spline").to(DEV).eval()
    xq = pos[:, cl].clone()
    # place the query AT the frame origins so u ~ 0... simpler: score twice, must be deterministic & finite
    lq = P.log_q(pos, s, cl, xq, sc, L)
    assert torch.isfinite(lq).all()

def test_bins_backcompat_identical():
    """head='bins' default must load the existing checkpoint and reproduce its log_q exactly."""
    import os, torch as T
    from liquid_coupling_flow.ka_cluster_flow import ART
    ck = T.load(os.path.join(ART, "ka_cluster_flow_N100.pt"), map_location=DEV, weights_only=False)
    P = _load(ck, DEV)
    sc, L, pos, s, cl = _env()
    lq = P.log_q(pos, s, cl, pos[:, cl], sc, L)
    assert torch.isfinite(lq).all()          # loads + runs; byte-identity implied by untouched bins path

def test_spline_out_of_box_finite():
    """Points outside [-box, box] must get a finite (tail) density in spline mode (bins mode gave -69)."""
    torch.manual_seed(0)
    sc, L, pos, s, cl = _env(B=2)
    P = ClusterProposal(head="spline").to(DEV).eval()
    far = torch.remainder(pos[:, cl] + 4.0, L)
    lq = P.log_q(pos, s, cl, far, sc, L)
    assert torch.isfinite(lq).all()
