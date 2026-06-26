import torch
from liquid_coupling_flow.ka_flowhead import KAFlowHeadModel
from liquid_coupling_flow.ka_localframe import _wrap_pm


def test_inertial_rotation_is_orthonormal_unit_det():
    m = KAFlowHeadModel(rho=1.2, n_bins=192, knn=8, frame_mode="inertial"); m.eval()
    torch.manual_seed(0)
    nbr_rel = torch.randn(4, 12, 8, 2)                       # [B,N,k,2] neighbour offsets
    R = m._inertial_R(nbr_rel)
    assert R.shape[-2:] == (2, 2)
    eye = torch.eye(2).expand_as(R)
    assert torch.allclose(R.transpose(-1, -2) @ R, eye, atol=1e-4)     # orthonormal
    det = R[..., 0, 0] * R[..., 1, 1] - R[..., 0, 1] * R[..., 1, 0]
    assert torch.allclose(det, torch.ones_like(det), atol=1e-4)        # |det| = 1 (rotation)


def test_inertial_R_logprob_matches_sample_step():
    """Genuine R-consistency test: the per-particle inertial R in log_prob must equal the R
    built step-by-step in sample/_step, for EVERY particle including j=0,1 and j in [2,knn).
    Before the valid-mask + j<2-identity fix this test FAILS for early particles."""
    torch.manual_seed(42)
    B, N = 3, 12
    m = KAFlowHeadModel(rho=1.2, n_bins=192, knn=8, frame_mode="inertial"); m.eval()
    L = m._Lof(N)
    device = "cpu"

    # Build a fixed curve-ordered config: generate positions and sort by curve order.
    x_raw = torch.rand(B, N, 2) * L
    order = m.geo._curve_order(x_raw, N)
    xo = torch.gather(x_raw, 1, order[..., None].expand(-1, -1, 2))   # curve-ordered

    sc = m.geo._scaffold(N, device)

    # Origins from _local (needed by _inertial_R_logprob).
    _ctx, origin = m._local(xo, torch.zeros(B, N, dtype=torch.long), sc, L, N)

    # R stack from log_prob path (uses valid-mask + j<2 identity after fix).
    R_lp = m._inertial_R_logprob(xo, origin, L, N)      # [B,N,2,2]

    # R stack from sample/_step path (step-by-step, uses placed prefix).
    R_samp = m._inertial_R_sample_step(xo, sc, L, N)    # [B,N,2,2]

    assert R_lp.shape == (B, N, 2, 2)
    assert R_samp.shape == (B, N, 2, 2)
    assert torch.allclose(R_lp, R_samp, atol=1e-5), (
        f"Inertial R mismatch between log_prob and sample paths.\n"
        f"Max abs diff: {(R_lp - R_samp).abs().max().item():.2e}\n"
        f"Per-particle max diff: {(R_lp - R_samp).abs().amax(dim=(0,2,3))}"
    )


if __name__ == "__main__":
    test_inertial_rotation_is_orthonormal_unit_det()
    test_inertial_R_logprob_matches_sample_step()
    print("FLOWHEAD-FRAME TESTS PASSED")
