import torch
from liquid_coupling_flow.ka_flowhead import KAFlowHeadModel


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


def test_inertial_exactness_gate_preserved():
    m = KAFlowHeadModel(rho=1.2, n_bins=192, knn=8, frame_mode="inertial"); m.eval()
    pos, sp, logq = m.sample(4, 12, n_B=4, device="cpu", return_logq=True)
    assert torch.allclose(logq, m.log_prob(pos, sp), atol=1e-4)


if __name__ == "__main__":
    test_inertial_rotation_is_orthonormal_unit_det()
    test_inertial_exactness_gate_preserved()
    print("FLOWHEAD-FRAME TESTS PASSED")
