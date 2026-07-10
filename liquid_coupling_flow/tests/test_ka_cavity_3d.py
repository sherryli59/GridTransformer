import torch

from liquid_coupling_flow.ka_cavity import cavity_inside
from liquid_coupling_flow.ka_cavity_3d import local_identity_swap
from liquid_coupling_flow.ka_energy import ka_energy


def test_ka_energy_and_hard_wall_support_3d():
    x = torch.tensor([[[5., 5., 5.], [7.9, 5., 5.], [8., 5., 5.]]])
    assert torch.equal(cavity_inside(x, torch.tensor([5., 5., 5.]), 3., 10.), torch.tensor([[True, True, False]]))
    assert torch.isfinite(ka_energy(x, torch.tensor([[0, 1, 0]]), 10.)).all()


def test_local_3d_identity_swap_tracks_exact_energy():
    torch.manual_seed(3); B, N, L = 3, 10, 9.
    x = torch.rand(B, N, 3) * L; s = torch.tensor([[0] * 5 + [1] * 5] * B); mobile = torch.ones(B, N, dtype=torch.bool)
    s2, U2, _ = local_identity_swap(x, s, ka_energy(x, s, L), mobile, 2., L)
    assert torch.allclose(U2, ka_energy(x, s2, L), atol=2e-4, rtol=2e-5)
