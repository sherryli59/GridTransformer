import torch

from liquid_coupling_flow.ka_pmc_3d import particle_energies, parallel_mc_disp
from liquid_coupling_flow.ka_energy import ka_energy


def test_particle_energies_match_ka_energy_3d():
    """The pmc move energetics use the SAME shifted min-image formula as ka_energy."""
    torch.manual_seed(0); B, N, L = 2, 24, 3.2
    x = torch.rand(B, N, 3) * L; s = torch.tensor([[0] * 16 + [1] * 8] * B)
    assert torch.allclose(ka_energy(x, s, L), 0.5 * particle_energies(x, s, L).sum(1), atol=1e-4)


def test_parallel_mc_zero_step_is_identity_3d():
    torch.manual_seed(1); B, N, L = 2, 16, 3.0
    x = torch.rand(B, N, 3) * L; s = torch.tensor([[0] * 10 + [1] * 6] * B)
    assert torch.equal(parallel_mc_disp(x, s, L, beta=2.0, step=0.0), x)


def test_parallel_mc_respects_mobile_mask_3d():
    """Frozen particles never move regardless of proposed noise."""
    torch.manual_seed(2); B, N, L = 2, 16, 3.0
    x = torch.rand(B, N, 3) * L; s = torch.tensor([[0] * 10 + [1] * 6] * B)
    mob = torch.zeros(B, N, dtype=torch.bool); mob[:, :4] = True
    x2 = parallel_mc_disp(x, s, L, beta=2.0, step=0.1, mobile=mob)
    assert torch.equal(x2[:, 4:], x[:, 4:])


def test_parallel_mc_lowers_energy_at_low_T_from_bad_start_3d():
    """Sanity: at low T a few sweeps from an overlapping start should not raise energy."""
    torch.manual_seed(3); B, N, L = 4, 64, (64 / 1.2) ** (1 / 3)
    x = torch.rand(B, N, 3) * L; s = torch.tensor([[0] * 51 + [1] * 13] * B)
    U0 = ka_energy(x, s, L)
    for _ in range(30):
        x = parallel_mc_disp(x, s, L, beta=2.0, step=0.05)
    assert (ka_energy(x, s, L) <= U0 + 1e-3).all()
