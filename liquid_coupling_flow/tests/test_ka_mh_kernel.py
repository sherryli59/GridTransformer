import torch, pytest
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow import ka_mh_kernel as K

def _setup(B=4, N=12, seed=0):
    torch.manual_seed(seed); L = (N / 1.2) ** 0.5
    pos = (torch.rand(B, N, 2) * L).double()
    s = torch.zeros(N, dtype=torch.long); s[: N // 3] = 1
    s = s[torch.randperm(N)]
    return pos, s, L

def test_site_dE_matches_full_energy():
    pos, s, L = _setup()
    B, N = pos.shape[:2]; j = 5
    xj_new = torch.remainder(pos[:, j] + 0.3 * torch.randn(B, 2), L)
    dE = K.site_dE(pos, s, j, xj_new, L)
    pos2 = pos.clone(); pos2[:, j] = xj_new
    dE_full = ka_energy(pos2, s, L) - ka_energy(pos, s, L)
    assert torch.allclose(dE, dE_full, atol=1e-4), (dE - dE_full).abs().max()
