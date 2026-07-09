import torch
from liquid_coupling_flow.ka_pin import pin_mask, masked_mala
from liquid_coupling_flow.ka_energy import ka_energy, ka_forces

def _setup(B=4, N=64, dev="cpu", seed=0):
    torch.manual_seed(seed)
    L = (N / 1.2) ** 0.5
    x = torch.rand(B, N, 2, device=dev) * L
    s = (torch.rand(B, N, device=dev) < 0.37).long()
    return x, s, L

def test_pin_mask_count_and_frozen_fixed():
    x, s, L = _setup()
    mobile = pin_mask(4, 64, c=0.25, device="cpu", generator=torch.Generator().manual_seed(1))
    assert mobile.dtype == torch.bool and mobile.shape == (4, 64)
    # exactly ceil(c*N)=16 pinned per row
    assert int((~mobile[0]).sum()) == 16
    U = ka_energy(x, s, L)
    efn = lambda a, b: ka_energy(a, b, L); ffn = lambda a, b: ka_forces(a, b, L)
    x2, U2, acc = masked_mala(x, s, U, mobile, beta=2.0, L=L, dt=0.01, energy_fn=efn, force_fn=ffn)
    # frozen particles are bit-for-bit unchanged
    assert torch.equal(x2[~mobile], x[~mobile])
    # energy recomputed from x2 equals the tracked U2 (accept bookkeeping is exact)
    assert torch.allclose(ka_energy(x2, s, L), U2, atol=1e-4)
    assert 0.0 <= acc <= 1.0
