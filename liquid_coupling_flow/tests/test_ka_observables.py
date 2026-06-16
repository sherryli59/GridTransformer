import math, torch
from liquid_coupling_flow.ka_observables import partial_gr, psi6
from liquid_coupling_flow.ka_mcmc import make_species


def test_ideal_gas_partial_gr_flat():
    torch.manual_seed(0)
    N, L = 80, 12.0
    s = make_species(N, 0.35)
    x = torch.rand(6000, N, 2) * L
    rc, g = partial_gr(x, s, L, rmax=L / 2, nbins=60, pair=(0, 0))
    plateau = g[rc > 0.6 * (L / 2)].mean().item()
    assert 0.9 < plateau < 1.1, plateau


def test_psi6_perfect_triangular_is_one():
    # triangular lattice patch -> |psi6| ~ 1 for interior particles
    rows = []
    a = 1.0
    for j in range(8):
        for i in range(8):
            rows.append([i * a + (j % 2) * a / 2, j * a * math.sqrt(3) / 2])
    x = torch.tensor(rows)[None]  # [1,64,2]
    L = 100.0  # large box, no PBC effects on interior
    val = psi6(x, L)              # [1] mean |psi6| over particles
    assert val.item() > 0.9, val.item()


if __name__ == "__main__":
    test_ideal_gas_partial_gr_flat()
    test_psi6_perfect_triangular_is_one()
    print("KA OBSERVABLES TESTS PASSED")
