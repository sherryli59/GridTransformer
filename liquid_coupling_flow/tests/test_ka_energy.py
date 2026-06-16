import math, torch
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA, EPS


def _pair(r, a, b):  # one-pair shifted LJ, analytic
    s, e, rc = SIGMA[a][b], EPS[a][b], 2.5 * SIGMA[a][b]
    if r >= rc:
        return 0.0
    sr6 = (s / r) ** 6
    src6 = (s / rc) ** 6
    return 4 * e * (sr6 ** 2 - sr6) - 4 * e * (src6 ** 2 - src6)


def test_two_body_all_species():
    L = 20.0
    for (a, b) in [(0, 0), (0, 1), (1, 1)]:
        x = torch.tensor([[[0.0, 0.0], [1.05, 0.0]]], dtype=torch.float64)
        s = torch.tensor([a, b], dtype=torch.int8)
        got = ka_energy(x, s, L).item()
        assert abs(got - _pair(1.05, a, b)) < 1e-9, (a, b, got)


def test_translation_invariant():
    # jittered 4x3 grid (no near-overlaps -> r^-12 stays O(1), round-off ~1e-15).
    # random configs would trip this via near-contacts where r^-12 ~ 1e12.
    torch.manual_seed(0)
    L = 10.0
    cx = (torch.arange(4) + 0.5) * L / 4
    cy = (torch.arange(3) + 0.5) * L / 3
    grid = torch.stack(torch.meshgrid(cx, cy, indexing="ij"), -1).reshape(-1, 2)
    x = (grid + 0.15 * torch.randn(12, 2)).to(torch.float64)[None]
    s = (torch.arange(12) % 3 == 0).to(torch.int8)  # ~33% B
    e0 = ka_energy(x, s, L)
    e1 = ka_energy(torch.remainder(x + 3.3, L), s, L)
    assert (e0 - e1).abs().max() < 1e-9


def test_per_particle_sums_to_total():
    torch.manual_seed(1)
    L = 10.0
    x = torch.rand(2, 16, 2, dtype=torch.float64) * L
    s = (torch.rand(16) < 0.35).to(torch.int8)
    tot = ka_energy(x, s, L)
    per = ka_energy(x, s, L, per_particle=True)   # [B,N]; sum over i = 2*total
    assert (per.sum(-1) - 2 * tot).abs().max() < 1e-9


if __name__ == "__main__":
    test_two_body_all_species()
    test_translation_invariant()
    test_per_particle_sums_to_total()
    print("KA ENERGY TESTS PASSED")
