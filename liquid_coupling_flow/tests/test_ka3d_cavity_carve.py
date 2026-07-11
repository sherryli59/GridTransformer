import pytest
import torch

from liquid_coupling_flow.ka3d_cavity_carve import carve, carve_batch, reassemble


def _bulk(B=32, N=512):
    torch.manual_seed(0); L = (N / 1.2) ** (1 / 3)
    x = torch.rand(B, N, 3) * L
    s = torch.zeros(B, N, dtype=torch.long); s[:, -round(.2 * N):] = 1
    return x, s, L


def test_carve_reassembles_exactly_and_partitions_species():
    x, s, L = _bulk(B=1)
    pair = carve(x[0], s[0], torch.tensor([L / 2] * 3), 2.2, L)
    xr, sr = reassemble(pair)
    assert torch.equal(xr, x[0]) and torch.equal(sr, s[0])
    assert pair["n_in"] + pair["x_out"].shape[0] == x.shape[1]
    assert torch.equal(pair["s_in"], s[0, pair["mobile"]])


def test_carve_batch_count_matches_uniform_density_in_mean():
    x, s, L = _bulk(B=64)
    pairs = carve_batch(x, s, L, [2.0], centers_per_config=2,
                        rng=torch.Generator().manual_seed(3))
    mean = sum(p["n_in"] for p in pairs) / len(pairs)
    expected = 1.2 * (4 / 3) * torch.pi * 2.0 ** 3
    assert abs(mean - float(expected)) / float(expected) < .08
    assert all(p["R"] == 2.0 for p in pairs)


def test_carve_rejects_periodically_oversized_radius():
    x, s, L = _bulk(B=1)
    with pytest.raises(AssertionError):
        carve(x[0], s[0], torch.zeros(3), 3.2, L)
