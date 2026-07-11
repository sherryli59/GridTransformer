import torch

from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_cavity_eval import assemble_generated, compare_overlap_distributions


def test_assemble_generated_preserves_boundary_and_species():
    torch.manual_seed(0); N = 256; L = (N / 1.2) ** (1 / 3)
    x = torch.rand(N, 3) * L; s = torch.zeros(N, dtype=torch.long); s[-51:] = 1
    center = x[0].clone(); pair = carve(x, s, center, 1.6, L)
    rel = pair["x_in"] - center; rel -= L * torch.round(rel / L)
    xf, sf = assemble_generated(pair, rel[None].expand(3, -1, -1).clone())
    assert torch.allclose(xf[0], x) and torch.equal(sf[0], s)
    assert torch.equal(xf[:, pair["idx_out"]], pair["x_out"][None].expand(3, -1, -1))


def test_distribution_comparison_detects_match_and_shift():
    q = torch.tensor([.2, .3, .4, .5, .6, .7])
    same = compare_overlap_distributions(q, q.clone(), bins=5)
    shifted = compare_overlap_distributions((q + .2).clamp_max(1), q, bins=5)
    assert same["mean_abs_diff"] == 0 and same["histogram_tv"] == 0
    assert shifted["mean_abs_diff"] > .1 and shifted["histogram_tv"] > 0
