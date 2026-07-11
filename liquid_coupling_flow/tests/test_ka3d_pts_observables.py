import pytest
import torch
from liquid_coupling_flow.ka3d_pts_observables import core_overlap, overlap_pdf, pts_susceptibility


def test_identical_configs_overlap_near_one():
    torch.manual_seed(0); B,N,L=3,80,4.0; Y=torch.rand(B,N,3)*L; s=torch.tensor([[0]*64+[1]*16]*B)
    centers = Y[:, 0].clone()  # guarantee at least one core particle in every row
    assert (core_overlap(Y,s,Y.clone(),s,centers,L,0.7,0.2) > 0.9).all()


def test_empty_core_is_guarded_to_zero():
    x = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
    s = torch.tensor([[0, 1]])
    q = core_overlap(x, s, x.clone(), s, torch.tensor([0.5, 0.5, 0.5]), 10.0, r_c=0.1)
    assert torch.equal(q, torch.zeros_like(q))


def test_independent_configs_low_overlap():
    torch.manual_seed(1); B,N,L=3,80,4.0
    X=torch.rand(B,N,3)*L; Y=torch.rand(B,N,3)*L; s=torch.tensor([[0]*64+[1]*16]*B)
    assert (core_overlap(X,s,Y,s,torch.tensor([L/2]*3),L,0.7,0.2) < 0.6).all()


def test_susceptibility_is_disorder_mean_of_thermal_variance():
    q = torch.tensor([[0.2,0.8,0.5],[0.9,0.85,0.95]])          # [2 centers, 3 pairs]
    expected = float(q.var(dim=1, unbiased=False).mean())       # per-center var, then mean
    assert abs(pts_susceptibility(q) - expected) < 1e-6


def test_batched_centers_and_pdf_density():
    torch.manual_seed(4); B, N, L = 2, 40, 4.0
    x = torch.rand(B, N, 3) * L
    centers = x[:, 0].clone()  # guaranteed non-empty cores, one distinct center per row
    s = torch.tensor([[0] * 32 + [1] * 8] * B)
    q = core_overlap(x, s, x.clone(), s, centers, L)
    assert torch.allclose(q, torch.ones_like(q), atol=1e-6)
    hist = overlap_pdf(torch.tensor([0.1, 0.2, 0.8, 0.9]), torch.linspace(0, 1, 6))
    assert hist.shape == (5,)
    assert abs(float(hist.sum() * 0.2) - 1.0) < 1e-6
    with pytest.raises(ValueError):
        overlap_pdf(torch.tensor([]), 5)
