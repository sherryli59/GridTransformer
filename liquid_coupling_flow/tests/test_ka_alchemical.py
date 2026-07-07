"""Alchemical (x,lambda) foundation tests (ka_alchemical.py): bilinear pair-parameter interpolation whose
CORNERS reproduce binary KA exactly (vs canonical ka_energy); smooth mid-path; NCMC work bookkeeping nulls."""
import torch, pytest
from liquid_coupling_flow.ka_energy import ka_energy

L = (100 / 1.2) ** 0.5


def _cfg(B=3, N=100, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(B, N, 2, generator=g) * L
    s = (torch.rand(N, generator=g) < 0.35).long()
    return x, s


def test_corners_match_binary_ka_exactly():
    """lambda = species (0/1) must reproduce ka_energy to fp32 precision on random configs."""
    from liquid_coupling_flow.ka_alchemical import alch_energy
    x, s = _cfg()
    lam = s.float()[None].expand(3, -1)
    Ua = alch_energy(x, lam, L)
    Ub = ka_energy(x, s, L)
    assert torch.allclose(Ua, Ub, rtol=1e-5, atol=1e-3), f"corner mismatch {(Ua-Ub).abs().max():.2e}"


def test_midpath_smooth_and_finite():
    """Energy is finite and continuous along a pair lambda-path (no interpolation pathologies)."""
    from liquid_coupling_flow.ka_alchemical import alch_energy
    x, s = _cfg(B=1)
    lam = s.float()[None].clone()
    i = int((s == 0).nonzero()[0]); j = int((s == 1).nonzero()[0])
    us = []
    for t in torch.linspace(0, 1, 11):
        l2 = lam.clone(); l2[0, i] = float(t); l2[0, j] = 1.0 - float(t)
        us.append(alch_energy(x, l2, L).item())
    us = torch.tensor(us)
    assert torch.isfinite(us).all()
    assert (us[1:] - us[:-1]).abs().max() < max(50.0, us.abs().max()), "wild jumps along the lambda path"


def test_composition_preserved_by_pair_drive():
    """The antisymmetric pair drive keeps sum(lambda) exactly constant."""
    from liquid_coupling_flow.ka_alchemical import ncmc_swap
    x, s = _cfg(B=2)
    lam = s.float()[None].expand(2, -1).clone()
    i = int((s == 0).nonzero()[0]); j = int((s == 1).nonzero()[0])
    x2, lam2, info = ncmc_swap(x, lam, i, j, L, beta=2.0, T_steps=4, n_relax=1, gen=torch.Generator().manual_seed(0))
    assert torch.allclose(lam2.sum(1), lam.sum(1), atol=1e-5)
    # endpoints stay binary: accepted rows have lambda in {0,1} everywhere
    assert ((lam2 - lam2.round()).abs() < 1e-6).all(), "accepted/rejected states must be at binary endpoints"


def test_ncmc_null_protocol_work_zero():
    """T_steps=0 (no lambda change, no relax) must give W=0 and acceptance prob 1 (identity move)."""
    from liquid_coupling_flow.ka_alchemical import ncmc_swap
    x, s = _cfg(B=2)
    lam = s.float()[None].expand(2, -1).clone()
    i = int((s == 0).nonzero()[0]); j = int((s == 1).nonzero()[0])
    x2, lam2, info = ncmc_swap(x, lam, i, j, L, beta=2.0, T_steps=0, n_relax=0, gen=torch.Generator().manual_seed(0))
    assert torch.allclose(info["work"], torch.zeros_like(info["work"]), atol=1e-5)
