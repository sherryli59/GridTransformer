import math, torch
from liquid_coupling_flow import ka_cluster_flow_b as B
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def test_anchor_base_logp_matches_closed_form():
    torch.manual_seed(0)
    q = torch.randn(4, 7, 2, device=DEV); z = q + 0.3 * torch.randn(4, 7, 2, device=DEV); sb = 0.747
    got = B.anchor_base_logp(z, q, sb)
    d2 = ((z - q) ** 2).sum(-1)                                   # [B,k]
    want = (-d2 / (2 * sb ** 2) - math.log(2 * math.pi * sb ** 2)).sum(-1)
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()

def test_sample_base_stats():
    q = torch.zeros(2000, 7, 2, device=DEV); sb = 0.747
    z = B.sample_base(q, sb)
    assert z.std().item() == __import__("pytest").approx(sb, rel=0.05)
    assert z.mean().abs().item() < 0.05
