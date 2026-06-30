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

def test_conditioner_masks_active_c():
    """Params for active particles must NOT depend on their own transformed coord c (coupling invertibility)."""
    torch.manual_seed(0)
    cond = B.ClusterConditioner(num_bins=8, n_species=2).to(DEV).eval()
    u = torch.randn(3, 7, 2, device=DEV); uc = torch.randn(3, 16, 2, device=DEV)
    sp = torch.randint(0, 2, (3, 7), device=DEV); spc = torch.randint(0, 2, (3, 16), device=DEV)
    b = 0; par, c = cond.meta[b]
    p0 = cond.params(b, u, uc, sp, spc, c, par)
    u2 = u.clone(); amask = (torch.arange(7, device=DEV) % 2 == par)
    u2[:, amask, c] += 5.0                                        # perturb ONLY the active, transformed coord
    p1 = cond.params(b, u2, uc, sp, spc, c, par)
    assert torch.allclose(p0, p1, atol=1e-5), (p0 - p1).abs().max()   # params unchanged
    # control: perturbing the OTHER coord DOES change params
    u3 = u.clone(); u3[:, amask, 1 - c] += 0.5
    assert (cond.params(b, u3, uc, sp, spc, c, par) - p0).abs().max() > 1e-3
