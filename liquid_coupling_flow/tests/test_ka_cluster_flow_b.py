import math, torch
from liquid_coupling_flow import ka_cluster_flow_b as B
from liquid_coupling_flow import ka_cluster as KC
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def _setup(Bsz=4):
    sc, L, geo = B._scaffold(100, DEV)
    ref = torch.load(f"{B.ART}/ka_reference_N100.pt", map_location=DEV, weights_only=False)
    pos, s = B.slot_order(ref["x"][:Bsz].to(DEV), ref["s"].to(DEV).long(), geo, 100)
    cl = KC.cluster_slots(5, sc, 7, L)
    return sc, L, pos, s, cl

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

def test_sampler_equals_scorer():
    """PRIMARY exactness guard: the density the sampler reports == log_q at the sampled point (untrained ok)."""
    sc, L, pos, s, cl = _setup()
    flow = B.ClusterFlow(sigma_b=0.747).to(DEV).eval()
    xC, logq = flow.sample(pos, s, cl, sc, L)
    logq2 = flow.log_q(pos, s, cl, xC, sc, L)
    assert xC.shape == (4, 7, 2)
    assert torch.allclose(logq, logq2, atol=1e-3), (logq - logq2).abs().max()

def test_logdet_matches_autograd():
    """Analytic flow log-det == autograd jacobian log|det| of xC_query -> log_q's z-mapping, small batch."""
    sc, L, pos, s, cl = _setup(Bsz=1)
    flow = B.ClusterFlow(sigma_b=0.747).to(DEV).eval()
    xC, _ = flow.sample(pos, s, cl, sc, L)
    x0 = xC[0].reshape(-1).double().requires_grad_(True)        # [2k]
    def to_z(xflat):                                            # map query -> base z (the flow inverse), in-frame
        return flow._x_to_z(pos[:1].double(), s[:1], cl, xflat.reshape(1, 7, 2), sc, L)[0].reshape(-1)
    J = torch.autograd.functional.jacobian(to_z, x0)           # [2k,2k]
    logdet_auto = torch.linalg.slogdet(J)[1]
    logdet_anal = flow._x_to_z(pos[:1].double(), s[:1], cl, xC[:1].double(), sc, L)[1][0]   # returns (z, sum_logdet)
    assert abs(logdet_auto.item() - logdet_anal.item()) < 1e-3

def test_frame_independent_of_xC_bit_exact():
    """Jacobian-1 exactness: moving the whole cluster leaves the frame identical to the bit."""
    sc, L, pos, s, cl = _setup(Bsz=1)
    o0, R0 = KC.cluster_frame(pos[0], cl, sc, L)
    p2 = pos[0].clone(); p2[cl] = torch.remainder(p2[cl] + torch.tensor([1.7, -1.1], device=DEV), L)
    o1, R1 = KC.cluster_frame(p2, cl, sc, L)
    assert (o1 - o0).abs().max().item() == 0.0 and (R1 - R0).abs().max().item() == 0.0
