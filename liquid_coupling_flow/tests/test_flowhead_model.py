# liquid_coupling_flow/tests/test_flowhead_model.py
import torch
from liquid_coupling_flow.ka_flowhead import KAFlowHeadModel


def _tiny(seed=0):
    torch.manual_seed(seed)
    m = KAFlowHeadModel(rho=1.2, n_bins=192, knn=8); m.eval()   # knn<N=12 required by _local topk
    return m


def test_exactness_gate_sample_logq_equals_log_prob():
    m = _tiny(0); N, B = 12, 4
    pos, sp, logq = m.sample(B, N, n_B=4, device="cpu", return_logq=True)
    assert torch.allclose(logq, m.log_prob(pos, sp), atol=1e-4)


def test_log_prob_finite_and_differentiable():
    m = _tiny(1); N, B = 12, 4
    x = torch.rand(B, N, 2) * m._Lof(N); s = (torch.rand(B, N) < 0.35).long()
    lp = m.log_prob(x, s)
    assert lp.shape == (B,) and torch.isfinite(lp).all()
    (-lp.mean()).backward()
    assert m.flow.head_a.weight.grad is not None and torch.isfinite(m.flow.head_a.weight.grad).all()


def test_no_vol_term_uses_flow_density():
    # The flow log_prob must equal (flow_pos_logdensity + species_logp).sum - arc_scale_jac, with NO bin_w
    # volume term (that was categorical-only). Recompute the expected value from the flow directly.
    import math
    import torch.nn.functional as F
    from liquid_coupling_flow.ka_localframe import _wrap_pm
    m = _tiny(2); N, B = 12, 3
    x = torch.rand(B, N, 2) * m._Lof(N); s = (torch.rand(B, N) < 0.35).long()
    order = m.geo._curve_order(x, N)
    xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s, 1, order)
    ctx, origin = m._local(xo, so, m.geo._scaffold(N, x.device), m._Lof(N), N)
    ab = _wrap_pm(xo - origin, m._Lof(N)) / m._arc_scale(N)
    lp_ab = m.flow.log_prob(ctx, ab)
    sl = m.head_species(ctx)
    oh = F.one_hot(so, m.n_species).float(); rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)
    lp_s = F.log_softmax(sl.masked_fill(rem <= 0, float("-inf")), -1).gather(-1, so[..., None]).squeeze(-1)
    expected = (lp_ab + lp_s).sum(1) - m.d * N * math.log(m._arc_scale(N))   # NO bin_w vol term
    assert torch.allclose(m.log_prob(x, s), expected, atol=1e-4)


if __name__ == "__main__":
    test_exactness_gate_sample_logq_equals_log_prob()
    test_log_prob_finite_and_differentiable()
    test_no_vol_term_uses_flow_density()
    print("FLOWHEAD-MODEL TESTS PASSED")
