# liquid_coupling_flow/tests/test_curveflow.py
import math, torch
from liquid_coupling_flow.ka_curveflow import KACurveFlowModel


def _tiny(seed=0):
    torch.manual_seed(seed)
    m = KACurveFlowModel(rho=1.2, n_bins=192, knn=8); m.eval()   # knn<N=12
    return m


def test_curve_feat_zero_at_init():
    # Zero-initialized curve projection => curve feature is exactly zero => KACurveFlowModel == B-alone at init.
    m = _tiny(0)
    cf = m._curve_feat(12, "cpu")
    assert cf.shape == (12, m.d_model)
    assert cf.abs().max().item() == 0.0


def test_curve_feat_deterministic_per_index():
    # The curve feature depends only on (N, j) — same for every config (it's the GPS coordinate).
    m = _tiny(1)
    with torch.no_grad():                                        # make it nonzero
        for p in m.curve_proj.parameters():
            p.add_(0.1 * torch.randn_like(p))
    a = m._curve_feat(12, "cpu"); b = m._curve_feat(12, "cpu")
    assert torch.equal(a, b)
    assert not torch.allclose(a[3], a[7])                        # different curve positions differ


def test_exactness_gate_preserved():
    m = _tiny(2); N, B = 12, 4
    with torch.no_grad():
        for p in m.curve_proj.parameters():
            p.add_(0.05 * torch.randn_like(p))                  # nonzero curve conditioning
    pos, sp, logq = m.sample(B, N, n_B=4, device="cpu", return_logq=True)
    assert torch.allclose(logq, m.log_prob(pos, sp), atol=1e-4)


def test_log_prob_matches_b_alone_when_curve_zero():
    # With the curve feature zero (init), log_prob must equal the same computation without the curve term.
    import torch.nn.functional as F
    from liquid_coupling_flow.ka_localframe import _wrap_pm
    m = _tiny(3); N, B = 12, 3
    x = torch.rand(B, N, 2) * m._Lof(N); s = (torch.rand(B, N) < 0.35).long()
    order = m.geo._curve_order(x, N)
    xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s, 1, order)
    ctx, origin = m._local(xo, so, m.geo._scaffold(N, x.device), m._Lof(N), N)
    ab = _wrap_pm(xo - origin, m._Lof(N)) / m._arc_scale(N)
    lp_ab = m.flow.log_prob(ctx, ab)                            # B-alone conditioning (curve feat is zero)
    sl = m.head_species(ctx)
    oh = F.one_hot(so, m.n_species).float(); rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)
    lp_s = F.log_softmax(sl.masked_fill(rem <= 0, float("-inf")), -1).gather(-1, so[..., None]).squeeze(-1)
    expected = (lp_ab + lp_s).sum(1) - m.d * N * math.log(m._arc_scale(N))
    assert torch.allclose(m.log_prob(x, s), expected, atol=1e-4)


if __name__ == "__main__":
    test_curve_feat_zero_at_init()
    test_curve_feat_deterministic_per_index()
    test_exactness_gate_preserved()
    test_log_prob_matches_b_alone_when_curve_zero()
    print("CURVEFLOW TESTS PASSED")
