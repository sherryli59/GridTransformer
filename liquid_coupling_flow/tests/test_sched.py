# liquid_coupling_flow/tests/test_sched.py
import math, torch
from liquid_coupling_flow.ka_sched import arc_pT, KALocalFrameSched


def _tiny(seed=0):
    torch.manual_seed(seed)
    m = KALocalFrameSched(rho=1.2, n_bins=192, knn=8); m.eval()  # knn<N=12 required by _local topk
    N, B = 12, 2
    x = torch.rand(B, N, 2) * m._Lof(N); s = (torch.rand(B, N) < 0.35).long()
    return m, x, s, N, B


def test_arc_pT_schedule():
    assert abs(arc_pT(0.0) - 1.0) < 1e-9
    assert abs(arc_pT(1.0, r=2.0, floor=0.5) - 0.5) < 1e-9          # floored
    assert abs(arc_pT(0.5, r=2.0, floor=0.0) - math.sqrt(3) / 2) < 1e-6
    assert arc_pT(0.3) >= arc_pT(0.6)                              # monotone non-increasing


def test_p_keep_one_parity_with_base_log_prob():
    m, x, s, N, B = _tiny(1)
    d = 2; vol = d * N * math.log(m.bin_w); jac = d * N * math.log(m._arc_scale(N))
    torch.manual_seed(123)
    lp = m.log_prob_sched(x, s, p_keep=1.0)                        # no self-conditioning
    assert torch.allclose(lp, m.log_prob(x, s) + vol + jac, atol=1e-4)


def test_inference_exactness_gate_preserved():
    m, _, _, N, B = _tiny(2)
    pos, sp, logq = m.sample(B, N, n_B=4, device="cpu", return_logq=True)
    assert torch.allclose(logq, m.log_prob(pos, sp), atol=1e-4)


def test_grad_flows_through_pass2_not_pass1():
    # Bug-4 guard: "not xmix.requires_grad" alone is vacuous (true by @torch.no_grad() regardless of
    # .detach()). Assert the real contract: (a) xmix is detached leaf data (no grad_fn), and (b) gradients
    # still reach the heads through PASS 2 even when fully self-conditioned (p_keep=0.0 => prefix is all x_hat).
    m, x, s, N, B = _tiny(3)
    s2 = s.expand(B, N).clone() if s.dim() == 1 else s
    order = m.geo._curve_order(x, N)
    xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s2, 1, order)
    xmix = m._self_prefix(xo, so, m.geo._scaffold(N, x.device), m._Lof(N), N, p_keep=0.0,
                          gen=torch.Generator().manual_seed(5))
    assert xmix.grad_fn is None and not xmix.requires_grad           # pass-1 output is detached data
    m.zero_grad()
    loss = (-m.log_prob_sched(x, s, p_keep=0.0, gen=torch.Generator().manual_seed(5))).mean()
    loss.backward()
    assert m.head_a.weight.grad is not None and torch.isfinite(m.head_a.weight.grad).all()
    assert m.head_a.weight.grad.abs().sum() > 0                      # grad genuinely flows via pass 2


if __name__ == "__main__":
    test_arc_pT_schedule()
    test_p_keep_one_parity_with_base_log_prob()
    test_inference_exactness_gate_preserved()
    test_grad_flows_through_pass2_not_pass1()
    print("SCHED TESTS PASSED")
