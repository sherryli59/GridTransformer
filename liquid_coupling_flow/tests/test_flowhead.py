# liquid_coupling_flow/tests/test_flowhead.py
import math, torch
from liquid_coupling_flow.ka_flowhead import SplineFlowHead

_LOG2PI = math.log(2 * math.pi)


def test_sample_logprob_round_trip_exact():
    # The flow's sample and log_prob must be mutually exact: log_prob at the sampled point == sample's logq.
    torch.manual_seed(0)
    head = SplineFlowHead(d_model=16); head.eval()
    h = torch.randn(5, 16)
    g = torch.Generator().manual_seed(7)
    ab, logq = head.sample(h, gen=g)
    lp = head.log_prob(h, ab)
    assert torch.allclose(logq, lp, atol=1e-5), (logq, lp)


def test_identity_init_is_gaussian_base():
    # At identity init the spline is the identity, so log_prob(ab) == standard-normal base log-density of ab.
    torch.manual_seed(1)
    head = SplineFlowHead(d_model=16); head.eval()
    h = torch.randn(4, 16)
    ab = torch.randn(4, 2) * 0.5            # inside the spline domain
    base = (-0.5 * ab ** 2 - 0.5 * _LOG2PI).sum(-1)
    assert torch.allclose(head.log_prob(h, ab), base, atol=1e-4)


def test_finite_on_extreme_offsets():
    # Linear tails => finite density far outside the spline domain (no NaN/inf).
    head = SplineFlowHead(d_model=16); head.eval()
    h = torch.randn(3, 16)
    ab = torch.tensor([[100.0, -100.0], [50.0, 50.0], [-7.0, 7.0]])
    lp = head.log_prob(h, ab)
    assert torch.isfinite(lp).all()


def test_log_prob_differentiable():
    head = SplineFlowHead(d_model=16)
    h = torch.randn(4, 16); ab = torch.randn(4, 2) * 0.5
    loss = -head.log_prob(h, ab).mean()
    loss.backward()
    assert head.head_a.weight.grad is not None and torch.isfinite(head.head_a.weight.grad).all()


if __name__ == "__main__":
    test_sample_logprob_round_trip_exact()
    test_identity_init_is_gaussian_base()
    test_finite_on_extreme_offsets()
    test_log_prob_differentiable()
    print("FLOWHEAD TESTS PASSED")
