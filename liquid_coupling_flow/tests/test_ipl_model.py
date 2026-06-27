import math, torch
from liquid_coupling_flow.ipl44.ipl_model import make_ipl_model, support_coverage


def test_model_box_is_ipl():
    m = make_ipl_model(device="cpu")
    assert abs(m._Lof(44) - math.sqrt(88)) < 1e-5          # rho=0.5 box
    assert abs(m._arc_scale(44) - 44 ** (1 / 6)) < 1e-5


def test_support_coverage_passes_on_ipl_reference():
    m = make_ipl_model(device="cpu")
    cov = support_coverage(m, device="cpu")
    print("IPL support coverage:", cov)
    # the offset must fit inside BOTH the arc_range bins and the flow tail_bound (else the target is truncated)
    assert cov["oor_arc_range"] < 1e-3, cov
    assert cov["oor_tail_bound"] < 1e-3, cov


if __name__ == "__main__":
    test_model_box_is_ipl(); test_support_coverage_passes_on_ipl_reference(); print("IPL MODEL TESTS PASSED")
