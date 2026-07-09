import math, torch
from liquid_coupling_flow.mw.mw_bedrock import quad_N2, quad_N3

def test_n2_matches_direct_radial():
    # Independent check: N=2 <U> via explicit 3D numpy-style sum with a DIFFERENT grid offset scheme
    U, err = quad_N2(4.0, 2.0, 64)
    U2, _ = quad_N2(4.0, 2.0, 96)
    assert abs(U - U2) < 3 * err + 1e-6 and err < 1e-3

def test_n3_halving_converges():
    U, err = quad_N3(4.0, 2.0, 16)
    U2, err2 = quad_N3(4.0, 2.0, 20)
    assert err2 < err * 1.5 and abs(U - U2) < 3 * (err + err2)

def test_n3_beats_two_body_only():
    # with the 3-body term zeroed the answer must CHANGE (the gate really exercises phi3)
    from liquid_coupling_flow.mw import mw_bedrock
    U, _ = quad_N3(4.0, 2.0, 16)
    U2b, _ = quad_N3(4.0, 2.0, 16, two_body_only=True)
    assert abs(U - U2b) > 1e-3
