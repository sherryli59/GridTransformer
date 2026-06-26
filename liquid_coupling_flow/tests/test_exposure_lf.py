import numpy as np, torch
from liquid_coupling_flow.ka_exposure_lf import clash_by_index


def test_clash_by_index_known_geometry():
    # 3 particles on a line at x=0,0.5,5.0 in a big box: j=1 clashes with j=0 (d=0.5<0.7),
    # j=2 clashes with neither (min dist 4.5). One batch element.
    L = 50.0
    pos = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [5.0, 0.0]]])
    out = clash_by_index(pos, pos, L, thr=0.7)
    assert out.shape == (3,)
    assert out[0] == 0.0          # j=0 has no predecessor
    assert out[1] == 1.0          # 0.5 < 0.7
    assert out[2] == 0.0          # 4.5 > 0.7


def test_clash_by_index_pbc_wrap():
    # j=1 at 49.9, predecessor at 0.0; min-image distance is 0.1 (<0.7) across the boundary.
    L = 50.0
    pos = torch.tensor([[[0.0, 0.0], [49.9, 0.0]]])
    out = clash_by_index(pos, pos, L, thr=0.7)
    assert out[1] == 1.0


if __name__ == "__main__":
    test_clash_by_index_known_geometry()
    test_clash_by_index_pbc_wrap()
    print("EXPOSURE-LF TESTS PASSED")
