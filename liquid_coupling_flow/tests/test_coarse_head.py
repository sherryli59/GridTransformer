import torch
from liquid_coupling_flow.ka3d_coarse_head import CoarseFineHead


def test_roundtrip_u_to_bins_to_u():
    h = CoarseFineHead(d_model=32)
    u = (torch.rand(500, 3) - 0.5) * 2 * 2.5 * 0.999
    ci = h.cell_index(u)
    fb = h.fine_bin(u, ci)
    ctr = h.fine_ctr(ci, fb)
    assert ci.min() >= 0 and ci.max() < 4096
    assert fb.min() >= 0 and fb.max() < 8
    # every point lies within half a fine bin of its reconstructed center
    assert (u - ctr).abs().max() <= h.bwf / 2 + 1e-6


def test_grid_constants():
    h = CoarseFineHead(d_model=32)
    assert abs(h.cw - 5.0 / 16) < 1e-9
    assert abs(h.bwf - 5.0 / 128) < 1e-9        # == current fl.bw -> same final resolution
    assert abs(h.half_diag - h.cw * 3 ** 0.5 / 2) < 1e-9


def test_cell_center_inverse():
    h = CoarseFineHead(d_model=32)
    idx = torch.arange(4096)
    assert torch.equal(h.cell_index(h.cell_center(idx)), idx)
