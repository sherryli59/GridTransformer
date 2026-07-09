import pytest
import torch

from liquid_coupling_flow.ka_cavity import assert_mobile_inside, cavity_inside


def test_cavity_inside_uses_minimum_image_and_strict_hard_wall():
    L = 10.0
    center = torch.tensor([5.0, 5.0])
    x = torch.tensor([[
        [5.0, 5.0],  # centre
        [7.9, 5.0],  # interior
        [8.0, 5.0],  # exactly on R=3 wall
        [0.1, 5.0],  # periodic distance 4.9
    ]])
    inside = cavity_inside(x, center, radius=3.0, L=L)
    assert torch.equal(inside, torch.tensor([[True, True, False, False]]))


def test_assert_mobile_inside_allows_frozen_exterior_but_rejects_escaped_mobile():
    L = 10.0
    center = torch.tensor([5.0, 5.0])
    x = torch.tensor([[[5.0, 5.0], [8.5, 5.0]]])
    mobile = torch.tensor([[True, False]])
    assert_mobile_inside(x, mobile, center, radius=3.0, L=L)

    with pytest.raises(AssertionError, match="outside cavity"):
        assert_mobile_inside(x, torch.tensor([[True, True]]), center, radius=3.0, L=L)
