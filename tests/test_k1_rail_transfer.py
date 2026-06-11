import math
import numpy as np
import pytest

from analyze_k1_rail_transfer import (
    choose_R,
    rail_waypoints_absolute,
    in_box_fraction,
    CELL_TRAIN,
)


def test_choose_R_pow2():
    assert choose_R(3.0, "pow2") == 64
    assert choose_R(4.0, "pow2") == 128
    assert choose_R(5.0, "pow2") == 128


def test_choose_R_exact():
    assert choose_R(3.0, "exact") == 64
    assert choose_R(4.0, "exact") == 85
    assert choose_R(5.0, "exact") == 107


def test_rail_waypoints_shape_and_determinism():
    wp1 = rail_waypoints_absolute(N=27, R=64, L=3.0)
    wp2 = rail_waypoints_absolute(N=27, R=64, L=3.0)
    assert wp1.shape == (26, 3)
    assert np.allclose(wp1, wp2)


def test_rail_waypoints_in_box_for_pow2():
    # reference="absolute" waypoints are min-imaged to [-L/2, L/2), not [0, L)
    wp = rail_waypoints_absolute(N=64, R=128, L=4.0)
    L = 4.0
    assert in_box_fraction(wp, L, low=-L / 2) == 1.0


def test_in_box_fraction_counts_out_of_box():
    pts = np.array([[0.1, 0.1, 0.1], [3.9, 0.1, 0.1], [4.5, 0.1, 0.1]])
    assert in_box_fraction(pts, 4.0) == pytest.approx(2.0 / 3.0)


def test_raw_cell_inbox_pow2_is_one():
    # pow2 R == curve side: every decoded waypoint cell sits in [0, R).
    from analyze_k1_rail_transfer import raw_cell_inbox_fraction
    assert raw_cell_inbox_fraction(64, 64) == 1.0
    assert raw_cell_inbox_fraction(64, 128) == 1.0


def test_raw_cell_inbox_exact_is_valid_fraction():
    # exact (non-pow2) R may leak cells outside [0, R); result is a valid fraction.
    from analyze_k1_rail_transfer import raw_cell_inbox_fraction
    frac = raw_cell_inbox_fraction(64, 85)
    assert 0.0 <= frac <= 1.0
