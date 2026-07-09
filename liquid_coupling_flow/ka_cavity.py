"""Shared geometry checks for hard-walled cavity point-to-set runs."""
from __future__ import annotations

import torch


def cavity_inside(x: torch.Tensor, center: torch.Tensor, radius: float, L: float) -> torch.Tensor:
    """Return whether each periodic position lies strictly inside a spherical cavity.

    ``x`` may have any leading shape as long as its last dimension is the
    coordinate dimension.  The strict inequality makes a proposed move onto
    the wall a rejection, which is the usual hard-wall convention.
    """
    delta = x - center
    delta = delta - L * torch.round(delta / L)
    return delta.square().sum(dim=-1) < radius * radius


def assert_mobile_inside(x: torch.Tensor, mobile: torch.Tensor, center: torch.Tensor,
                         radius: float, L: float) -> None:
    """Raise if a movable particle has escaped a hard-walled cavity."""
    inside = cavity_inside(x, center, radius, L)
    if not bool(inside[mobile].all()):
        n_outside = int((mobile & ~inside).sum())
        raise AssertionError(f"{n_outside} mobile particle(s) lie outside cavity R={radius}")
