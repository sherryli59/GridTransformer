"""Shared construction helpers for the mW eRSI flow base.

``learndiffeq`` is intentionally vendored rather than installed.  Keep the
path manipulation here, local to the mW integration, so importing the normal
mW stack never changes the process import path.
"""
from __future__ import annotations

import os
import sys

from liquid_coupling_flow.mw.mw_energy import RHO_STAR


_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEARNDIFFEQ_ROOT = os.path.join(_REPO, "liquid_coupling_flow", "ipl44", "learndiffeq")


def _add_learndiffeq_path() -> str:
    """Add the vendored package root to ``sys.path`` once and return it."""
    package = os.path.join(LEARNDIFFEQ_ROOT, "learndiffeq")
    if not os.path.isdir(package):
        raise ImportError(f"vendored learndiffeq package not found at {package}")
    if LEARNDIFFEQ_ROOT not in sys.path:
        sys.path.insert(0, LEARNDIFFEQ_ROOT)
    return LEARNDIFFEQ_ROOT


def L_for_N(N: int) -> float:
    """mW cubic box length at the certified ambient reduced density."""
    if N <= 1:
        raise ValueError("eRSI requires at least two particles")
    return (float(N) / RHO_STAR) ** (1.0 / 3.0)
