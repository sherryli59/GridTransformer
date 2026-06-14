"""liquid_coupling_flow — a local, exact-likelihood coupling normalizing flow
for size-transferable Boltzmann generation of disordered liquids.

Design (see reports/2026-06-14-getting-unstuck-new-directions.md):
  base (uniform-on-torus / Gaussian) --[stack of coupling layers]--> data
  * exact log_prob in one pass (triangular Jacobian)  -> importance reweighting / ESS
  * local geometric conditioner (cutoff)              -> size transfer for homogeneous liquids
  * circular-spline transforms                        -> periodic (torus) coordinates

Build order (proof-of-concept first):
  Phase A  affine coupling + framework, non-periodic 2D toy   [this file's first commit]
  Phase B  rational-quadratic spline coupling, 2D toy
  Phase C  circular spline coupling, periodic torus toy
  Phase D  ESS / importance-reweighting demo on a known target
  Phase E  particle flow (geometric conditioner) on a small LJ-like system
  + SMC/annealing corrector (separate module)
"""

from .base import DiagGaussian, UniformTorus
from .transforms import AffineElementwise
from .coupling import CouplingLayer, alternating_masks
from .conditioner import MLPConditioner
from .flow import Flow

__all__ = [
    "DiagGaussian",
    "UniformTorus",
    "AffineElementwise",
    "CouplingLayer",
    "alternating_masks",
    "MLPConditioner",
    "Flow",
]
