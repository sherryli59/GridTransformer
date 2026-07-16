"""Full-support anchor-centered cube chart and Cat3 position head."""
from __future__ import annotations

import torch
import torch.nn as nn

from liquid_coupling_flow.ka3d_scaffold_ar import Cat3Head


def clip_anchor(anchor: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    """Apply the pinned numerical anchor clipping used by sample and score."""
    if not torch.is_floating_point(anchor):
        raise TypeError("anchor must be floating point")
    if eps is None:
        # A dtype-epsilon clip makes the narrow chart branch numerically
        # unrecoverable after an oriented-torus round trip in float64.  This
        # fixed geometric clip is shared by sample and score and still leaves
        # every point in the cube in support.
        eps = 1.0e-2
    if not 0.0 < float(eps) < 0.5:
        raise ValueError("anchor eps must lie in (0, 0.5)")
    return anchor.clamp(float(eps), 1.0 - float(eps))


def _width_tensor(width, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(width, device=reference.device, dtype=reference.dtype)
    if bool((value <= 0).any()) or not bool(torch.isfinite(value).all()):
        raise ValueError("cube width must be finite and positive")
    return value


def anchored_cube_forward(u: torch.Tensor, anchor: torch.Tensor, width=1.0,
                          eps: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Map ``u in [-1,1]^3`` to cell-local coordinates; return forward logdet."""
    if u.shape[-1] != 3:
        raise ValueError("u must end in dimension 3")
    if bool((u < -1.0).any()) or bool((u > 1.0).any()):
        raise ValueError("u outside [-1,1]")
    anchor = clip_anchor(torch.as_tensor(anchor, device=u.device, dtype=u.dtype), eps)
    width_t = _width_tensor(width, u)
    frac = torch.where(u < 0.0, anchor * (u + 1.0), anchor + (1.0 - anchor) * u)
    slope = torch.where(u < 0.0, anchor, 1.0 - anchor) * width_t
    return frac * width_t, torch.log(slope).sum(-1)


def anchored_cube_inverse(local: torch.Tensor, anchor: torch.Tensor, width=1.0,
                          eps: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse cube chart; return ``u`` and ``log|du/dlocal|``."""
    if local.shape[-1] != 3:
        raise ValueError("local must end in dimension 3")
    width_t = _width_tensor(width, local)
    if bool((local < 0.0).any()) or bool((local > width_t).any()):
        raise ValueError("local coordinate outside the closed cube")
    anchor = clip_anchor(torch.as_tensor(anchor, device=local.device, dtype=local.dtype), eps)
    frac = local / width_t
    lower = frac < anchor
    u = torch.where(lower, frac / anchor - 1.0, (frac - anchor) / (1.0 - anchor))
    slope = torch.where(lower, anchor, 1.0 - anchor) * width_t
    return u, -torch.log(slope).sum(-1)


class AnchoredCubeCat3Head(nn.Module):
    """July-14 Cat3 head composed with the exact full-cube chart.

    ``anchor`` is fractional in the cube and ``local`` is in physical length
    units.  The underlying Cat3 range is exactly one, so it covers the complete
    latent cube ``[-1,1)^3``.
    """

    def __init__(self, d_model: int, num_bins: int = 128, anchor_eps: float | None = None):
        super().__init__()
        self.base = Cat3Head(d_model, num_bins=num_bins, u_range=1.0)
        self.num_bins = int(num_bins)
        self.anchor_eps = anchor_eps

    def log_prob(self, context: torch.Tensor, local: torch.Tensor, anchor: torch.Tensor,
                 width=1.0) -> torch.Tensor:
        width_t = _width_tensor(width, local)
        valid = ((local >= 0.0) & (local < width_t)).all(-1)
        upper = torch.nextafter(width_t, torch.zeros_like(width_t))
        safe = torch.minimum(local.clamp_min(0.0), upper)
        u, inverse_logdet = anchored_cube_inverse(
            safe, anchor, width_t, eps=self.anchor_eps
        )
        result = self.base.log_prob(context, u) + inverse_logdet
        return torch.where(valid, result, torch.full_like(result, float("-inf")))

    def sample(self, context: torch.Tensor, anchor: torch.Tensor, width=1.0,
               gen=None) -> tuple[torch.Tensor, torch.Tensor]:
        u, latent_log_prob = self.base.sample(context, gen=gen)
        local, forward_logdet = anchored_cube_forward(
            u, anchor, width, eps=self.anchor_eps
        )
        return local, latent_log_prob - forward_logdet

    @torch.no_grad()
    def mode(self, context: torch.Tensor, anchor: torch.Tensor, width=1.0) -> torch.Tensor:
        u = self.base.mode(context)
        return anchored_cube_forward(u, anchor, width, eps=self.anchor_eps)[0]
