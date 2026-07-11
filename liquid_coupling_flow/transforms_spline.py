"""Rational-quadratic spline transforms (Durkan et al., Neural Spline Flows 2019).

Two elementwise transforms matching the AffineElementwise interface
(forward/inverse take (x, params[..., P]) and return (y, logdet_elementwise)):

  RQSplineElementwise         unconstrained spline on R with linear (identity)
                              tails outside [-B, B]; params_per_dim = 3K - 1.
                              Use with a Gaussian base (non-periodic toys).

  CircularRQSplineElementwise spline on the circle [0, L] (a diffeomorphism of the
                              torus coordinate); params_per_dim = 3K. Boundary
                              derivative is shared so the map is smooth across the
                              seam. Use with a UniformTorus base (periodic liquids).

The closed-form forward-derivative below is what the autograd log-det test checks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3
# softplus(const)+min_deriv == 1: boundary-derivative pad value, computed ONCE (avoids a per-call
# host->device tensor creation + sync in the hot path).
_BOUNDARY_DERIV_CONST = float(torch.log(torch.expm1(torch.tensor(1.0 - DEFAULT_MIN_DERIVATIVE))))


def _searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6):
    bl = bin_locations.clone()
    bl[..., -1] = bl[..., -1] + eps
    return torch.sum(inputs[..., None] >= bl, dim=-1) - 1


def _rqs_core(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
              inverse, left, right, bottom, top,
              min_bin_width=DEFAULT_MIN_BIN_WIDTH, min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
              min_derivative=DEFAULT_MIN_DERIVATIVE):
    """Core RQS on inputs known to lie in [left, right] (fwd) / [bottom, top] (inv).

    inputs: [M]; unnormalized_widths/heights: [M, K]; unnormalized_derivatives: [M, K+1].
    Returns (outputs [M], logabsdet [M]).
    """
    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_locations = cumheights if inverse else cumwidths
    bin_idx = _searchsorted(bin_locations, inputs).clamp(0, num_bins - 1)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta) \
            + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives \
            - (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        c = -input_delta * (inputs - input_cumheights)
        discriminant = (b ** 2 - 4 * a * c).clamp_min(0.0)
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        derivative_numerator = input_delta ** 2 * (
            input_derivatives_plus_one * root ** 2
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root) ** 2)
        logabsdet = -(torch.log(derivative_numerator) - 2 * torch.log(denominator))
        return outputs, logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)
        numerator = input_heights * (input_delta * theta ** 2 + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        outputs = input_cumheights + numerator / denominator
        derivative_numerator = input_delta ** 2 * (
            input_derivatives_plus_one * theta ** 2
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta) ** 2)
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet


class RQSplineElementwise:
    """Unconstrained RQ spline on R: spline inside [-B, B], identity linear tails."""

    def __init__(self, num_bins: int = 8, tail_bound: float = 5.0):
        self.num_bins = int(num_bins)
        self.tail_bound = float(tail_bound)

    @property
    def params_per_dim(self) -> int:
        # K widths + K heights + (K-1) interior derivatives
        return 3 * self.num_bins - 1

    def _apply(self, x: torch.Tensor, params: torch.Tensor, inverse: bool):
        B = self.tail_bound
        K = self.num_bins
        shape = x.shape
        x_flat = x.reshape(-1)
        p_flat = params.reshape(-1, self.params_per_dim)
        uw = p_flat[:, :K]
        uh = p_flat[:, K:2 * K]
        ud_interior = p_flat[:, 2 * K:]  # [M, K-1]
        # Pad boundary derivatives with the precomputed constant (softplus+min_deriv == 1).
        pad = x_flat.new_full((ud_interior.shape[0], 1), _BOUNDARY_DERIV_CONST)
        ud = torch.cat([pad, ud_interior, pad], dim=-1)  # [M, K+1]

        # Compute the spline for ALL elements (inputs clamped into [-B,B] so _rqs_core is well-defined),
        # then blend identity linear tails via where. Avoids a data-dependent .any() SYNC and
        # masked_scatter in the hot path; exact for inside elements (tails overwritten anyway).
        inside = (x_flat > -B) & (x_flat < B)
        o, l = _rqs_core(x_flat.clamp(-B, B), uw, uh, ud,
                         inverse=inverse, left=-B, right=B, bottom=-B, top=B)
        out = torch.where(inside, o, x_flat)
        ld = torch.where(inside, l, torch.zeros_like(l))
        return out.reshape(shape), ld.reshape(shape)

    def forward(self, x, params):
        return self._apply(x, params, inverse=False)

    def inverse(self, y, params):
        return self._apply(y, params, inverse=True)


class CircularRQSplineElementwise:
    """RQ spline on the circle [0, L]: a diffeomorphism of a periodic coordinate."""

    def __init__(self, num_bins: int = 8, L: float = 1.0):
        self.num_bins = int(num_bins)
        self.L = float(L)

    @property
    def params_per_dim(self) -> int:
        # K widths + K heights + K derivatives (boundary derivative shared)
        return 3 * self.num_bins

    def _apply(self, x: torch.Tensor, params: torch.Tensor, inverse: bool):
        K = self.num_bins
        L = self.L
        shape = x.shape
        x_flat = torch.remainder(x.reshape(-1), L)
        p_flat = params.reshape(-1, self.params_per_dim)
        uw = p_flat[:, :K]
        uh = p_flat[:, K:2 * K]
        ud_k = p_flat[:, 2 * K:]  # [M, K]
        ud = torch.cat([ud_k, ud_k[:, :1]], dim=-1)  # tie last to first -> [M, K+1]
        out, ld = _rqs_core(x_flat, uw, uh, ud, inverse=inverse,
                            left=0.0, right=L, bottom=0.0, top=L)
        return out.reshape(shape), ld.reshape(shape)

    def forward(self, x, params):
        return self._apply(x, params, inverse=False)

    def inverse(self, y, params):
        return self._apply(y, params, inverse=True)
