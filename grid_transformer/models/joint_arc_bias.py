from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..utils.spatial import wrap_min_image


class JointEdgeBias(nn.Module):
    """Variant A: additive attention bias over (Euclidean distance × arc separation).

    The defining geometric correlation of Hilbert-ordered LJ data is the joint law of
    pair distance d_ij and curve separation Δs_ij = s_i − s_j (normalized arc length).
    The standard EdgeBias sees only d_ij; this module feeds the MLP the *joint*
    [RBF(d_ij) ‖ RBF(log1p(Δs_ij))] so it can express "spatially close but curve-far ⇒
    strong constraint" — invisible to distance-only biases (architecture doc §A).

    Warm-start safe: the final linear layer is **zero-initialized**, so adding this
    module to a trained checkpoint is an exact no-op until fine-tuning moves it.
    Used *additively* on top of the existing EdgeBias.
    """

    def __init__(
        self,
        n_head: int,
        *,
        n_rbf_dist: int = 32,
        n_rbf_arc: int = 16,
        max_dist: float = 4.0,
        max_arc: float = 64.0,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_head = int(n_head)
        self.max_dist = max(float(max_dist), 1e-8)
        self.max_log_arc = math.log1p(max(float(max_arc), 1e-8))

        dist_centers = torch.linspace(0.0, self.max_dist, steps=int(n_rbf_dist))
        arc_centers = torch.linspace(0.0, self.max_log_arc, steps=int(n_rbf_arc))
        self.register_buffer("dist_centers", dist_centers, persistent=True)
        self.register_buffer("arc_centers", arc_centers, persistent=True)
        d_delta = self.max_dist / max(int(n_rbf_dist) - 1, 1)
        a_delta = self.max_log_arc / max(int(n_rbf_arc) - 1, 1)
        self.dist_gamma = 1.0 / max(d_delta * d_delta, 1e-8)
        self.arc_gamma = 1.0 / max(a_delta * a_delta, 1e-8)

        self.mlp = nn.Sequential(
            nn.Linear(int(n_rbf_dist) + int(n_rbf_arc), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.n_head),
        )
        # Zero-init output: bias starts at exactly 0 (warm-start no-op).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _features(self, dist: torch.Tensor, arc_sep: torch.Tensor) -> torch.Tensor:
        """[..., Q, K] distance + arc separation -> [..., Q, K, n_rbf_d + n_rbf_a]."""
        dc = self.dist_centers.to(device=dist.device, dtype=dist.dtype)
        rbf_d = torch.exp(-self.dist_gamma * (dist.unsqueeze(-1) - dc) ** 2)
        log_arc = torch.log1p(arc_sep.abs())
        ac = self.arc_centers.to(device=dist.device, dtype=dist.dtype)
        rbf_a = torch.exp(-self.arc_gamma * (log_arc.unsqueeze(-1) - ac) ** 2)
        return torch.cat([rbf_d, rbf_a], dim=-1)

    def _bias(self, coords_q, coords_k, arc_q, arc_k, box_size, torus) -> torch.Tensor:
        diff = coords_q[:, :, None, :] - coords_k[:, None, :, :]  # [B,Q,K,3]
        if torus:
            if box_size is None:
                raise ValueError("box_size is required when torus=True")
            diff = wrap_min_image(diff, box_size)
        dist = torch.linalg.norm(diff, dim=-1)
        arc_sep = arc_q[:, :, None] - arc_k[:, None, :]  # [B,Q,K]
        feat = self._features(dist, arc_sep)
        return self.mlp(feat).permute(0, 3, 1, 2).contiguous()  # [B,nH,Q,K]

    def forward(
        self,
        coords: torch.Tensor,
        arc_s: torch.Tensor,
        causal_mask: torch.Tensor,
        *,
        box_size: torch.Tensor | None = None,
        torus: bool = False,
    ) -> torch.Tensor:
        """coords [B,T,3], arc_s [B,T] cumulative normalized arc; -> [B,nH,T,T]."""
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(f"coords must be [B,T,3], got {tuple(coords.shape)}")
        if arc_s.shape != coords.shape[:2]:
            raise ValueError(f"arc_s must be [B,T], got {tuple(arc_s.shape)}")
        bias = self._bias(coords, coords, arc_s, arc_s, box_size, torus)
        return bias.masked_fill(~causal_mask[:, None, :, :], float("-inf"))

    def bias_row(
        self,
        coords: torch.Tensor,
        arc_s: torch.Tensor,
        *,
        box_size: torch.Tensor | None = None,
        torus: bool = False,
    ) -> torch.Tensor:
        """Last causal row for KV-cached generation: [B,nH,1,P] (no mask needed)."""
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(f"coords must be [B,P,3], got {tuple(coords.shape)}")
        if arc_s.shape != coords.shape[:2]:
            raise ValueError(f"arc_s must be [B,P], got {tuple(arc_s.shape)}")
        return self._bias(
            coords[:, -1:, :], coords, arc_s[:, -1:], arc_s, box_size, torus
        )
