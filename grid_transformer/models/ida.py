from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.spatial import wrap_min_image


class PeriodicIDA(nn.Module):
    """
    Interatomic Distance Attention (IDA) adapted for periodic boundary
    conditions (torus) and arbitrary spatial dimensions (2D or 3D).
    """

    def __init__(self, d_model: int, num_heads: int = 8, spatial_dim: int = 2, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.spatial_dim = spatial_dim

        self.layer_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, spatial_dim * num_heads * 3, bias=bias)
        self.per_head_scalar = nn.Parameter(torch.ones(num_heads))
        self.out_proj = nn.Linear(spatial_dim * num_heads, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        causal_mask: torch.Tensor,
        box_size: Optional[torch.Tensor] = None,
        torus: bool = False,
    ) -> torch.Tensor:
        bsz, t_steps, _ = x.shape

        q, k, v = self.proj(self.layer_norm(x)).chunk(3, dim=-1)
        q = q.view(bsz, t_steps, self.num_heads, self.spatial_dim).permute(0, 2, 1, 3)
        k = k.view(bsz, t_steps, self.num_heads, self.spatial_dim).permute(0, 2, 1, 3)
        v = v.view(bsz, t_steps, self.num_heads, self.spatial_dim).permute(0, 2, 1, 3)

        shift = coords.unsqueeze(1)
        q = q + shift
        k = k + shift

        if torus and box_size is not None:
            diff = q.unsqueeze(3) - k.unsqueeze(2)
            d = torch.linalg.norm(wrap_min_image(diff, box_size), dim=-1)
        else:
            d = torch.cdist(q, k, p=2)

        d = 1.0 / (math.sqrt(self.spatial_dim) * (d + 1e-8))
        a = F.softplus(self.per_head_scalar).view(1, self.num_heads, 1, 1) * d

        if causal_mask.ndim == 2:
            causal = causal_mask.view(1, 1, t_steps, t_steps).expand(bsz, self.num_heads, -1, -1)
        elif causal_mask.ndim == 3:
            causal = causal_mask[:, None, :, :].expand(bsz, self.num_heads, -1, -1)
        else:
            raise ValueError(f"causal_mask must be [T,T] or [B,T,T], got {tuple(causal_mask.shape)}")

        a = a.masked_fill(~causal, float("-inf"))
        a = F.softmax(a, dim=-1)

        o = torch.einsum("bhij,bhjd->bhid", a, v)
        o = o.permute(0, 2, 1, 3).contiguous().view(bsz, t_steps, -1)
        return x + self.out_proj(o)
