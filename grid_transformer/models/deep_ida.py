from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.spatial import wrap_min_image


class DeepIDABias(nn.Module):
    """Deep non-monotonic continuous distance bias for attention logits."""

    def __init__(self, n_head: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.n_head = int(n_head)
        h = int(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(1, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, self.n_head),
        )

    def _prepare_coords_and_box(
        self,
        coords: torch.Tensor,
        box_size: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if coords.ndim != 3:
            raise ValueError(f"coords must be [B,T,C], got {tuple(coords.shape)}")
        bsz, steps, dim = coords.shape
        if dim == 2:
            zeros = torch.zeros(bsz, steps, 1, device=coords.device, dtype=coords.dtype)
            coords3 = torch.cat([coords, zeros], dim=-1)
            if box_size is None:
                return coords3, None
            bs = torch.as_tensor(box_size, device=coords.device, dtype=coords.dtype)
            if bs.ndim == 1:
                bs = torch.cat([bs, torch.ones(1, device=bs.device, dtype=bs.dtype)], dim=0)
            elif bs.ndim == 2:
                pad = torch.ones(bs.shape[0], 1, device=bs.device, dtype=bs.dtype)
                bs = torch.cat([bs, pad], dim=1)
            return coords3, bs
        if dim == 3:
            bs = None if box_size is None else torch.as_tensor(box_size, device=coords.device, dtype=coords.dtype)
            return coords, bs
        raise ValueError(f"coords last dim must be 2 or 3, got {dim}")

    def forward(
        self,
        coords: torch.Tensor,
        causal_mask: torch.Tensor,
        *,
        box_size: torch.Tensor | None = None,
        torus: bool = False,
    ) -> torch.Tensor:
        coords3, box = self._prepare_coords_and_box(coords, box_size)
        diff = coords3[:, :, None, :] - coords3[:, None, :, :]  # [B,T,T,3]
        if torus:
            if box is None:
                raise ValueError("box_size is required when torus=True")
            diff = wrap_min_image(diff, box)
        dist = torch.linalg.norm(diff, dim=-1).unsqueeze(-1)  # [B,T,T,1]
        bias = self.mlp(dist).permute(0, 3, 1, 2).contiguous()  # [B,nH,T,T]
        return bias.masked_fill(~causal_mask[:, None, :, :], float("-inf"))

