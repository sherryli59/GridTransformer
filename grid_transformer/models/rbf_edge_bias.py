from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.spatial import wrap_min_image


class RBFEdgeBias(nn.Module):
    """Continuous distance-to-bias mapping via Gaussian RBF expansion."""

    def __init__(
        self,
        n_head: int,
        n_rbf_centers: int = 64,
        max_dist: float = 4.0,
        hidden_dim: int = 64,
        gamma: float | None = None,
        learnable_centers: bool = False,
    ) -> None:
        super().__init__()
        if n_rbf_centers <= 0:
            raise ValueError(f"n_rbf_centers must be positive, got {n_rbf_centers}")
        self.n_head = int(n_head)
        self.n_rbf_centers = int(n_rbf_centers)
        self.max_dist = max(float(max_dist), 1e-8)

        centers = torch.linspace(0.0, self.max_dist, steps=self.n_rbf_centers)
        if learnable_centers:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers, persistent=True)

        if gamma is None:
            if self.n_rbf_centers > 1:
                delta = self.max_dist / float(self.n_rbf_centers - 1)
            else:
                delta = self.max_dist
            gamma = 1.0 / max(delta * delta, 1e-8)
        self.gamma = float(gamma)

        self.mlp = nn.Sequential(
            nn.Linear(self.n_rbf_centers, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.n_head),
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
        dist = torch.linalg.norm(diff, dim=-1)  # [B,T,T]

        centers = self.centers.to(device=dist.device, dtype=dist.dtype)
        rbf = torch.exp(-self.gamma * (dist.unsqueeze(-1) - centers.view(1, 1, 1, -1)) ** 2)  # [B,T,T,K]
        bias = self.mlp(rbf).permute(0, 3, 1, 2).contiguous()  # [B,nH,T,T]
        return bias.masked_fill(~causal_mask[:, None, :, :], float("-inf"))

