from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CurveRailAttention(nn.Module):
    """Ordered, geometric cross-attention onto the Hilbert "curve rail" (GPS guide).

    Each token (one placed particle) cross-attends to its own ``K`` look-ahead
    waypoints — the deterministic Hilbert-curve cell centers at log-spaced arc-length
    offsets, supplied as min-image vectors *relative to that particle*. This is kept
    deliberately separate from the neighborhood geometric attention (``EdgeBias`` /
    ``PeriodicIDA``), which looks backward at placed particles; the rail looks forward
    along the curve.

    Key properties requested by design:
      * **Geometric** — waypoints enter through an RBF expansion of their distance plus
        a unit-direction MLP, mirroring the existing RBF/IDA attention style.
      * **Ordered (not permutation-invariant)** — a learned per-rail-index embedding is
        added so waypoint ``k=0`` (nearest along the curve) is distinguished from the
        farther ones; shuffling the rail changes the output.
      * **Local / causal-safe** — a token attends only to its *own* waypoints (no cross-
        token mixing), and the waypoints depend only on the current particle's cell, so
        no causal masking is needed and nothing about future particles leaks.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_rail: int,
        *,
        n_rbf: int = 32,
        max_dist: float = 4.0,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_head={n_head}")
        if n_rail <= 0:
            raise ValueError(f"n_rail must be positive, got {n_rail}")
        if n_rbf <= 0:
            raise ValueError(f"n_rbf must be positive, got {n_rbf}")

        self.d_model = int(d_model)
        self.n_head = int(n_head)
        self.d_head = int(d_model) // int(n_head)
        self.n_rail = int(n_rail)
        self.n_rbf = int(n_rbf)
        self.max_dist = max(float(max_dist), 1e-8)

        centers = torch.linspace(0.0, self.max_dist, steps=self.n_rbf)
        self.register_buffer("centers", centers, persistent=True)
        delta = self.max_dist / max(self.n_rbf - 1, 1)
        self.gamma = 1.0 / max(delta * delta, 1e-8)

        # Waypoint featurizer: RBF(distance) + unit direction -> d_model.
        self.feat_mlp = nn.Sequential(
            nn.Linear(self.n_rbf + 3, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.d_model),
        )
        # Order embedding makes the rail ordered (breaks permutation invariance over K).
        self.order_emb = nn.Embedding(self.n_rail, self.d_model)

        self.ln = nn.LayerNorm(self.d_model)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.kv_proj = nn.Linear(self.d_model, 2 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def _featurize(self, waypoints: torch.Tensor) -> torch.Tensor:
        """[B,T,K,3] relative vectors -> [B,T,K,d_model] geometric+order features."""
        dist = torch.linalg.norm(waypoints, dim=-1)  # [B,T,K]
        unit_dir = waypoints / dist.clamp_min(1e-6).unsqueeze(-1)  # [B,T,K,3]
        centers = self.centers.to(device=dist.device, dtype=dist.dtype)
        rbf = torch.exp(-self.gamma * (dist.unsqueeze(-1) - centers.view(1, 1, 1, -1)) ** 2)
        feat = self.feat_mlp(torch.cat([rbf, unit_dir], dim=-1))  # [B,T,K,Dm]
        order_ids = torch.arange(self.n_rail, device=feat.device)
        return feat + self.order_emb(order_ids).view(1, 1, self.n_rail, self.d_model)

    def forward(self, x: torch.Tensor, waypoints: torch.Tensor) -> torch.Tensor:
        """
        x:         [B, T, d_model] token hidden states.
        waypoints: [B, T, K, 3] min-image vectors (particle -> future curve cell).
        returns:   [B, T, d_model] (residual already added).
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [B,T,d_model], got {tuple(x.shape)}")
        if waypoints.ndim != 4 or waypoints.shape[-1] != 3:
            raise ValueError(f"waypoints must be [B,T,K,3], got {tuple(waypoints.shape)}")
        if waypoints.shape[2] != self.n_rail:
            raise ValueError(f"waypoints K={waypoints.shape[2]} != n_rail={self.n_rail}")
        if waypoints.shape[:2] != x.shape[:2]:
            raise ValueError(f"x/waypoints [B,T] mismatch: {tuple(x.shape[:2])} vs {tuple(waypoints.shape[:2])}")

        B, T, _ = x.shape
        K = self.n_rail
        feat = self._featurize(waypoints.to(dtype=x.dtype))  # [B,T,K,Dm]

        q = self.q_proj(self.ln(x)).view(B, T, self.n_head, self.d_head)        # [B,T,H,dh]
        kv = self.kv_proj(feat).view(B, T, K, 2, self.n_head, self.d_head)
        k, v = kv.unbind(dim=3)                                                  # [B,T,K,H,dh]

        scores = torch.einsum("bthd,btkhd->bthk", q, k) / (self.d_head ** 0.5)   # [B,T,H,K]
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out = torch.einsum("bthk,btkhd->bthd", attn, v).reshape(B, T, self.d_model)
        return x + self.out_proj(out)
