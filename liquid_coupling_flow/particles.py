"""Augmented coupling flow over particle configurations on a periodic box.

State is (x, a), both [B, N, d] on the torus [0,L). Layers alternate:
  update x  conditioned on a   (neighbour graph + features from the FIXED a)
  update a  conditioned on x
so the conditioner is LOCAL (cutoff GNN) yet the layer stays invertible (params
depend only on the fixed set). Exact joint log-density (triangular Jacobian); the
liquid x-marginal is recovered by reweighting in the joint space:

    log pi(x, a) = -U(x)/kT + log r(a)         r(a) = uniform on the torus (const)
    log w        = log pi(x, a) - log q(x, a)   (exact; partition fn cancels)

Conditioner uses min-image relative features -> translation invariant; cutoff +
liquid homogeneity -> size transfer. (Rotation equivariance is a v1 refinement.)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .transforms_spline import CircularRQSplineElementwise


def min_image(pos: torch.Tensor, L: float):
    """pos [B,N,d] -> (diff [B,N,N,d], r [B,N,N]) with minimum-image convention."""
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    diff = diff - L * torch.round(diff / L)
    r = torch.sqrt((diff ** 2).sum(-1) + 1e-12)
    return diff, r


class ParticleGNNConditioner(nn.Module):
    """Local message passing on the conditioning set -> per-particle spline params."""

    def __init__(self, d: int, params_per_dim: int, L: float, cutoff: float = 2.5,
                 n_rbf: int = 16, hidden: int = 64):
        super().__init__()
        self.d = d
        self.ppd = params_per_dim
        self.L = float(L)
        self.cutoff = float(cutoff)
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = (n_rbf / cutoff) ** 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(n_rbf + d, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d * params_per_dim),
        )
        nn.init.zeros_(self.node_mlp[-1].weight)
        nn.init.zeros_(self.node_mlp[-1].bias)

    def forward(self, cond_pos: torch.Tensor) -> torch.Tensor:
        B, N, d = cond_pos.shape
        diff, r = min_image(cond_pos, self.L)                       # [B,N,N,d],[B,N,N]
        rbf = torch.exp(-self.gamma * (r[..., None] - self.centers) ** 2)  # [B,N,N,n_rbf]
        rel = diff / self.cutoff                                     # bounded relative position
        edge = torch.cat([rbf, rel], dim=-1)
        msg = self.edge_mlp(edge)                                    # [B,N,N,hidden]
        eye = torch.eye(N, device=cond_pos.device, dtype=torch.bool)
        mask = (r < self.cutoff) & (~eye)                           # [B,N,N]
        h = (msg * mask[..., None]).sum(dim=2)                       # aggregate neighbours -> [B,N,hidden]
        params = self.node_mlp(h)                                    # [B,N,d*ppd]
        return params.view(B, N, d, self.ppd)


class AugmentedCouplingLayer(nn.Module):
    def __init__(self, update: str, conditioner: ParticleGNNConditioner,
                 spline: CircularRQSplineElementwise):
        super().__init__()
        assert update in ("x", "a")
        self.update = update
        self.conditioner = conditioner
        self.spline = spline

    def _run(self, x, a, inverse: bool):
        if self.update == "x":
            params = self.conditioner(a)
            fn = self.spline.inverse if inverse else self.spline.forward
            xn, ld = fn(x, params)
            return xn, a, ld.sum(dim=(1, 2))
        else:
            params = self.conditioner(x)
            fn = self.spline.inverse if inverse else self.spline.forward
            an, ld = fn(a, params)
            return x, an, ld.sum(dim=(1, 2))

    def forward(self, x, a):
        return self._run(x, a, inverse=False)

    def inverse(self, x, a):
        return self._run(x, a, inverse=True)


class ParticleFlow(nn.Module):
    def __init__(self, layers, N: int, d: int, L: float):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.N, self.d, self.L = N, d, float(L)
        self.base_logp = -2 * N * d * math.log(self.L)  # x and a uniform on torus

    def sample(self, B: int, device=None, dtype=None):
        x = torch.rand(B, self.N, self.d, device=device, dtype=dtype) * self.L
        a = torch.rand(B, self.N, self.d, device=device, dtype=dtype) * self.L
        logdet = torch.zeros(B, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, a, ld = layer.forward(x, a)
            logdet = logdet + ld
        return x, a, self.base_logp - logdet

    def log_prob(self, x, a):
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in reversed(self.layers):
            x, a, ld = layer.inverse(x, a)
            logdet = logdet + ld
        return self.base_logp + logdet


def build_particle_flow(N: int, d: int = 2, L: float = 6.0, n_layers: int = 8,
                        num_bins: int = 8, cutoff: float = 2.5, hidden: int = 64,
                        n_rbf: int = 16) -> ParticleFlow:
    layers = []
    for i in range(n_layers):
        update = "x" if i % 2 == 0 else "a"
        spline = CircularRQSplineElementwise(num_bins=num_bins, L=L)
        cond = ParticleGNNConditioner(d, spline.params_per_dim, L=L, cutoff=cutoff,
                                      n_rbf=n_rbf, hidden=hidden)
        layers.append(AugmentedCouplingLayer(update, cond, spline))
    return ParticleFlow(layers, N=N, d=d, L=L)
