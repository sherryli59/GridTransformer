"""Positions-only boundary-conditioned flow-matching prototype for 3D cavities.

Coordinates consumed and produced here are minimum-image coordinates relative
to the cavity center.  Interior species are fixed inputs.  The prototype is not
an exact-density model; spherical projection during sampling is therefore an
explicit G2 engineering guard, not an MH-compatible transform.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def _relative(x, center, L):
    d = x - center
    return d - L * torch.round(d / L)


def _uniform_ball(B, N, R, device, dtype):
    z = torch.randn(B, N, 3, device=device, dtype=dtype)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rad = torch.rand(B, N, 1, device=device, dtype=dtype).pow(1 / 3) * R[:, None, None]
    return z * rad


def collate_cavities(pairs, n_max=None, n_ctx_max=256, r_ctx=2.5):
    """Pad variable interiors and the nearest frozen boundary shell."""
    if not pairs:
        raise ValueError("cannot collate an empty cavity list")
    B = len(pairs); n_max = n_max or max(p["n_in"] for p in pairs)
    if max(p["n_in"] for p in pairs) > n_max:
        raise ValueError("n_max is smaller than an interior")
    dev, dt = pairs[0]["x_in"].device, pairs[0]["x_in"].dtype
    xin = torch.zeros(B, n_max, 3, device=dev, dtype=dt)
    sin = torch.zeros(B, n_max, device=dev, dtype=torch.long)
    mask = torch.zeros(B, n_max, device=dev, dtype=torch.bool)
    xctx = torch.zeros(B, n_ctx_max, 3, device=dev, dtype=dt)
    sctx = torch.zeros(B, n_ctx_max, device=dev, dtype=torch.long)
    cmask = torch.zeros(B, n_ctx_max, device=dev, dtype=torch.bool)
    radii = torch.empty(B, device=dev, dtype=dt)
    temps = torch.empty(B, device=dev, dtype=dt)
    for b, p in enumerate(pairs):
        n = p["n_in"]; rel_in = _relative(p["x_in"], p["center"], p["L"])
        xin[b, :n] = rel_in; sin[b, :n] = p["s_in"]; mask[b, :n] = True
        rel_out = _relative(p["x_out"], p["center"], p["L"])
        rr = rel_out.norm(dim=-1)
        keep = torch.nonzero(rr < p["R"] + r_ctx).squeeze(1)
        if keep.numel() > n_ctx_max:
            keep = keep[torch.argsort(rr[keep])[:n_ctx_max]]
        nc = keep.numel()
        xctx[b, :nc] = rel_out[keep]; sctx[b, :nc] = p["s_out"][keep]; cmask[b, :nc] = True
        radii[b] = p["R"]; temps[b] = float(p.get("T", 0.5))
    return {"x_in": xin, "s_in": sin, "mask": mask, "x_ctx": xctx,
            "s_ctx": sctx, "ctx_mask": cmask, "R": radii, "T": temps}


class _EquivariantLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.node = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.coord = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, h, x, valid):
        # i receives from j. Relative cavity coordinates do not need a box wrap
        # for the bounded pilot: their straight chord is the natural FM geometry.
        diff = x[:, :, None] - x[:, None, :]
        r2 = diff.square().sum(-1, keepdim=True)
        hi = h[:, :, None].expand(-1, -1, h.shape[1], -1)
        hj = h[:, None, :].expand(-1, h.shape[1], -1, -1)
        m = self.edge(torch.cat([hi, hj, r2], -1))
        edge_mask = valid[:, :, None] & valid[:, None, :]
        eye = torch.eye(h.shape[1], device=h.device, dtype=torch.bool)[None]
        edge_mask = edge_mask & ~eye
        mf = m * edge_mask[..., None]
        denom = edge_mask.sum(2, keepdim=True).clamp_min(1).to(h.dtype)
        agg = mf.sum(2) / denom
        h = h + self.node(torch.cat([h, agg], -1)) * valid[..., None]
        velocity = (diff * self.coord(m) * edge_mask[..., None]).sum(2) / denom
        return h, velocity


class CavityGenerator(nn.Module):
    def __init__(self, n_max, hidden_nf=64, n_layers=3, n_ctx_max=256):
        super().__init__(); self.n_max = n_max; self.n_ctx_max = n_ctx_max
        self.species = nn.Embedding(2, 8); self.kind = nn.Embedding(2, 4)
        self.embed = nn.Sequential(nn.Linear(8 + 4 + 4, hidden_nf), nn.SiLU(), nn.Linear(hidden_nf, hidden_nf))
        self.layers = nn.ModuleList([_EquivariantLayer(hidden_nf) for _ in range(n_layers)])
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, t, x_in, mask, s_in, x_ctx, s_ctx, ctx_mask, R, T):
        B, Ni = x_in.shape[:2]; Nc = x_ctx.shape[1]
        if Ni != self.n_max:
            raise ValueError(f"expected padded interior length {self.n_max}, got {Ni}")
        x = torch.cat([x_in, x_ctx], 1); valid = torch.cat([mask, ctx_mask], 1)
        species = torch.cat([s_in, s_ctx], 1)
        kind = torch.cat([torch.zeros(B, Ni, dtype=torch.long, device=x.device),
                          torch.ones(B, Nc, dtype=torch.long, device=x.device)], 1)
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        if t.ndim == 0: t = t.expand(B)
        scalars = torch.stack([t, R, T, 1.0 / R.clamp_min(.1)], -1)[:, None].expand(-1, Ni + Nc, -1)
        h = self.embed(torch.cat([self.species(species), self.kind(kind), scalars], -1))
        vel = torch.zeros_like(x)
        for layer in self.layers:
            h, dv = layer(h, x, valid); vel = vel + dv
        return self.out_scale * vel[:, :Ni] * mask[..., None]

    @torch.no_grad()
    def sample(self, s_in, mask, x_ctx, s_ctx, ctx_mask, R, T, n_steps=40):
        B = mask.shape[0]; x = _uniform_ball(B, self.n_max, R, mask.device, x_ctx.dtype)
        x = x * mask[..., None]; dt = 1.0 / n_steps
        for k in range(n_steps):
            v = self(k * dt, x, mask, s_in, x_ctx, s_ctx, ctx_mask, R, T)
            x = x + dt * v
            norm = x.norm(dim=-1, keepdim=True)
            cap = (R[:, None, None] - 1e-4).clamp_min(1e-4)
            x = torch.where(norm > cap, x / norm.clamp_min(1e-8) * cap, x)
            x = x * mask[..., None]
        return x


def cavity_fm_loss(model, batch):
    x1, mask, R = batch["x_in"], batch["mask"], batch["R"]
    B = x1.shape[0]; x0 = _uniform_ball(B, model.n_max, R, x1.device, x1.dtype) * mask[..., None]
    t = torch.rand(B, device=x1.device, dtype=x1.dtype)
    xt = ((1 - t[:, None, None]) * x0 + t[:, None, None] * x1) * mask[..., None]
    target = (x1 - x0) * mask[..., None]
    pred = model(t, xt, mask, batch["s_in"], batch["x_ctx"], batch["s_ctx"],
                 batch["ctx_mask"], R, batch["T"])
    denom = (mask.sum() * 3).clamp_min(1)
    return ((pred - target).square() * mask[..., None]).sum() / denom
