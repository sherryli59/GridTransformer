from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineCoupling(nn.Module):
    """Diagonal affine map on [...,3] with tanh-bounded log-scale (identity when params=0).
    forward:  u_out = u * exp(s) + t,  s = smax*tanh(params[...,:3]),  t = params[...,3:]
    logdet = sum(s). inverse: u = (u_out - t) * exp(-s)."""
    def __init__(self, smax: float = 2.0):
        super().__init__()
        self.smax = float(smax)

    def _st(self, params):
        s = self.smax * torch.tanh(params[..., :3])
        t = params[..., 3:]
        return s, t

    def forward(self, u, params):
        s, t = self._st(params)
        return u * torch.exp(s) + t, s.sum(-1)

    def inverse(self, u_out, params):
        s, t = self._st(params)
        return (u_out - t) * torch.exp(-s)


class CageConditioner(nn.Module):
    """Per-active-particle affine params from a distance-RBF + species-pair message over the conditioning
    set (other block particles + cage). Physical positions; batched [M,A,*] x [M,C,*] -> [M,A,6]. Last
    layer zero-init => identity coupling at init (composition starts == base)."""
    def __init__(self, d_model: int = 96, n_rbf: int = 12, rbf_max: float = 3.0, n_species: int = 2):
        super().__init__()
        self.n_species = n_species
        mu = torch.linspace(0.0, rbf_max, n_rbf)
        self.register_buffer("mu", mu); self.w = float(mu[1] - mu[0])
        self.pair = nn.Embedding(n_species * n_species, 8)
        self.msg = nn.Sequential(nn.Linear(n_rbf + 8, d_model), nn.SiLU(),
                                 nn.Linear(d_model, d_model), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 6))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def params(self, active_x, active_s, cond_x, cond_s, R):
        d = (active_x[:, :, None, :] - cond_x[:, None, :, :]).norm(dim=-1)          # [M,A,C]
        r = torch.exp(-((d[..., None] - self.mu) ** 2) / (2 * self.w ** 2))         # [M,A,C,n_rbf]
        pr = (active_s[:, :, None] * self.n_species + cond_s[:, None, :]).clamp(0, self.n_species ** 2 - 1)
        msg = self.msg(torch.cat([r, self.pair(pr)], -1))                          # [M,A,C,d]
        agg = msg.mean(2)                                                          # permutation-invariant over cond
        return self.head(agg)                                                     # [M,A,6]


from liquid_coupling_flow.ka3d_scaffold_ar import ball_squash


class BlockCorrector(nn.Module):
    """Stacked conditional affine-coupling flow over the K block particles' u-coords. Layer L transforms
    the parity-L half of block particles conditioned on the other half + the frozen cage."""
    def __init__(self, n_layers: int = 6, d_model: int = 96, n_rbf: int = 12, smax: float = 2.0):
        super().__init__()
        self.n_layers = n_layers
        self.couple = AffineCoupling(smax=smax)
        self.conds = nn.ModuleList([CageConditioner(d_model, n_rbf) for _ in range(n_layers)])

    def _phys(self, u, anchor_y, R):
        x, _ = ball_squash(u + anchor_y[None], R)                                 # [M,K,3]
        return x

    def _run(self, u, s_blk, anchor_y, cage_x, cage_s, R, invert):
        M, K, _ = u.shape
        dev = u.device
        idx = torch.arange(K, device=dev)
        order = range(self.n_layers - 1, -1, -1) if invert else range(self.n_layers)
        logdet = u.new_zeros(M)
        # Fixed (non-evolving) per-slot anchor position, used as the ACTIVE side's own query location so
        # the conditioner is a function only of quantities invariant to this layer's transform (the
        # inactive half + cage + each active particle's fixed anchor) -- required for exact invertibility:
        # using the evolving xu[:, active] instead would differ between forward (pre-transform) and
        # inverse (post-transform, same layer) since that IS the value being solved for.
        anchor_x = self._phys(u.new_zeros(u.shape), anchor_y, R)                  # [M,K,3], layer-invariant
        for L in order:
            active = (idx % 2) == (L % 2)                                         # alternating half
            passive = ~active
            xu = self._phys(u, anchor_y, R)                                       # current physical block
            cond_x = torch.cat([xu[:, passive], cage_x], 1)                       # other block + cage
            cond_s = torch.cat([s_blk[:, passive], cage_s], 1)
            params_full = u.new_zeros(M, K, 6)
            params_full[:, active] = self.conds[L].params(
                anchor_x[:, active], s_blk[:, active], cond_x, cond_s, R)
            if invert:
                u = u.clone()
                u[:, active] = self.couple.inverse(u[:, active], params_full[:, active])
                logdet = logdet - self.couple.forward(u[:, active] * 0, params_full[:, active])[1].sum(-1)  # -sum s
            else:
                ua, ld = self.couple.forward(u[:, active], params_full[:, active])
                u = u.clone(); u[:, active] = ua
                logdet = logdet + ld.sum(-1)
        return u, logdet

    def forward(self, u_blk, s_blk, anchor_y_blk, cage_x, cage_s, R):
        return self._run(u_blk, s_blk, anchor_y_blk, cage_x, cage_s, R, invert=False)

    def inverse(self, u_out, s_blk, anchor_y_blk, cage_x, cage_s, R):
        return self._run(u_out, s_blk, anchor_y_blk, cage_x, cage_s, R, invert=True)
