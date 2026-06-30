"""Proposal-B: ClusterFlow — exact-likelihood, E(2)-equivariant coupling flow over the k cluster particles'
in-frame coordinates. Anchor-centered Gaussian base + per-block geometric distance-attention conditioner.
See docs/superpowers/specs/2026-06-30-ka-cluster-flow-proposal-b-design.md. Interface mirrors ClusterProposal."""
from __future__ import annotations
import os, math, torch, torch.nn as nn, torch.nn.functional as F
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold
from liquid_coupling_flow.ka_flow_coupling import GeomAttnLayer

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def anchor_base_logp(z, q_scaf, sigma_b):                     # z,q_scaf [B,k,2] -> [B]
    d2 = ((z - q_scaf) ** 2).sum(-1)                          # [B,k]
    return (-d2 / (2 * sigma_b ** 2) - math.log(2 * math.pi * sigma_b ** 2)).sum(-1)


def sample_base(q_scaf, sigma_b, gen=None):                  # [B,k,2]
    return q_scaf + sigma_b * torch.randn(q_scaf.shape, generator=gen, device=q_scaf.device, dtype=q_scaf.dtype)


@torch.no_grad()
def compute_sigma_b(data, s, geo, sc, L, k):
    """Isotropic std of each cluster particle's min-imaged LAB displacement from its own scaffold slot. This is
    the FIXED base width (rotation-invariant, so no frame needed). Expected ~0.747 on ka_reference_N100."""
    from liquid_coupling_flow.ka_gridformer import _wrap_pm
    N = data.shape[1]; pos0, _ = slot_order(data, s, geo, N); disp = []
    for seed in range(0, N, 3):
        cl = KC.cluster_slots(seed, sc, k, L)
        disp.append(_wrap_pm(pos0[:, cl] - sc[cl][None], L).reshape(-1, 2))
    return float(torch.cat(disp, 0).std().item())


class _GeomEdgeBiasB(nn.Module):
    """GeomEdgeBias with PER-BATCH species (s [B,M]). Bias from loc-coord distance + pair-type embedding."""
    def __init__(self, n_head, n_species, n_rbf=16, cutoff=2.4):
        super().__init__()
        self.n_species = n_species
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = (n_rbf / cutoff) ** 2
        self.pt_emb = nn.Embedding(n_species * n_species, n_rbf)
        self.mlp = nn.Sequential(nn.Linear(n_rbf, n_rbf), nn.SiLU(), nn.Linear(n_rbf, n_head))

    def forward(self, d, s):                                      # d [B,M,M], s [B,M] long
        rbf = torch.exp(-self.gamma * (d[..., None] - self.centers) ** 2)         # [B,M,M,R]
        pt = (s[:, :, None] * self.n_species + s[:, None, :])                     # [B,M,M]
        feat = rbf + self.pt_emb(pt)                                              # [B,M,M,R]
        return self.mlp(feat).permute(0, 3, 1, 2)                                 # [B,nH,M,M]


class ClusterConditioner(nn.Module):
    """Per coupling block, builds one token per {cluster ∪ x_R} particle and runs geometric attention to emit
    spline params for the ACTIVE cluster particles. Active cluster coord c is MASKED -> params independent of it."""
    def __init__(self, num_bins=8, n_cycles=4, n_species=2, d_model=192, n_head=6, n_layer=4, box=4.0,
                 n_rbf=16, cutoff=2.4):
        super().__init__()
        self.P = 3 * num_bins - 1; self.box = box
        self.meta = [(par, c) for _ in range(n_cycles) for par in (0, 1) for c in (0, 1)]
        self.enc = nn.Linear(4, d_model)                         # featpos(pos) = [p, sin(pi p/box)] per coord -> 4
        self.sp_emb = nn.Embedding(n_species, d_model)
        self.role_emb = nn.Embedding(3, d_model)                 # 0=x_R, 1=frozen cluster, 2=active cluster
        self.block_emb = nn.Embedding(len(self.meta), d_model)
        self.cmask = nn.Parameter(torch.randn(1) * 0.02)         # stands in for the masked active c
        self.edge_bias = _GeomEdgeBiasB(n_head, n_species, n_rbf, cutoff)
        self.layers = nn.ModuleList([GeomAttnLayer(d_model, n_head) for _ in range(n_layer)])
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.P))
                                    for _ in self.meta])

    def _featpos(self, p):                                       # [...,2] -> [...,4]
        return torch.cat([p, torch.sin(math.pi * p / self.box)], -1)

    def params(self, b, u, u_ctx, sp_cl, sp_ctx, c, par):
        Bsz, k, _ = u.shape; nctx = u_ctx.shape[1]; dev = u.device
        amask_cl = (torch.arange(k, device=dev) % 2 == par)                      # [k] active cluster
        # mask the active cluster's transformed coord c
        u_m = u.clone()
        u_m[:, amask_cl, c] = self.cmask
        # token features for cluster then x_R
        feat_cl = self._featpos(u_m); feat_ctx = self._featpos(u_ctx)
        tok_cl = self.enc(feat_cl) + self.sp_emb(sp_cl)
        tok_ctx = self.enc(feat_ctx) + self.sp_emb(sp_ctx) + self.role_emb.weight[0]
        role_cl = torch.where(amask_cl, 2, 1)                                     # [k]
        tok_cl = tok_cl + self.role_emb(role_cl)[None]
        tok = torch.cat([tok_cl, tok_ctx], 1) + self.block_emb.weight[b]          # [B,M,d], M=k+nctx
        # loc-coord (1-c) distance among all tokens (known for all; non-periodic)
        loc = 1 - c
        loc_all = torch.cat([u[:, :, loc], u_ctx[:, :, loc]], 1)                  # [B,M]
        d = (loc_all[:, :, None] - loc_all[:, None, :]).abs()                     # [B,M,M]
        s_all = torch.cat([sp_cl, sp_ctx], 1)                                     # [B,M]
        bias = self.edge_bias(d, s_all)
        h = tok
        for layer in self.layers:
            h = layer(h, bias)
        return self.heads[b](h[:, :k][:, amask_cl])                              # [B,Na,P]
