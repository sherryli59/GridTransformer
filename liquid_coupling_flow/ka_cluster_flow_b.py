"""Proposal-B: ClusterFlow — exact-likelihood, E(2)-equivariant coupling flow over the k cluster particles'
in-frame coordinates. Anchor-centered Gaussian base + per-block geometric distance-attention conditioner.
See docs/superpowers/specs/2026-06-30-ka-cluster-flow-proposal-b-design.md. Interface mirrors ClusterProposal."""
from __future__ import annotations
import os, math, torch, torch.nn as nn, torch.nn.functional as F
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold  # re-exported for tests
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
        dtype = u.dtype                                                            # may be float64 (autograd test)
        amask_cl = (torch.arange(k, device=dev) % 2 == par)                      # [k] active cluster
        # mask the active cluster's transformed coord c; run network in fp32 (weight dtype)
        u_m = u.float().clone()
        u_m[:, amask_cl, c] = self.cmask
        u_f = u.float(); u_ctx_f = u_ctx.float()
        # loc-coord (1-c) distance among all tokens (known for all; non-periodic)
        loc = 1 - c
        loc_all = torch.cat([u_f[:, :, loc], u_ctx_f[:, :, loc]], 1)             # [B,M]
        d = (loc_all[:, :, None] - loc_all[:, None, :]).abs()                     # [B,M,M]
        s_all = torch.cat([sp_cl, sp_ctx], 1)                                     # [B,M]
        # Disable TF32 so the attention/matmul results are reproducible across forward and inverse
        # passes (TF32 introduces ~1e-3 non-determinism that accumulates over n_cycles*4 blocks).
        old_mm = torch.backends.cuda.matmul.allow_tf32
        old_nn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            # token features for cluster then x_R
            feat_cl = self._featpos(u_m); feat_ctx = self._featpos(u_ctx_f)
            tok_cl = self.enc(feat_cl) + self.sp_emb(sp_cl)
            tok_ctx = self.enc(feat_ctx) + self.sp_emb(sp_ctx) + self.role_emb.weight[0]
            role_cl = torch.where(amask_cl, 2, 1)                                 # [k]
            tok_cl = tok_cl + self.role_emb(role_cl)[None]
            tok = torch.cat([tok_cl, tok_ctx], 1) + self.block_emb.weight[b]     # [B,M,d], M=k+nctx
            bias = self.edge_bias(d, s_all)
            h = tok
            for layer in self.layers:
                h = layer(h, bias)
            return self.heads[b](h[:, :k][:, amask_cl]).to(dtype)                # [B,Na,P] in target dtype
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_mm
            torch.backends.cudnn.allow_tf32 = old_nn


from liquid_coupling_flow.transforms_spline import RQSplineElementwise


class ClusterFlow(nn.Module):
    def __init__(self, sigma_b, num_bins=8, n_cycles=4, n_ctx=32, box=4.0, tail_bound=4.0,
                 d_model=192, n_head=6, n_layer=4, n_species=2):
        super().__init__()
        self.sigma_b = float(sigma_b); self.n_ctx = n_ctx; self.box = box
        self.tail_bound = float(tail_bound)
        self.spline = RQSplineElementwise(num_bins=num_bins, tail_bound=tail_bound)
        self.cond = ClusterConditioner(num_bins=num_bins, n_cycles=n_cycles, n_species=n_species,
                                       d_model=d_model, n_head=n_head, n_layer=n_layer, box=box)

    def _safe_tail_bound(self, L):
        """Largest tail_bound s.t. from_frame(u) -> to_frame round-trip is exact.
        In-frame coords u have |u|_2 <= sqrt(2)*tail_bound; the round-trip through KC.from_frame/to_frame
        is exact iff u @ R + origin stays in [0, L]^2. Worst case: |u@R|_inf <= |u|_2 so we need
        sqrt(2)*tail_bound < L/2, i.e. tail_bound < L/(2*sqrt(2)). Use 0.99 safety margin."""
        return min(self.tail_bound, float(L) / (2 * math.sqrt(2)) * 0.99)

    def _frame_ctx(self, pos, s, cluster_idx, sc, L):
        """Frame (from the cluster_frame's own 16 slots) + the conditioner's n_ctx x_R tokens + q_scaf anchors."""
        origin, R = KC.cluster_frame(pos, cluster_idx, sc, L)                     # x_R-only frame (16 slots)
        # cluster_frame forces fp32 internally; cast to match pos dtype (e.g. double in the autograd test)
        origin = origin.to(pos.dtype); R = R.to(pos.dtype)
        slots = KC.frame_ctx_slots(cluster_idx, sc, L, self.n_ctx)
        u_ctx = KC.to_frame(pos[:, slots, :], origin, R, L)                       # [B,n_ctx,2]
        sp_ctx = s[:, slots]
        q_scaf = KC.to_frame(sc[cluster_idx].to(pos.dtype)[None].expand(pos.shape[0], -1, -1), origin, R, L)
        return origin, R, u_ctx, sp_ctx, q_scaf

    def _amask(self, k, par, dev):
        return (torch.arange(k, device=dev) % 2 == par)

    def _z_to_u(self, z, u_ctx, sp_cl, sp_ctx, tail_bound):
        """Base z [B,k,2] -> cluster in-frame u, accumulating forward log-det (sum over active coords/blocks)."""
        u = z.clone(); ld = torch.zeros(u.shape[0], device=u.device)
        old_tb = self.spline.tail_bound; self.spline.tail_bound = tail_bound
        try:
            for b in range(len(self.cond.meta)):
                par, c = self.cond.meta[b]; am = self._amask(u.shape[1], par, u.device)
                p = self.cond.params(b, u, u_ctx, sp_cl, sp_ctx, c, par)          # [B,Na,P]
                y, d = self.spline.forward(u[:, am, c], p)
                u = u.clone(); u[:, am, c] = y; ld = ld + d.sum(-1)
        finally:
            self.spline.tail_bound = old_tb
        return u, ld

    def _x_to_z(self, pos, s, cluster_idx, xC_query, sc, L):
        """Cluster lab xC_query -> base z (flow inverse), accumulating inverse log-det. Returns (z, sum_logdet)."""
        origin, R, u_ctx, sp_ctx, _ = self._frame_ctx(pos, s, cluster_idx, sc, L)
        sp_cl = s[:, cluster_idx]
        u = KC.to_frame(xC_query, origin, R, L); ld = torch.zeros(u.shape[0], device=u.device)
        tail_bound = self._safe_tail_bound(L)
        old_tb = self.spline.tail_bound; self.spline.tail_bound = tail_bound
        try:
            for b in reversed(range(len(self.cond.meta))):
                par, c = self.cond.meta[b]; am = self._amask(u.shape[1], par, u.device)
                p = self.cond.params(b, u, u_ctx, sp_cl, sp_ctx, c, par)
                z, d = self.spline.inverse(u[:, am, c], p)
                u = u.clone(); u[:, am, c] = z; ld = ld + d.sum(-1)
        finally:
            self.spline.tail_bound = old_tb
        return u, ld

    @torch.no_grad()
    def sample(self, pos, s, cluster_idx, sc, L):
        origin, R, u_ctx, sp_ctx, q_scaf = self._frame_ctx(pos, s, cluster_idx, sc, L)
        sp_cl = s[:, cluster_idx]
        z = sample_base(q_scaf, self.sigma_b)
        tail_bound = self._safe_tail_bound(L)
        u, ld = self._z_to_u(z, u_ctx, sp_cl, sp_ctx, tail_bound)
        xC_lab = KC.from_frame(u, origin, R, L)
        logq = anchor_base_logp(z, q_scaf, self.sigma_b) - ld
        return xC_lab, logq

    def log_q(self, pos, s, cluster_idx, xC_query, sc, L):
        _, _, _, _, q_scaf = self._frame_ctx(pos, s, cluster_idx, sc, L)
        z, ld = self._x_to_z(pos, s, cluster_idx, xC_query, sc, L)
        return anchor_base_logp(z, q_scaf, self.sigma_b) + ld
