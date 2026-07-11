"""LEARNED-POTENTIAL (energy-like embedding) head for the local-frame AR conditional.

Motivation: the block-MTM clash is dominated by block<->RETAINED overlaps (39% in 3D) -- the factorized
categorical can't sharply represent excluded-volume against every cage neighbour. A hand-coded -beta*U tilt
would fix it but bakes in the LJ form (kills generality/transfer). Instead we learn a GENERAL energy-like
embedding: a pairwise, radial potential V_theta(x | cage) = sum_i phi_theta(|x - x_i|, s, s_i) summed over
cage neighbours, and TILT the proposal by it. phi_theta is an MLP on (distance, species-pair) -- no LJ, no
sigma/eps; it learns the potential-of-mean-force from data. It inherits size-transfer for free (a pairwise
sum-with-cutoff is the same computational shape as the energy) and it is the exact object the campaign's
heat-bath lever used (ka-heatbath-depth-lever), but LEARNED rather than given.

Exactness/tractability: a full 2D joint-grid EBM needs a 192x192 normaliser per particle (infeasible in a
training batch). We keep the factorized head and tilt only the FINAL axis b|a by the true 2D potential
evaluated at the chosen a-bin over the 192 b-bins: q(b|a) ~ exp(head_b(b|a) - V(a,b)). The particle's last
coordinate dodges its neighbours; normaliser is 192 evals; any per-bin tilt is a valid categorical so
log_q is closed-form both directions (sample()==log_prob() to fp precision)."""
from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from liquid_coupling_flow.ka_localframe import KALocalFrameModel, SIGMA
from liquid_coupling_flow.ka_gridformer import _wrap_pm

KNN_POT = 8            # nearest neighbours the potential sums over (exclusion is dominated by the nearest)


class KALocalFrameEBM(KALocalFrameModel):
    """Local-frame AR whose b|a categorical is tilted by a LEARNED pairwise radial potential over the cage.
    a-axis and species use the inherited factorized heads unchanged; only b|a gains the energy-like tilt."""
    def __init__(self, n_rbf=12, rbf_max=3.0, phi_hidden=32, pair_emb=8, **kw):
        super().__init__(**kw)
        self.n_rbf, self.knn_pot = n_rbf, KNN_POT
        mu = torch.linspace(0.0, rbf_max, n_rbf)
        self.register_buffer("rbf_mu", mu)
        self.rbf_w = float(mu[1] - mu[0])                               # RBF width = spacing
        self.pair_emb = nn.Embedding(self.n_species * self.n_species, pair_emb)   # (s_j, s_k) -> vec
        self.phi = nn.Sequential(nn.Linear(n_rbf + pair_emb, phi_hidden), nn.SiLU(),
                                 nn.Linear(phi_hidden, phi_hidden), nn.SiLU(),
                                 nn.Linear(phi_hidden, 1))
        # init phi near zero so the model starts == the factorized head (stable warm start)
        for p in self.phi[-1].parameters():
            nn.init.zeros_(p)

    # ---- learned pairwise potential over the b-column at a fixed a-bin ----
    def _phi_pair(self, dist, sj, nbr_sp):
        """dist [...,K], sj [...], nbr_sp [...,K] -> phi [...,K] (learned pairwise energy contribution)."""
        r = torch.exp(-((dist[..., None] - self.rbf_mu) ** 2) / (2 * self.rbf_w ** 2))       # [...,K,n_rbf]
        pair = (sj[..., None] * self.n_species + nbr_sp).clamp(0, self.n_species ** 2 - 1)   # [...,K]
        pe = self.pair_emb(pair)                                                             # [...,K,pair_emb]
        return self.phi(torch.cat([r, pe], -1)).squeeze(-1)                                  # [...,K]

    def _V_b(self, ba, nbr_rel, nbr_sp, valid, sj, arc, bcenters):
        """V over the 192 b-bins at the chosen a-bin `ba`. nbr_rel [...,K,2] neighbour coords in the LOCAL
        physical frame (rel to origin); returns V [...,n_bins]. Short-range -> only nearest knn_pot used."""
        K = min(self.knn_pot, nbr_rel.shape[-2])
        nr, ns, vv = nbr_rel[..., :K, :], nbr_sp[..., :K], valid[..., :K].float()
        a_phys = (self._bin_center(ba) * arc)[..., None]                                     # [...,1] fixed a
        b_phys = bcenters * arc                                                              # [n_bins]
        dx = a_phys[..., None] - nr[..., None, :, 0]                                          # [...,n_bins,K]
        dy = b_phys[..., None] - nr[..., None, :, 1]                                          # [...,n_bins,K]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-12)
        phi = self._phi_pair(dist, sj[..., None].expand(dist.shape[:-1]), ns[..., None, :].expand(dist.shape))
        return (phi * vv[..., None, :]).sum(-1)                                              # [...,n_bins]

    # ---- teacher-forced context that ALSO returns neighbour coords for the potential ----
    def _local_ebm(self, pos, sp, sc, L, N):
        B = pos.shape[0]
        d = _wrap_pm(pos[:, None, :, :] - sc[None, :, None, :], L)
        dist2 = (d ** 2).sum(-1)
        jj = torch.arange(N, device=pos.device)
        causal = jj[None, None, :] < jj[None, :, None]
        w = torch.exp(-dist2.masked_fill(~causal, 1e9) / (2 * SIGMA ** 2)) * causal.float()
        wsum = w.sum(-1, keepdim=True).clamp_min(1e-12)
        origin = torch.remainder(sc[None] + (w[..., None] * d).sum(2) / wsum, L)
        origin = torch.where((jj < 2)[None, :, None], sc[None].expand(B, N, 2), origin)
        idx = dist2.masked_fill(~causal, 1e9).topk(self.knn, dim=2, largest=False).indices
        valid = torch.gather(causal.expand(B, N, N), 2, idx)
        nbr_pos = torch.gather(pos[:, None].expand(B, N, N, 2), 2, idx[..., None].expand(-1, -1, -1, 2))
        nbr_rel = _wrap_pm(nbr_pos - origin[:, :, None, :], L)
        nbr_sp = torch.gather(sp[:, None].expand(B, N, N), 2, idx)
        feat = self.nbr_proj(self._periodic(nbr_rel)) + self.sp_emb(nbr_sp)
        feat = feat * valid[..., None]
        q = self.query.expand(B * N, 1, -1)
        seq = torch.cat([q, feat.reshape(B * N, self.knn, -1)], 1)
        pad = torch.cat([torch.zeros(B * N, 1, dtype=torch.bool, device=pos.device),
                         ~valid.reshape(B * N, self.knn)], 1)
        h = self.tr(seq, src_key_padding_mask=pad)[:, 0].reshape(B, N, -1)
        return h, origin, nbr_rel, nbr_sp, valid

    def log_prob(self, x, s, canonical=None, preordered=False, n_chunk=16):
        B, N = x.shape[0], x.shape[1]; s = s.long(); s = s.expand(B, N).clone() if s.dim() == 1 else s
        L = self._Lof(N); arc = self._arc_scale(N)
        if preordered:
            xo, so = x, s
        else:
            order = self.geo._curve_order(x, N)
            xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s, 1, order)
        sc = self.geo._scaffold(N, x.device)
        context, origin, nbr_rel, nbr_sp, valid = self._local_ebm(xo, so, sc, L, N)
        ab = _wrap_pm(xo - origin, L) / arc
        ba, bb = self._bin(ab[..., 0]), self._bin(ab[..., 1])
        la = F.log_softmax(self.head_a(context), -1)
        base_b = self.head_b(context + self.bin_a_emb(ba))                                    # [B,N,n_bins]
        bc = self._bin_center(torch.arange(self.n_bins, device=x.device))
        V = torch.zeros_like(base_b)
        for c0 in range(0, N, n_chunk):                                                       # chunk N for memory
            sl = slice(c0, c0 + n_chunk)
            V[:, sl] = self._V_b(ba[:, sl], nbr_rel[:, sl], nbr_sp[:, sl], valid[:, sl], so[:, sl], arc, bc)
        lb = F.log_softmax(base_b - V, -1)
        lp_ab = la.gather(-1, ba[..., None]).squeeze(-1) + lb.gather(-1, bb[..., None]).squeeze(-1)
        s_logits = self.head_species(context)
        if self.canonical if canonical is None else canonical:
            oh = F.one_hot(so, self.n_species).to(s_logits.dtype)
            rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)
            s_logits = s_logits.masked_fill(rem <= 0, float("-inf"))
        lp_s = F.log_softmax(s_logits, -1).gather(-1, so[..., None]).squeeze(-1)
        vol = self.d * N * math.log(self.bin_w); jac = self.d * N * math.log(arc)
        return (lp_ab + lp_s).sum(1) - vol - jac

    # ---- per-step context for sampling that ALSO returns neighbour coords ----
    def _step_ebm(self, pos, sp, sc_j, j, L):
        B = pos.shape[0]
        d = _wrap_pm(pos[:, :j] - sc_j[None, None], L); dist2 = (d ** 2).sum(-1)
        if j < 2:
            origin = sc_j[None].expand(B, 2)
        else:
            w = torch.exp(-dist2 / (2 * SIGMA ** 2)); wsum = w.sum(1, keepdim=True).clamp_min(1e-12)
            origin = torch.remainder(sc_j[None] + (w[..., None] * d).sum(1) / wsum, L)
        k = min(self.knn, j)
        idx = dist2.topk(k, dim=1, largest=False).indices
        nbr_pos = torch.gather(pos[:, :j], 1, idx[..., None].expand(-1, -1, 2))
        nbr_rel = _wrap_pm(nbr_pos - origin[:, None, :], L)
        nbr_sp = torch.gather(sp[:, :j], 1, idx)
        feat = self.nbr_proj(self._periodic(nbr_rel)) + self.sp_emb(nbr_sp)
        h = self.tr(torch.cat([self.query.expand(B, 1, -1), feat], 1))[:, 0]
        valid = torch.ones(B, k, dtype=torch.bool, device=pos.device)
        return h, origin, nbr_rel, nbr_sp, valid
