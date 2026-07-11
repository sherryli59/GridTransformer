"""3D learned-potential (energy-like embedding) cavity infiller — port of ka_localframe_ebm to 3D.

Tilts the FINAL axis c|a,b of the Cat3Head by a GENERAL learned pairwise radial potential
V_theta(x|cage) = sum_{i in cage} phi_theta(|x - x_i|, s_j, s_i), phi = MLP on RBF(dist) + species-pair
embedding. NO LJ / sigma / eps. cage = [boundary UNION placed interior] pooled with NO kind distinction
-> the user directive "boundary treated like surrounding particles" is automatic (the potential needs no
ordering and no anchor, unlike the AR context). Radius: reuse the per-slot _slot_features R embedding +
a zero-init global Fourier(R) token (R_embed) added to the context; potential itself is size-invariant.

Exact: any per-bin tilt is a valid categorical, so sample_block logq == block_log_prob scorer to fp
precision, independent of phi and of the frozen boundary. Design:
docs/superpowers/specs/2026-07-11-ka3d-learned-potential-cavity-design.md
"""
from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from liquid_coupling_flow.ka3d_scaffold_ar import (KA3DScaffoldCatAR, fixed_ball_scaffold,
                                                   ball_squash, ball_unsquash)

KNN_POT = 8


class KA3DScaffoldEBM(KA3DScaffoldCatAR):
    def __init__(self, *args, n_rbf=12, rbf_max=3.0, phi_hidden=32, pair_emb=8, r_fourier=6, **kw):
        super().__init__(*args, **kw)
        self.n_rbf, self.knn_pot = n_rbf, KNN_POT
        mu = torch.linspace(0.0, rbf_max, n_rbf)
        self.register_buffer("rbf_mu", mu); self.rbf_w = float(mu[1] - mu[0])
        self.pair_emb = nn.Embedding(self.n_species ** 2, pair_emb)
        self.phi = nn.Sequential(nn.Linear(n_rbf + pair_emb, phi_hidden), nn.SiLU(),
                                 nn.Linear(phi_hidden, phi_hidden), nn.SiLU(),
                                 nn.Linear(phi_hidden, 1))
        for p in self.phi[-1].parameters():
            nn.init.zeros_(p)                                       # phi ~ 0 -> starts == the base model
        self.register_buffer("r_freqs", torch.arange(1, r_fourier + 1).float())
        self.R_embed = nn.Sequential(nn.Linear(2 * r_fourier, self.d_model), nn.SiLU(),
                                     nn.Linear(self.d_model, self.d_model))
        nn.init.zeros_(self.R_embed[-1].weight); nn.init.zeros_(self.R_embed[-1].bias)   # start neutral

    # ---- radius token ----
    def _rfeat(self, R, dev, dtype):
        ang = torch.as_tensor(float(R), device=dev, dtype=dtype) * self.r_freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)])         # [2*r_fourier]

    def _r_bias(self, R, dev, dtype):
        return self.R_embed(self._rfeat(R, dev, dtype))            # [d_model]

    # ---- learned pairwise potential ----
    def _phi_pair(self, dist, sj, cage_s):
        r = torch.exp(-((dist[..., None] - self.rbf_mu) ** 2) / (2 * self.rbf_w ** 2))
        pair = (sj[..., None] * self.n_species + cage_s).clamp(0, self.n_species ** 2 - 1)
        return self.phi(torch.cat([r, self.pair_emb(pair)], -1)).squeeze(-1)

    def _cage_knn(self, combined, scomb, valid_row, anchor):
        """kNN nearest cage particles to each slot anchor, pooled over boundary+interior (no kind flag).
        combined [P,3], scomb [P], valid_row [S,P] bool, anchor [S,3]. Returns x/s/v [S,K,*]."""
        d2 = (combined[None] - anchor[:, None]).square().sum(-1).masked_fill(~valid_row, 1e18)
        K = min(self.knn_pot, combined.shape[0])
        idx = d2.topk(K, dim=1, largest=False).indices             # [S,K]
        return combined[idx], scomb[idx], torch.gather(valid_row, 1, idx)

    def _V_c(self, ba, bb, frame, anchor_y, cage_x, cage_s, cage_v, sj, R):
        """V over the c-bins at chosen (ba,bb): candidate u=(ua,ub,uc_grid) -> y=anchor_y+R_f^T u ->
        x=ball_squash -> pairwise phi to the cage. [S] in, [S,n_bins] out."""
        S, nb = ba.shape[0], self.flow.n
        uc = self.flow._ctr(torch.arange(nb, device=ba.device))    # [nb] u-space c centres
        ua, ub = self.flow._ctr(ba), self.flow._ctr(bb)            # [S]
        u = torch.stack([ua[:, None].expand(S, nb), ub[:, None].expand(S, nb), uc[None].expand(S, nb)], -1)
        y = anchor_y[:, None, :] + torch.einsum("saj,sba->sbj", frame, u)          # [S,nb,3]
        x, _ = ball_squash(y, R)                                                    # [S,nb,3]
        d = (x[:, :, None, :] - cage_x[:, None, :, :]).norm(dim=-1)                 # [S,nb,K]
        phi = self._phi_pair(d, sj[:, None].expand(S, nb),
                             cage_s[:, None, :].expand(S, nb, cage_s.shape[1]))     # [S,nb,K]
        return (phi * cage_v[:, None, :]).sum(-1)                                   # [S,nb]

    def _tilted_lp_u(self, h_e, u, frame, anchor_y, cage_x, cage_s, cage_v, sj, R, n_chunk=16):
        """Exact log q(u) with the c-axis tilted (teacher-forced, S slots). Mirrors Cat3Head.log_prob."""
        fl = self.flow
        ba, bb, bc = fl._bin(u[..., 0]), fl._bin(u[..., 1]), fl._bin(u[..., 2])
        la = F.log_softmax(fl.head_a(h_e), -1).gather(-1, ba[..., None]).squeeze(-1)
        lb = F.log_softmax(fl.head_b(h_e + fl.bin_a_emb(ba)), -1).gather(-1, bb[..., None]).squeeze(-1)
        base_c = fl.head_c(h_e + fl.bin_a_emb(ba) + fl.bin_b_emb(bb))               # [S,n_bins]
        Vc = torch.zeros_like(base_c)
        S = ba.shape[0]
        for c0 in range(0, S, n_chunk):
            sl = slice(c0, c0 + n_chunk)
            Vc[sl] = self._V_c(ba[sl], bb[sl], frame[sl], anchor_y[sl], cage_x[sl], cage_s[sl],
                               cage_v[sl], sj[sl], R)
        lc = F.log_softmax(base_c - Vc, -1).gather(-1, bc[..., None]).squeeze(-1)
        return la + lb + lc - fl._logbw3

    # ---- training: exact log-density of one cavity interior (c-tilted) ----
    def log_prob_pair(self, interior_rel, s_in, bnd_rel, s_bnd, R, preordered=False, n_chunk=16):
        xo, so = self._labeled(interior_rel, s_in, R, preordered)
        n, m, dev = xo.shape[0], bnd_rel.shape[0], xo.device
        anchors = fixed_ball_scaffold(n, R, dev, xo.dtype)
        combined = torch.cat([bnd_rel, xo], 0); scomb = torch.cat([s_bnd, so], 0)
        kind = torch.cat([torch.ones(m, dtype=torch.long, device=dev),
                          torch.zeros(n, dtype=torch.long, device=dev)])
        idxp = torch.arange(m + n, device=dev)
        valid = idxp[None] < (m + torch.arange(n, device=dev))[:, None]
        slot_feat = self._slot_features(anchors, torch.arange(n, device=dev), n, R)
        h, frame = self._frame_context(combined, scomb, kind, valid, anchors, anchors, slot_feat=slot_feat)
        h = h + self._r_bias(R, dev, h.dtype)
        oh = F.one_hot(so, self.n_species).to(h.dtype)
        rem = oh.sum(0, keepdim=True) - (oh.cumsum(0) - oh)
        lp_s = F.log_softmax(self.head_species(h).masked_fill(rem <= 0, float("-inf")), -1).gather(-1, so[:, None]).squeeze(-1)
        y, logdet_yx = ball_unsquash(xo, R); anchor_y, _ = ball_unsquash(anchors, R)
        u = torch.einsum("naj,nj->na", frame, y - anchor_y)
        cage_x, cage_s, cage_v = self._cage_knn(combined, scomb, valid, anchors)
        lp_u = self._tilted_lp_u(h + self.sp_out_emb(so), u, frame, anchor_y, cage_x, cage_s, cage_v, so, R, n_chunk)
        return (lp_s + lp_u + logdet_yx).sum()

    # ---- exact block-MTM (c-tilted); mirrors ka3d_block with the potential added ----
    def block_log_prob(self, xo_full, so_full, block_mask, bnd, s_bnd, R):
        n, m, dev = xo_full.shape[0], bnd.shape[0], xo_full.device
        anchors_full = fixed_ball_scaffold(n, R, dev, xo_full.dtype)
        order = torch.argsort(block_mask.to(torch.uint8), stable=True)
        xo, so, anchors = xo_full[order], so_full[order], anchors_full[order]
        reordered_block = block_mask[order].to(xo.dtype)
        combined = torch.cat([bnd, xo], 0); scomb = torch.cat([s_bnd, so], 0)
        kind = torch.cat([torch.ones(m, dtype=torch.long, device=dev),
                          torch.zeros(n, dtype=torch.long, device=dev)])
        idxp = torch.arange(m + n, device=dev)
        valid = idxp[None] < (m + torch.arange(n, device=dev))[:, None]
        slot_feat = self._slot_features(anchors, torch.arange(n, device=dev), n, R)
        h, frame = self._frame_context(combined, scomb, kind, valid, anchors, anchors, slot_feat=slot_feat)
        h = h + self._r_bias(R, dev, h.dtype)
        oh = F.one_hot(so, self.n_species).to(h.dtype)
        rem = oh.sum(0, keepdim=True) - (oh.cumsum(0) - oh)
        lp_s = F.log_softmax(self.head_species(h).masked_fill(rem <= 0, float("-inf")), -1).gather(-1, so[:, None]).squeeze(-1)
        y, logdet_yx = ball_unsquash(xo, R); anchor_y, _ = ball_unsquash(anchors, R)
        u = torch.einsum("naj,nj->na", frame, y - anchor_y)
        cage_x, cage_s, cage_v = self._cage_knn(combined, scomb, valid, anchors)
        lp_u = self._tilted_lp_u(h + self.sp_out_emb(so), u, frame, anchor_y, cage_x, cage_s, cage_v, so, R)
        return ((lp_s + lp_u + logdet_yx) * reordered_block).sum()

    @torch.no_grad()
    def sample_block(self, xo_full, so_full, block_mask, bnd, s_bnd, R, gen=None):
        fl = self.flow
        n, m, dev = xo_full.shape[0], bnd.shape[0], xo_full.device
        anchors_full = fixed_ball_scaffold(n, R, dev, xo_full.dtype)
        order = torch.argsort(block_mask.to(torch.uint8), stable=True)
        n_ret = int((~block_mask).sum())
        xo, so, anchors = xo_full[order].clone(), so_full[order].clone(), anchors_full[order]
        anchor_y, _ = ball_unsquash(anchors, R)
        combined = torch.cat([bnd, xo], 0); scomb = torch.cat([s_bnd, so], 0)
        kind = torch.cat([torch.ones(m, dtype=torch.long, device=dev),
                          torch.zeros(n, dtype=torch.long, device=dev)])
        idxp = torch.arange(m + n, device=dev)
        blk_s = so_full[block_mask]
        rem = torch.tensor([[float((blk_s == 0).sum()), float((blk_s == 1).sum())]], device=dev)
        rbias = self._r_bias(R, dev, xo.dtype)
        logq = xo.new_zeros(())
        for jj in range(n_ret, n):
            valid = (idxp < (m + jj))[None]
            slot_feat = self._slot_features(anchors[jj:jj + 1], jj, n, R)
            h, frame = self._frame_context(combined, scomb, kind, valid, anchors[jj:jj + 1], anchors[jj:jj + 1],
                                           slot_feat=slot_feat)
            h = h + rbias
            ls = F.log_softmax(self.head_species(h).masked_fill(rem <= 0, float("-inf")), -1)
            sj = torch.multinomial(ls.exp(), 1, generator=gen).squeeze(-1)
            he = h + self.sp_out_emb(sj)
            la = F.log_softmax(fl.head_a(he), -1); ba = torch.multinomial(la.exp(), 1, generator=gen).squeeze(-1)
            lb = F.log_softmax(fl.head_b(he + fl.bin_a_emb(ba)), -1); bb = torch.multinomial(lb.exp(), 1, generator=gen).squeeze(-1)
            cage_x, cage_s, cage_v = self._cage_knn(combined, scomb, valid, anchors[jj:jj + 1])
            Vc = self._V_c(ba, bb, frame, anchor_y[jj:jj + 1], cage_x, cage_s, cage_v, sj, R)
            lc = F.log_softmax(fl.head_c(he + fl.bin_a_emb(ba) + fl.bin_b_emb(bb)) - Vc, -1)
            bc = torch.multinomial(lc.exp(), 1, generator=gen).squeeze(-1)
            dith = (torch.rand(1, 3, device=dev, dtype=xo.dtype, generator=gen) - 0.5) * fl.bw
            u = torch.stack([fl._ctr(ba), fl._ctr(bb), fl._ctr(bc)], -1) + dith
            y = anchor_y[jj] + torch.einsum("naj,na->nj", frame, u)[0]
            pos, logdet_xy = ball_squash(y, R)
            xo[jj], so[jj] = pos, sj
            combined[m + jj], scomb[m + jj] = pos, sj
            rem[0, sj] -= 1
            lp_u = (la.gather(1, ba[:, None]).squeeze() + lb.gather(1, bb[:, None]).squeeze()
                    + lc.gather(1, bc[:, None]).squeeze() - fl._logbw3)
            logq = logq + ls.gather(1, sj[:, None]).squeeze() + lp_u - logdet_xy.squeeze()
        xo_new, so_new = xo_full.clone(), so_full.clone()
        xo_new[order], so_new[order] = xo, so
        return xo_new, so_new, logq
