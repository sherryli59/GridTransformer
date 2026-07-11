"""Batched-over-M block methods for KA3DScaffoldEBM: run M independent cavity interiors (SAME frozen
boundary, SAME cavity/radius) through the AR generator IN PARALLEL, so the transformer sees M*S slots at
once instead of S slots M times. This turns the SMC's per-step cost (M full-scores + M*nmut block moves)
into a handful of big GPU calls -> fixes the ~10% GPU utilisation (CPU/launch-bound) of the per-particle loop.

Assumptions (all hold for the cavity models): FRAMELESS (use_frame=False -> Rf=I) and the scaffold model's
origin == anchors (shared across M). Only positions/species carry the M dimension; anchors, valid mask,
kind, slot features are shared. Exactness gate at bottom: batched == stacked-unbatched to fp precision."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from liquid_coupling_flow.ka3d_scaffold_ebm import KA3DScaffoldEBM
from liquid_coupling_flow.ka3d_scaffold_ar import fixed_ball_scaffold, ball_squash, ball_unsquash


class KA3DScaffoldEBMBatched(KA3DScaffoldEBM):
    # ---- batched KNN transformer context (frameless): [M,P,3] -> h [M,S,d] ----
    def _fc_batched(self, combined, scomb, kind, valid_row, anchors, slot_feat):
        M, P, _ = combined.shape
        S = valid_row.shape[0]
        d2 = (combined[:, None] - anchors[None, :, None]).square().sum(-1)          # [M,S,P]
        is_bnd = (kind == 1)[None, None].expand(M, S, P)
        vr = valid_row[None].expand(M, S, P)
        valid_int = vr & ~is_bnd
        valid_bnd = vr & is_bnd & (d2 < self.bnd_cutoff ** 2)

        def gather(mask, k_max):
            k = min(k_max, P)
            idx = d2.masked_fill(~mask, 1e18).topk(k, dim=2, largest=False).indices  # [M,S,k]
            return idx, torch.gather(mask, 2, idx)

        cexp = combined[:, None].expand(M, S, P, 3)

        def feats(idx, v):
            nbr = torch.gather(cexp, 2, idx[..., None].expand(-1, -1, -1, 3)) - anchors[None, :, None, :]
            sp = torch.gather(scomb[:, None].expand(M, S, P), 2, idx)
            f = self.nbr_proj(self._periodic(nbr)) + self.sp_emb(sp) + self.kind_emb(kind[idx])
            return f * v[..., None]

        idx_i, v_i = gather(valid_int, self.knn)
        idx_b, v_b = gather(valid_bnd, self.knn_bnd)
        fi, fb = feats(idx_i, v_i), feats(idx_b, v_b)
        q = self.query.reshape(1, 1, -1) + self.slot_proj(slot_feat)[None]          # [1,S,d] (broadcast M)
        q = q.expand(M, S, -1).reshape(M * S, 1, -1)
        ki, kb = idx_i.shape[2], idx_b.shape[2]
        seq = torch.cat([q, fi.reshape(M * S, ki, -1), fb.reshape(M * S, kb, -1)], 1)
        pad = torch.cat([torch.zeros(M * S, 1, dtype=torch.bool, device=combined.device),
                         ~v_i.reshape(M * S, ki), ~v_b.reshape(M * S, kb)], 1)
        return self.tr(seq, src_key_padding_mask=pad)[:, 0].reshape(M, S, -1)       # [M,S,d]

    def _cage_knn_b(self, combined, scomb, valid_row, anchors):
        M, P, _ = combined.shape
        S = valid_row.shape[0]
        d2 = (combined[:, None] - anchors[None, :, None]).square().sum(-1).masked_fill(~valid_row[None], 1e18)
        K = min(self.knn_pot, P)
        idx = d2.topk(K, dim=2, largest=False).indices                             # [M,S,K]
        cage_x = torch.gather(combined[:, None].expand(M, S, P, 3), 2, idx[..., None].expand(-1, -1, -1, 3))
        cage_s = torch.gather(scomb[:, None].expand(M, S, P), 2, idx)
        cage_v = torch.gather(valid_row[None].expand(M, S, P), 2, idx)
        return cage_x, cage_s, cage_v

    def _V_axis_b(self, u_fixed, grid_axis, anchor_y, cage_x, cage_s, cage_v, sj, R, net, emb):
        """u_fixed [M,S,3]; grid swept along grid_axis; frameless (y=anchor_y+u). Returns V [M,S,n_bins]."""
        M, S, _ = u_fixed.shape
        nb = self.flow.n
        grid = self.flow._ctr(torch.arange(nb, device=u_fixed.device))
        u = u_fixed[:, :, None, :].expand(M, S, nb, 3).clone()
        u[..., grid_axis] = grid[None, None]
        y = anchor_y[None, :, None, :] + u                                          # frameless Rf=I
        x, _ = ball_squash(y, R)                                                     # [M,S,nb,3]
        d = (x[:, :, :, None, :] - cage_x[:, :, None, :, :]).norm(dim=-1)            # [M,S,nb,K]
        phi = self._phi_pair(d, sj[:, :, None].expand(M, S, nb),
                             cage_s[:, :, None, :].expand(M, S, nb, cage_s.shape[-1]), net, emb)
        return (phi * cage_v[:, :, None, :]).sum(-1)                                # [M,S,nb]

    def _tilted_lp_u_b(self, h_e, u, anchor_y, cage_x, cage_s, cage_v, sj, R, s_chunk=4):
        fl = self.flow
        M, S, _ = u.shape
        ba, bb, bc = fl._bin(u[..., 0]), fl._bin(u[..., 1]), fl._bin(u[..., 2])
        base_a = fl.head_a(h_e)
        base_b = fl.head_b(h_e + fl.bin_a_emb(ba))
        base_c = fl.head_c(h_e + fl.bin_a_emb(ba) + fl.bin_b_emb(bb))
        Va, Vb, Vc = (torch.zeros_like(base_a), torch.zeros_like(base_b), torch.zeros_like(base_c))
        z = torch.zeros(M, S, 3, device=u.device, dtype=anchor_y.dtype)
        ufb = z.clone(); ufb[..., 0] = fl._ctr(ba)
        ufc = z.clone(); ufc[..., 0] = fl._ctr(ba); ufc[..., 1] = fl._ctr(bb)
        for c0 in range(0, S, s_chunk):                                             # chunk slots for memory
            sl = slice(c0, c0 + s_chunk)
            ar = (anchor_y[sl], cage_x[:, sl], cage_s[:, sl], cage_v[:, sl], sj[:, sl], R)
            Va[:, sl] = self._V_axis_b(z[:, sl], 0, *ar, self.phi_a, self.pair_emb_a)
            Vb[:, sl] = self._V_axis_b(ufb[:, sl], 1, *ar, self.phi_b, self.pair_emb_b)
            Vc[:, sl] = self._V_axis_b(ufc[:, sl], 2, *ar, self.phi, self.pair_emb)
        la = F.log_softmax(base_a - Va, -1).gather(-1, ba[..., None]).squeeze(-1)
        lb = F.log_softmax(base_b - Vb, -1).gather(-1, bb[..., None]).squeeze(-1)
        lc = F.log_softmax(base_c - Vc, -1).gather(-1, bc[..., None]).squeeze(-1)
        return la + lb + lc - fl._logbw3                                            # [M,S]

    def _prep(self, xo, so, block_mask, bnd, s_bnd, R):
        """Shared reorder + static tensors for a batch of M configs (xo [M,n,3])."""
        M, n, _ = xo.shape
        m, dev = bnd.shape[0], xo.device
        anchors_full = fixed_ball_scaffold(n, R, dev, xo.dtype)
        order = torch.argsort(block_mask.to(torch.uint8), stable=True)
        anchors = anchors_full[order]
        reordered_block = block_mask[order].to(xo.dtype)
        kind = torch.cat([torch.ones(m, dtype=torch.long, device=dev), torch.zeros(n, dtype=torch.long, device=dev)])
        idxp = torch.arange(m + n, device=dev)
        valid = idxp[None] < (m + torch.arange(n, device=dev))[:, None]
        slot_feat = self._slot_features(anchors, torch.arange(n, device=dev), n, R)
        anchor_y, _ = ball_unsquash(anchors, R)
        return order, anchors, anchor_y, reordered_block, kind, valid, slot_feat, m, n

    def block_log_prob_b(self, xo, so, block_mask, bnd, s_bnd, R):
        """Exact log q(block | retained+boundary) for M configs at once. xo [M,n,3] -> [M]."""
        M = xo.shape[0]
        order, anchors, anchor_y, rblock, kind, valid, slot_feat, m, n = self._prep(xo, so, block_mask, bnd, s_bnd, R)
        xo, so = xo[:, order], so[:, order]
        combined = torch.cat([bnd[None].expand(M, m, 3), xo], 1)
        scomb = torch.cat([s_bnd[None].expand(M, m), so], 1)
        h = self._fc_batched(combined, scomb, kind, valid, anchors, slot_feat) + self._r_bias(R, xo.device, xo.dtype)
        oh = F.one_hot(so, self.n_species).to(h.dtype)
        rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)
        lp_s = F.log_softmax(self.head_species(h).masked_fill(rem <= 0, float("-inf")), -1).gather(-1, so[..., None]).squeeze(-1)
        y, logdet_yx = ball_unsquash(xo, R)
        u = y - anchor_y[None]                                                       # frameless
        cage_x, cage_s, cage_v = self._cage_knn_b(combined, scomb, valid, anchors)
        lp_u = self._tilted_lp_u_b(h + self.sp_out_emb(so), u, anchor_y, cage_x, cage_s, cage_v, so, R)
        return ((lp_s + lp_u + logdet_yx) * rblock[None]).sum(1)                     # [M]

    @torch.no_grad()
    def sample_block_b(self, xo, so, block_mask, bnd, s_bnd, R, gen=None):
        """Regenerate the block for M configs in parallel. Returns xo_new [M,n,3], so_new [M,n], logq [M]."""
        fl = self.flow
        M = xo.shape[0]
        order, anchors, anchor_y, rblock, kind, valid, slot_feat, m, n = self._prep(xo, so, block_mask, bnd, s_bnd, R)
        n_ret = int((~block_mask).sum())
        xo, so = xo[:, order].clone(), so[:, order].clone()
        combined = torch.cat([bnd[None].expand(M, m, 3), xo], 1).clone()
        scomb = torch.cat([s_bnd[None].expand(M, m), so], 1).clone()
        blk_s = so[:, n_ret:]                                                        # per-config block species budget
        rem = torch.stack([(blk_s == 0).sum(1).float(), (blk_s == 1).sum(1).float()], -1)   # [M,2]
        rbias = self._r_bias(R, xo.device, xo.dtype)
        ar = torch.arange(M, device=xo.device)
        logq = xo.new_zeros(M)
        for jj in range(n_ret, n):
            vj = (torch.arange(m + n, device=xo.device) < (m + jj))[None]            # [1, m+n]
            h = self._fc_batched(combined, scomb, kind, vj, anchors[jj:jj + 1], slot_feat[jj:jj + 1])[:, 0] + rbias  # [M,d]
            ls = F.log_softmax(self.head_species(h).masked_fill(rem <= 0, float("-inf")), -1)
            sj = torch.multinomial(ls.exp(), 1, generator=gen).squeeze(-1)           # [M]
            he = h + self.sp_out_emb(sj)
            cage_x, cage_s, cage_v = self._cage_knn_b(combined, scomb, vj, anchors[jj:jj + 1])   # [M,1,K,*]
            cx, cs, cv = cage_x[:, 0], cage_s[:, 0], cage_v[:, 0]

            def Vax(u_fixed, axis, net, emb):
                uf = u_fixed[:, None]                                                # [M,1,3]
                return self._V_axis_b(uf, axis, anchor_y[jj:jj + 1], cage_x, cage_s, cage_v, sj[:, None], R, net, emb)[:, 0]
            z = torch.zeros(M, 3, device=xo.device, dtype=xo.dtype)
            la = F.log_softmax(fl.head_a(he) - Vax(z, 0, self.phi_a, self.pair_emb_a), -1)
            ba = torch.multinomial(la.exp(), 1, generator=gen).squeeze(-1)
            ufb = z.clone(); ufb[:, 0] = fl._ctr(ba)
            lb = F.log_softmax(fl.head_b(he + fl.bin_a_emb(ba)) - Vax(ufb, 1, self.phi_b, self.pair_emb_b), -1)
            bb = torch.multinomial(lb.exp(), 1, generator=gen).squeeze(-1)
            ufc = ufb.clone(); ufc[:, 1] = fl._ctr(bb)
            lc = F.log_softmax(fl.head_c(he + fl.bin_a_emb(ba) + fl.bin_b_emb(bb)) - Vax(ufc, 2, self.phi, self.pair_emb), -1)
            bc = torch.multinomial(lc.exp(), 1, generator=gen).squeeze(-1)
            dith = (torch.rand(M, 3, device=xo.device, dtype=xo.dtype, generator=gen) - 0.5) * fl.bw
            u = torch.stack([fl._ctr(ba), fl._ctr(bb), fl._ctr(bc)], -1) + dith      # [M,3]
            y = anchor_y[jj][None] + u                                               # frameless
            pos, logdet_xy = ball_squash(y, R)                                       # [M,3], [M]
            xo[:, jj], so[:, jj] = pos, sj
            combined[:, m + jj], scomb[:, m + jj] = pos, sj
            rem[ar, sj] -= 1
            lp_u = (la.gather(1, ba[:, None]).squeeze(1) + lb.gather(1, bb[:, None]).squeeze(1)
                    + lc.gather(1, bc[:, None]).squeeze(1) - fl._logbw3)
            logq = logq + ls.gather(1, sj[:, None]).squeeze(1) + lp_u - logdet_xy
        xo_new, so_new = xo.new_empty(M, n, 3), so.new_empty(M, n)
        xo_new[:, order], so_new[:, order] = xo, so
        return xo_new, so_new, logq
