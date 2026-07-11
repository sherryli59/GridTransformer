"""Exact block-update Metropolis kernel for the fixed-scaffold AR cavity model.

A block move regenerates a subset of interior slots (positions + species) conditioned on the
frozen boundary AND all retained interior slots. Fixed Hungarian labels make this a pure reorder:
place the interior in [retained; block] order and the model's prefix-based machinery gives the
exact block conditional (each block slot sees ALL retained particles = full cage). Species
count-masking over [retained; block] restricts the block to its own A/B budget -> in-block A<->B
swaps are automatic. Exact logq both directions -> exact MH. No rollout drift (small block =
clean conditional). k=1 == full-cage single-site heat-bath; k>1 == collective move.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from liquid_coupling_flow.ka3d_scaffold_ar import fixed_ball_scaffold, ball_squash, ball_unsquash


def _reordered_context(model, xo_ord, so_ord, anchors_ord, bnd, s_bnd, R):
    """Run the model context for interior already in [retained; block] order with GIVEN anchors."""
    n, m, dev = xo_ord.shape[0], bnd.shape[0], xo_ord.device
    combined = torch.cat([bnd, xo_ord], 0)
    scomb = torch.cat([s_bnd, so_ord], 0)
    kind = torch.cat([torch.ones(m, dtype=torch.long, device=dev),
                      torch.zeros(n, dtype=torch.long, device=dev)])
    idx = torch.arange(m + n, device=dev)
    valid = idx[None] < (m + torch.arange(n, device=dev))[:, None]      # prefix: boundary + retained + block<j
    slot_feat = model._slot_features(anchors_ord, torch.arange(n, device=dev), n, R)
    h, frame = model._frame_context(combined, scomb, kind, valid, anchors_ord, anchors_ord, slot_feat=slot_feat)
    return h, frame


def _order(block_mask):
    """Stable [retained; block] permutation (retained first) with NO data-dependent sync."""
    return torch.argsort(block_mask.to(torch.uint8), stable=True)


def block_log_prob(model, xo_full, so_full, block_mask, bnd, s_bnd, R):
    """Exact log q(block positions+species | retained + boundary). Scalar."""
    n = xo_full.shape[0]
    anchors_full = fixed_ball_scaffold(n, R, xo_full.device, xo_full.dtype)
    order = _order(block_mask)
    xo, so, anchors = xo_full[order], so_full[order], anchors_full[order]
    reordered_block = block_mask[order].to(xo.dtype)                   # [0..0, 1..1]; sum-mask, no .item()
    h, frame = _reordered_context(model, xo, so, anchors, bnd, s_bnd, R)
    oh = F.one_hot(so, model.n_species).to(h.dtype)
    rem = oh.sum(0, keepdim=True) - (oh.cumsum(0) - oh)                 # remaining budget before each slot
    lp_s = F.log_softmax(model.head_species(h).masked_fill(rem <= 0, float("-inf")), -1).gather(-1, so[:, None]).squeeze(-1)
    y, logdet_yx = ball_unsquash(xo, R)
    anchor_y, _ = ball_unsquash(anchors, R)
    u = torch.einsum("naj,nj->na", frame, y - anchor_y)
    lp_u = model.flow.log_prob(h + model.sp_out_emb(so), u)
    per_slot = lp_s + lp_u + logdet_yx
    return (per_slot * reordered_block).sum()                          # only the block (suffix) slots


@torch.no_grad()
def sample_block(model, xo_full, so_full, block_mask, bnd, s_bnd, R, gen=None):
    """Regenerate the block conditioned on retained + boundary. Returns (xo_new, so_new, logq_fwd)."""
    n, m, dev = xo_full.shape[0], bnd.shape[0], xo_full.device
    anchors_full = fixed_ball_scaffold(n, R, dev, xo_full.dtype)
    order = _order(block_mask)
    n_ret = int((~block_mask).sum())                                   # eval/MH path: sync acceptable
    xo = xo_full[order].clone(); so = so_full[order].clone(); anchors = anchors_full[order]
    anchor_y, _ = ball_unsquash(anchors, R)
    combined = torch.cat([bnd, xo], 0)
    scomb = torch.cat([s_bnd, so], 0)
    kind = torch.cat([torch.ones(m, dtype=torch.long, device=dev), torch.zeros(n, dtype=torch.long, device=dev)])
    idx = torch.arange(m + n, device=dev)
    # block species budget = counts among the block's current particles (count-preserving in-block swap)
    blk_s = so_full[block_mask]
    rem = torch.tensor([[float((blk_s == 0).sum()), float((blk_s == 1).sum())]], device=dev)
    logq = xo.new_zeros(())
    for jj in range(n_ret, n):                                         # only regenerate block slots (suffix)
        valid = (idx < (m + jj))[None]
        slot_feat = model._slot_features(anchors[jj:jj + 1], jj, n, R)
        h, frame = model._frame_context(combined, scomb, kind, valid, anchors[jj:jj + 1], anchors[jj:jj + 1],
                                        slot_feat=slot_feat)
        ls = F.log_softmax(model.head_species(h).masked_fill(rem <= 0, float("-inf")), -1)
        sj = torch.multinomial(ls.exp(), 1, generator=gen).squeeze(-1)
        u, lp_u = model.flow.sample(h + model.sp_out_emb(sj), gen=gen)
        y = anchor_y[jj] + torch.einsum("naj,na->nj", frame, u)[0]
        pos, logdet_xy = ball_squash(y, R)
        xo[jj], so[jj] = pos, sj
        combined[m + jj], scomb[m + jj] = pos, sj
        rem[0, sj] -= 1
        logq = logq + ls.gather(1, sj[:, None]).squeeze() + lp_u.squeeze() - logdet_xy.squeeze()
    # scatter block back into label order
    xo_new, so_new = xo_full.clone(), so_full.clone()
    xo_new[order], so_new[order] = xo, so
    return xo_new, so_new, logq


@torch.no_grad()
def block_mh_step(model, xo_full, so_full, block_mask, bnd, s_bnd, R, beta, energy_fn, gen=None):
    """One exact Metropolis block move. energy_fn(xo, so)->scalar U (interior+boundary). Returns
    (xo, so, accepted, dU)."""
    u_old = energy_fn(xo_full, so_full)
    logq_rev = block_log_prob(model, xo_full, so_full, block_mask, bnd, s_bnd, R)   # q(old block|ret)
    xo_new, so_new, logq_fwd = sample_block(model, xo_full, so_full, block_mask, bnd, s_bnd, R, gen)
    u_new = energy_fn(xo_new, so_new)
    log_alpha = -beta * (u_new - u_old) + (logq_rev - logq_fwd)
    accept = torch.rand((), device=xo_full.device, generator=gen).log() < log_alpha
    if bool(accept):
        return xo_new, so_new, True, float(u_new - u_old)
    return xo_full, so_full, False, float(u_new - u_old)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Exactness gate: sample_block's logq == block_log_prob re-evaluated on the sampled block
    # (a delta-check of the block chain-rule + Jacobians), for random blocks and radii.
    from liquid_coupling_flow.ka3d_scaffold_ar import KA3DScaffoldAR, label_to_scaffold
    from liquid_coupling_flow.ka3d_cavity_carve import carve
    from liquid_coupling_flow.ka3d_cavity_ar import _mic

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
    X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
    m = KA3DScaffoldAR().to(dev); m.eval()
    torch.manual_seed(0)
    empty_x, empty_s = torch.zeros(0, 3, device=dev), torch.zeros(0, dtype=torch.long, device=dev)
    worst = 0.0
    for t in range(6):
        R = float([1.6, 2.0, 2.4][t % 3])
        center = X[t][17 * t % X.shape[1]]
        p = carve(X[t], S[t], center, R, L)
        xr = _mic(p["x_in"], center, L)
        x, s, _ = label_to_scaffold(xr, p["s_in"], R)
        n = x.shape[0]
        kblk = max(1, n // 5)
        block = torch.zeros(n, dtype=torch.bool, device=dev)
        block[torch.randperm(n, device=dev)[:kblk]] = True
        xn, sn, lqf = sample_block(m, x, s, block, empty_x, empty_s, R)
        lqf2 = block_log_prob(m, xn, sn, block, empty_x, empty_s, R)
        err = float((lqf - lqf2).abs())
        worst = max(worst, err)
        print(f"R={R} n={n} k={int(block.sum())} logq_fwd={float(lqf):+.3f} reeval={float(lqf2):+.3f} "
              f"err={err:.2e} {'OK' if err < 1e-3 else 'MISMATCH'}", flush=True)
    print(f"worst block-logq err={worst:.2e} -> {'EXACT' if worst < 1e-3 else 'FAIL'}", flush=True)
