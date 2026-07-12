"""Task 4: conditional flow-matching DATA pipeline with minibatch-OT coupling.

Produces training batches that pair AR-sampled cavity blocks (the flow's base, x0) with the
cavity's data block (the flow's target, x1) for `liquid_coupling_flow.ka3d_cavity_egnn.CavityBlockFlow`
(Task 1/2/3). See docs/superpowers/specs/2026-07-12-egnn-cavity-block-flow-corrector-design.md
("Training objective") -- conditional flow-matching, rectified-flow interpolant
`x_t = (1-t) x0 + t x1`, target velocity `x1 - x0`.

OT coupling choice (documented per the task-4 brief's three sketched options)
------------------------------------------------------------------------------
A single carved cavity has exactly ONE physically-valid data block (x1). That is unlike a generic
OT-CFM setup, where source and target are both drawn i.i.d. from their marginals and a minibatch
naturally contains M DISTINCT targets. Two ways to manufacture M distinct targets were considered
and rejected:

  (1) Cross-cavity OT: batch M different cavities (same fixed K), draw one AR block per cavity, and
      Hungarian-match cavity i's AR block against cavity j's data block (j maybe != i). REJECTED: the
      flow is CAGE-conditioned (`CavityBlockFlow.flow(x0_block, cage_x, ...)`); pairing an AR sample
      drawn under cage_i with a data target that is only physically valid under cage_j teaches the
      network to transport toward the WRONG cage's answer whenever the assignment reassigns i != j.
      It also requires every cavity in the batch to share an identical K (block size), which the
      brief itself flags as awkward for a general carve.

  (2) SO(3)-rotate just the block (not the cage) to fabricate M "distinct" targets from the one data
      block. REJECTED: rotating the movers without rotating the boundary/retained interior they were
      relaxed against manufactures mover/cage clashes that never occurred in the data -- the
      "target" would no longer be a valid low-energy configuration for that cage, polluting the
      training signal with physically wrong velocity labels.

CHOSEN: OT at the PARTICLE level -- "Hungarian on the K-block" (the brief's own phrasing for the
`Produces` line). For each of M independent AR draws of ONE cavity's block (all sharing the SAME
cage and the SAME single data target x1), solve a K x K assignment problem between the AR block's K
mover positions and the data block's K target slots, minimizing total squared distance on CENTERED
positions (mean-subtracted over the whole K-block, so the match compares relative geometry, not
absolute position in the cage), SEPARATELY per species. Species-count matching is exact by
construction: `KA3DScaffoldEBMBatched.sample_block_b` draws each block position's species from a
remaining-count budget copied directly from `so[block_mask]` of THIS SAME cavity (see
ka3d_ebm_batched.py, `blk_s = so[:, n_ret:]` / `rem = ...`), so AR-species-A count == data-species-A
count for every draw -- the per-species Hungarian submatrices are always square (never degenerate on
species mismatch for a single-cavity draw; the identity fallback below exists for robustness, e.g. an
external caller reusing this on a differently-constructed block).

This is the standard fix for the "arbitrary scaffold-slot label" problem: two same-species movers
are physically exchangeable, so regressing x0's mover at slot j toward x1's slot j (identity
coupling) forces the flow to learn a transport that crosses the whole block whenever the AR happened
to seat species differently across slots than the data did -- an artifact of the (arbitrary) fixed
scaffold order, not a real displacement. Hungarian removes this artifact and gives the SHORT-
transport target that minibatch-OT is meant to produce.

Why this does not repeat the earlier deterministic-OT corrector's mean-collapse failure: that
corrector collapsed because it was NOT cage-conditioned (all AR samples, across ALL cavities, were
pulled toward one global average target). Here x1 is always the single physically-correct answer for
the SAME cage x0 was drawn under -- mapping M diverse AR realizations of one cavity to that one
cavity's one data target is not a collapse, it is the correct (cage-conditional) supervision signal;
diversity across CAVITIES (different cages -> different targets) is what a real training run adds by
looping `fm_batch` over many carved cavities and concatenating along dim 0.
"""
from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic

RCTX = 2.5   # boundary-shell context radius beyond R, matches frontier_threearm.py / mtm_early_read.py


def carve_cavity_block(X, S, L, ci, center, R, K, gen):
    """Carve one cavity from chain `ci` of a bulk config batch (X[B,N,3], S[B,N], L) at `center`,
    radius `R`, and pick a K-slot block (the K nearest fixed-scaffold anchors to a random seed slot).
    Mirrors the carve pattern in reports/logs-2026-07-12/frontier_threearm.py and mtm_early_read.py
    exactly. Returns None if the cavity is too small for K, else a dict:
      xo[n,3], so[n]        -- labeled DATA interior (fixed-scaffold order, via label_to_scaffold)
      block_mask[n] bool    -- True at the K mover slots
      bnd[m,3], s_bnd[m]    -- frozen boundary shell (|x|<R+RCTX, outside R)
      R, n
    """
    p = carve(X[ci], S[ci], center, R, L)
    if p["n_in"] < K + 1:
        return None
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], center, L), p["s_in"], R)
    xout = _mic(p["x_out"], center, L)
    bm = xout.norm(dim=-1) < (R + RCTX)
    bnd, s_bnd = xout[bm], p["s_out"][bm]
    n = xo.shape[0]
    a = fixed_ball_scaffold(n, R, xo.device)
    seed = int(torch.randint(n, (), generator=gen, device=xo.device))
    block_mask = torch.zeros(n, dtype=torch.bool, device=xo.device)
    block_mask[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    return {"xo": xo, "so": so, "block_mask": block_mask, "bnd": bnd, "s_bnd": s_bnd, "R": R, "n": n}


def _match_species(x0, sp0, x1, sp1):
    """Per-species Hungarian assignment of x1's K slots onto x0's K movers.

    x0[K,3] (AR movers), sp0[K] (AR species); x1[K,3] (data movers), sp1[K] (data species). Cost is
    squared distance between x0 and x1 after centering BOTH point sets on their own centroid (compare
    relative geometry, not absolute position in the cage -- "Hungarian on centered positions").
    Returns perm[K] (long) such that `x1[perm]` is slot-aligned with `x0` and `sp1[perm] == sp0`
    exactly, or None if some species' counts differ between x0 and x1 (degenerate -- caller falls
    back to identity coupling for that draw)."""
    K = x0.shape[0]
    x0c = x0 - x0.mean(0, keepdim=True)
    x1c = x1 - x1.mean(0, keepdim=True)
    perm = torch.full((K,), -1, dtype=torch.long, device=x0.device)
    species = torch.unique(torch.cat([sp0, sp1])).tolist()
    for sp in species:
        i0 = (sp0 == sp).nonzero(as_tuple=True)[0]
        i1 = (sp1 == sp).nonzero(as_tuple=True)[0]
        if i0.numel() != i1.numel():
            return None
        if i0.numel() == 0:
            continue
        if i0.numel() == 1:
            perm[i0] = i1
            continue
        cost = torch.cdist(x0c[i0], x1c[i1]).pow(2).detach().cpu().numpy()
        row, col = linear_sum_assignment(cost)
        perm[i0[row]] = i1[col]
    if (perm < 0).any():
        return None
    return perm


@torch.no_grad()
def fm_batch(ar_model, cav, gen, pos_temp=1.0, M=8, t=None):
    """Build one cavity's conditional flow-matching training rows.

    `cav` is a dict from `carve_cavity_block` (xo, so, block_mask, bnd, s_bnd, R). Draws M
    independent AR block samples (the base x0) for this SAME cavity, matches each against the
    cavity's single data block (the target x1) with a per-species Hungarian assignment (see module
    docstring), and builds the rectified-flow interpolant / target velocity.

    `t`: None -> t ~ U(0,1) per row (training default). Pass a scalar or a length-M tensor to force a
    fixed t (used by the boundary test: t=0 -> x_t==x0_block, t=1 -> x_t==x1_block).

    Returns a dict of tensors, all with a leading M dimension (except `R`, a python float):
      x0_block[M,K,3], x1_block[M,K,3]  -- OT-matched base/target (diagnostics)
      x_t[M,K,3], t[M], target_v[M,K,3] -- flow-matching training tensors
      cage_x[M,n_cage,3], sp_cage[M,n_cage] -- frozen cage = boundary + retained interior (broadcast)
      sp_block[M,K]                     -- mover species (fixed through the flow; AR == matched data)
      R                                 -- cavity radius (python float)
      logq_ar[M]                        -- AR base log-density (diagnostic; not used by the FM loss)
      perm[M,K]                         -- the OT permutation applied to x1 (diagnostics/tests)
      n_identity_fallback               -- count of draws where OT degenerated to identity (diagnostic)
    """
    xo, so, block_mask = cav["xo"], cav["so"], cav["block_mask"]
    bnd, s_bnd, R = cav["bnd"], cav["s_bnd"], cav["R"]
    n = xo.shape[0]
    dev = xo.device
    K = int(block_mask.sum())

    Xb = xo[None].expand(M, n, 3).contiguous()
    Sb = so[None].expand(M, n).contiguous()
    xo_ar, so_ar, logq_ar = ar_model.sample_block_b(Xb, Sb, block_mask, bnd, s_bnd, R, gen=gen, pos_temp=pos_temp)

    x0_block = xo_ar[:, block_mask]            # [M,K,3] AR-sampled movers (base)
    sp_block = so_ar[:, block_mask]            # [M,K]   AR species (fixed through the flow)
    x1_data = xo[block_mask]                   # [K,3]   the single data target for this cavity
    sp_data = so[block_mask]                   # [K]

    x1_block = x0_block.new_empty(M, K, 3)
    perms = torch.empty(M, K, dtype=torch.long, device=dev)
    n_identity_fallback = 0
    identity = torch.arange(K, device=dev)
    for mi in range(M):
        perm = _match_species(x0_block[mi], sp_block[mi], x1_data, sp_data)
        if perm is None:
            perm = identity
            n_identity_fallback += 1
        perms[mi] = perm
        x1_block[mi] = x1_data[perm]

    ret_x = xo[~block_mask]                     # [n_ret,3] retained interior -- identical for AR and data
    ret_s = so[~block_mask]                     # (sample_block_b never touches retained slots)
    n_ret = ret_x.shape[0]
    n_cage = bnd.shape[0] + n_ret
    cage_x = torch.cat([bnd, ret_x], 0)[None].expand(M, n_cage, 3).contiguous()
    sp_cage = torch.cat([s_bnd, ret_s], 0)[None].expand(M, n_cage).contiguous()

    if t is None:
        tt = torch.rand(M, device=dev, generator=gen, dtype=x0_block.dtype)
    else:
        tt = torch.as_tensor(t, device=dev, dtype=x0_block.dtype)
        if tt.ndim == 0:
            tt = tt.expand(M).contiguous()
    x_t = (1.0 - tt)[:, None, None] * x0_block + tt[:, None, None] * x1_block
    target_v = x1_block - x0_block

    return {"x0_block": x0_block, "x1_block": x1_block, "x_t": x_t, "t": tt, "target_v": target_v,
            "cage_x": cage_x, "sp_cage": sp_cage, "sp_block": sp_block, "R": R,
            "logq_ar": logq_ar, "perm": perms, "n_identity_fallback": n_identity_fallback}
