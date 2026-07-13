"""Gated AR base for the cavity block-flow corrector: exact truncated-base construction.

Motivation (2026-07-12 structure-gate finding): raw AR@0.4 draws are bimodal — most are usable, a
tail is garbage (worst-pair r->0; e.g. the gate's cav-7 input, AR inherent-structure floor +55/particle,
which no corrector can rescue). Garbage draws (a) dominate the flow-matching MSE with huge x1-x0
transport targets, starving the fine-placement regime the MTM acceptance actually needs, (b) waste
ODE solves at deployment, and (c) are exactly the stiff inputs that collapse dopri5's step size.

Construction: rejection-sample the AR base against a STATE-INDEPENDENT screen — redraw each row until
its block passes `min_{pairs} r_ij / sigma_ij >= cut` over block-block and block-cage pairs. The
resulting base density is q_gated(x0) = q_AR(x0) * 1[pass] / Z_A(cage). Z_A is unknown but depends
ONLY on the (cage, gate) — fixed during a move — so it multiplies every MTM/MH weight identically
(all trials AND the current-state reverse evaluation) and cancels exactly in every acceptance ratio.
We therefore return the UNGATED per-row logq_ar unchanged; callers must use weights only in ratios
that share the same cage and gate on both sides.

Exactness rules for callers:
  - The gate must never see the current MCMC state — only the candidate block and the frozen cage.
  - A trial that fails the gate is OUTSIDE the proposal support: drop it (training) or give it
    weight 0 (MTM). `last_pass_mask` marks rows that never passed within `max_rounds`.
  - Reverse (current-state) evaluation: reverse-flow the current block to its base image x0_rev and
    check `gate_pass(x0_rev, ...)`. If it fails, q_gated(current) = 0 -> the move is REJECTED (valid,
    conservative). Measure the pass rate of reverse images of DATA blocks before hardening the cut —
    see reports/logs-2026-07-12/diag_reverse_gate.py.
  - If no draw passes after max_rounds the caller may skip the move entirely: the skip probability
    depends only on (cage, RNG), not the current state, so detailed balance is untouched.
"""
from __future__ import annotations
import torch

# KA sigma table, indices 0=A, 1=B (must match liquid_coupling_flow.ka_energy.SIGMA)
_SIGMA = [[1.0, 0.8], [0.8, 0.88]]


def sigma_min_ratio(xb, sp_blk, cage_x, cage_s):
    """Per-row min over all block-involved pairs of r_ij / sigma_ij.

    xb[M,K,3], sp_blk[M,K], cage_x[M,m,3], cage_s[M,m] -> [M]. Pairs = block-block (i<j) and
    block-cage. Depends only on the candidate block and the frozen cage (state-independent).
    """
    M, K = xb.shape[0], xb.shape[1]
    sig = torch.tensor(_SIGMA, device=xb.device, dtype=xb.dtype)
    allx = torch.cat([xb, cage_x], 1)                                  # [M,K+m,3]
    alls = torch.cat([sp_blk, cage_s], 1).long()                       # [M,K+m]
    d = torch.cdist(xb, allx)                                          # [M,K,K+m]
    s_pair = sig[sp_blk.long()[:, :, None], alls[:, None, :]]          # [M,K,K+m]
    ratio = d / s_pair
    # mask self-pairs (block row i vs cloud column i)
    ratio[:, torch.arange(K), torch.arange(K)] = torch.inf
    return ratio.flatten(1).min(1).values                              # [M]


def gate_pass(xb, sp_blk, cage_x, cage_s, cut):
    """Boolean [M]: True where the block passes the sigma-scaled worst-pair screen."""
    return sigma_min_ratio(xb, sp_blk, cage_x, cage_s) >= cut


class GatedARBase:
    """Decorator around a KA3DScaffoldEBMBatched-style base: same `sample_block_b` signature, but each
    returned row is redrawn until it passes the gate (truncated base, see module docstring).

    After each call: `last_pass_mask[M]` (False = row never passed within max_rounds; the row holds
    the final failed draw and must be dropped/zero-weighted by the caller), `last_rounds` (redraw
    rounds used), `last_min_ratio[M]` (final per-row worst-pair ratio, diagnostic).
    """

    def __init__(self, ar_model, cut=0.8, max_rounds=32):
        self.ar = ar_model
        self.cut = float(cut)
        self.max_rounds = int(max_rounds)
        self.last_pass_mask = None
        self.last_rounds = 0
        self.last_min_ratio = None

    def __getattr__(self, name):
        # transparent pass-through (eval(), use_frame, log_prob_pair, ...) for non-overridden attrs
        return getattr(self.ar, name)

    @torch.no_grad()
    def sample_block_b(self, xo, so, block_mask, bnd, s_bnd, R, gen=None, pos_temp=1.0):
        M = xo.shape[0]
        xo_out, so_out, lq_out = self.ar.sample_block_b(xo, so, block_mask, bnd, s_bnd, R,
                                                        gen=gen, pos_temp=pos_temp)
        bnd_b = bnd[None].expand(M, -1, -1)
        sb_b = s_bnd[None].expand(M, -1)

        def _cage(xf, sf):
            return (torch.cat([bnd_b[:xf.shape[0]], xf[:, ~block_mask]], 1),
                    torch.cat([sb_b[:xf.shape[0]], sf[:, ~block_mask]], 1))

        cage_x, cage_s = _cage(xo_out, so_out)
        ratio = sigma_min_ratio(xo_out[:, block_mask], so_out[:, block_mask], cage_x, cage_s)
        ok = ratio >= self.cut
        rounds = 0
        while (~ok).any() and rounds < self.max_rounds:
            rounds += 1
            idx = (~ok).nonzero(as_tuple=True)[0]
            xf, sf, lqf = self.ar.sample_block_b(xo[idx], so[idx], block_mask, bnd, s_bnd, R,
                                                 gen=gen, pos_temp=pos_temp)
            cxf, csf = _cage(xf, sf)
            rf = sigma_min_ratio(xf[:, block_mask], sf[:, block_mask], cxf, csf)
            keep = rf >= self.cut
            # scatter fresh draws back (also keep the final failed draw so shapes stay [M,...])
            xo_out[idx], so_out[idx], lq_out[idx], ratio[idx] = xf, sf, lqf, rf
            ok[idx] = keep
        self.last_pass_mask = ok
        self.last_rounds = rounds
        self.last_min_ratio = ratio
        return xo_out, so_out, lq_out
