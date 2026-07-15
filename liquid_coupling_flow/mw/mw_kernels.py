"""Exact collective MH kernels for the mW lambda-bridge (spec 2026-07-14).

Conventions (load-bearing, see spec):
- pi_lambda uses the CANONICAL-LIFT q0: q0_model.log_prob(x, L) with preordered=False,
  permutation-invariant over unordered configs -> storage relabeling is always legal.
- suffix_move requires canonical storage on entry and rejects noncanonical regenerated
  tails; forward/reverse truncation normalizers share the same prefix and cancel, so the
  reduced ratio  -lam*(beta*dU + logq_fwd - logq_rev)  is exact
  (test_suffix_reduction_equals_general_bridge_formula is the guard).
- two_blob_move draws centers state-independently and REJECTS proposals whose union
  block is not re-selected by the same centers under the proposed config: the center-set
  selecting a given index block is symmetric in (x, x'), so selection probs cancel.
"""
from __future__ import annotations
import torch

from liquid_coupling_flow.mw.mw_energy import mw_energy
from liquid_coupling_flow.mw.mw_generator import mw_scaffold, canonical_order, wrap_pm
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept


def recanonicalize(x, L):
    """Sort each config into canonical slot order (pure relabeling)."""
    B, N, _ = x.shape
    x = torch.remainder(x, L)
    t, rank, R = mw_scaffold(N, L, x.device)
    perm = canonical_order(x, L, R, rank)
    return torch.gather(x, 1, perm[..., None].expand(-1, -1, 3))


def is_canonical(x, L):
    B, N, _ = x.shape
    t, rank, R = mw_scaffold(N, L, torch.remainder(x, L).device)
    perm = canonical_order(torch.remainder(x, L), L, R, rank)
    return (perm == torch.arange(N, device=x.device)[None]).all(1)


def suffix_move(x, U, q0_model, m_lo, m_hi, lam, beta, L, gen):
    """One suffix-resample MH move on every walker. x MUST be canonical storage."""
    B, N, _ = x.shape
    m = int(torch.randint(m_lo, m_hi + 1, (1,), device=x.device, generator=gen).item())
    with torch.no_grad():
        lq_rev = q0_model.suffix_log_prob(x, m, L)
        xp, lq_fwd = q0_model.sample_suffix(x, m, L, gen=gen)
    ok = is_canonical(xp, L)
    Up = U.clone()
    if ok.any():
        Up[ok] = mw_energy(xp[ok], L)          # evals only for live proposals
    la = -lam * (beta * (Up - U) + (lq_fwd - lq_rev))
    r = torch.rand(B, device=x.device, generator=gen).clamp_min(1e-38).log()
    acc = ok & (r < la)
    x = torch.where(acc[:, None, None], xp, x)
    U = torch.where(acc, Up, U)
    return x, U, {"acc": float(acc.float().mean()), "m": m,
                  "reject_canon": float((~ok).float().mean()), "reject_reverse": 0.0}


def conveyor_regions(N, L, device, m_frac=0.4):
    """(prefix cells, suffix cells) split of the curve-ordered anchor list -- fixed and
    state-independent, so conveyor center draws keep MH selection symmetric."""
    t, rank, R = mw_scaffold(N, L, device)
    cut = N - int(round(m_frac * N))
    return t[:cut], t[cut:]


def draw_centers(L, min_sep, gen, device, regions=None, max_tries=64):
    """Two centers, min-image separation >= min_sep. regions=None: both uniform in the box.
    regions=(cells_a, cells_b): random cell center + uniform jitter within the cell
    (conveyor placement) -- both distributions are fixed, so selection stays symmetric."""
    for _ in range(max_tries):
        if regions is None:
            cA = torch.rand(3, device=device, generator=gen) * L
            cB = torch.rand(3, device=device, generator=gen) * L
        else:
            n_cells = regions[0].shape[0] + regions[1].shape[0]
            cw = L / round(n_cells ** (1.0 / 3.0))          # cell width of the scaffold grid
            outs = []
            for cells in regions:
                i = int(torch.randint(cells.shape[0], (1,), device=device, generator=gen).item())
                jit = (torch.rand(3, device=device, generator=gen) - 0.5) * cw
                outs.append(torch.remainder(cells[i] + jit, L))
            cA, cB = outs
        if wrap_pm(cA - cB, L).norm() >= min_sep:
            return cA, cB
    raise RuntimeError(f"draw_centers: no pair with sep>={min_sep} after {max_tries} tries")


def _union_block(x1, cA, cB, K, L):
    """Union of the K nearest to each center; None if the two sets overlap."""
    iA = wrap_pm(x1 - cA[None], L).norm(dim=-1).topk(K, largest=False).indices
    iB = wrap_pm(x1 - cB[None], L).norm(dim=-1).topk(K, largest=False).indices
    idx = torch.cat([iA, iB]).unique().sort().values
    return idx if idx.numel() == 2 * K else None


def two_blob_move(x, U, lq0, q0_model, block_model, K, lam, beta, L, min_sep, gen,
                  regions=None):
    """One two-blob union-regen MH move per walker (walker loop: block model takes one
    shared index block). lq0 = canonical q0_model.log_prob(x, L) [B]; None iff lam==1."""
    if lam < 1.0 and lq0 is None:
        raise ValueError("lq0 required when lam < 1")
    B, N, _ = x.shape
    xp = x.clone()
    lq_f = torch.zeros(B, device=x.device)
    lq_r = torch.zeros(B, device=x.device)
    live = torch.zeros(B, dtype=torch.bool, device=x.device)
    n_rev = 0
    with torch.no_grad():
        for b in range(B):
            cA, cB = draw_centers(L, min_sep, gen, x.device, regions)
            idx = _union_block(x[b], cA, cB, K, L)
            if idx is None:
                continue
            xb, f = block_model.sample_block(x[b:b + 1], idx, L, gen=gen)
            idx2 = _union_block(xb[0], cA, cB, K, L)
            if idx2 is None or not torch.equal(idx2, idx):
                n_rev += 1
                continue
            lq_r[b] = block_model.block_log_prob(x[b:b + 1], idx, L)[0]
            xp[b] = xb[0]
            lq_f[b] = f[0]
            live[b] = True
        Up = U.clone()
        lq0p = lq0.clone() if lq0 is not None else None
        if live.any():
            Up[live] = mw_energy(xp[live], L)
            if lam < 1.0:
                lq0p[live] = q0_model.log_prob(xp[live], L)
    # geometric_bridge_log_accept natively handles lam==1 (q0 term dropped, lq0p may be None)
    la = geometric_bridge_log_accept(
        log_q0_current=lq0 if lq0 is not None else torch.zeros_like(U),
        log_q0_proposed=lq0p, energy_current=U, energy_proposed=Up,
        log_r_reverse=lq_r, log_r_forward=lq_f, lam=lam, beta=beta)
    r = torch.rand(B, device=x.device, generator=gen).clamp_min(1e-38).log()
    acc = live & (r < la)
    x = torch.where(acc[:, None, None], xp, x)
    U = torch.where(acc, Up, U)
    if lq0 is not None:
        lq0 = torch.where(acc, lq0p, lq0)
    return x, U, lq0, {"acc": float(acc.float().mean()),
                       "reject_reverse": n_rev / B, "reject_canon": 0.0,
                       "live": float(live.float().mean())}
