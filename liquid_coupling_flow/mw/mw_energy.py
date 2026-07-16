"""Batched torch mW (Stillinger-Weber, Molinero-Moore 2009) in reduced units (sigma=eps=1)."""
from __future__ import annotations
import torch

A_SW, B_SW = 7.049556277, 0.6022245584
A_CUT, GAMMA, LAMBDA3, COS0 = 1.8, 1.2, 23.15, -1.0 / 3.0
KB_KCAL, EPS_KCAL, SIGMA_A = 0.0019872, 6.189, 2.3925
T_STAR = KB_KCAL * 300.0 / EPS_KCAL          # 0.09632 (ambient 300 K)
RHO_STAR = 0.4564                             # 0.997 g/cm^3
KMAX = 32                                     # 3-body neighbor cap; guarded by assert below


def _pair(x, L):
    d = x[:, :, None, :] - x[:, None, :, :]
    d = d - L * torch.round(d / L)
    return d, d.norm(dim=-1)


def _phi2(r):
    m = r < A_CUT
    rm = torch.where(m, r, torch.full_like(r, A_CUT + 1.0)).clamp_min(1e-9)
    v = A_SW * (B_SW * rm ** -4 - 1.0) * torch.exp(1.0 / (rm - A_CUT))
    return torch.where(m, v, torch.zeros_like(r))


def _phi3_centers(d, r, centers, device):
    """Sum of phi3 over the given boolean center mask [B,N]. Single source of truth for the
    3-body term; both mw_energy (all-true mask) and du_move (locally-affected mask) call this."""
    B, N = centers.shape
    eye = torch.eye(N, dtype=torch.bool, device=device)
    rr = r.masked_fill(eye[None], 1e9)
    k = min(KMAX, N - 1)
    dist, idx = rr.topk(k, dim=2, largest=False)
    within = dist < A_CUT
    if k < N - 1:
        assert not within[..., -1].any(), "KMAX too small"
    dn = torch.gather(d, 2, idx[..., None].expand(-1, -1, -1, 3))
    h = torch.where(within, torch.exp(GAMMA / (dist - A_CUT).clamp_max(-1e-9)), torch.zeros_like(dist))
    dnu = dn / dist.clamp_min(1e-12)[..., None]
    cos = torch.einsum("bikd,bild->bikl", dnu, dnu)
    tri = torch.triu(torch.ones(k, k, dtype=torch.bool, device=device), 1)
    term = LAMBDA3 * (cos - COS0) ** 2 * h[:, :, :, None] * h[:, :, None, :]
    per_center = (term * (within[:, :, :, None] & within[:, :, None, :]) * tri[None, None]).sum((2, 3))
    return (per_center * centers).sum(1)


def mw_energy(x, L):
    """x [B,N,3] -> U [B]. Two-body + three-body within a=1.8; PBC min-image."""
    B, N, _ = x.shape
    d, r = _pair(x, L)
    eye = torch.eye(N, dtype=torch.bool, device=x.device)
    u2 = _phi2(r.masked_fill(eye[None], A_CUT + 1.0)).sum((1, 2)) / 2
    centers = torch.ones(B, N, dtype=torch.bool, device=x.device)
    u3 = _phi3_centers(d, r, centers, x.device)
    return u2 + u3


def mw_energy_chunked(x, L, chunk=128):
    """The [n,N,N,3] tensors blow up on big stacks (10.4 GiB lesson) — chunk measurement evals."""
    return torch.cat([mw_energy(x[i:i + chunk], L) for i in range(0, x.shape[0], chunk)])


def du_move(x, i, xi_new, L):
    """Exact dU for moving particle i -> xi_new [B,3], via local re-summation.
    Affected 3-body centers = i itself + every particle within cutoff of i's OLD or NEW position;
    recompute the phi2 row and the phi3 sums of affected centers before/after. Correctness anchor:
    tests assert equality with full recompute (1e-8)."""
    x2 = x.clone(); x2[:, i] = xi_new
    aff_old = (_pair(x, L)[1][:, i] < A_CUT)
    aff_new = (_pair(x2, L)[1][:, i] < A_CUT)
    aff = aff_old | aff_new                                            # [B,N]; includes i via <A_CUT self? no:
    aff[:, i] = True

    # 3-body energy restricted to affected centers, plus phi2 row of i (each pair once: row sum).
    def _local(xc):
        d, r = _pair(xc, L)
        row = _phi2(r[:, i].scatter(1, torch.full((xc.shape[0], 1), i, device=xc.device, dtype=torch.long),
                                    A_CUT + 1.0)).sum(1)
        u3 = _phi3_centers(d, r, aff, xc.device)
        return row + u3

    return _local(x2) - _local(x)


def capped_mw_candidate_pair_increment(candidate, cage, cage_valid=None,
                                       pair_cap=5.0, L=None):
    """Capped candidate--cage pair cost, optionally under PBC.

    This is the cheap exact subset of :func:`capped_mw_candidate_increment`.
    It matters for proposal training because a nonzero pair tilt and a zero
    three-body tilt must not accidentally construct all angular triplets.
    """
    if candidate.shape[:-2] != cage.shape[:-2] or candidate.shape[-1] != 3 or cage.shape[-1] != 3:
        raise ValueError(f"incompatible candidate/cage shapes {candidate.shape} and {cage.shape}")
    if cage_valid is None:
        cage_valid = torch.ones(cage.shape[:-1], dtype=torch.bool, device=cage.device)
    cage_valid = cage_valid.bool()
    if cage_valid.shape != cage.shape[:-1]:
        raise ValueError(f"cage_valid {cage_valid.shape} != {cage.shape[:-1]}")
    pc = cage[..., None, :, :] - candidate[..., :, None, :]
    if L is not None:
        L_t = torch.as_tensor(L, dtype=pc.dtype, device=pc.device)
        pc = pc - L_t * torch.round(pc / L_t)
    rpc = pc.norm(dim=-1)
    within_pc = (rpc < A_CUT) & cage_valid[..., None, :]
    safe_rpc = torch.where(within_pc, rpc, torch.full_like(rpc, A_CUT + 1.0))
    return _phi2(safe_rpc).clamp(0.0, float(pair_cap)).sum(-1)


def capped_mw_candidate_increment(candidate, cage, cage_valid=None,
                                  pair_cap=5.0, three_cap=5.0, L=None):
    """Capped mW cost added by candidate particles to an existing local cage.

    ``candidate`` has shape ``[..., C, 3]`` and ``cage`` has shape
    ``[..., K, 3]``.  The returned pair and three-body tensors have shape
    ``[..., C]``.  The three-body increment contains both angular terms
    centered on the candidate and the new terms centered on cage particles.
    By default no periodic wrapping is applied, preserving the local-cavity
    API.  Passing ``L`` applies the minimum-image convention and is used by
    the global cell generator.

    This is intentionally a deterministic *proposal feature*, not an energy
    used by MH.  Applying the same feature in sampling and scoring therefore
    preserves the normalized proposal density.
    """
    if candidate.shape[:-2] != cage.shape[:-2] or candidate.shape[-1] != 3 or cage.shape[-1] != 3:
        raise ValueError(f"incompatible candidate/cage shapes {candidate.shape} and {cage.shape}")
    K = cage.shape[-2]
    if cage_valid is None:
        cage_valid = torch.ones(cage.shape[:-1], dtype=torch.bool, device=cage.device)
    cage_valid = cage_valid.bool()
    if cage_valid.shape != cage.shape[:-1]:
        raise ValueError(f"cage_valid {cage_valid.shape} != {cage.shape[:-1]}")

    # Candidate--cage pairs. Retain pc/rpc for the angular terms below.
    pc = cage[..., None, :, :] - candidate[..., :, None, :]                 # [...,C,K,3]
    if L is not None:
        L_t = torch.as_tensor(L, dtype=pc.dtype, device=pc.device)
        pc = pc - L_t * torch.round(pc / L_t)
    rpc = pc.norm(dim=-1)
    within_pc = (rpc < A_CUT) & cage_valid[..., None, :]
    safe_rpc = torch.where(within_pc, rpc, torch.full_like(rpc, A_CUT + 1.0))
    pair = _phi2(safe_rpc).clamp(0.0, float(pair_cap)).sum(-1)

    # Three-body terms centered on the candidate.
    hpc = torch.where(within_pc,
                      torch.exp(GAMMA / (rpc - A_CUT).clamp_max(-1e-9)),
                      torch.zeros_like(rpc))
    upc = pc / rpc.clamp_min(1e-12)[..., None]
    cos_p = torch.einsum("...cid,...cjd->...cij", upc, upc)
    tri = torch.triu(torch.ones(K, K, dtype=torch.bool, device=cage.device), 1)
    mask_p = within_pc[..., :, :, None] & within_pc[..., :, None, :] & tri
    term_p = LAMBDA3 * (cos_p - COS0).square() * hpc[..., :, :, None] * hpc[..., :, None, :]
    three_p = term_p.clamp(max=float(three_cap)).masked_fill(~mask_p, 0.0).sum((-1, -2))

    # New three-body terms centered on cage i, with arms (candidate, cage j).
    # Existing cage-only terms cancel in the incremental cost.
    cc = cage[..., None, :, :] - cage[..., :, None, :]                       # [...,i,j,3]
    if L is not None:
        cc = cc - L_t * torch.round(cc / L_t)
    rcc = cc.norm(dim=-1)
    eye = torch.eye(K, dtype=torch.bool, device=cage.device)
    within_cc = ((rcc < A_CUT) & ~eye
                 & cage_valid[..., :, None] & cage_valid[..., None, :])
    hcc = torch.where(within_cc,
                      torch.exp(GAMMA / (rcc - A_CUT).clamp_max(-1e-9)),
                      torch.zeros_like(rcc))
    ucc = cc / rcc.clamp_min(1e-12)[..., None]
    # vector cage_i -> candidate is the negative of candidate -> cage_i
    uip = -upc
    cos_c = torch.einsum("...cid,...ijd->...cij", uip, ucc)
    mask_c = within_pc[..., :, :, None] & within_cc[..., None, :, :]
    term_c = LAMBDA3 * (cos_c - COS0).square() * hpc[..., :, :, None] * hcc[..., None, :, :]
    three_c = term_c.clamp(max=float(three_cap)).masked_fill(~mask_c, 0.0).sum((-1, -2))
    return pair, three_p + three_c
