"""Occupancy overlap Q(t) on a square cell grid for point-to-set. n_i in {0,1}; occupancy-based so Q is
invariant under species exchange (Berthier-Kob definition). Cells holding a pinned particle are excluded."""
import torch


def _ncells(L, a_c):
    return max(1, int(round(L / a_c)))


def cell_occupancy(x, L, a_c=0.3):
    """x [B,N,2] in [0,L)^2 -> occ bool [B, g*g] (g=round(L/a_c)); True if >=1 particle in the cell."""
    B, N, _ = x.shape
    g = _ncells(L, a_c)
    xi = torch.remainder(x, L)
    ci = torch.clamp((xi / L * g).long(), 0, g - 1)                    # [B,N,2] cell coords
    idx = ci[..., 0] * g + ci[..., 1]                                  # [B,N] flat cell index
    occ = torch.zeros(B, g * g, dtype=torch.bool, device=x.device)
    occ.scatter_(1, idx, torch.ones_like(idx, dtype=torch.bool))
    return occ


def pinned_cells(x, mobile, L, a_c=0.3):
    """Cells containing >=1 pinned particle in config x -> excluded bool [B, g*g]."""
    g = _ncells(L, a_c)
    xi = torch.remainder(x, L)
    ci = torch.clamp((xi / L * g).long(), 0, g - 1)
    idx = (ci[..., 0] * g + ci[..., 1])
    excl = torch.zeros(x.shape[0], g * g, dtype=torch.bool, device=x.device)
    frozen = ~mobile
    for b in range(x.shape[0]):
        excl[b].scatter_(0, idx[b][frozen[b]], torch.ones(int(frozen[b].sum()), dtype=torch.bool, device=x.device))
    return excl


def q_rand(a_c=0.3, rho=1.2):
    """Uncorrelated-occupancy baseline Q_rand = rho * a_c^2."""
    return rho * a_c * a_c


def overlap_Q(occ_t, occ_ref, excluded):
    """Q = sum_i n_i(t) n_i(0) / sum_i n_i(0) over NON-excluded cells, averaged over the batch."""
    keep = ~excluded
    num = (occ_t & occ_ref & keep).sum(1).float()
    den = (occ_ref & keep).sum(1).float().clamp_min(1.0)
    return float((num / den).mean())


def overlap_Q_perchain(occ_t, occ_ref, excluded):
    """Per-config occupancy overlap: Q_b = sum_i n_i(t)n_i(0) / sum_i n_i(0) over non-excluded cells, per row.
    Returns a float tensor [B] (no batch reduction). mean(overlap_Q_perchain(...)) == overlap_Q(...)."""
    keep = ~excluded
    num = (occ_t & occ_ref & keep).sum(1).float()
    den = (occ_ref & keep).sum(1).float().clamp_min(1.0)
    return num / den
