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
