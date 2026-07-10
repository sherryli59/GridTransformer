"""Shrinkage-ensemble KABLJ pair energy (Berthier-Charbonneau-Yaida point-to-set cavity method).

At shrinkage parameter lam in [0,1], each pair's sigma is scaled by lambda_tilde(lam, mobile_i, mobile_j):
lam if both particles are mobile, (1+lam)/2 if exactly one is mobile (one mobile, one pinned), 1.0 if both are
pinned (unshrunk). The cutoff scales with the shrunk sigma: rc = RCUT_FACTOR * lambda_tilde * sigma_ab. eps is
left unscaled (only the core radius shrinks). At lam=1.0, lambda_tilde==1.0 everywhere, so this collapses exactly
onto the standard shifted-LJ/minimum-image convention used by `ka_energy`/`ka_pmc_3d.particle_energies`.

`shrink_pair_row` mirrors `ka_energy.ka_pair_row_scatter`'s O(N) single-particle-row pattern (idx=(rows, i),
candidate position xi) for use in single-site Metropolis moves: delta = shrink_pair_row(new) - shrink_pair_row(old)."""
from __future__ import annotations
import torch
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR
from liquid_coupling_flow.ka_cavity import cavity_inside


def _lt_values(lam, mi, mj, dtype):
    """mi, mj: broadcastable bool tensors (mobile masks). Returns a `dtype` tensor of the broadcast shape with
    lam where both mobile, (1+lam)/2 where exactly one mobile, 1.0 where both pinned."""
    both = mi & mj
    one = mi ^ mj
    lt = torch.ones_like(both, dtype=dtype)
    lt = torch.where(one, torch.full_like(lt, (1.0 + lam) / 2.0), lt)
    lt = torch.where(both, torch.full_like(lt, float(lam)), lt)
    return lt


def lambda_tilde(lam, mobile_i, mobile_j):
    """mobile_i, mobile_j: [B,N] bool mobile masks. Returns [B,N,N]."""
    B, Ni = mobile_i.shape
    Nj = mobile_j.shape[1]
    mi = mobile_i[:, :, None].expand(B, Ni, Nj)
    mj = mobile_j[:, None, :].expand(B, Ni, Nj)
    return _lt_values(lam, mi, mj, torch.get_default_dtype())


def shrink_particle_energy(x, s, L, lam, mobile):
    """pe_i = sum_{j!=i} e(x_i, x_j) under shrinkage lam, shifted-LJ, minimum image. Returns [B, N].
    At lam=1.0 exactly equals `ka_pmc_3d.particle_energies` (lambda_tilde==1.0 everywhere)."""
    B, N, _ = x.shape
    dtype = x.dtype
    si = s.long()
    t_sig = torch.tensor(SIGMA, device=x.device, dtype=dtype)
    t_eps = torch.tensor(EPS, device=x.device, dtype=dtype)
    a = si[:, :, None].expand(-1, -1, N); b = si[:, None, :].expand(-1, N, -1)
    sig = t_sig[a, b]; eps = t_eps[a, b]
    mi = mobile[:, :, None].expand(-1, -1, N); mj = mobile[:, None, :].expand(-1, N, -1)
    lt = _lt_values(lam, mi, mj, dtype)
    sig_s = lt * sig
    rc = RCUT_FACTOR * sig_s
    diff = x[:, :, None, :] - x[:, None, :, :]; diff = diff - L * torch.round(diff / L)
    eye = torch.eye(N, device=x.device, dtype=torch.bool)[None]
    r2 = (diff ** 2).sum(-1).masked_fill(eye, 1e12); inv6 = (sig_s ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6); src6 = (sig_s / rc) ** 6
    return torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e)).sum(-1)


def shrink_pair_row(x, s, idx, xi, L, lam, mobile):
    """O(N) row energy of the single particle idx=(rows,i) placed at candidate position xi, against all
    other particles, under shrinkage lam. Same shifted/cutoff/min-image convention as `shrink_particle_energy`
    (mirrors `ka_energy.ka_pair_row_scatter`'s signature/pattern). rows,i: [K]; xi: [K,3]. Returns [K]."""
    b_ids, p_ids = idx
    B, N, _ = x.shape
    dtype = x.dtype
    si = s.long()
    if si.dim() == 1:
        si = si[None].expand(B, -1)
    t_sig = torch.tensor(SIGMA, device=x.device, dtype=dtype)
    t_eps = torch.tensor(EPS, device=x.device, dtype=dtype)
    s_row = si[b_ids]                                          # [K,N]
    s_i = si[b_ids, p_ids]                                     # [K]
    sig = t_sig[s_i[:, None].expand(-1, N), s_row]             # [K,N]
    eps = t_eps[s_i[:, None].expand(-1, N), s_row]

    mobile_row = mobile[b_ids]                                 # [K,N]
    mobile_i = mobile[b_ids, p_ids][:, None].expand(-1, N)     # [K,N]
    lt = _lt_values(lam, mobile_i, mobile_row, dtype)          # [K,N]

    sig_s = lt * sig
    rc = RCUT_FACTOR * sig_s
    diff = xi[:, None, :] - x[b_ids]                           # [K,N,3]
    diff = diff - L * torch.round(diff / L)
    r2 = (diff ** 2).sum(-1)                                   # [K,N]
    r2[torch.arange(r2.shape[0], device=x.device), p_ids] = 1e12
    inv6 = (sig_s ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (sig_s / rc) ** 6
    e = torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e))
    return e.sum(-1)                                           # [K]


def _pick_mobile(mobile):
    """Multinomial-sample one mobile-particle index per batch row (mirrors `ka_cavity_3d._pick`).
    Rows with no mobile particles fall back to index 0 and are flagged invalid."""
    valid = mobile.any(1)
    weights = mobile.float()
    weights[~valid, 0] = 1.0
    return torch.multinomial(weights, 1).squeeze(1), valid


def cavity_move(x, s, U, mobile, center, R, L, beta, lam, step):
    """One exact O(N) single-site hard-wall Metropolis move per batch row, under shrinkage `lam`.

    Picks one mobile particle per row, proposes a wrapped Gaussian step, rejects any proposal that
    lands outside the spherical cavity (`cavity_inside`, strict hard wall), else accepts/rejects via
    Metropolis on the O(N) `shrink_pair_row` energy delta. `U` is each row's total shrinkage energy
    (`0.5*shrink_particle_energy(...).sum(1)`); since `shrink_pair_row` uses the same not-double-counted
    convention as `shrink_particle_energy`'s 0.5*sum, moving one particle changes that total by exactly
    `new_e - old_e`. Mirrors `ka_cavity_3d.local_displacement`, scored under shrinkage instead of the
    unshrunk `ka_pair_row_scatter` energy. x is mutated in-place (caller owns the chain state)."""
    B = x.shape[0]
    rows = torch.arange(B, device=x.device)
    i, valid = _pick_mobile(mobile)
    old = x[rows, i]
    prop = torch.remainder(old + step * torch.randn_like(old), L)
    inside = cavity_inside(prop, center, R, L)
    old_e = shrink_pair_row(x, s, (rows, i), old, L, lam, mobile)
    new_e = shrink_pair_row(x, s, (rows, i), prop, L, lam, mobile)
    acc = valid & inside & (torch.log(torch.rand(B, device=x.device)) < -beta * (new_e - old_e))
    x[rows[acc], i[acc]] = prop[acc]
    return x, torch.where(acc, U + new_e - old_e, U), acc


def replica_exchange(x, s, mobile, center, R, L, betas, lams):
    """One sweep of adjacent-pair replica exchange over a shrinkage/temperature ladder of `n_rep`
    replicas sharing one cavity (`center`, `R`). For every adjacent pair (a, a+1) the exact log
    Metropolis ratio for swapping the two replicas' full configs (positions AND species) is:

        log A = -beta_a*[H_a(x_b) - H_a(x_a)] - beta_b*[H_b(x_a) - H_b(x_b)]

    where H_r(cfg) = 0.5*shrink_particle_energy(cfg_x, cfg_s, L, lam_r, mobile_r).sum(1) — the
    Hamiltonian (lam, mobile) stays with the replica slot `r`; the swapped quantity is the config
    (x together with its species s). `log_acc` is computed once, deterministically, from the configs
    passed in — independent of the random accept coin, which is drawn separately per pair so the
    returned ratio is directly testable. All pairwise ratios are computed from the ORIGINAL input
    configs before any swap is applied (so `log_acc` never depends on another pair's accept coin);
    swaps are then applied sequentially in ascending pair order. Species channel only moves here as
    part of a whole-config swap (no A/B identity exchange). `center`/`R` are accepted for interface
    symmetry with the shared cavity the ladder lives in, but the swap ratio depends only on `mobile`
    (already encodes the pinned/mobile partition) — this function does not itself enforce the hard
    wall (that is `cavity_move`'s job on each site update); it is valid to call on any pair of configs.
    Returns (x, s, log_acc[n_rep-1])."""
    n_rep = x.shape[0]
    log_acc = x.new_zeros(n_rep - 1)
    for a in range(n_rep - 1):
        b = a + 1
        lam_a, lam_b = float(lams[a]), float(lams[b])
        Ha_xa = 0.5 * shrink_particle_energy(x[a:a + 1], s[a:a + 1], L, lam_a, mobile[a:a + 1]).sum(1)
        Ha_xb = 0.5 * shrink_particle_energy(x[b:b + 1], s[b:b + 1], L, lam_a, mobile[a:a + 1]).sum(1)
        Hb_xb = 0.5 * shrink_particle_energy(x[b:b + 1], s[b:b + 1], L, lam_b, mobile[b:b + 1]).sum(1)
        Hb_xa = 0.5 * shrink_particle_energy(x[a:a + 1], s[a:a + 1], L, lam_b, mobile[b:b + 1]).sum(1)
        log_acc[a] = -(betas[a] * (Ha_xb - Ha_xa) + betas[b] * (Hb_xa - Hb_xb))
    for a in range(n_rep - 1):
        b = a + 1
        if torch.log(torch.rand((), device=x.device, dtype=x.dtype)) < log_acc[a]:
            x[a], x[b] = x[b].clone(), x[a].clone()
            s[a], s[b] = s[b].clone(), s[a].clone()
    return x, s, log_acc
