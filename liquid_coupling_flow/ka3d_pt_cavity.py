"""Minimal hard-wall Hamiltonian-PT driver for 3D KABLJ cavities.

This is the small G1 yardstick, not a reproduction of the full 50-boundary
Berthier-Charbonneau-Yaida campaign.  The physical bottom replica has
``(T, lambda)=(T_target, 1)``; auxiliary replicas are hotter and smaller.
"""
from __future__ import annotations

import math
import torch

from liquid_coupling_flow.ka_cavity import assert_mobile_inside
from liquid_coupling_flow.ka3d_pts_observables import core_overlap
from liquid_coupling_flow.ka3d_shrink import cavity_move, replica_exchange, shrink_particle_energy


def assert_cavity_valid(R: float, L: float, sigma_max: float = 1.0) -> None:
    """Require a full maximum-cutoff frozen buffer between periodic cavity images."""
    needed = 2.0 * float(R) + 2.5 * float(sigma_max)
    if needed > float(L) + 1e-12:
        raise AssertionError(f"periodic cavity invalid: 2R+r_cut={needed:.4f} exceeds L={float(L):.4f}")


def build_ladder(T: float, n_rep: int, T_hot: float = 1.0, lam_min: float = 0.8,
                 device=None, dtype=torch.float32):
    if n_rep < 2:
        raise ValueError("Hamiltonian PT needs at least two replicas")
    if not (0 < T <= T_hot and 0 < lam_min <= 1):
        raise ValueError("need 0<T<=T_hot and 0<lam_min<=1")
    u = torch.linspace(0, 1, n_rep, device=device, dtype=dtype)
    # Paper Appendix Eq. A1: temperature and shrinkage follow one coupled
    # straight path.  For T~0.5 and a small cavity the reported endpoint is
    # approximately (T_dec, lambda_dec)=(1.0, 0.8).
    lams = 1.0 + (float(lam_min) - 1.0) * u
    temps = float(T) + (float(T_hot) - float(T)) * u
    return 1.0 / temps, lams


def _energy_rows(x, s, mobile, L, lams):
    return 0.5 * shrink_particle_energy(x, s, L, lams, mobile).sum(1)


def _randomized_start(x_ref, s_ref, mobile, center, R, L, sweeps, step):
    """Hot/shrunk melt used only to seed the independent low-overlap arm."""
    x = x_ref.clone(); s = s_ref.clone()
    lam = x.new_tensor([0.6]); beta = x.new_tensor([1.0])
    U = _energy_rows(x, s, mobile, L, lam)
    nmove = max(1, int(mobile.sum(1).item()))
    for _ in range(int(sweeps) * nmove):
        x, U, _ = cavity_move(x, s, U, mobile, center, R, L, beta, lam, step)
    assert_mobile_inside(x, mobile, center, R, L)
    return x, s


def equilibrate_cavity(x_ref, s_ref, mobile, center, R, L, T, n_rep=6, n_sweep=4000,
                       exch_every=50, init="ref", t_rec=100, step=0.08,
                       T_hot=1.0, lam_min=0.8, randomize_sweeps=200, seed=0):
    """Run one PT arm and return bottom-replica overlap trajectory and samples.

    ``x_ref/s_ref/mobile`` describe exactly one frozen-boundary realization
    (leading dimension 1).  Species never change within a replica; they move
    only with a whole configuration during replica exchange.
    """
    assert_cavity_valid(R, L)
    if init not in ("ref", "random"):
        raise ValueError("init must be 'ref' or 'random'")
    if x_ref.shape[0] != 1 or s_ref.shape[0] != 1 or mobile.shape[0] != 1:
        raise ValueError("equilibrate_cavity accepts one boundary realization at a time")
    torch.manual_seed(seed)
    if init == "ref":
        x0, s0 = x_ref.clone(), s_ref.clone()
    else:
        x0, s0 = _randomized_start(x_ref, s_ref, mobile, center, R, L,
                                   randomize_sweeps, max(step, 0.12))
    x = x0.expand(n_rep, -1, -1).clone()
    s = s0.expand(n_rep, -1).clone()
    mob = mobile.expand(n_rep, -1).clone()
    betas, lams = build_ladder(T, n_rep, T_hot, lam_min, x.device, x.dtype)
    U = _energy_rows(x, s, mob, L, lams)
    nmove = max(1, int(mobile.sum(1).item()))
    q_traj, times, xs, ss, exch_prob = [], [], [], [], []
    for sweep in range(n_sweep + 1):
        if sweep % t_rec == 0:
            q_traj.append(float(core_overlap(x[:1], s[:1], x_ref, s_ref, center, L)[0]))
            times.append(sweep); xs.append(x[0].detach().cpu().clone()); ss.append(s[0].detach().cpu().clone())
        if sweep == n_sweep:
            break
        for _ in range(nmove):
            x, U, _ = cavity_move(x, s, U, mob, center, R, L, betas, lams, step)
        if (sweep + 1) % exch_every == 0:
            x, s, log_acc = replica_exchange(x, s, mob, center, R, L, betas, lams)
            exch_prob.append(torch.exp(log_acc.clamp(max=0)).detach().cpu())
            U = _energy_rows(x, s, mob, L, lams)
    assert_mobile_inside(x, mob, center, R, L)
    return {"t": times, "q_c_traj": q_traj, "x_final": x[0].detach().cpu(),
            "s_final": s[0].detach().cpu(), "x_samples": torch.stack(xs),
            "s_samples": torch.stack(ss), "exchange_prob":
            (torch.stack(exch_prob).mean(0) if exch_prob else torch.empty(n_rep - 1)),
            "init": init, "R": float(R), "T": float(T)}


def two_arm_converged(q_ref_traj, q_random_traj, q_tol=0.1, tail_fraction=1 / 3):
    """Compare tail means of the descending and ascending overlap arms."""
    qr = torch.as_tensor(q_ref_traj, dtype=torch.float64)
    qx = torch.as_tensor(q_random_traj, dtype=torch.float64)
    if qr.numel() < 3 or qx.numel() < 3:
        return False
    nr = max(3, math.ceil(qr.numel() * tail_fraction))
    nx = max(3, math.ceil(qx.numel() * tail_fraction))
    return bool((qr[-nr:].mean() - qx[-nx:].mean()).abs() <= q_tol)
