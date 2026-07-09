"""Random-pinning point-to-set: masked (frozen-subset) exact kernels + constrained-run driver.
Pinned particles never move but contribute to the full KA energy; every learned component is behind
exact Metropolis. See docs/superpowers/specs/2026-07-08-ka-point-to-set-pinning-design.md."""
import math, torch
from liquid_coupling_flow.ipl44.ipl_swap_smc import _tame
from liquid_coupling_flow.ka_energy import ka_energy, ka_forces


def pin_mask(B, N, c, device, generator=None):
    """Random pinning: freeze ceil(c*N) particles per config. Returns mobile bool[B,N] (True = mobile)."""
    n_pin = math.ceil(c * N)
    mobile = torch.ones(B, N, dtype=torch.bool, device=device)
    for b in range(B):
        perm = torch.randperm(N, generator=generator, device=device if generator is None else generator.device)
        mobile[b, perm[:n_pin].to(device)] = False
    return mobile


def masked_mala(x, s, U, mobile, beta, L, dt, energy_fn, force_fn, fmax=60.0):
    """Tamed-MALA restricted to mobile particles: drift & noise are masked to zero on frozen sites, so their
    proposal residuals vanish and the MH ratio involves only mobile DOFs. Frozen contribute to energy/forces.
    Returns (x, U, acc_rate)."""
    m = mobile[..., None].to(x.dtype)                                  # [B,N,1]
    F = _tame(force_fn(x, s), fmax) * m
    mu = x + 0.5 * dt * dt * beta * F
    xp = torch.remainder(mu + dt * torch.randn_like(x) * m, L)         # frozen: mu=x, noise=0 -> xp=x
    Up = energy_fn(xp, s)
    Fp = _tame(force_fn(xp, s), fmax) * m
    mup = xp + 0.5 * dt * dt * beta * Fp
    d_f = xp - mu; d_f = d_f - L * torch.round(d_f / L)                # frozen residual = 0
    d_r = x - mup; d_r = d_r - L * torch.round(d_r / L)
    logq = (-(d_r ** 2).sum((1, 2)) + (d_f ** 2).sum((1, 2))) / (2 * dt * dt)
    log_ratio = -beta * (Up - U) + logq
    acc = torch.log(torch.rand(x.shape[0], device=x.device)) < log_ratio
    x = torch.where(acc[:, None, None], xp, x); U = torch.where(acc, Up, U)
    return x, U, float(acc.float().mean())
