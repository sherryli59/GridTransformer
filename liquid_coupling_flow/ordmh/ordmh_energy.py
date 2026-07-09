"""Pure-numpy WCA (Weeks-Chandler-Andersen) energy for a 2D single-species
periodic liquid.

WCA = Lennard-Jones truncated and shifted at its minimum r_c = 2^(1/6) sigma,
so it is purely repulsive and continuous (value and derivative -> 0 at r_c):

    u(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ] + eps   for r < r_c
         = 0                                            for r >= r_c

This is the ONLY energy used by the whole probe.  It is deliberately written in
pure numpy so the reference stack (Task 1) shares no code with the torch
campaign; it is verified against two independent implementations in
``test_ordmh_energy.py`` before anything downstream trusts it.

Minimum-image is valid (unambiguous) only when L/2 >= r_c, which the probe's
density (rho* = 0.5) guarantees at every N used.
"""
import numpy as np

R_C = 2.0 ** (1.0 / 6.0)  # WCA cutoff for sigma = 1


def wca_pair_energy(r, eps=1.0, sigma=1.0):
    """WCA pair potential u(r).  Accepts scalar or array r; zero beyond r_c."""
    r = np.asarray(r, dtype=float)
    rc = R_C * sigma
    out = np.zeros_like(r)
    inside = r < rc
    # (sigma/r)^6 evaluated only where inside the cutoff (avoids 0**-6 warnings)
    sr6 = np.zeros_like(r)
    sr6[inside] = (sigma / r[inside]) ** 6
    out[inside] = 4.0 * eps * (sr6[inside] ** 2 - sr6[inside]) + eps
    return out[()] if out.ndim == 0 else out


def wca_energy(pos, L, eps=1.0, sigma=1.0):
    """Total WCA energy of one config (N,2) or a batch (B,N,2).

    Square periodic box of side ``L``; minimum-image via displacement rounding.
    Coincident particles (r == 0) give +inf energy (handled cleanly downstream
    via exp(-beta*inf) == 0).  Returns a scalar for (N,2) input, or (B,) array.
    """
    pos = np.asarray(pos, dtype=float)
    single = pos.ndim == 2
    if single:
        pos = pos[None]
    B, N, d = pos.shape
    rc = R_C * sigma
    i, j = np.triu_indices(N, k=1)
    diff = pos[:, i, :] - pos[:, j, :]            # (B, P, d)
    diff -= L * np.round(diff / L)                # minimum image
    r2 = np.einsum("bpd,bpd->bp", diff, diff)     # (B, P) squared distances
    u = np.zeros_like(r2)
    inside = r2 < rc * rc
    core = r2 <= 0.0
    ok = inside & ~core
    sr6 = (sigma * sigma / r2[ok]) ** 3           # (sigma/r)^6
    u[ok] = 4.0 * eps * (sr6 * sr6 - sr6) + eps
    u[inside & core] = np.inf
    U = u.sum(axis=1)
    return float(U[0]) if single else U
