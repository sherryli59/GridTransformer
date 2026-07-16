"""Per-rung interaction tables for the identity-bridge (u) and BCY-shrinkage (lam) ladders.

u-arm: entrywise interpolation of the KA sigma/eps matrices toward common values (sig_bar, eps_bar).
At u=0 mobile-mobile pairs are species-blind (identity swaps are FREE); at u=1 physical KA.
Mobile-pinned uses interpolation strength (1+u)/2 (pinned particles keep physical character),
mirroring BCY's lam~=(1+lam)/2 convention for mobile-pinned pairs.
lam-arm: BCY Eq.2 verbatim (sig scaled, eps untouched) — verified correct 2026-07-15."""
import numpy as np
from liquid_coupling_flow.ka_energy import SIGMA, EPS

KA_SIG = np.array(SIGMA, dtype=np.float64)
KA_EPS = np.array(EPS, dtype=np.float64)


def build_tables_u(u, sig_bar=0.95, eps_bar=1.0):
    u = float(u)
    w_mm = u
    w_mp = 0.5 * (1.0 + u)
    sig_mm = w_mm * KA_SIG + (1 - w_mm) * sig_bar
    eps_mm = w_mm * KA_EPS + (1 - w_mm) * eps_bar
    sig_mp = w_mp * KA_SIG + (1 - w_mp) * sig_bar
    eps_mp = w_mp * KA_EPS + (1 - w_mp) * eps_bar
    return sig_mm, eps_mm, sig_mp, eps_mp


def build_tables_lam(lam):
    lam = float(lam)
    sig_mm = lam * KA_SIG
    sig_mp = 0.5 * (1.0 + lam) * KA_SIG
    return sig_mm, KA_EPS.copy(), sig_mp, KA_EPS.copy()


def t_of(x, x_bot, x_top, T_bot, T_top):
    """Linear T along the ladder coordinate (x = u or lam); x_bot -> T_bot, x_top -> T_top."""
    if x_top == x_bot:
        return T_bot
    f = (x - x_bot) / (x_top - x_bot)
    return T_bot + (T_top - T_bot) * f
