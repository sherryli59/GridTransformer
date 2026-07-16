"""Count-preserving identity-swap move: exchange the species LABELS of one mobile A and one
mobile B (positions fixed), exact MH on dU under the rung's tables. At u=0 (species-blind
tables) dU==0 -> free; at u=1 this is the known-dead plain swap. Pinned labels never touched."""
import numpy as np
from numba import njit
from liquid_coupling_flow.ptu.kernels import row_e


@njit(cache=True, fastmath=True)
def identity_sweep(allx, alls, n, n_tot, beta, n_try, sig_mm, eps_mm, sig_mp, eps_mp):
    acc = 0
    att = 0
    for _ in range(n_try):
        ia = np.random.randint(n)
        ib = np.random.randint(n)
        if alls[ia] == alls[ib]:
            continue
        att += 1
        e0 = (row_e(allx, alls, ia, allx[ia, 0], allx[ia, 1], allx[ia, 2], n, n_tot,
                    sig_mm, eps_mm, sig_mp, eps_mp)
              + row_e(allx, alls, ib, allx[ib, 0], allx[ib, 1], allx[ib, 2], n, n_tot,
                      sig_mm, eps_mm, sig_mp, eps_mp))
        sa = alls[ia]
        alls[ia] = alls[ib]; alls[ib] = sa
        e1 = (row_e(allx, alls, ia, allx[ia, 0], allx[ia, 1], allx[ia, 2], n, n_tot,
                    sig_mm, eps_mm, sig_mp, eps_mp)
              + row_e(allx, alls, ib, allx[ib, 0], allx[ib, 1], allx[ib, 2], n, n_tot,
                      sig_mm, eps_mm, sig_mp, eps_mp))
        if np.random.random() < np.exp(-beta * (e1 - e0)):
            acc += 1
        else:
            sb = alls[ia]
            alls[ia] = alls[ib]; alls[ib] = sb          # revert
    return acc, att
