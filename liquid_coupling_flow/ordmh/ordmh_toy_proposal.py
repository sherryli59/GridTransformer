"""Task 2 --- a deliberately-mediocre, fully-tractable ordered proposal.

This is NOT the learned generator.  It is a hand-written autoregressive proposal
over the ordered tuple (x_1, ..., x_N), used for the CORRECTNESS test so that any
bias must come from the MH construction, not from a good generator.

Per-step conditional (2D periodic box, side L, area A = L^2):

    q(x_j | x_<j) = w_u * Uniform(box)
                  + (1 - w_u) * (1/(j-1)) sum_{i<j} WrappedNormal(x_j; c_i, tau^2)

where c_i = (x_i + (L/2, L/2)) mod L is the torus-antipode of already-placed
particle i (a crude, *uncalibrated* "place it away from what's there").  The
uniform component (w_u > 0) guarantees full support everywhere pi has support and
keeps the proposal non-degenerate; the antipode bumps make the density genuinely
order-dependent (needed for the ordering-mismatch diagnostic).  It is mediocre on
purpose --- tau = L/4 is broad and nothing is tuned to the WCA excluded volume.

The FULL mixture density is what gets recorded at each drawn point, regardless of
which mixture component actually produced the draw --- the standard correctness
requirement for an importance/MH proposal.
"""
import numpy as np

_SQRT2PI = np.sqrt(2.0 * np.pi)


def _wrapped_gauss_2d(diff, tau, L, K=3):
    """2D wrapped-normal density from displacement ``diff`` (..., 2) = x - center.

    Wrapped normal on the torus: sum over periodic images k in {-K..K}.  Each 1D
    factor integrates to 1 over [0, L); the product is a proper torus density.
    """
    ks = np.arange(-K, K + 1) * L                         # (2K+1,)
    dd = diff[..., None] - ks                             # (..., 2, 2K+1)
    g = np.exp(-0.5 * (dd / tau) ** 2) / (tau * _SQRT2PI)
    wn = g.sum(axis=-1)                                   # (..., 2)
    return wn[..., 0] * wn[..., 1]                        # (...,)


class ToyOrderedProposal:
    def __init__(self, N, L, w_u=0.5, tau_frac=0.25, seed=0):
        self.N = N
        self.L = float(L)
        self.A = float(L) * float(L)
        self.w_u = float(w_u)
        self.tau = tau_frac * float(L)
        self.shift = np.array([L / 2.0, L / 2.0])
        self.rng = np.random.default_rng(seed)

    # -- per-step density (shared low level; used by BOTH sample and eval) -----
    def _step_logpdf(self, xj, placed):
        """log q(x_j | placed) for a batch.

        xj: (B, 2).  placed: (B, m, 2) already-placed positions (m may be 0).
        Returns (B,).  The full mixture density.
        """
        B = xj.shape[0]
        m = placed.shape[1]
        unif = np.full(B, 1.0 / self.A)
        if m == 0:
            return np.log(unif)
        centers = (placed + self.shift) % self.L           # (B, m, 2) antipodes
        diff = xj[:, None, :] - centers                    # (B, m, 2)
        bump = _wrapped_gauss_2d(diff, self.tau, self.L).mean(axis=1)  # (B,)
        q = self.w_u * unif + (1.0 - self.w_u) * bump
        return np.log(q)

    # -- single-prefix conditional (for the normalization test) ---------------
    def conditional_logpdf(self, xj, placed):
        """log q(x_j | placed) for xj (M,2) and a SINGLE prefix placed (m,2)."""
        xj = np.asarray(xj, dtype=float)
        placed = np.asarray(placed, dtype=float).reshape(-1, 2)
        M = xj.shape[0]
        placed_b = np.broadcast_to(placed[None], (M,) + placed.shape).copy()
        return self._step_logpdf(xj, placed_b)

    # -- draw ordered tuples, accumulating log_q inline as we sample ----------
    def sample_ordered(self, n):
        L, A, w_u, tau = self.L, self.A, self.w_u, self.tau
        rng = self.rng
        x = np.empty((n, self.N, 2))
        logq = np.zeros(n)
        for jj in range(self.N):
            if jj == 0:
                x[:, 0, :] = rng.uniform(0.0, L, size=(n, 2))
                logq += -np.log(A)
                continue
            placed = x[:, :jj, :]                           # (n, jj, 2)
            centers = (placed + self.shift) % L
            # choose mixture component per sample
            use_unif = rng.uniform(size=n) < w_u
            i_idx = rng.integers(0, jj, size=n)             # which antipode bump
            chosen_c = centers[np.arange(n), i_idx, :]      # (n, 2)
            xj_unif = rng.uniform(0.0, L, size=(n, 2))
            xj_bump = (chosen_c + tau * rng.normal(size=(n, 2))) % L
            xj = np.where(use_unif[:, None], xj_unif, xj_bump)
            x[:, jj, :] = xj
            logq += self._step_logpdf(xj, placed)           # FULL mixture density
        return x, logq

    # -- evaluate log q~ on arbitrary ordered tuples (independent loop) -------
    def log_q_ordered(self, x):
        x = np.asarray(x, dtype=float)
        single = x.ndim == 2
        if single:
            x = x[None]
        n = x.shape[0]
        logq = np.zeros(n)
        for jj in range(self.N):
            if jj == 0:
                logq += -np.log(self.A)
                continue
            logq += self._step_logpdf(x[:, jj, :], x[:, :jj, :])
        return logq[0] if single else logq
