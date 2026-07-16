"""Deployment-matched block pairs: freeze env, permute the block's own sigmas (2 random
transpositions), re-equilibrate block positions by block-local displacement MC (RELAX_SW sweeps).
Pair = (old positions + old assignment) -> (relaxed positions + new assignment). Short transport,
same-particle correspondence (no OT). Usage as module (make_block_pair) or script (build dataset)."""
import sys
import numpy as np
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.poly.model import row_e, seed_numba
from numba import njit

RELAX_SW = 400


@njit(cache=True, fastmath=True)
def _block_relax(x, sig, L, beta, idx, n_sw, step):
    for _ in range(n_sw):
        for t in range(idx.shape[0]):
            i = idx[np.random.randint(idx.shape[0])]
            xn0 = (x[i, 0] + step * np.random.randn()) % L
            xn1 = (x[i, 1] + step * np.random.randn()) % L
            xn2 = (x[i, 2] + step * np.random.randn()) % L
            e0 = row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
            e1 = row_e(x, sig, i, xn0, xn1, xn2, L)
            if np.random.random() < np.exp(-beta * (e1 - e0)):
                x[i, 0] = xn0; x[i, 1] = xn1; x[i, 2] = xn2


def make_block_pair(x, sig, L, beta, k, seed):
    rng = np.random.default_rng(seed)
    seed_numba(seed)
    n = x.shape[0]
    s0 = rng.integers(n)
    d = x - x[s0]
    d -= L * np.round(d / L)
    idx = np.argsort((d ** 2).sum(1))[:k].astype(np.int64)
    mask = np.zeros(n, bool); mask[idx] = True
    xw, sw = x.copy(), sig.copy()
    x_old = xw[idx].copy(); sig_old = sw[idx].copy()
    perm = rng.permutation(k)
    while np.all(perm == np.arange(k)):
        perm = rng.permutation(k)
    sw[idx] = sig_old[perm]
    _block_relax(xw, sw, L, beta, idx, RELAX_SW, 0.1)
    cen = x_old.mean(0)
    def cent(a):
        b = a - cen
        return b - L * np.round(b / L)
    return {"idx": idx, "env_x": cent(xw[~mask]), "env_sig": sw[~mask],
            "x_old": cent(x_old), "sig_old": sig_old, "perm": perm,
            "sig_new": sw[idx].copy(), "x_new": cent(xw[idx]), "L": L, "beta": beta}


if __name__ == "__main__":
    import torch, time
    T = float(sys.argv[1]); K = int(sys.argv[2]); NPAIRS = int(sys.argv[3])
    bank = torch.load("reports/logs-2026-07-17/poly_tau_curves.pt", weights_only=False)
    train_runs = [0, 1, 2]                                     # run 3 held out
    pairs = []
    t0 = time.time()
    for m in range(NPAIRS):
        run = train_runs[m % len(train_runs)]
        rec = bank[(run, T)]
        fr = rec["x"][m % rec["x"].shape[0]]
        pairs.append(make_block_pair(fr, rec["sig"], rec["L"], 1.0 / T, K, seed=9000 + m))
        if (m + 1) % 2000 == 0:
            torch.save(pairs, f"reports/logs-2026-07-17/poly_blockpairs_T{T}_k{K}.pt")
            print(f"{m+1}/{NPAIRS} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(pairs, f"reports/logs-2026-07-17/poly_blockpairs_T{T}_k{K}.pt")
    print("PAIRS DONE", flush=True)
