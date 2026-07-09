"""Task 1 --- Independent ground-truth reference for the 2D WCA liquid.

Two methods that share NO code with the ordered-space sampler under test:

  * ``displacement_mc`` --- textbook single-particle-displacement Metropolis MC
    on the physical (unordered) system.  Trivially correct, standard local
    moves, vectorised over many independent chains (chains give both parallelism
    and a clean error bar / init-independence check).

  * ``grid_quadrature`` --- direct numerical quadrature of the configurational
    Boltzmann integral for N=2 (2D) and N=3 (4D).  A second, fully non-MC ground
    truth.  For N=3 this is the *bedrock* correctness oracle of the whole probe.

Gate R: the two must agree on <U> within error.  If not, the ground truth is
broken and nothing downstream means anything --- so this file also runs the
gate in __main__ at heavy settings and saves the reference.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ordmh_energy import wca_energy, wca_pair_energy, R_C


# ---------------------------------------------------------------------------
# Boltzmann-weighted reduction (shared estimator, exactly unit-tested)
# ---------------------------------------------------------------------------
def weighted_boltzmann_mean(U, beta):
    """<U> = sum U e^{-beta U} / sum e^{-beta U}, with +inf entries contributing
    exactly 0 (never nan)."""
    U = np.asarray(U, dtype=float)
    w = np.exp(-beta * U)                       # inf -> 0
    finite = np.isfinite(U)
    Ufin = np.where(finite, U, 0.0)             # kill inf BEFORE multiply (no nan)
    num = np.sum(Ufin * w)
    den = np.sum(w)
    return num / den


# ---------------------------------------------------------------------------
# Grid quadrature (non-MC ground truth), N in {2, 3}
# ---------------------------------------------------------------------------
def _midpoints(L, n):
    return (np.arange(n) + 0.5) * (L / n)


def grid_quadrature(N, L, beta, n_grid, chunk=200000, u_bins=60):
    """Direct quadrature of the configurational Boltzmann integral.

    Particle 0 is pinned at the origin (translation removed); the remaining
    N-1 particles are integrated over the box on a midpoint grid with ``n_grid``
    points per axis.  Returns <U>, an energy histogram P(U), and a grid-
    discretisation error estimate from halving the resolution.

    Only N in {2, 3} (2D / 4D integrals) are supported --- higher N is not
    grid-tractable, which is exactly why we only claim the non-MC oracle there.
    """
    if N not in (2, 3):
        raise ValueError("grid quadrature only supports N in {2,3}")

    def _accumulate(ng):
        ax = _midpoints(L, ng)
        if N == 2:
            X1, Y1 = np.meshgrid(ax, ax, indexing="ij")
            free = np.stack([X1.ravel(), Y1.ravel()], axis=1)[:, None, :]  # (M,1,2)
        else:  # N == 3: two free particles -> 4D grid
            X1, Y1, X2, Y2 = np.meshgrid(ax, ax, ax, ax, indexing="ij")
            free = np.stack([X1.ravel(), Y1.ravel(),
                             X2.ravel(), Y2.ravel()], axis=1).reshape(-1, 2, 2)
        M = free.shape[0]
        num = den = 0.0
        u_lo, u_hi = 0.0, 0.0
        hist = None
        edges = None
        for s in range(0, M, chunk):
            fr = free[s:s + chunk]
            B = fr.shape[0]
            pos = np.concatenate([np.zeros((B, 1, 2)), fr], axis=1)  # pin p0 at 0
            U = wca_energy(pos, L)
            w = np.exp(-beta * U)
            fin = np.isfinite(U)
            num += np.sum(np.where(fin, U, 0.0) * w)  # kill inf before multiply
            den += np.sum(w)
            if edges is None:
                # energy histogram range from a cheap pre-scan of this chunk
                uf = U[fin & (w > 0)]
                u_hi = np.quantile(uf, 0.9995) if uf.size else 1.0
                edges = np.linspace(0.0, max(u_hi, 1e-6), u_bins + 1)
                hist = np.zeros(u_bins)
            uf = U[fin]
            wf = w[fin]
            h, _ = np.histogram(uf, bins=edges, weights=wf)
            hist = hist + h
        mean_U = num / den
        centers = 0.5 * (edges[:-1] + edges[1:])
        probs = hist / hist.sum()
        return mean_U, centers, probs

    mean_hi, centers, probs = _accumulate(n_grid)
    mean_lo, _, _ = _accumulate(max(4, n_grid // 2))
    # grid error ~ |fine - coarse| (midpoint rule, conservative)
    err = abs(mean_hi - mean_lo)
    return {"mean_U": mean_hi, "mean_U_err": err, "n_grid": n_grid,
            "PU": (centers, probs), "mean_U_coarse": mean_lo}


# ---------------------------------------------------------------------------
# Displacement Metropolis MC (independent reference sampler)
# ---------------------------------------------------------------------------
def _wca_single_particle_energy(pos, k, L):
    """Energy of particle k with all others, per chain.  pos: (B,N,2) -> (B,)."""
    diff = pos[:, k, :][:, None, :] - pos            # (B,N,2)
    diff -= L * np.round(diff / L)
    r2 = np.einsum("bnd,bnd->bn", diff, diff)        # (B,N)
    r2[:, k] = np.inf                                # exclude self
    rc2 = R_C * R_C
    u = np.zeros_like(r2)
    inside = r2 < rc2
    core = r2 <= 0.0
    ok = inside & ~core
    sr6 = (1.0 / r2[ok]) ** 3
    u[ok] = 4.0 * (sr6 * sr6 - sr6) + 1.0
    u[inside & core] = np.inf
    return u.sum(axis=1)


def displacement_mc(N, L, beta, n_chains=384, n_equil=500, n_collect=400,
                    thin=3, step=0.22, seed=0, x0=None, sp_energy_fn=None):
    """Vectorised single-particle-displacement Metropolis MC over independent
    chains.  Returns collected configs (M,N,2), per-chain <U> (for a clean SEM
    and init-independence), and the acceptance rate."""
    rng = np.random.default_rng(seed)
    sp_energy = sp_energy_fn if sp_energy_fn is not None else _wca_single_particle_energy
    if x0 is None:
        pos = rng.uniform(0.0, L, size=(n_chains, N, 2))
    else:
        pos = np.array(x0, dtype=float)
        n_chains = pos.shape[0]
    B = pos.shape[0]

    n_acc = 0
    n_prop = 0
    collected = []
    chain_U_sum = np.zeros(B)
    chain_U_cnt = 0
    total_sweeps = n_equil + n_collect * thin
    for sweep in range(total_sweeps):
        for k in range(N):
            u_old = sp_energy(pos, k, L)
            trial = (pos[:, k, :] + step * rng.uniform(-1.0, 1.0, size=(B, 2))) % L
            saved = pos[:, k, :].copy()
            pos[:, k, :] = trial
            u_new = sp_energy(pos, k, L)
            dU = u_new - u_old
            accept = np.log(rng.uniform(0.0, 1.0, size=B)) < -beta * dU
            # revert rejected
            pos[:, k, :] = np.where(accept[:, None], trial, saved)
            n_acc += int(accept.sum())
            n_prop += B
        if sweep >= n_equil and (sweep - n_equil) % thin == 0:
            collected.append(pos.copy())
            chain_U_sum += wca_energy(pos, L)
            chain_U_cnt += 1

    configs = np.concatenate(collected, axis=0) if collected else pos.copy()
    chain_meanU = chain_U_sum / max(chain_U_cnt, 1)          # (B,) per-chain <U>
    return {
        "configs": configs,
        "acc_rate": n_acc / max(n_prop, 1),
        "chain_meanU": chain_meanU,
        "mean_U": float(chain_meanU.mean()),
        "mean_U_err": float(chain_meanU.std(ddof=1) / np.sqrt(B)),
        "n_chains": B,
    }


def observables_from_configs(configs, L, beta=1.0, gr_nbins=80, u_bins=60,
                             n_chains=None):
    """Permutation-invariant observables from a bag of configs: <U> (+SEM via
    chain blocks if n_chains given), energy histogram P(U), and g(r)."""
    U = wca_energy(configs, L)
    mean_U = float(U.mean())
    if n_chains and configs.shape[0] % n_chains == 0:
        blocks = U.reshape(n_chains, -1).mean(axis=1)
        mean_U_err = float(blocks.std(ddof=1) / np.sqrt(n_chains))
    else:
        mean_U_err = float(U.std(ddof=1) / np.sqrt(U.shape[0]))
    # energy histogram
    u_hi = np.quantile(U, 0.999)
    edges = np.linspace(0.0, max(u_hi, 1e-6), u_bins + 1)
    hist, _ = np.histogram(U, bins=edges, density=True)
    ucent = 0.5 * (edges[:-1] + edges[1:])
    # g(r)
    gr_c, gr = radial_distribution(configs, L, nbins=gr_nbins)
    return {"mean_U": mean_U, "mean_U_err": mean_U_err,
            "PU": (ucent, hist), "gr": (gr_c, gr)}


def radial_distribution(configs, L, nbins=80, rmax=None):
    """2D radial distribution function g(r) with minimum-image PBC, plateau ->1."""
    configs = np.asarray(configs, dtype=float)
    B, N, d = configs.shape
    if rmax is None:
        rmax = L / 2.0
    i, j = np.triu_indices(N, k=1)
    diff = configs[:, i, :] - configs[:, j, :]
    diff -= L * np.round(diff / L)
    r = np.sqrt(np.einsum("bpd,bpd->bp", diff, diff)).ravel()
    r = r[r < rmax]
    edges = np.linspace(0.0, rmax, nbins + 1)
    counts, _ = np.histogram(r, bins=edges)
    # ideal-gas normalisation in 2D: shell area pi(r2^2-r1^2), density (N-1)/A
    area_box = L * L
    rho = (N - 1) / area_box
    shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    n_pairs_per_config = B  # each of B configs contributes to counts once
    ideal = shell * rho * n_pairs_per_config
    g = counts / ideal
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, g


# ---------------------------------------------------------------------------
# Gate R driver: heavy run + save reference
# ---------------------------------------------------------------------------
def _gate_R(N, L, beta, n_grid, mc_kwargs, label):
    grid = grid_quadrature(N=N, L=L, beta=beta, n_grid=n_grid)
    mc = displacement_mc(N=N, L=L, beta=beta, **mc_kwargs)
    obs = observables_from_configs(mc["configs"], L=L, beta=beta,
                                   n_chains=mc["n_chains"])
    diff = abs(obs["mean_U"] - grid["mean_U"])
    tol = 5 * obs["mean_U_err"] + grid["mean_U_err"]
    ok = diff < tol
    print(f"[{label}] N={N} L={L:.4f} beta={beta}")
    print(f"   grid   <U> = {grid['mean_U']:.5f}  (grid-halving err {grid['mean_U_err']:.5f},"
          f" n_grid={n_grid})")
    print(f"   MC     <U> = {obs['mean_U']:.5f} +/- {obs['mean_U_err']:.5f}"
          f"  (acc {mc['acc_rate']:.3f}, chains {mc['n_chains']})")
    print(f"   |diff| = {diff:.5f}   tol = {tol:.5f}   -> {'PASS' if ok else 'FAIL'}")
    return ok, grid, mc, obs


if __name__ == "__main__":
    import pickle
    L3 = float(np.sqrt(6.0))   # N=3, rho*=0.5
    L2 = 2.0
    beta = 1.0
    results = {}
    ok_all = True

    ok2, grid2, mc2, obs2 = _gate_R(
        N=2, L=L2, beta=beta, n_grid=600,
        mc_kwargs=dict(n_chains=512, n_equil=800, n_collect=1000, thin=3,
                       step=0.25, seed=101),
        label="Gate R / N=2 (2D quad)")
    ok_all &= ok2

    ok3, grid3, mc3, obs3 = _gate_R(
        N=3, L=L3, beta=beta, n_grid=64,
        mc_kwargs=dict(n_chains=768, n_equil=1000, n_collect=1500, thin=3,
                       step=0.22, seed=202),
        label="Gate R / N=3 (4D quad, BEDROCK)")
    ok_all &= ok3

    print(f"\nGATE R: {'PASS' if ok_all else 'FAIL'}")

    # Save the reference (configs + observables) so downstream checks reuse it.
    out = {
        "params": {"beta": beta, "L2": L2, "L3": L3, "rho_star": 0.5, "T_star": 1.0},
        "N2": {"grid": grid2, "mc_meanU": obs2["mean_U"], "mc_err": obs2["mean_U_err"],
               "gr": obs2["gr"], "PU": obs2["PU"], "configs": mc2["configs"]},
        "N3": {"grid": grid3, "mc_meanU": obs3["mean_U"], "mc_err": obs3["mean_U_err"],
               "gr": obs3["gr"], "PU": obs3["PU"], "configs": mc3["configs"]},
    }
    art_dir = os.path.join(HERE, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    path = os.path.join(art_dir, "ordmh_reference.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"saved reference -> {path}")
    sys.exit(0 if ok_all else 1)
