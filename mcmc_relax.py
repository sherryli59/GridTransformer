"""Short MCMC relaxation test: can 200 Metropolis sweeps refine transformer samples to Boltzmann?

Starts from candidate configs (rail / baseline / uniform), runs per-particle Metropolis
under the SAME LJ energy the data was generated with (periodic, kT=1, eps=sig=1), and
compares energy + g(r) against the MCMC target before and after relaxation.

The point of the rail result is refinability: a good starting point should reach Boltzmann
in a short relaxation, while a cold (uniform) start should not. This is a more physical
test than OTgap.

Env: RELAX_SIZE (L3/L4/L5, default L4), RELAX_SWEEPS (200), RELAX_STEP (0.08).
"""
import os

import h5py
import numpy as np
import torch

from grid_transformer.physics.energy import LJ

SIZE = os.environ.get("RELAX_SIZE", "L4")
SWEEPS = int(os.environ.get("RELAX_SWEEPS", "200"))
STEP = float(os.environ.get("RELAX_STEP", "0.08"))
# DATA-GENERATING sigma (mcmc/sample_lj.py default): r_min = 2^(1/6)*sigma = 1.0 at rho=1.
# (benchmark_lj27 uses sigma=1.0 for its refinability metric, but to relax toward the data's
# Boltzmann distribution we MUST use the potential the data was sampled from.)
SIGMA = 2 ** (-1 / 6)
CORE = 0.8 * SIGMA  # clash/clamp core in absolute units (0.8 sigma)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPEC = {"L3": (27, 3.0), "L4": (64, 4.0), "L5": (125, 5.0)}[SIZE]
N, L = SPEC
H5 = f"/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_{SIZE}_rho1.0_N{N}_T1.0.h5"
CANDS = {
    "rail": f"reports/multisize_arc/samples_rail_final/arc_{SIZE}_N{N}.npz",
    "baseline": f"reports/multisize_arc/samples_baseline/arc_{SIZE}_N{N}.npz",
}


def lj_for(n):
    return LJ(nparticles=n, dim=3, boxlength=L, kT=1.0, epsilon=1.0, sigma=SIGMA,
              periodic=True, spring_constant=0.5, device=DEV)


def single_particle_energy(lj, x, i):
    """LJ pair energy of particle i with all others (periodic, no harmonic). x: [B,N,3]."""
    xi = x[:, i:i + 1, :]                                   # [B,1,3]
    disp = xi - x                                           # [B,N,3]
    disp = disp - torch.round(disp / L) * L
    r = torch.linalg.norm(disp, dim=-1)                    # [B,N]
    r = r.clone()
    r[:, i] = float("inf")                                 # exclude self
    inv6 = (SIGMA / r) ** 6
    return (4.0 * (inv6 * inv6 - inv6)).sum(dim=-1)        # [B]


def metropolis(lj, x0, sweeps, step, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = x0.clone()
    B, n, _ = x.shape
    acc = 0
    traj_E = [lj.potential(x, min_dist=CORE, turn_off_harmonic=True).mean().item() / x.shape[1]]
    for s in range(sweeps):
        for i in range(n):
            e_old = single_particle_energy(lj, x, i)
            prop = x.clone()
            prop[:, i, :] = x[:, i, :] + step * torch.randn(B, 3, generator=g, device=DEV)
            prop[:, i, :] = prop[:, i, :] - torch.floor(prop[:, i, :] / L) * L
            e_new = single_particle_energy(lj, prop, i)
            dE = e_new - e_old
            paccept = torch.exp(torch.clamp(-dE, max=0.0))      # min(1, exp(-dE))
            u = torch.rand(B, generator=g, device=DEV)
            take = u < paccept
            x[take, i, :] = prop[take, i, :]
            acc += take.float().mean().item()
        if (s + 1) % 25 == 0:
            traj_E.append(lj.potential(x, min_dist=CORE, turn_off_harmonic=True).mean().item() / x.shape[1])
    return x, acc / (sweeps * n), traj_E


def _gr_hist(x, bins=60):
    """Normalized g(r)-like histogram of min-image pair distances on a FIXED [0, L/2] grid."""
    disp = x.unsqueeze(-2) - x.unsqueeze(-3)            # [B,N,N,3]
    disp = disp - torch.round(disp / L) * L
    r = torch.linalg.norm(disp, dim=-1)                # [B,N,N]
    iu = torch.triu_indices(x.shape[1], x.shape[1], offset=1)
    rr = r[:, iu[0], iu[1]].reshape(-1).detach().cpu().numpy()
    edges = np.linspace(0.0, L / 2, bins + 1)
    counts, _ = np.histogram(rr, bins=edges)
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho = x.shape[1] / (L ** 3)
    npairs = x.shape[0] * x.shape[1]
    return counts / (npairs * shell * rho * 0.5)


def gr_l1(x, target, bins=60):
    return float(np.abs(_gr_hist(x, bins) - _gr_hist(target, bins)).mean())


def ener_stats(lj, x):
    # core-clamped (r floored at 0.8σ) like benchmark_lj27: raw r^-12 explodes on a few
    # healable overlaps, so the clamped energy is the stable structural metric.
    e = lj.potential(x, min_dist=CORE, turn_off_harmonic=True) / N
    return float(e.mean()), float(e.std())


def clash_pct(x, rmin=CORE):
    disp = x.unsqueeze(-2) - x.unsqueeze(-3)
    disp = disp - torch.round(disp / L) * L
    r = torch.linalg.norm(disp, dim=-1)
    n = x.shape[1]
    iu = torch.triu_indices(n, n, offset=1)
    rr = r[:, iu[0], iu[1]]
    return 100.0 * (rr < rmin).any(dim=1).float().mean().item()


def main():
    torch.manual_seed(0)
    lj = lj_for(N)
    with h5py.File(H5, "r") as f:
        traj = f["traj"]  # (n_traj, n_step, N, 3)
        nt, ns = traj.shape[0], traj.shape[1]
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(nt * ns, size=512, replace=False))
        ti, si = np.divmod(idx, ns)
        xt = np.empty((len(idx), N, 3), dtype=np.float32)
        for t in np.unique(ti):
            m = ti == t
            xt[m] = traj[t, si[m]]
        tgt = torch.tensor(xt, dtype=torch.float32, device=DEV)
    et_m, et_s = ener_stats(lj, tgt)
    print(f"# {SIZE} N={N} L={L}  sweeps={SWEEPS} step={STEP} device={DEV}")
    print(f"# TARGET (MCMC): Uc/N mean={et_m:.3f} std={et_s:.3f}")

    # candidates: rail, baseline (gen if needed), uniform cold-start control
    cand = {}
    rail = np.load(CANDS["rail"])["x_base"]
    cand["rail"] = torch.tensor(rail, dtype=torch.float32, device=DEV)
    if os.path.exists(CANDS["baseline"]):
        cand["baseline"] = torch.tensor(np.load(CANDS["baseline"])["x_base"], dtype=torch.float32, device=DEV)
    B = cand["rail"].shape[0]
    cand["uniform"] = torch.rand(B, N, 3, generator=torch.Generator(device=DEV).manual_seed(3), device=DEV) * L
    # CONTROL: relax the target configs themselves. A correct T=1 chain leaves them at
    # equilibrium (~-0.32); drift to a much lower energy ⇒ the move scheme is over-cooling.
    cand["TARGET-ctl"] = tgt[:B].clone()

    gt_clash = clash_pct(tgt)
    print(f"# TARGET clash%={gt_clash:.1f}")
    print(f"\n{'candidate':10s} {'Uc/N init':>10s} {'Uc/N fin':>9s} {'(tgt)':>7s} "
          f"{'gr_L1 i':>8s} {'gr_L1 f':>8s} {'clash%i':>8s} {'clash%f':>8s} {'acc':>5s}")
    for name, x0 in cand.items():
        ei_m, _ = ener_stats(lj, x0)
        gi = gr_l1(x0, tgt); ci = clash_pct(x0)
        xf, accept, trajE = metropolis(lj, x0, SWEEPS, STEP, seed=1)
        ef_m, _ = ener_stats(lj, xf)
        gf = gr_l1(xf, tgt); cf = clash_pct(xf)
        print(f"{name:10s} {ei_m:10.3f} {ef_m:9.3f} {et_m:7.3f} {gi:8.4f} {gf:8.4f} "
              f"{ci:8.1f} {cf:8.1f} {accept:5.2f}")
        print(f"           clamped Uc/N traj (0,25,50,...sweeps): {[round(v,2) for v in trajE]}")
    print(f"\n# target Uc/N = {et_m:.3f}: a candidate 'recovers Boltzmann' if final Uc/N and")
    print(f"# gr_L1 approach the target's, AND the cold uniform start does NOT (good init matters).")


if __name__ == "__main__":
    main()
