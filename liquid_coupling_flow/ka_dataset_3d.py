"""Generate a 3D 80:20 KABLJ equilibrium DATASET at T=0.5 (training data for a 3D generator, and/or a
better-sampled cavity reference than the 8-config lean screen).

Fast path = batched parallel-Metropolis (ka_pmc_3d, ~1 GPU launch/sweep) for the expensive DECORRELATION, then a
short EXACT single-site polish (ka_cavity_3d.local_displacement) to remove the pmc deep-plateau bias measured in
reports/logs-2026-07-09/pmc_validate_3d (~0.058/particle, cleared by ~40 exact sweeps). Cross-chain independence
gives decorrelation; each snapshot is pmc-decorrelated then exact-polished, so it is unbiased AND spread.

Honesty guard: per-config energies + mean/std recorded; this is a fast DATASET, NOT a certified 2-seed PT
reference -- for a publication xi number use ka_reference_3d (or gate the exchange band there)."""
from __future__ import annotations
import time
from pathlib import Path
import torch
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_pmc_3d import parallel_mc_disp
from liquid_coupling_flow.ka_cavity_3d import local_displacement, local_identity_swap, RHO, T as T_DEFAULT, X_B
from liquid_coupling_flow.ka_reference_3d import _lattice_start, _species


def _swaps(x, s, U, mob, beta, L, n):
    for _ in range(n):
        s, U, _ = local_identity_swap(x, s, U, mob, beta, L)
    return s, U


def _pmc_block(x, s, mob, beta, L, step, sweeps):
    for _ in range(sweeps):
        x = parallel_mc_disp(x, s, L, beta, step)
        U = ka_energy(x, s, L)
        s, U = _swaps(x, s, U, mob, beta, L, max(1, x.shape[1] // 8))
    return x, s


def _exact_polish(x, s, mob, beta, L, step, sweeps):
    U = ka_energy(x, s, L); N = x.shape[1]
    for _ in range(sweeps):
        for _ in range(N):
            x, U, _ = local_displacement(x, s, U, mob, beta, L, step)
        s, U = _swaps(x, s, U, mob, beta, L, max(1, N // 8))
    return x, s


def generate_dataset(out, N=512, n_chains=64, equil_sweeps=3000, decorr_sweeps=200, polish_sweeps=40,
                     per_chain=16, step=0.08, T=T_DEFAULT, device="cuda", seed=0):
    """Independent chains -> pmc-equilibrate -> per snapshot: pmc-decorrelate + exact-polish. Saves
    {x [n_total,N,3], s [n_total,N], L, rho, T, composition_B, energy_per_N, ...}. n_total = n_chains*per_chain."""
    out = Path(out); L = (N / RHO) ** (1 / 3); beta = 1.0 / T
    torch.manual_seed(seed)
    x = _lattice_start(N, L, n_chains, device, seed); s = _species(N, n_chains, device, seed + 17)
    mob = torch.ones(n_chains, N, dtype=torch.bool, device=device)
    t0 = time.time()
    x, s = _pmc_block(x, s, mob, beta, L, step, equil_sweeps)
    x, s = _exact_polish(x, s, mob, beta, L, step, polish_sweeps)
    print(f"[3d-data] equilibrated {n_chains} chains N={N} L={L:.2f} T={T} in {(time.time()-t0)/60:.1f} min "
          f"| U/N={float((ka_energy(x,s,L)/N).mean()):.4f}", flush=True)
    xs, ss, es = [], [], []
    for j in range(per_chain):
        x, s = _pmc_block(x, s, mob, beta, L, step, decorr_sweeps)
        x, s = _exact_polish(x, s, mob, beta, L, step, polish_sweeps)
        e = (ka_energy(x, s, L) / N).cpu()
        xs.append(x.cpu().clone()); ss.append(s.cpu().clone()); es.append(e)
        print(f"[3d-data] snapshot {j+1}/{per_chain}: {n_chains} configs U/N={float(e.mean()):.4f}"
              f"+-{float(e.std()):.4f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        X = torch.cat(xs, 0); S = torch.cat(ss, 0); E = torch.cat(es, 0)      # incremental save (checkpoint)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"x": X, "s": S, "L": L, "rho": RHO, "T": T, "composition_B": X_B, "N": N,
                    "energy_per_N": E, "energy_mean": float(E.mean()), "energy_std": float(E.std()),
                    "n_chains": n_chains, "per_chain": j + 1, "step": step, "seed": seed,
                    "method": "pmc-equil + exact-polish (fast dataset, not certified PT ref)"}, out)
    print(f"[3d-data] DONE {X.shape[0]} configs -> {out}  <U/N>={float(E.mean()):.4f}+-{float(E.std()):.4f}", flush=True)
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt"))
    p.add_argument("--N", type=int, default=512); p.add_argument("--n-chains", type=int, default=64)
    p.add_argument("--equil-sweeps", type=int, default=3000); p.add_argument("--decorr-sweeps", type=int, default=200)
    p.add_argument("--polish-sweeps", type=int, default=40); p.add_argument("--per-chain", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    generate_dataset(a.out, a.N, a.n_chains, a.equil_sweeps, a.decorr_sweeps, a.polish_sweeps, a.per_chain,
                     device=a.device)


if __name__ == "__main__":
    main()
