"""Validated bulk parallel-tempering reference for the 3D 80:20 KABLJ cavity gate.

This is intentionally separate from ``ka_cavity_3d``: a cavity result is only
interpretable if its frozen exterior came from a bulk equilibrium reference.
The sampler uses exact one-site Metropolis displacements plus exact A/B identity
swaps and temperature exchanges.  It persists each independent seed before the
second one starts, then records whether the seed agreement and collection-tail
drift gates pass.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from liquid_coupling_flow.ka_cavity_3d import D, RHO, T, X_B, local_displacement, local_identity_swap
from liquid_coupling_flow.ka_energy import ka_energy


def _species(N, B, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.zeros(N, dtype=torch.long)
    s[torch.randperm(N, generator=g)[:round(X_B * N)]] = 1
    return s.to(device)[None].expand(B, -1).clone()


def _lattice_start(N, L, B, device, seed):
    side = math.ceil(N ** (1 / 3))
    a = (torch.arange(side, device=device, dtype=torch.float32) + .5) * L / side
    grid = torch.stack(torch.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, D)[:N]
    g = torch.Generator(device=device).manual_seed(seed)
    # Independent random lattice relabellings and small jitter avoid a shared
    # artificial origin while retaining a finite-energy start.
    x = grid[torch.stack([torch.randperm(N, generator=g, device=device) for _ in range(B)])]
    return torch.remainder(x + .04 * torch.randn(B, N, D, generator=g, device=device), L)


def _pt_seed(N, L, seed, device, n_equil, n_collect, every, n_temp=8, exchange_every=10, track_every=200):
    """One exact PT run. Returns cold snapshots, their species, and energy diagnostics."""
    torch.manual_seed(seed)
    temps = T * (1.5 / T) ** (torch.arange(n_temp, device=device) / (n_temp - 1))
    beta = 1 / temps
    x = _lattice_start(N, L, n_temp, device, seed)
    s = _species(N, n_temp, device, seed + 17)
    U = ka_energy(x, s, L)
    mobile = torch.ones(n_temp, N, device=device, dtype=torch.bool)
    snapshots, species, track, collection_u, exchange = [], [], [], [], []
    for sweep in range(n_equil + n_collect):
        # One local sweep: each temperature replica attempts N positions and N/8 swaps.
        for _ in range(N):
            x, U, _ = local_displacement(x, s, U, mobile, beta, L, .08)
        for _ in range(max(1, N // 8)):
            s, U, _ = local_identity_swap(x, s, U, mobile, beta, L)
        if (sweep + 1) % exchange_every == 0:
            for parity in (0, 1):
                for rung in range(parity, n_temp - 1, 2):
                    p = torch.exp((beta[rung] - beta[rung + 1]) * (U[rung] - U[rung + 1])).clamp(max=1)
                    if bool(torch.rand((), device=device) < p):
                        x[[rung, rung + 1]] = x[[rung + 1, rung]]
                        s[[rung, rung + 1]] = s[[rung + 1, rung]]
                        U[[rung, rung + 1]] = U[[rung + 1, rung]]
                        exchange.append(1.0)
                    else:
                        exchange.append(0.0)
        if (sweep + 1) % track_every == 0:
            # Full recomputation is an exact bookkeeping audit, not a sampling move.
            exact = ka_energy(x, s, L)
            if not torch.allclose(U, exact, atol=3e-3, rtol=2e-5):
                raise RuntimeError("local-MC energy bookkeeping drifted from ka_energy")
            U = exact
            track.append((sweep + 1, float((U[0] / N).item())))
        if sweep >= n_equil and (sweep - n_equil) % every == 0:
            snapshots.append(x[0].cpu().clone()); species.append(s[0].cpu().clone()); collection_u.append(float((U[0] / N).item()))
    return torch.stack(snapshots), torch.stack(species), track, collection_u, float(sum(exchange) / max(1, len(exchange)))


def build_reference(out: Path, N=512, device="cuda", n_equil=5000, n_collect=2000, every=500):
    L = (N / RHO) ** (1 / 3)
    if N < 64: raise ValueError("N is too small for a meaningful 3D bulk reference")
    seed_rows, records = [], []
    for seed in (0, 1):
        x, s, track, coll_u, ex = _pt_seed(N, L, seed, device, n_equil, n_collect, every)
        partial = out.with_name(f"{out.stem}.seed{seed}.partial.pt")
        torch.save({"x": x, "s": s, "L": L, "rho": RHO, "T": T, "composition_B": X_B,
                    "converged": False, "seed": seed, "track": track, "collection_U_per_N": coll_u,
                    "exchange_acceptance": ex}, partial)
        seed_rows.append((x, s)); records.append((track, coll_u, ex))
        print(f"[3d ref seed={seed}] cold U/N={sum(coll_u)/len(coll_u):.4f} exch={ex:.3f} saved={partial}", flush=True)
    means = [sum(v[1]) / len(v[1]) for v in records]
    drift = [abs(sum(v[1][:len(v[1]) // 2]) / max(1, len(v[1]) // 2) - sum(v[1][len(v[1]) // 2:]) / max(1, len(v[1]) - len(v[1]) // 2)) for v in records]
    agree = abs(means[0] - means[1]) < .03
    flat = max(drift) < .02
    exchange_ok = all(.10 < v[2] < .95 for v in records)
    converged = agree and flat and exchange_ok
    x = torch.cat([v[0] for v in seed_rows]); s = torch.cat([v[1] for v in seed_rows])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "s": s, "L": L, "rho": RHO, "T": T, "composition_B": X_B, "converged": converged,
                "N": N, "seed_means_U_per_N": means, "tail_drift": drift, "exchange_acceptance": [v[2] for v in records],
                "gates": {"seed_agreement": agree, "tail_flat": flat, "exchange_in_band": exchange_ok},
                "min_snapshot_separation_sweeps": every}, out)
    print(f"[3d ref] agree={agree} flat={flat} exchange={exchange_ok} -> {'CONVERGED' if converged else 'NOT CONVERGED'} saved={out}", flush=True)
    return converged


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=Path("liquid_coupling_flow/artifacts/ka3d_pt_N512_T0.5.pt")); p.add_argument("--N", type=int, default=512); p.add_argument("--n-equil", type=int, default=5000); p.add_argument("--n-collect", type=int, default=2000); p.add_argument("--every", type=int, default=500); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); build_reference(a.out, a.N, a.device, a.n_equil, a.n_collect, a.every)


if __name__ == "__main__": main()
