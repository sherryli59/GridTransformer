"""FINITE-SIZE SCALING of the cold equilibrium energy — decides under-convergence vs genuine 2D finite-size
physics (user challenge, spec addendum to 2026-07-07-pt-augmentation-and-scale-crossover-design.md).

Question: converged U/N vs N — if U/N(N) has a real slope toward -3.23 at N=256, the N=256 'under-convergence'
is partly finite-size physics and the cross-N equality gate is wrong; if flat within ~0.01, the 0.05 gap is
protocol shortfall (=> M=16 ladder rerun is worth it). Also measures SPECIES-REALIZATION scatter (2-3 seeded
realizations per N) — the uncontrolled variable in all past cross-N comparisons.

Protocol per (N, realization): plain PT (the measured-best sampler), M=10 ladder, n_equil=36000 + n_collect=4000
(the budget that reached -3.284 at N=100). CONVERGENCE PROOF per run, no extra cost: budget-halving flatness
(tracked cold U at half-budget vs final) + collection-window drift. Each unit SAVES IMMEDIATELY
(checkpoint-incrementally rule).

Run: python -m liquid_coupling_flow.ka_finite_size [Ns...]   (default 64 100 144)
"""
from __future__ import annotations
import os, sys, time, torch
import numpy as np
from liquid_coupling_flow.ka_reference import parallel_tempering
from liquid_coupling_flow.ka_mcmc import make_species
from liquid_coupling_flow.ka_energy import ka_energy

ART = os.path.join(os.path.dirname(__file__), "artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def fixed_count_species(N, frac_B=0.35, seed=0):
    """EXACT composition (nB = round(frac*N)), arrangement shuffled by seed. Bernoulli draws (make_species)
    let composition fluctuate ~±6% at N=64 (measured: seed 0 gave 42% B) — that would dominate realization
    scatter and inject composition-rounding artifacts into the cross-N trend. Fixed count isolates ARRANGEMENT."""
    nB = round(frac_B * N)
    g = torch.Generator().manual_seed(seed)
    s = torch.zeros(N, dtype=torch.int8)
    s[torch.randperm(N, generator=g)[:nB]] = 1
    return s


def run_unit(N, real, n_equil=36000, n_collect=4000, nB=None):
    L = (N / 1.2) ** 0.5
    nB_eff = nB if nB is not None else round(0.35 * N)
    done = os.path.join(ART, f"fss_N{N}_r{real}_nB{nB_eff}.pt")
    if os.path.exists(done):                                     # resume support: unit already banked
        out = torch.load(done, map_location="cpu", weights_only=False)
        print(f"FSS N={N} r={real}: SKIP (banked: U/N {out['U']:.4f})", flush=True)
        return out
    if nB is None:
        sd = fixed_count_species(N, 0.35, seed=real).to(DEV)
    else:                                                   # composition-sensitivity unit: exact nB, seed=real
        g = torch.Generator().manual_seed(real)
        sd = torch.zeros(N, dtype=torch.int8); sd[torch.randperm(N, generator=g)[:nB]] = 1; sd = sd.to(DEV)
    T_ladder = 0.5 * (1.25 / 0.5) ** (torch.arange(10) / 9)
    t0 = time.time()
    cfg, traj, ex, _ = parallel_tempering(N, L, sd, T_ladder, DEV, n_per=8, n_equil=n_equil,
                                          n_collect=n_collect, every=8, track_every=500, seed=0)
    wall = time.time() - t0
    U = (ka_energy(cfg, sd, L) / N)
    Um, se = U.mean().item(), U.std().item() / np.sqrt(cfg.shape[0])
    # convergence proofs: budget-halving flatness + collection drift
    sw = np.array([t for t, _ in traj]); u = np.array([x for x, in [(v,) for _, v in traj]])
    half_mask = (sw > 0.4 * n_equil) & (sw < 0.6 * n_equil)
    u_half = float(u[half_mask].mean()) if half_mask.any() else float("nan")
    coll = u[sw > n_equil]
    drift = float(abs(coll[len(coll)//2:].mean() - coll[:len(coll)//2].mean())) if len(coll) >= 4 else float("nan")
    flat_budget = abs(Um - u_half)
    out = {"N": N, "real": real, "U": Um, "se": se, "exch": ex, "wall": wall,
           "u_half_budget": u_half, "budget_flatness": flat_budget, "coll_drift": drift,
           "traj": traj, "n_B": int(sd.sum())}
    torch.save(out, os.path.join(ART, f"fss_N{N}_r{real}_nB{int(sd.sum())}.pt"))          # save-per-unit, immediately
    print(f"FSS N={N} r={real}: U/N {Um:.4f}+/-{se:.4f}  exch {ex:.2f}  budget-flat {flat_budget:.4f}  "
          f"coll-drift {drift:.4f}  nB {int(sd.sum())}  {wall:.0f}s", flush=True)
    return out


def main(Ns=(64, 100, 144), reals=(0,)):
    res = []
    for N in Ns:
        for r in reals:
            res.append(run_unit(N, r))
    # composition sensitivity at N=100: nB = 37 and 39 (35 comes from the trend family). HISTORICAL DISCOVERY:
    # seeded-Bernoulli make_species gave 39%B at N=100 and 37.1%B at N=256 — all past cross-N comparisons
    # carried a ~2-point composition difference. dU/dx_B from these units converts that into an energy correction.
    for nB in (39,):
        res.append(run_unit(100, 0, nB=nB))
    # trend: converged units only (budget-flat < 0.01)
    print("\n=== FINITE-SIZE TREND (converged units: budget-flat < 0.01) ===", flush=True)
    for N in Ns:
        us = [o["U"] for o in res if o["N"] == N and o["budget_flatness"] < 0.01]
        alln = [o["U"] for o in res if o["N"] == N]
        if us:
            print(f"N={N}: mean {np.mean(us):.4f}  realization-scatter {np.std(alln):.4f}  "
                  f"({len(us)}/{len(alln)} converged)", flush=True)
        else:
            print(f"N={N}: NO unit passed budget-flatness — protocol insufficient at this N", flush=True)
    torch.save(res, os.path.join(ART, "fss_all.pt"))
    print("VERDICT guide: slope of converged U/N vs 1/N pointing to ~-3.23 at N=256 => finite-size physics; "
          "flat within ~0.01 => the N=256 gap is protocol shortfall (M=16 rerun justified).", flush=True)


if __name__ == "__main__":
    main(Ns=tuple(int(a) for a in sys.argv[1:]) if len(sys.argv) > 1 else (64, 100, 144))
