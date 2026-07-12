"""BIG-BOX N=4096 3D KA reference at the NATIVE Bapst condition (T=0.5, P=1.55 -> rho~1.15) — user decision:
adopt their state point instead of compressing to rho=1.2. Their configs are ALREADY equilibrated there, so
this is a VALIDATION + POLISH pass, not a re-equilibration:

1. Each NPT config has its own box (spread ~0.3%); rescale each to the COMMON MEAN density (per-file nudge
   <=0.3%, negligible perturbation) so the dataset has one L like all our pipelines expect.
2. EQUILIBRIUM/HAMILTONIAN-MATCH CHECK: run our pmc+swap MC at T=0.5 and watch U/N from sweep 0. If their
   LAMMPS Hamiltonian == our ka_energy, the trace is FLAT from the start (equilibrium in -> equilibrium out).
   Any initial relaxation quantifies the Hamiltonian mismatch our polish absorbs.
3. Exact single-site polish (clears pmc plateau bias) -> snapshot; 2 more snapshots with pmc decorrelation.
Saves 48 reference configs. Incremental saves per snapshot. ~1h wall."""
import glob
import pickle
import time
from pathlib import Path
import torch

from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_pmc_3d import parallel_mc_disp
from liquid_coupling_flow.ka_cavity_3d import local_displacement, local_identity_swap

DEV = "cuda"; T = 0.5; BETA = 1.0 / T; STEP = 0.08
CHECK_SWEEPS = 400; CHECK_EVERY = 50; POLISH = 40; DECORR = 300; PER_CHAIN = 3
OUT = Path("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt")


def _swaps(x, s, U, mob, L, n):
    for _ in range(n):
        s, U, _ = local_identity_swap(x, s, U, mob, BETA, L)
    return s, U


def _pmc_block(x, s, mob, L, sweeps):
    for _ in range(sweeps):
        x = parallel_mc_disp(x, s, L, BETA, STEP)
        U = ka_energy(x, s, L)
        s, U = _swaps(x, s, U, mob, L, max(1, x.shape[1] // 8))
    return x, s


def _polish(x, s, mob, L, sweeps):
    U = ka_energy(x, s, L); N = x.shape[1]
    for _ in range(sweeps):
        for _ in range(N):
            x, U, _ = local_displacement(x, s, U, mob, BETA, L, STEP)
        s, U = _swaps(x, s, U, mob, L, max(1, N // 8))
    return x, s


def main():
    t0 = time.time()
    files = sorted(glob.glob("datasets/bapst_ka3d/t050_test_*.pickle"))
    xs, ss, Ls = [], [], []
    for f in files:
        d = pickle.load(open(f, "rb"))
        xs.append(torch.tensor(d["positions"], dtype=torch.float32))
        ss.append(torch.tensor(d["types"], dtype=torch.long))
        Ls.append(float(d["box"][0]))
    x = torch.stack(xs).to(DEV); s = torch.stack(ss).to(DEV)
    B, N, _ = x.shape
    Ls_t = torch.tensor(Ls)
    rho_each = N / Ls_t ** 3
    L = float((N / rho_each.mean()) ** (1 / 3))                             # common box = mean density
    RHO = N / L ** 3
    print(f"[native] {B} chains N={N}  box spread {Ls_t.min():.4f}..{Ls_t.max():.4f} "
          f"(rho {rho_each.min():.4f}..{rho_each.max():.4f}) -> common L={L:.4f} rho={RHO:.4f}", flush=True)
    for b in range(B):                                                      # per-file nudge to common L
        x[b] = torch.remainder(x[b] * (L / Ls[b]), L)
    mob = torch.ones(B, N, dtype=torch.bool, device=DEV)
    e = (ka_energy(x, s, L) / N)
    print(f"[native] U/N after common-box nudge: {e.mean():.4f}+-{e.std():.4f}", flush=True)

    # equilibrium / Hamiltonian-match check: flat-from-sweep-0 trace
    hist = [(0, float(e.mean()))]
    for k in range(CHECK_SWEEPS // CHECK_EVERY):
        x, s = _pmc_block(x, s, mob, L, CHECK_EVERY)
        e = float((ka_energy(x, s, L) / N).mean()); hist.append(((k + 1) * CHECK_EVERY, e))
        print(f"[native] check {hist[-1][0]:4d} sweeps  U/N {e:.4f}  ({(time.time()-t0)/60:.0f} min)", flush=True)
    drift = hist[-1][1] - hist[0][1]
    print(f"[native] EQUILIBRIUM CHECK: drift over {CHECK_SWEEPS} sweeps = {drift:+.4f}/particle "
          f"({'FLAT -> their equilibrium == ours' if abs(drift) < 0.01 else 'RELAXED -> Hamiltonian mismatch absorbed'})", flush=True)

    xs_out, ss_out, es_out = [], [], []
    for j in range(PER_CHAIN):
        if j > 0:
            x, s = _pmc_block(x, s, mob, L, DECORR)
        x, s = _polish(x, s, mob, L, POLISH)
        e = (ka_energy(x, s, L) / N).cpu()
        xs_out.append(x.cpu().clone()); ss_out.append(s.cpu().clone()); es_out.append(e)
        X = torch.cat(xs_out, 0); S = torch.cat(ss_out, 0); E = torch.cat(es_out, 0)
        torch.save({"x": X, "s": S, "L": L, "rho": RHO, "T": T, "N": N, "energy_per_N": E,
                    "energy_mean": float(E.mean()), "energy_std": float(E.std()),
                    "n_chains": B, "per_chain": j + 1, "check_hist": hist, "check_drift": drift,
                    "source": "Bapst/DeepMind public_dataset/temperature_050 NATIVE condition (P=1.55), "
                              "per-file nudge to common mean rho, our-Hamiltonian check + exact polish"}, OUT)
        print(f"[native] snapshot {j+1}/{PER_CHAIN}: {B} configs U/N={float(e.mean()):.4f}+-{float(e.std()):.4f} "
              f"SAVED ({(time.time()-t0)/60:.0f} min)", flush=True)
    print(f"[native] DONE {X.shape[0]} configs -> {OUT}  <U/N>={float(E.mean()):.4f}+-{float(E.std()):.4f}  "
          f"L={L:.4f} rho={RHO:.4f}", flush=True)


if __name__ == "__main__":
    main()
