"""BIG-BOX (N=4096) 3D KA reference at rho=1.2, T=0.5 — WARM-STARTED from the public DeepMind/Bapst
equilibrated configs (Nature Physics 2020 bucket, public_dataset/temperature_050, LAMMPS-equilibrated
N=4096 80:20 KA at T=0.5 but P=1.55 => rho=1.150).

Why: the N=512 box caps cavities at R<=2.51 < xi_PTS~3.8 (all cavities PTS-pinned; GATE_VERDICT_fullcage.md).
N=4096 -> L=15.06 at rho=1.2 -> R up to ~6.3 = the full Berthier-2016 overlap-decay range.

Protocol: rescale positions by (L_ours/L_bapst) (1.4% compression, measured +0.096/particle above our
equilibrium under OUR Hamiltonian) -> pmc+swap re-equilibrate with a DRIFT-PLATEAU gate (the N=512 dataset
and the 2D reference both showed drifting tails; gate on the last-window slope, don't trust a fixed budget)
-> exact single-site polish (clears the pmc deep-plateau bias ~0.058/part, ka_dataset_3d docstring) ->
snapshots with pmc decorrelation + polish. INCREMENTAL SAVES: state + every snapshot saved the moment it
exists (CLAUDE.md checkpoint directive).

16 independent Bapst files = 16 independent chains; 3 snapshots/chain -> 48 reference configs."""
import glob
import pickle
import time
from pathlib import Path
import torch

from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_pmc_3d import parallel_mc_disp
from liquid_coupling_flow.ka_cavity_3d import local_displacement, local_identity_swap

DEV = "cuda"; RHO = 1.2; T = 0.5; BETA = 1.0 / T; STEP = 0.08
EQUIL_MAX = 6000; EQUIL_MIN = 1500; CHECK_EVERY = 250; SLOPE_TOL = 0.004   # dU/N per 1000 sweeps, plateau gate
POLISH = 40; DECORR = 300; PER_CHAIN = 3
OUT = Path("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5.pt")
STATE = Path("liquid_coupling_flow/artifacts/ka3d_bigbox_state.pt")        # resumable equilibration state


def _swaps(x, s, U, mob, beta, L, n):
    for _ in range(n):
        s, U, _ = local_identity_swap(x, s, U, mob, beta, L)
    return s, U


def _pmc_block(x, s, mob, L, sweeps):
    for _ in range(sweeps):
        x = parallel_mc_disp(x, s, L, BETA, STEP)
        U = ka_energy(x, s, L)
        s, U = _swaps(x, s, U, mob, BETA, L, max(1, x.shape[1] // 8))
    return x, s


def _polish(x, s, mob, L, sweeps):
    U = ka_energy(x, s, L); N = x.shape[1]
    for _ in range(sweeps):
        for _ in range(N):
            x, U, _ = local_displacement(x, s, U, mob, BETA, L, STEP)
        s, U = _swaps(x, s, U, mob, BETA, L, max(1, N // 8))
    return x, s


def main():
    t0 = time.time()
    files = sorted(glob.glob("datasets/bapst_ka3d/t050_test_*.pickle"))
    xs, ss = [], []
    for f in files:
        d = pickle.load(open(f, "rb"))
        xs.append(torch.tensor(d["positions"], dtype=torch.float32))
        ss.append(torch.tensor(d["types"], dtype=torch.long))
        L_b = float(d["box"][0])
    x = torch.stack(xs).to(DEV); s = torch.stack(ss).to(DEV)
    B, N, _ = x.shape
    L = (N / RHO) ** (1 / 3)
    x = torch.remainder(x * (L / L_b), L)                                   # compress 1.4% to rho=1.2
    mob = torch.ones(B, N, dtype=torch.bool, device=DEV)
    e = (ka_energy(x, s, L) / N)
    print(f"[bigbox] {B} Bapst chains N={N} L_bapst={L_b:.3f} -> L={L:.4f} rho={RHO} | "
          f"U/N after rescale {e.mean():.4f}+-{e.std():.4f} (N=512 equilibrium ~ -6.758)", flush=True)
    torch.manual_seed(0)

    hist = []
    done_sweeps = 0
    if STATE.exists():                                                      # resume support
        st = torch.load(STATE, map_location=DEV, weights_only=False)
        x, s, hist, done_sweeps = st["x"].to(DEV), st["s"].to(DEV), st["hist"], st["sweeps"]
        print(f"[bigbox] RESUMED at {done_sweeps} sweeps", flush=True)
    while done_sweeps < EQUIL_MAX:
        x, s = _pmc_block(x, s, mob, L, CHECK_EVERY)
        done_sweeps += CHECK_EVERY
        e = float((ka_energy(x, s, L) / N).mean()); hist.append((done_sweeps, e))
        torch.save({"x": x.cpu(), "s": s.cpu(), "hist": hist, "sweeps": done_sweeps, "L": L}, STATE)
        # plateau gate: slope of U/N over the trailing ~1000 sweeps, in units of dU/N per 1000 sweeps
        tail = [h for h in hist if h[0] > done_sweeps - 1000]
        slope = (tail[-1][1] - tail[0][1]) / max(1, (tail[-1][0] - tail[0][0])) * 1000 if len(tail) > 1 else 1e9
        print(f"[bigbox] equil {done_sweeps:5d} sweeps  U/N {e:.4f}  slope {slope:+.4f}/1k  ({(time.time()-t0)/60:.0f} min)", flush=True)
        if done_sweeps >= EQUIL_MIN and abs(slope) < SLOPE_TOL:
            print(f"[bigbox] PLATEAU at {done_sweeps} sweeps (|slope| < {SLOPE_TOL})", flush=True)
            break
    x, s = _polish(x, s, mob, L, POLISH)
    e = (ka_energy(x, s, L) / N)
    print(f"[bigbox] polished: U/N {e.mean():.4f}+-{e.std():.4f}  ({(time.time()-t0)/60:.0f} min)", flush=True)

    xs_out, ss_out, es_out = [], [], []
    for j in range(PER_CHAIN):
        x, s = _pmc_block(x, s, mob, L, DECORR)
        x, s = _polish(x, s, mob, L, POLISH)
        e = (ka_energy(x, s, L) / N).cpu()
        xs_out.append(x.cpu().clone()); ss_out.append(s.cpu().clone()); es_out.append(e)
        X = torch.cat(xs_out, 0); S = torch.cat(ss_out, 0); E = torch.cat(es_out, 0)
        torch.save({"x": X, "s": S, "L": L, "rho": RHO, "T": T, "N": N, "energy_per_N": E,
                    "energy_mean": float(E.mean()), "energy_std": float(E.std()),
                    "n_chains": B, "per_chain": j + 1, "equil_sweeps": done_sweeps, "hist": hist,
                    "source": "Bapst/DeepMind public_dataset/temperature_050 warm start, rescaled 1.150->1.2, "
                              "pmc+swap re-equil (drift-plateau gated) + exact polish"}, OUT)
        print(f"[bigbox] snapshot {j+1}/{PER_CHAIN}: {B} configs U/N={float(e.mean()):.4f}+-{float(e.std()):.4f} "
              f"SAVED ({(time.time()-t0)/60:.0f} min)", flush=True)
    print(f"[bigbox] DONE {X.shape[0]} configs -> {OUT}  <U/N>={float(E.mean()):.4f}+-{float(E.std()):.4f}", flush=True)


if __name__ == "__main__":
    main()
