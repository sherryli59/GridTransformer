"""Prep the 112 Bapst TRAIN-split configs as TRAINING data at the campaign box (L=15.2781, rho=1.1486 —
same common box as the reference dataset): per-file nudge to common L, short pmc+swap (100 sweeps, absorbs
the nudge + pmc-bias transient), exact polish (40 sweeps, removes the documented pmc plateau bias), save.

Chunked (8 chains/chunk) to coexist with the PT run on the GPU; INCREMENTAL SAVE per chunk. These 112
chains are for TRAINING ONLY — the 16 test-split chains behind the PTS reference dataset are held out
entirely (chain-level split, zero leakage into the measurement)."""
import glob
import pickle
import time
from pathlib import Path
import torch

from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_pmc_3d import parallel_mc_disp
from liquid_coupling_flow.ka_cavity_3d import local_displacement, local_identity_swap

DEV = "cuda"; T = 0.5; BETA = 1.0 / T; STEP = 0.08
L = None                                                     # set in main from the reference dataset
PMC = 100; POLISH = 40; CHUNK = 8
OUT = Path("liquid_coupling_flow/artifacts/ka3d_train_N4096_T0.5_rho1.15.pt")


def _swaps(x, s, U, mob, n):
    for _ in range(n):
        s, U, _ = local_identity_swap(x, s, U, mob, BETA, L)
    return s, U


def main():
    global L
    t0 = time.time()
    ref = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location="cpu",
                     weights_only=False)
    L = float(ref["L"])                                                  # EXACT common box of the reference
    files = sorted(glob.glob("datasets/bapst_ka3d/t050_train_*.pickle"))
    print(f"[prep] {len(files)} train-split files -> common L={L:.4f}", flush=True)
    xs_out, ss_out, es_out = [], [], []
    for c0 in range(0, len(files), CHUNK):
        xs, ss = [], []
        for f in files[c0:c0 + CHUNK]:
            dd = pickle.load(open(f, "rb"))
            xb = torch.tensor(dd["positions"], dtype=torch.float32)
            Lb = float(dd["box"][0])
            xs.append(torch.remainder(xb * (L / Lb), L))
            ss.append(torch.tensor(dd["types"], dtype=torch.long))
        x = torch.stack(xs).to(DEV); s = torch.stack(ss).to(DEV)
        B, N, _ = x.shape
        mob = torch.ones(B, N, dtype=torch.bool, device=DEV)
        for _ in range(PMC):
            x = parallel_mc_disp(x, s, L, BETA, STEP)
            U = ka_energy(x, s, L)
            s, U = _swaps(x, s, U, mob, max(1, N // 8))
        U = ka_energy(x, s, L)
        for _ in range(POLISH):
            for _ in range(N):
                x, U, _ = local_displacement(x, s, U, mob, BETA, L, STEP)
            s, U = _swaps(x, s, U, mob, max(1, N // 8))
        e = (ka_energy(x, s, L) / N).cpu()
        xs_out.append(x.cpu()); ss_out.append(s.cpu()); es_out.append(e)
        X = torch.cat(xs_out, 0); S = torch.cat(ss_out, 0); E = torch.cat(es_out, 0)
        torch.save({"x": X, "s": S, "L": L, "rho": N / L ** 3, "T": T, "N": N, "energy_per_N": E,
                    "energy_mean": float(E.mean()), "energy_std": float(E.std()),
                    "source": "Bapst train-split (112 chains), nudged to common L + pmc100 + polish40; "
                              "TRAINING ONLY (16 test-split reference chains held out entirely)"}, OUT)
        print(f"[prep] chunk {c0//CHUNK + 1}/{(len(files)+CHUNK-1)//CHUNK}: {X.shape[0]} configs "
              f"U/N={float(e.mean()):.4f}+-{float(e.std()):.4f} SAVED ({(time.time()-t0)/60:.0f} min)", flush=True)
    print(f"[prep] DONE {X.shape[0]} configs -> {OUT}  <U/N>={float(E.mean()):.4f}+-{float(E.std()):.4f} "
          f"(reference: {float(ref['energy_mean']):.4f})", flush=True)


if __name__ == "__main__":
    main()
