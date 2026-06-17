"""Fatten the N=256 KA reference: one long PT run with more chains + a long collection
phase -> ~2.5k equilibrated cold-level configs (training + validation data). Convergence
is already validated (ka_reference.py); here we just harvest many decorrelated samples."""
from __future__ import annotations
import os, numpy as np, torch
from liquid_coupling_flow.ka_reference import parallel_tempering
from liquid_coupling_flow.ka_mcmc import make_species
from liquid_coupling_flow.ka_energy import ka_energy

ART = os.path.join(os.path.dirname(__file__), "artifacts")
N, RHO, T, FRACB = 256, 1.2, 0.5, 0.35


def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    L = (N / RHO) ** 0.5
    sd = make_species(N, FRACB).to(device)
    T_ladder = 0.5 * (1.25 / 0.5) ** (torch.arange(10) / 9)
    cfg, traj, ex = parallel_tempering(N, L, sd, T_ladder, device, n_per=16,
                                       n_equil=8000, n_collect=4000, every=25,
                                       track_every=1000, seed=0)
    U = (ka_energy(cfg, sd, L) / N)
    drift = abs(np.mean([u for _, u in traj][-2:]) - np.mean([u for _, u in traj][-4:-2]))
    print(f"fattened reference: {cfg.shape[0]} configs, <U>/N {U.mean():.4f}+/-{U.std()/np.sqrt(len(U)):.4f}, "
          f"exch {ex:.2f}, plateau drift {drift:.4f}", flush=True)
    torch.save({"x": cfg.cpu(), "s": sd.cpu(), "N": N, "L": L, "T": T,
                "U_per_N": U.mean().item()}, os.path.join(ART, "ka_reference_N256.pt"))
    print(f"saved -> artifacts/ka_reference_N256.pt", flush=True)


if __name__ == "__main__":
    main()
