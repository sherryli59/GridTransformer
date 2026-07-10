"""DECISIVE 3D cavity G-stick re-run on the well-mixed dataset (fixes both confounds of the n=8 run):
  (1) reference = last 96 configs of ka3d_dataset (snapshot 8, U/N -6.777, deepest/best-equilibrated, independent
      pmc-chains) instead of the 8-config under-mixed lean screen;
  (2) 96 centers instead of 8 -> per-arm SEM ~0.026 instead of ~0.09.
Reports mean +- SEM and gap/SEM (the honest significance test; the earlier bootstrap |diff| was positive-biased).
Saves per-center Q for real 96-point histograms. EXACT single-site interior dynamics (no pmc bias). Run ALONE."""
import time
from pathlib import Path
import numpy as np
import torch
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_cavity_3d import (local_displacement, local_identity_swap, occupancy,
                                               _deep_excluded, _melt, RHO, T)

dev = "cuda"; N = 512; beta = 1.0 / T; SHELL = 1.0; CELL = 0.3; STEP = 0.08
RADII = (1.6, 1.8, 2.0); N_ITER = 2000; NC = 96; SEED = 0
DATA = "liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt"
OUT = Path("liquid_coupling_flow/artifacts/pts/cavity3d_stick2_N512_T0.5.pt")

d = torch.load(DATA, map_location=dev, weights_only=False)
xall = d["x"][-NC:].to(dev); sall = d["s"][-NC:].to(dev).long(); L = float(d["L"])
q_rand = RHO * CELL ** 3
print(f"[stick2] N={N} L={L:.2f} T={T} centers={NC} (dataset snapshot-8, U/N~-6.777) Q_rand~{q_rand:.3f}", flush=True)


def q_percenter(x, ref_occ, excl):
    now = occupancy(x, L, CELL); keep = ~excl
    return (now & ref_occ & keep).sum(1).float() / (ref_occ & keep).sum(1).clamp_min(1).float()


def run_arm(x_init, s0, mobile, center, R, ref_occ, excl, n_iter=N_ITER, rec=None):
    rec = rec or max(1, n_iter // 10)
    nmove = max(1, int(mobile.sum(1).float().mean().round()))
    x = x_init.clone(); s = s0.clone(); U = ka_energy(x, s, L); qc = []
    for it in range(n_iter + 1):
        if it % rec == 0:
            qc.append(q_percenter(x, ref_occ, excl).cpu())
        if it == n_iter:
            break
        for _ in range(nmove):
            x, U, _ = local_displacement(x, s, U, mobile, beta, L, STEP, center, R)
        for _ in range(8):
            s, U, _ = local_identity_swap(x, s, U, mobile, beta, L)
    return torch.stack(qc[-3:]).mean(0)                                 # per-center plateau [NC]


def stats(p):
    p = p.numpy(); return float(p.mean()), float(p.std(ddof=1) / np.sqrt(len(p)))


torch.manual_seed(SEED)
ref_occ0 = occupancy(xall, L, CELL)
center = torch.rand(NC, 3, device=dev) * L
allres = {}; t0 = time.time()
for ri, R in enumerate(RADII):
    delta = xall - center[:, None]; delta = delta - L * torch.round(delta / L)
    mobile = delta.square().sum(-1) < R * R
    excl = _deep_excluded(center, R, SHELL, L, CELL)
    scrA, _ = _melt(xall, sall, mobile, center, R, L, 400, SEED + 100 * ri + 1)
    scrB, _ = _melt(xall, sall, mobile, center, R, L, 400, SEED + 100 * ri + 2)
    pr = run_arm(xall, sall, mobile, center, R, ref_occ0, excl)
    pa = run_arm(scrA, sall, mobile, center, R, ref_occ0, excl)
    pb = run_arm(scrB, sall, mobile, center, R, ref_occ0, excl)
    (mr, er), (ma, ea), (mb, eb) = stats(pr), stats(pa), stats(pb)
    gap_ab = abs(ma - mb); sem_ab = np.hypot(ea, eb); gap_ra = abs(mr - ma); sem_ra = np.hypot(er, ea)
    sig = max(gap_ab / sem_ab, gap_ra / sem_ra)
    verdict = "H2-NICHE (>3 SEM split)" if sig > 3 else ("H1-CONVERGED (<2 SEM)" if sig < 2 else "MARGINAL")
    nmob = float(mobile.sum(1).float().mean()); nd = (4 / 3) * np.pi * (R - SHELL) ** 3 * RHO
    print(f"[stick2 R={R:.1f}] mob~{nmob:.0f} ~{nd:.1f}p/center | ref {mr:.3f}+-{er:.3f} scrA {ma:.3f}+-{ea:.3f} "
          f"scrB {mb:.3f}+-{eb:.3f} | scrA-scrB {gap_ab:.3f} ({gap_ab/sem_ab:.1f}SEM) ref-scrA {gap_ra:.3f} "
          f"({gap_ra/sem_ra:.1f}SEM) | excess~{mr-q_rand:.3f} -> {verdict}  ({(time.time()-t0)/60:.1f}min)", flush=True)
    allres[R] = {"ref": pr, "scrA": pa, "scrB": pb, "means": (mr, ma, mb), "sems": (er, ea, eb), "n_mob": nmob}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"res": allres, "meta": {"N": N, "T": T, "L": L, "radii": RADII, "centers": NC, "n_iter": N_ITER,
               "reference": DATA + " [-96: snapshot8]", "q_rand": q_rand, "shell": SHELL}}, OUT)
print("[stick2] DONE", flush=True)
