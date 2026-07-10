"""Validate the vectorized parallel-Metropolis 3D sampler against the EXACT single-site sampler, in the one unit
that matters for a frozen exterior: equilibrium U/N. Two diagnostics:

  (A) DRIFT TEST (primary): equilibrate with pmc (fast), then run EXACT single-site from that config. If pmc's
      plateau is the true equilibrium, exact holds it flat; if pmc is biased, exact drifts off it. drift = the
      bias, measured against the tail-flat 0.02/particle tolerance.
  (B) MATCHED RELAXATION: pmc vs exact from the SAME lattice start, same sweep budget -> do the U/N curves track?

Also reports ms/sweep and the speedup. N=512, T=0.5, B=8 replicas, identity swaps in both."""
import time
import torch
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_pmc_3d import parallel_mc_disp
from liquid_coupling_flow.ka_cavity_3d import local_displacement, local_identity_swap, RHO, T
from liquid_coupling_flow.ka_reference_3d import _lattice_start, _species

dev = "cuda"; N = 512; L = (N / RHO) ** (1 / 3); B = 8; beta = 1.0 / T; STEP = 0.08
mob = torch.ones(B, N, dtype=torch.bool, device=dev)


def sweep_pmc(x, s):
    x = parallel_mc_disp(x, s, L, beta, STEP)
    U = ka_energy(x, s, L)
    for _ in range(max(1, N // 8)):
        s, U, _ = local_identity_swap(x, s, U, mob, beta, L)
    return x, s


def sweep_exact(x, s, U):
    for _ in range(N):
        x, U, _ = local_displacement(x, s, U, mob, beta, L, STEP)
    for _ in range(max(1, N // 8)):
        s, U, _ = local_identity_swap(x, s, U, mob, beta, L)
    return x, s, U


def run_pmc(x, s, nsweep, rec):
    t0 = time.time(); traj = []
    for it in range(nsweep + 1):
        if it % rec == 0:
            traj.append((it, float((ka_energy(x, s, L) / N).mean())))
        if it == nsweep:
            break
        x, s = sweep_pmc(x, s)
    return x, s, traj, (time.time() - t0) / nsweep


def run_exact(x, s, nsweep, rec):
    t0 = time.time(); U = ka_energy(x, s, L); traj = []
    for it in range(nsweep + 1):
        if it % rec == 0:
            traj.append((it, float((ka_energy(x, s, L) / N).mean())))
        if it == nsweep:
            break
        x, s, U = sweep_exact(x, s, U)
    return x, s, traj, (time.time() - t0) / nsweep


torch.manual_seed(0)
x0 = _lattice_start(N, L, B, dev, 0); s0 = _species(N, B, dev, 17)
print(f"[pmc-val] N={N} L={L:.2f} T={T} B={B} step={STEP}", flush=True)

# --- pmc equilibrate, then exact-polish drift test (A) ---
xp, sp, tp, ms_pmc = run_pmc(x0, s0, 3000, rec=500)
pmc_plateau = tp[-1][1]
xe, se, te, ms_ex = run_exact(xp, sp, 120, rec=20)
exact_after = te[-1][1]
drift = abs(exact_after - pmc_plateau)
print(f"[pmc-val A] pmc plateau U/N={pmc_plateau:.4f} ({ms_pmc*1000:.0f} ms/sw) "
      f"-> exact-polish U/N={exact_after:.4f} ({ms_ex*1000:.0f} ms/sw) | DRIFT={drift:.4f} "
      f"-> {'UNBIASED (<0.02)' if drift < 0.02 else 'BIASED'}  speedup={ms_ex/ms_pmc:.0f}x", flush=True)
print(f"[pmc-val A] pmc traj:   " + " ".join(f"{u:.3f}" for _, u in tp), flush=True)
print(f"[pmc-val A] exact-poli: " + " ".join(f"{u:.3f}" for _, u in te), flush=True)

# --- matched relaxation from identical lattice start (B) ---
_, _, tb_p, _ = run_pmc(x0, s0, 120, rec=20)
_, _, tb_e, _ = run_exact(x0, s0, 120, rec=20)
print(f"[pmc-val B] from-lattice pmc:   " + " ".join(f"{u:.3f}" for _, u in tb_p), flush=True)
print(f"[pmc-val B] from-lattice exact: " + " ".join(f"{u:.3f}" for _, u in tb_e), flush=True)
print("[pmc-val] DONE", flush=True)
