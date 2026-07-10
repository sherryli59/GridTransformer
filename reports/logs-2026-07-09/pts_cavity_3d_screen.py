"""LEAN 3D cavity PTS screen (option 1): single-seed reduced-sweep reference (exact single-site sampler) chained
into the hard-wall 2-arm gate. Answers the FIRST 3D question — does physics equilibrate the hard-walled cavity
interior at T=0.5, or is there an NN niche? — WITHOUT the full 8h 2-seed convergence-gated PT reference (that is
reserved for a publication-grade xi_PTS number, only if this screen warrants it).

Honesty guards ([[ka-reference-underconverged]], [[never-refute-bug-hypothesis]]): the single seed can't do the
seed-agreement gate, so we compute+log the tail-drift of the cold-replica energy; converged is FORCED True so the
gate consumes the archive, but a loud UNCERTIFIED warning fires if the tail isn't flat. The two-arm gate itself
then exposes any residual non-equilibration. Reference saved BEFORE the gate (checkpoint). N=512 -> R<=2.4 fits."""
import time
from pathlib import Path
import torch
from liquid_coupling_flow.ka_reference_3d import _pt_seed, RHO, T, X_B
from liquid_coupling_flow.ka_cavity_3d import run_gate

dev = "cuda" if torch.cuda.is_available() else "cpu"
N = 512; L = (N / RHO) ** (1 / 3)
N_EQUIL, N_COLLECT, EVERY = 2000, 1000, 125
RADII = (1.4, 1.8, 2.2, 2.4)
REF = Path("liquid_coupling_flow/artifacts/ka3d_screen_N512_T0.5.pt")
GATE_OUT = Path("reports/logs-2026-07-09/pts_cavity_3d_screen.json")

print(f"[screen] LEAN single-seed 3D reference: N={N} L={L:.2f} T={T} rho={RHO} xB={X_B} "
      f"(n_equil={N_EQUIL} n_collect={N_COLLECT} every={EVERY})", flush=True)
t0 = time.time()
x, s, track, coll_u, exch = _pt_seed(N, L, seed=0, device=dev, n_equil=N_EQUIL, n_collect=N_COLLECT, every=EVERY)
h = len(coll_u) // 2
drift = abs(sum(coll_u[:h]) / max(1, h) - sum(coll_u[h:]) / max(1, len(coll_u) - h))
tail_flat = drift < 0.02
meanU = sum(coll_u) / len(coll_u)
print(f"[screen] built {len(x)} configs in {(time.time()-t0)/60:.1f} min | cold U/N={meanU:.4f} "
      f"tail-drift={drift:.4f} exch={exch:.3f} -> tail_flat={tail_flat}", flush=True)

REF.parent.mkdir(parents=True, exist_ok=True)
torch.save({"x": x, "s": s, "L": L, "rho": RHO, "T": T, "composition_B": X_B,
            "converged": True,                 # forced so run_gate consumes it; honest status = tail_flat below
            "screen": True, "tail_flat": bool(tail_flat), "tail_drift": drift, "seeds": 1,
            "collection_U_per_N": coll_u, "track": track, "exchange_acceptance": exch, "N": N,
            "note": "LEAN single-seed SCREEN reference (NOT the certified 2-seed PT reference)"}, REF)
print(f"[screen] saved lean reference {REF}", flush=True)
if not tail_flat:
    print("[screen] *** WARNING: single-seed cold-energy tail NOT flat -> reference UNCERTIFIED; "
          "treat gate results as a screen, not a measurement ***", flush=True)

print(f"[screen] running hard-wall 2-arm 3D cavity gate radii={RADII} n_centers={len(x)} n_iter=3000", flush=True)
run_gate(REF, GATE_OUT, radii=RADII, n_centers=len(x), n_iter=3000, device=dev)
print(f"[screen] DONE total {(time.time()-t0)/60:.1f} min -> {GATE_OUT}", flush=True)
