"""D0: supercooled regime probe for the mW kernel campaign (spec 2026-07-14).
T* ladder, fixed-density N=64. Per-T* result saved THE MOMENT it finishes."""
import os, time, torch
from liquid_coupling_flow.mw.mw_energy import RHO_STAR, count_energy_evals
from liquid_coupling_flow.mw.mw_reference import mc_run, g_r

N = 64
L = (N / RHO_STAR) ** (1.0 / 3.0)
ART = "liquid_coupling_flow/mw/artifacts"
LADDER = [0.0963, 0.085, 0.075, 0.065, 0.055]

for tstar in LADDER:
    beta = 1.0 / tstar
    tag = f"{tstar:.4f}"
    ck = f"{ART}/d0_probe_T{tag}_N64_ckpt.pt"
    t0 = time.time()
    with count_energy_evals() as counter:
        res = mc_run(N, L, beta, n_equil=20_000, n_collect=20_000, every=100,
                     seed=0, B=8, track_every=100, ckpt_path=ck)
    r, g = g_r(res["cfgs"].cuda() if torch.cuda.is_available() else res["cfgs"], L)
    res.update({"tstar": tstar, "L": L, "gr_r": r.cpu(), "gr": g.cpu(),
                "evals": counter.as_dict(), "wall_s": time.time() - t0})
    torch.save(res, f"{ART}/d0_probe_T{tag}_N64.pt")
    print(f"T*={tag}: U/N final={float(res['U'].mean()/N):+.4f} acc={res['acc']:.3f} "
          f"coll_drift={res['coll_drift']:.4f} wall={res['wall_s']:.0f}s", flush=True)
print("D0 ladder DONE", flush=True)
