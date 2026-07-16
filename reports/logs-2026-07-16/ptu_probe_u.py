"""PROBE: identity-swap acceptance and displacement fluidity vs u — fixes U_MIN, T_TOP, N_RUNGS.
For u in a grid, equilibrate one R=2.0 cavity at (u, T(u)) with displacement sweeps, then measure
identity-swap acceptance. Also check top-rung fluidity (disp-acc in [0.15, 0.6]; U/N stationary --
guards against monodisperse freezing/crystallization at low u). Full curves saved."""
import sys, time
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.ptu.tables import build_tables_u, t_of
from liquid_coupling_flow.ptu.kernels import disp_sweep, mobile_U, seed_numba
from liquid_coupling_flow.ptu.identity import identity_sweep
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_cavity_ar import _mic
torch.set_grad_enabled(False)

R = 2.0; T_BOT = 0.5
T_TOP_GRID = [0.7, 0.9]
U_GRID = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.35, 0.2, 0.0]
EQ_SW = 3000; MEAS_SW = 1000
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location="cpu", weights_only=False)
X, S, L = D["x"].float(), D["s"].long(), float(D["L"])
g = torch.Generator().manual_seed(3)
ci = 950
c = torch.rand(3, generator=g) * L
p = carve(X[ci], S[ci], c, R, L)
xin = _mic(p["x_in"], c, L); xout = _mic(p["x_out"], c, L)
bm = xout.norm(dim=-1) < (R + 2.5)
allx0 = np.concatenate([xin.double().numpy(), xout[bm].double().numpy()])
alls0 = np.concatenate([p["s_in"].numpy(), p["s_out"][bm].numpy()]).astype(np.int64)
n = xin.shape[0]; n_tot = allx0.shape[0]
print(f"probe: cav {ci} n={n} n_tot={n_tot} | u grid {U_GRID} x T_top {T_TOP_GRID}", flush=True)
out = {}
for T_TOP in T_TOP_GRID:
    for u in U_GRID:
        T = t_of(u, 1.0, 0.0, T_BOT, T_TOP)
        beta = 1.0 / T
        tabs = build_tables_u(u)
        allx = allx0.copy(); alls = alls0.copy()
        seed_numba(7)
        t0 = time.time()
        dacc = 0
        us = []
        for sw in range(EQ_SW):
            dacc += disp_sweep(allx, alls, n, n_tot, beta, R, 0.3, *tabs)
            if sw % 200 == 0:
                us.append(mobile_U(allx, alls, n, n_tot, *tabs) / n)
        iacc = iatt = 0
        for sw in range(MEAS_SW):
            disp_sweep(allx, alls, n, n_tot, beta, R, 0.3, *tabs)
            a, t = identity_sweep(allx, alls, n, n_tot, beta, 10, *tabs)
            iacc += a; iatt += t
        drift = abs(us[-1] - us[len(us) // 2])
        out[(T_TOP, u)] = {"id_acc": iacc / max(iatt, 1), "disp_acc": dacc / (EQ_SW * n),
                           "U_trace": us, "drift_half": drift}
        torch.save(out, "reports/logs-2026-07-16/ptu_probe_u.pt")
        print(f"  T_top={T_TOP} u={u:.2f} T={T:.3f}: identity-acc {iacc/max(iatt,1):.3f} "
              f"disp-acc {dacc/(EQ_SW*n):.2f} U/n {us[-1]:+.3f} drift {drift:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
print("PROBE DONE", flush=True)
