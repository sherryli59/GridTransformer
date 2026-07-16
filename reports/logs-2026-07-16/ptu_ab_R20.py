# reports/logs-2026-07-16/ptu_ab_R20.py
"""A/B GATE: BCY (T,lam) shrinkage ladder vs (T,u) identity-bridge ladder, matched compute,
R=2.0, NCAV held-out cavities. Metrics per arm: TRIPS, Katzgraber flow linearity, per-rung
exchange acc, dual-init q_c gap trajectory + closure sweep, bottom-rung label-overlap decay
(the identity channel's direct signature). Matched compute = same (rungs x chains x sweeps)
budget; lam-arm uses the verbatim Table-IV R=2.0 ladder midpoint-densified to 21 rungs
(the measured-best lam configuration); u-arm uses N_RUNGS linear rungs over [U_MIN, 1.0].
Usage: ptu_ab_R20.py --u_min 0.35 --t_top 0.9 --sw 60000 --ncav 2"""
import sys, time, argparse
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.ptu.tables import t_of
from liquid_coupling_flow.ptu.ladder import Ladder
from liquid_coupling_flow.ptu.qc import bcy_qc
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_cavity_ar import _mic
torch.set_grad_enabled(False)

p = argparse.ArgumentParser()
p.add_argument("--u_min", type=float, required=True)
p.add_argument("--t_top", type=float, required=True)
p.add_argument("--sw", type=int, default=60000)
p.add_argument("--ncav", type=int, default=2)
p.add_argument("--rand_sw", type=int, default=4000)
a = p.parse_args()
from liquid_coupling_flow.ptu.kernels import seed_numba
seed_numba(1620)
R = 2.0; T_BOT = 0.5; Q_TOL = 0.1; REC = 500
_BASE = [1.0000, 0.9825, 0.9640, 0.9450, 0.9250, 0.9050, 0.8850, 0.8640, 0.8423, 0.8200, 0.7960]
LAM = []
for i, v in enumerate(_BASE):
    LAM.append(v)
    if i + 1 < len(_BASE):
        LAM.append(0.5 * (v + _BASE[i + 1]))
T_LAM = [T_BOT + (1.0 - T_BOT) * (1.0 - l) / (1.0 - 0.8) for l in LAM]
N_RUNGS = 12
US = list(np.linspace(1.0, a.u_min, N_RUNGS))
T_US = [t_of(u, 1.0, a.u_min, T_BOT, a.t_top) for u in US]
# matched compute: sweeps_u = sw * (21 * nch) / (N_RUNGS * nch)
SW_LAM = a.sw
SW_U = int(a.sw * len(LAM) / N_RUNGS)
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location="cpu", weights_only=False)
X, S, L = D["x"].float(), D["s"].long(), float(D["L"])
gsel = torch.Generator().manual_seed(1620)
gq = torch.Generator().manual_seed(7)
results = {"args": vars(a)}
print(f"A/B R={R}: lam-arm 21 rungs x {SW_LAM} sw | u-arm {N_RUNGS} rungs x {SW_U} sw "
      f"(matched compute) | u in [{a.u_min},1.0], T_top {a.t_top}", flush=True)
ncav = 0
for _ in range(60):
    ci = int(torch.randint(921, 1024, (1,), generator=gsel))
    c = torch.rand(3, generator=gsel) * L
    pr = carve(X[ci], S[ci], c, R, L)
    if pr["n_in"] < 14:
        continue
    xin = _mic(pr["x_in"], c, L); xout = _mic(pr["x_out"], c, L)
    bm = xout.norm(dim=-1) < (R + 2.5)
    allx0 = np.concatenate([xin.double().numpy(), xout[bm].double().numpy()])
    alls0 = np.concatenate([pr["s_in"].numpy(), pr["s_out"][bm].numpy()]).astype(np.int64)
    n = xin.shape[0]; n_tot = allx0.shape[0]
    print(f"=== cav {ci} (n={n}) ===", flush=True)
    for mode, coords, temps, sw_budget in (("lam", LAM, T_LAM, SW_LAM), ("u", US, T_US, SW_U)):
        t0 = time.time()
        lad = Ladder(allx0, alls0, n, n_tot, mode, coords, temps,
                     nch=2, exch_mean=10, R=R, seed=100 + ci)
        lad.randomize_stack_B(a.rand_sw)
        res = lad.run(sw_budget, REC, qc_fn=lambda x, y: bcy_qc(x, y, gq))
        # closure sweep: first record where running-mean gap < Q_TOL
        qa, qb = np.array(res["qcA"]), np.array(res["qcB"])
        closure = -1
        for k in range(4, len(qa)):
            if abs(qa[k - 4:k].mean() - qb[k - 4:k].mean()) < Q_TOL:
                closure = res["rec_sw"][k]
                break
        ea = res["exch_acc"] / np.maximum(res["exch_att"], 1)
        lin = np.linspace(1, 0, len(res["flow"]))
        res.update({"closure": closure, "mode": mode, "n": n})
        results[(ci, mode)] = res
        torch.save(results, "reports/logs-2026-07-16/ptu_ab_R20.pt")
        print(f"  {mode:>3}-arm: TRIPS={res['trips']:>4} | closure sweep {closure} "
              f"| final qcA {qa[-1]:+.3f} qcB {qb[-1]:+.3f} gap {abs(qa[-1]-qb[-1]):.3f} "
              f"| labB(end) {res['labB'][-1]:.2f} | exch min/med {ea.min():.2f}/{np.median(ea):.2f} "
              f"| flow-dev {np.abs(res['flow']-lin).max():.2f} ({time.time()-t0:.0f}s)", flush=True)
    ncav += 1
    if ncav >= a.ncav:
        break
print("A/B DONE", flush=True)
