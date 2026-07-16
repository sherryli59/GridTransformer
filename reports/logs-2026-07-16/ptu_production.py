# reports/logs-2026-07-16/ptu_production.py
"""PRODUCTION: certified equilibrium cavity sampling with the u-ladder (identity-bridge PT,
the winner arm per Task 5's matched-compute A/B gate). Runs dual-init (stack A = data init,
stack B = randomized init) replica ladders per held-out cavity to a q_c closure certificate,
then pools the post-closure window into a per-cavity q_c distribution. Reports G_PTS(R) (mean
core overlap across converged cavities) and chi_T(R) (BCY's susceptibility = mean per-cavity
q_c variance). Modeled closely on reports/logs-2026-07-16/ptu_ab_R20.py (data load, cavity
carve, Ladder construction, closure rule, incremental saves) -- u-mode only, CLI-selectable R,
adds an end_gap convergence certificate + pooled production-window q_c per cavity.
Usage: ptu_production.py --r 2.0 --sw 60000 --ncav 6 --rand_sw 4000"""
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
p.add_argument("--r", type=float, required=True, help="cavity radius")
p.add_argument("--sw", type=int, default=60000, help="sweeps per rung per cavity")
p.add_argument("--ncav", type=int, default=6, help="target number of CONVERGED cavities")
p.add_argument("--u_min", type=float, default=0.2)
p.add_argument("--t_top", type=float, default=0.7)
p.add_argument("--rand_sw", type=int, default=4000, help="stack-B randomization sweeps")
p.add_argument("--out", type=str, default=None)
p.add_argument("--max_attempt", type=int, default=None, help="stop after this many cavities attempted")
a = p.parse_args()
if a.out is None:
    a.out = f"reports/logs-2026-07-16/ptu_production_R{a.r}.pt"
if a.max_attempt is None:
    a.max_attempt = a.ncav + 4
from liquid_coupling_flow.ptu.kernels import seed_numba
seed_numba(1620)

T_BOT = 0.5
Q_TOL = 0.1
REC = 500
N_RUNGS = 12
US = list(np.linspace(1.0, a.u_min, N_RUNGS))
T_US = [t_of(u, 1.0, a.u_min, T_BOT, a.t_top) for u in US]

D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location="cpu", weights_only=False)
X, S, L = D["x"].float(), D["s"].long(), float(D["L"])
assert 2 * a.r + 2.5 < L, f"cavity R={a.r} + context shell too large for box L={L:.3f}"

gsel = torch.Generator().manual_seed(1620)
gq = torch.Generator().manual_seed(7)
results = {"args": vars(a)}
print(f"PRODUCTION u-arm R={a.r}: {N_RUNGS} rungs x {a.sw} sw | u in [{a.u_min},1.0], T_top {a.t_top} "
      f"| target {a.ncav} converged cavities, max_attempt {a.max_attempt}", flush=True)

converged_qc_means, converged_qc_vars = [], []
ncav = 0
attempt = 0
MAX_TRIES = max(500, a.max_attempt * 50)  # safety net against a pathological selection stall
for _try in range(MAX_TRIES):
    if ncav >= a.ncav or attempt >= a.max_attempt:
        break
    ci = int(torch.randint(921, 1024, (1,), generator=gsel))
    c = torch.rand(3, generator=gsel) * L
    pr = carve(X[ci], S[ci], c, a.r, L)
    if pr["n_in"] < 14:
        continue
    attempt += 1
    xin = _mic(pr["x_in"], c, L); xout = _mic(pr["x_out"], c, L)
    bm = xout.norm(dim=-1) < (a.r + 2.5)
    allx0 = np.concatenate([xin.double().numpy(), xout[bm].double().numpy()])
    alls0 = np.concatenate([pr["s_in"].numpy(), pr["s_out"][bm].numpy()]).astype(np.int64)
    n = xin.shape[0]; n_tot = allx0.shape[0]
    t0 = time.time()
    lad = Ladder(allx0, alls0, n, n_tot, "u", US, T_US,
                 nch=2, exch_mean=10, R=a.r, seed=100 + ci)
    lad.randomize_stack_B(a.rand_sw)
    res = lad.run(a.sw, REC, qc_fn=lambda x, y: bcy_qc(x, y, gq))
    qa, qb = np.array(res["qcA"]), np.array(res["qcB"])
    rec_sw = np.array(res["rec_sw"])
    # closure sweep: first record where running-mean gap < Q_TOL
    closure = -1
    for k in range(4, len(qa)):
        if abs(qa[k - 4:k].mean() - qb[k - 4:k].mean()) < Q_TOL:
            closure = res["rec_sw"][k]
            break
    end_gap = float(abs(qa[-4:].mean() - qb[-4:].mean()))
    converged = bool(end_gap < Q_TOL)
    thresh = max(closure, a.sw // 2 if closure < 0 else closure)
    mask = rec_sw > thresh
    qc_prod = np.concatenate([qa[mask], qb[mask]]).tolist()
    res.update({"closure": closure, "mode": "u", "n": n, "end_gap": end_gap,
                "converged": converged, "qc_prod": qc_prod})
    results[ci] = res
    torch.save(results, a.out)  # incremental save -- never end-only
    m = float(np.mean(qc_prod)) if len(qc_prod) else float("nan")
    tag = "CONV" if converged else "NOT-CONV"
    print(f"cav {ci} (n={n}): TRIPS={res['trips']:>4} closure={closure} end_gap={end_gap:.3f} "
          f"{tag} | qc_prod mean {m:+.3f} (k={len(qc_prod)} records) ({time.time()-t0:.0f}s)", flush=True)
    if converged:
        ncav += 1
        converged_qc_means.append(m)
        converged_qc_vars.append(float(np.var(qc_prod, ddof=1)) if len(qc_prod) > 1 else float("nan"))

if converged_qc_means:
    g_pts = float(np.mean(converged_qc_means))
    sem = (float(np.std(converged_qc_means, ddof=1) / np.sqrt(len(converged_qc_means)))
           if len(converged_qc_means) > 1 else float("nan"))
    chi_t = float(np.nanmean(converged_qc_vars))
else:
    g_pts = float("nan"); sem = float("nan"); chi_t = float("nan")
summary = {"G_PTS": g_pts, "G_PTS_sem": sem, "chi_T": chi_t,
           "n_converged": ncav, "n_attempted": attempt, "r": a.r}
results["summary"] = summary
torch.save(results, a.out)
print(f"G_PTS(R={a.r}) = {g_pts:+.3f} +- {sem:.3f} | chi_T = {chi_t:.4f} "
      f"| {ncav}/{attempt} cavities converged", flush=True)
print("PRODUCTION DONE", flush=True)
