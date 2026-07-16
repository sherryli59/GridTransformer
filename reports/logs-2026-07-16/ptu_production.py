# reports/logs-2026-07-16/ptu_production.py
"""PRODUCTION: certified equilibrium cavity sampling with the u-ladder (identity-bridge PT,
the winner arm per Task 5's matched-compute A/B gate). Runs dual-init (stack A = data init,
stack B = randomized init) replica ladders per held-out cavity to a q_c closure certificate,
then pools the post-closure window into a per-cavity q_c distribution. Reports G_PTS(R) (mean
core overlap across converged cavities) and chi_T(R) (BCY's susceptibility = mean per-cavity
q_c variance). Modeled closely on reports/logs-2026-07-16/ptu_ab_R20.py (data load, cavity
carve, Ladder construction, closure rule, incremental saves) -- u-mode only, CLI-selectable R.

BCY-paper-budget runs (--sw up to 1e7, ~28h/cavity) are executed in CHUNKS of --chunk sweeps:
Ladder state (replica configs, trips, flow, exchange counters) is instance state on `lad` and
persists across repeated lad.run() calls, so chunking is exact (not an approximation) -- only
each chunk's own qcA/qcB/labA/labB/rec_sw trace lists reset per call and must be stitched back
together with a running sweep offset. After EVERY chunk the full per-cavity record is saved to
--out (never end-only), so a 28h run is fully recoverable at any interruption.

--cav_offset lets independent parallel processes (same RNG seeds) work disjoint held-out
cavities: each process skips the first `cav_offset` cavities that pass the n_in>=14 filter,
so process k picks up where process k-1 would have started its (k+1)-th cavity.
Usage: ptu_production.py --r 2.0 --sw 10000000 --chunk 200000 --ncav 1 --max_attempt 1 --cav_offset 0"""
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
p.add_argument("--sw", type=int, default=60000, help="total sweeps per rung per cavity")
p.add_argument("--chunk", type=int, default=200000, help="sweeps per lad.run() call; checkpoints after each")
p.add_argument("--ncav", type=int, default=6, help="target number of CONVERGED cavities")
p.add_argument("--u_min", type=float, default=0.2)
p.add_argument("--t_top", type=float, default=0.7)
p.add_argument("--rand_sw", type=int, default=4000, help="stack-B randomization sweeps")
p.add_argument("--out", type=str, default=None)
p.add_argument("--max_attempt", type=int, default=None, help="stop after this many cavities attempted")
p.add_argument("--cav_offset", type=int, default=0, help="skip this many valid cavities from the selection stream (parallel disjointness)")
a = p.parse_args()
if a.out is None:
    a.out = f"reports/logs-2026-07-16/ptu_production_R{a.r}_off{a.cav_offset}.pt"
if a.max_attempt is None:
    a.max_attempt = a.ncav + 4
from liquid_coupling_flow.ptu.kernels import seed_numba
seed_numba(1620)

T_BOT = 0.5
Q_TOL = 0.1
REC = 1000
N_RUNGS = 12
US = list(np.linspace(1.0, a.u_min, N_RUNGS))
T_US = [t_of(u, 1.0, a.u_min, T_BOT, a.t_top) for u in US]

D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location="cpu", weights_only=False)
X, S, L = D["x"].float(), D["s"].long(), float(D["L"])
assert 2 * a.r + 2.5 < L, f"cavity R={a.r} + context shell too large for box L={L:.3f}"

gsel = torch.Generator().manual_seed(1620)
gq = torch.Generator().manual_seed(7)
results = {"args": vars(a)}
print(f"PRODUCTION u-arm R={a.r}: {N_RUNGS} rungs x {a.sw} sw (chunk {a.chunk}) | "
      f"u in [{a.u_min},1.0], T_top {a.t_top} | target {a.ncav} converged cavities, "
      f"max_attempt {a.max_attempt}, cav_offset {a.cav_offset}", flush=True)


def closure_and_gap(qa, qb, rec_sw):
    """First record where the 4-record running-mean gap < Q_TOL, and the final 4-record gap."""
    closure = -1
    for k in range(4, len(qa)):
        if abs(qa[k - 4:k].mean() - qb[k - 4:k].mean()) < Q_TOL:
            closure = int(rec_sw[k])
            break
    end_gap = float(abs(qa[-4:].mean() - qb[-4:].mean())) if len(qa) else float("nan")
    return closure, end_gap


converged_qc_means, converged_qc_vars = [], []
ncav = 0
attempt = 0
valid_seen = 0
MAX_TRIES = max(500, (a.max_attempt + a.cav_offset) * 50)  # safety net against a pathological selection stall
for _try in range(MAX_TRIES):
    if ncav >= a.ncav or attempt >= a.max_attempt:
        break
    ci = int(torch.randint(921, 1024, (1,), generator=gsel))
    c = torch.rand(3, generator=gsel) * L
    pr = carve(X[ci], S[ci], c, a.r, L)
    if pr["n_in"] < 14:
        continue
    if valid_seen < a.cav_offset:
        valid_seen += 1
        continue  # another process's cavity -- consume the RNG draw, skip the work
    valid_seen += 1
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

    qcA, qcB, labA, labB, rec_sw = [], [], [], [], []
    done = 0
    K = -(-a.sw // a.chunk)  # ceil
    last_res = None
    for k in range(1, K + 1):
        this_chunk = min(a.chunk, a.sw - done)
        res = lad.run(this_chunk, REC, qc_fn=lambda x, y: bcy_qc(x, y, gq))
        qcA.extend(res["qcA"]); qcB.extend(res["qcB"])
        labA.extend(res["labA"]); labB.extend(res["labB"])
        rec_sw.extend([done + s for s in res["rec_sw"]])
        done += this_chunk
        last_res = res
        qa, qb, rsw = np.array(qcA), np.array(qcB), np.array(rec_sw)
        closure, end_gap = closure_and_gap(qa, qb, rsw)
        converged = bool(end_gap < Q_TOL) if not np.isnan(end_gap) else False
        in_progress = done < a.sw
        thresh = max(closure, a.sw // 2 if closure < 0 else closure)
        mask = rsw > thresh
        qc_prod = np.concatenate([qa[mask], qb[mask]]).tolist() if mask.any() else []
        res_full = {"qcA": qcA, "qcB": qcB, "labA": labA, "labB": labB, "rec_sw": rec_sw,
                    "trips": last_res["trips"], "flow": last_res["flow"],
                    "exch_acc": last_res["exch_acc"], "exch_att": last_res["exch_att"],
                    "closure": closure, "mode": "u", "n": n, "end_gap": end_gap,
                    "converged": converged, "qc_prod": qc_prod, "in_progress": in_progress}
        results[ci] = res_full
        torch.save(results, a.out)  # incremental save after EVERY chunk -- never end-only
        print(f"cav {ci} chunk {k}/{K}: sw {done}/{a.sw} trips={last_res['trips']} "
              f"end_gap={end_gap:.3f}", flush=True)

    m = float(np.mean(qc_prod)) if len(qc_prod) else float("nan")
    tag = "CONV" if converged else "NOT-CONV"
    print(f"cav {ci} (n={n}): TRIPS={last_res['trips']:>4} closure={closure} end_gap={end_gap:.3f} "
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
           "n_converged": ncav, "n_attempted": attempt, "r": a.r, "cav_offset": a.cav_offset}
results["summary"] = summary
torch.save(results, a.out)
print(f"G_PTS(R={a.r}) = {g_pts:+.3f} +- {sem:.3f} | chi_T = {chi_t:.4f} "
      f"| {ncav}/{attempt} cavities converged", flush=True)
print("PRODUCTION DONE", flush=True)
