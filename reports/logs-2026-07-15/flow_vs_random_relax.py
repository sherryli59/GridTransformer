"""FLOW-SAMPLE + SHORT RELAX vs RANDOM + SHORT RELAX -- the head-to-head.
Hypothesis (from g(r): flow learns coarse shells but leaves a soft core): a flow sample is structurally
~right, so a SHORT relaxation declashes it into the reference basin, while a random start must find the
basin from scratch (measured tau ~60k-114k sweeps, p_success 0.04 @R=2.0 / 0.00 @R=2.3).

Per cavity (held-out): 2 REF controls (define equilibrium band) + NSAMP FLOW (uniform->RK4) + NSAMP RANDOM,
ALL relaxed by identical plain T=0.5 MC (numba). Record U/n densely; report p_success and median tau-to-band
for FLOW vs RANDOM at several SHORT budgets. The win = flow reaches the band at a budget where random can't.
Usage: flow_vs_random_relax.py [ckpt] [R] [NSAMP] [SW] [NCAV]"""
import sys, math, time, statistics as st
from pathlib import Path
import numpy as np
import torch

REPO = Path("/mnt/ssd/GridTransformer")
sys.path.insert(0, str(REPO))
from numba import njit, prange
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka_energy import SIGMA, EPS
torch.set_grad_enabled(False)

CKPT = sys.argv[1] if len(sys.argv) > 1 else "ka3d_cavity_ersi_rho12"
R = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
NSAMP = int(sys.argv[3]) if len(sys.argv) > 3 else 16
SW = int(sys.argv[4]) if len(sys.argv) > 4 else 50_000
NCAV = int(sys.argv[5]) if len(sys.argv) > 5 else 4
RCTX = 2.5; T = 0.50; BETA = 1.0 / T; STEP = 0.3; UREC = 500; TOL = 0.06
RK4_STEPS = 30; DUMMY0, DUMMY_D = 50.0, 5.0
B_OV = 0.2; RC_CORE = 0.5; P_MC = 1500; K_NN = 6
BUDGETS = [b for b in [1000, 2000, 5000, 10000, 20000, 50000] if b <= SW]
dev = "cuda"
sig_t = np.array(SIGMA, dtype=np.float64); eps_t = np.array(EPS, dtype=np.float64)
ck = torch.load(REPO / f"liquid_coupling_flow/artifacts/{CKPT}.pt", map_location=dev, weights_only=False)
A = ck["args"]; K_MAX = A["K_MAX"]; N_CAGE = A["N_CAGE"]
flow = CavityBlockFlow(n_cage=N_CAGE, k=K_MAX, r_c=2.5, hidden_nf=A["hidden"], n_layers=A["layers"],
                       n_species=2, max_neighbors=16).to(dev)
flow.load_state_dict(ck["state_dict"]); flow.eval()
D = torch.load(REPO / "liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location="cpu", weights_only=False)
Xds, Sds, L = D["x"].float(), D["s"].long(), float(D["L"])
n_train = int(Xds.shape[0] * 0.9)
print(f"FLOW vs RANDOM relax: {CKPT} step {ck['step']} FM {ck['fm']:.4f} | R={R} NSAMP={NSAMP} SW={SW} "
      f"budgets {BUDGETS}", flush=True)


@njit(cache=True, fastmath=True)
def row_e(allx, alls, i, x0, x1, x2, n_tot):
    e = 0.0
    for j in range(n_tot):
        if j == i:
            continue
        dx = allx[j, 0] - x0; dy = allx[j, 1] - x1; dz = allx[j, 2] - x2
        r2 = dx * dx + dy * dy + dz * dz
        s = sig_t[alls[i], alls[j]]; rc = 2.5 * s
        if r2 >= rc * rc:
            continue
        ep = eps_t[alls[i], alls[j]]
        sr6 = (s * s / r2) ** 3; sc6 = (1.0 / 2.5) ** 6
        e += 4.0 * ep * ((sr6 * sr6 - sr6) - (sc6 * sc6 - sc6))
    return e


@njit(cache=True, fastmath=True)
def mobile_U(allx, alls, n, n_tot):
    u = 0.0
    for i in range(n):
        u += 0.5 * row_e(allx, alls, i, allx[i, 0], allx[i, 1], allx[i, 2], n_tot)
        eb = 0.0
        for j in range(n, n_tot):
            dx = allx[j, 0] - allx[i, 0]; dy = allx[j, 1] - allx[i, 1]; dz = allx[j, 2] - allx[i, 2]
            r2 = dx * dx + dy * dy + dz * dz
            s = sig_t[alls[i], alls[j]]; rc = 2.5 * s
            if r2 >= rc * rc:
                continue
            ep = eps_t[alls[i], alls[j]]
            sr6 = (s * s / r2) ** 3; sc6 = (1.0 / 2.5) ** 6
            eb += 4.0 * ep * ((sr6 * sr6 - sr6) - (sc6 * sc6 - sc6))
        u += 0.5 * eb
    return u


@njit(cache=True, fastmath=True, parallel=True)
def relax_all(X0, alls, n, n_tot, bndx, beta, Rc, step, nsw, urec, seeds, Utr, Xfin):
    W = X0.shape[0]
    for w in prange(W):
        np.random.seed(seeds[w])
        allx = np.empty((n_tot, 3))
        for i in range(n):
            allx[i] = X0[w, i]
        for j in range(n_tot - n):
            allx[n + j] = bndx[j]
        for sw in range(nsw):
            for i in range(n):
                l = step * np.random.random()
                v0 = np.random.randn(); v1 = np.random.randn(); v2 = np.random.randn()
                vn = (v0 * v0 + v1 * v1 + v2 * v2) ** 0.5
                xn0 = allx[i, 0] + l * v0 / vn; xn1 = allx[i, 1] + l * v1 / vn; xn2 = allx[i, 2] + l * v2 / vn
                if xn0 * xn0 + xn1 * xn1 + xn2 * xn2 >= Rc * Rc:
                    continue
                e0 = row_e(allx, alls, i, allx[i, 0], allx[i, 1], allx[i, 2], n_tot)
                e1 = row_e(allx, alls, i, xn0, xn1, xn2, n_tot)
                if np.random.random() < np.exp(-beta * (e1 - e0)):
                    allx[i, 0] = xn0; allx[i, 1] = xn1; allx[i, 2] = xn2
            if (sw + 1) % urec == 0:
                Utr[w, (sw + 1) // urec - 1] = mobile_U(allx, alls, n, n_tot) / n
        for i in range(n):
            Xfin[w, i, 0] = allx[i, 0]; Xfin[w, i, 1] = allx[i, 1]; Xfin[w, i, 2] = allx[i, 2]


def bcy_qc(X, Y, gen):
    X = torch.as_tensor(X, dtype=torch.float32); Y = torch.as_tensor(Y, dtype=torch.float32)
    def fc(vpos, vq):
        u = torch.randn(P_MC, 3, generator=gen); u = u / u.norm(dim=-1, keepdim=True)
        r = RC_CORE * torch.rand(P_MC, generator=gen) ** (1.0 / 3.0); pmc = u * r[:, None]
        d2 = ((pmc[:, None] - vpos[None]) ** 2).sum(-1); w = 1.0 / (d2 + 1e-6)
        kk = min(K_NN, vpos.shape[0]); tw, idx = w.topk(kk, dim=1)
        return float(((tw * vq[idx]).sum(1) / tw.sum(1)).mean())
    qX = torch.exp(-(torch.cdist(X, Y).min(1).values / B_OV) ** 2)
    qY = torch.exp(-(torch.cdist(Y, X).min(1).values / B_OV) ** 2)
    return 0.5 * (fc(X, qX) + fc(Y, qY))


def dummies(m):
    d = torch.zeros(m, 3); d[:, 0] = DUMMY0 + DUMMY_D * torch.arange(m, dtype=torch.float32); return d


def sample_rk4(x0_block, cage_x, sp_block, sp_cage, nsteps):
    k = flow.k; B = x0_block.shape[0]; cl = x0_block.clone(); cage = cage_x
    sp = torch.cat([sp_block, sp_cage], 1); dt = 1.0 / nsteps
    def vel(x, tv):
        tt = torch.full((B,), tv, device=x.device, dtype=x.dtype)
        v, _ = flow.ce.vel_div(torch.cat([x, cage], 1), tt, sp, k); return v[:, :k]
    for s in range(nsteps):
        t = s * dt
        k1 = vel(cl, t); k2 = vel(cl + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = vel(cl + 0.5 * dt * k2, t + 0.5 * dt); k4 = vel(cl + dt * k3, t + dt)
        cl = cl + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return cl


gen = torch.Generator().manual_seed(4242); gq = torch.Generator().manual_seed(7)
agg = {b: {"flow": [], "rand": []} for b in BUDGETS}
tau = {"flow": [], "rand": []}; qc_final = {"flow": [], "rand": []}; Uraw_flow = []
for cav_i in range(NCAV):
    cav = None
    for _ in range(80):
        ci = int(torch.randint(n_train, Xds.shape[0], (1,), generator=gen))
        c = torch.rand(3, generator=gen) * L
        pr = carve(Xds[ci], Sds[ci], c, R, L)
        if pr["n_in"] < 14 or pr["n_in"] > K_MAX:
            continue
        xin = _mic(pr["x_in"], c, L); sin = pr["s_in"].long()
        xout = _mic(pr["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX)
        bx, bs = xout[bm], pr["s_out"][bm].long()
        if bx.shape[0] < 8:
            continue
        cav = (ci, xin, sin, bx, bs); break
    if cav is None:
        continue
    ci, xin, sin, bx, bs = cav; n = xin.shape[0]
    idx = bx.norm(dim=-1).argsort()[:N_CAGE]; cage_x = bx[idx]; sp_cage = bs[idx]; nc = cage_x.shape[0]
    if nc < N_CAGE:
        cage_x = torch.cat([cage_x, dummies(N_CAGE - nc)]); sp_cage = torch.cat([sp_cage, torch.zeros(N_CAGE - nc, dtype=torch.long)])
    # ---- FLOW samples ----
    x0 = torch.zeros(NSAMP, K_MAX, 3); spb = torch.zeros(NSAMP, K_MAX, dtype=torch.long)
    dpad = dummies(K_MAX - n) if K_MAX > n else None
    for s in range(NSAMP):
        u = torch.randn(n, 3, generator=gen); u = u / u.norm(dim=-1, keepdim=True)
        rr = R * torch.rand(n, 1, generator=gen) ** (1.0 / 3.0)
        x0[s, :n] = u * rr; spb[s, :n] = sin
        if dpad is not None:
            x0[s, n:] = dpad
    Xflow = sample_rk4(x0.to(dev), cage_x[None].expand(NSAMP, -1, -1).to(dev),
                       spb.to(dev), sp_cage[None].expand(NSAMP, -1).to(dev), RK4_STEPS)[:, :n].cpu().double().numpy()
    # ---- RANDOM samples (same n, same cavity) ----
    Xrand = np.zeros((NSAMP, n, 3))
    for s in range(NSAMP):
        u = torch.randn(n, 3, generator=gen); u = u / u.norm(dim=-1, keepdim=True)
        rr = R * torch.rand(n, 1, generator=gen) ** (1.0 / 3.0)
        Xrand[s] = (u * rr).numpy()
    alls = np.concatenate([sin.numpy(), bs.numpy()]).astype(np.int64)
    bnd_np = bx.numpy().astype(np.float64); n_tot = n + bnd_np.shape[0]
    Uraw_flow += [mobile_U(np.concatenate([Xflow[s], bnd_np]), alls, n, n_tot) / n for s in range(NSAMP)]
    # ---- relax ALL: 2 REF + flow + random ----
    W = 2 + 2 * NSAMP
    X0r = np.zeros((W, n, 3)); X0r[0] = xin.numpy(); X0r[1] = xin.numpy()
    for s in range(NSAMP):
        X0r[2 + s] = Xflow[s]; X0r[2 + NSAMP + s] = Xrand[s]
    seeds = np.array([abs(hash((ci, w))) % 2**31 for w in range(W)], dtype=np.int64)
    nu = SW // UREC; Utr = np.zeros((W, nu)); Xfin = np.zeros((W, n, 3))
    t0 = time.time()
    relax_all(X0r, alls, n, n_tot, bnd_np, BETA, R, STEP, SW, UREC, seeds, Utr, Xfin)
    ref_tail = Utr[:2, nu // 2:]; Ueq = float(ref_tail.mean()); band = Ueq + max(TOL, 2 * float(ref_tail.std()))
    fl = list(range(2, 2 + NSAMP)); rd = list(range(2 + NSAMP, W))
    for b in BUDGETS:
        kk = b // UREC; lo = max(1, int(0.8 * kk))
        sm = Utr[:, lo:kk].mean(axis=1)
        agg[b]["flow"].append(np.mean([sm[w] <= band for w in fl]))
        agg[b]["rand"].append(np.mean([sm[w] <= band for w in rd]))
    for arm, ws in (("flow", fl), ("rand", rd)):
        for w in ws:
            hit = np.argmax(Utr[w] <= band) if (Utr[w] <= band).any() else -1
            tau[arm].append(int(hit * UREC) if hit >= 0 else -1)
        surv = [w for w in ws if Utr[w, 3 * nu // 4:].mean() <= band]
        qc_final[arm] += [bcy_qc(Xfin[w], xin.numpy(), gq) for w in surv]
    print(f"  cav {ci} (n={n}): band {band:+.3f} | flow raw U/n med "
          f"{st.median([mobile_U(np.concatenate([Xflow[s], bnd_np]), alls, n, n_tot)/n for s in range(NSAMP)]):+.1f} "
          f"| relax {time.time()-t0:.0f}s", flush=True)

print(f"\n=== p_success vs relaxation budget (R={R}, {NCAV} cavities x {NSAMP} samples) ===", flush=True)
print(f"{'budget':>8} | {'FLOW':>6} | {'RANDOM':>7}", flush=True)
for b in BUDGETS:
    print(f"{b:>8} | {np.mean(agg[b]['flow']):>6.2f} | {np.mean(agg[b]['rand']):>7.2f}", flush=True)
for arm in ("flow", "rand"):
    tp = [t for t in tau[arm] if t >= 0]
    print(f"  {arm}: reached band {len(tp)}/{len(tau[arm])} | median tau {int(st.median(tp)) if tp else -1} | "
          f"q_c vs ref (survivors) {st.mean(qc_final[arm]) if qc_final[arm] else float('nan'):+.3f}", flush=True)
print(f"  flow raw U/n (pre-relax) median {st.median(Uraw_flow):+.1f}", flush=True)
torch.save({"agg": agg, "tau": tau, "qc": qc_final, "budgets": BUDGETS, "R": R, "step": ck["step"]},
           REPO / "reports/logs-2026-07-15/flow_vs_random_relax.pt")
