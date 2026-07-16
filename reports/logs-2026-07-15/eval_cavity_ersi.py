"""EVAL the cavity-conditioned eRSI against MEASURED baselines. FM loss is not the number that matters.

GATES (each vs a baseline measured this session):
  (1) CLASH: raw-sample energy U/n. AR base gave +36..+1e11 (relax_race.out); data sits at ~-9.6.
      A flow from a smooth base should land near the data manifold, not in hard r^-12 overlap.
      RISK (recorded, egnn-flow-corrector-poc): this architecture was previously "under-trained --
      fixes coarse basin, not fine declash". 0.1-sigma residual overlaps still blow up r^-12.
  (2) p_success after IDENTICAL plain MC relaxation (numba, same gate as relax_sweep.py):
      beat random's 0.04 @R=2.0 / 0.00 @R=2.3. This is the ONLY thing that makes the model useful.
  (3) q_c vs ref of relaxed survivors: should be ~0.66 (BCY-consistent) -> survivors are in the RIGHT
      basin, not a look-alike.
Also reports EXACT log q (base uniform-in-sphere density + flow logdet) -> the ingredient for an
exact-MH basin-crossing move (no FL weights, no PT ladder).
Usage: eval_cavity_ersi.py [ckpt] [R] [NSAMP] [SW]"""
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

CKPT = sys.argv[1] if len(sys.argv) > 1 else "ka3d_cavity_ersi_rho12_best"
R = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
NSAMP = int(sys.argv[3]) if len(sys.argv) > 3 else 16
SW = int(sys.argv[4]) if len(sys.argv) > 4 else 200_000
NCAV = 3; T = 0.50; BETA = 1.0 / T; RCTX = 2.5; STEP = 0.3; UREC = 1000; TOL = 0.06
DUMMY0, DUMMY_D = 50.0, 5.0
B_OV = 0.2; RC_CORE = 0.5; P_MC = 1500; K_NN = 6
dev = "cuda"
sig_t = np.array(SIGMA, dtype=np.float64); eps_t = np.array(EPS, dtype=np.float64)
ck = torch.load(REPO / f"liquid_coupling_flow/artifacts/{CKPT}.pt", map_location=dev, weights_only=False)
A = ck["args"]; K_MAX = A["K_MAX"]; N_CAGE = A["N_CAGE"]
flow = CavityBlockFlow(n_cage=N_CAGE, k=K_MAX, r_c=2.5, hidden_nf=A["hidden"], n_layers=A["layers"],
                       n_species=2, max_neighbors=16).to(dev)
flow.load_state_dict(ck["state_dict"]); flow.eval()
print(f"eval cavity eRSI: {CKPT} (step {ck['step']}, FM {ck['fm']:.4f}) | R={R} NSAMP={NSAMP} SW={SW}", flush=True)
D = torch.load(REPO / "liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location="cpu", weights_only=False)
Xds, Sds, L = D["x"].float(), D["s"].long(), float(D["L"])
n_train = int(Xds.shape[0] * 0.9)


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


def dummies(m, device):
    d = torch.zeros(m, 3, device=device)
    d[:, 0] = DUMMY0 + DUMMY_D * torch.arange(m, device=device, dtype=torch.float32)
    return d


gen = torch.Generator().manual_seed(4242); gq = torch.Generator().manual_seed(7)
all_pass, all_qc, all_Uraw = [], [], []
for cav_i in range(NCAV):
    cav = None
    for _ in range(80):
        ci = int(torch.randint(n_train, Xds.shape[0], (1,), generator=gen))   # HELD-OUT configs
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
    idx = bx.norm(dim=-1).argsort()[:N_CAGE]
    cage_x = bx[idx]; sp_cage = bs[idx]; nc = cage_x.shape[0]
    if nc < N_CAGE:
        cage_x = torch.cat([cage_x, dummies(N_CAGE - nc, cage_x.device)])
        sp_cage = torch.cat([sp_cage, torch.zeros(N_CAGE - nc, dtype=torch.long)])
    # ---- sample: uniform-in-sphere base -> flow ----
    x0 = torch.zeros(NSAMP, K_MAX, 3); spb = torch.zeros(NSAMP, K_MAX, dtype=torch.long)
    dpad = dummies(K_MAX - n, torch.device("cpu")) if K_MAX > n else None
    for s in range(NSAMP):
        u = torch.randn(n, 3, generator=gen); u = u / u.norm(dim=-1, keepdim=True)
        rr = R * torch.rand(n, 1, generator=gen) ** (1.0 / 3.0)
        x0[s, :n] = u * rr; spb[s, :n] = sin
        if dpad is not None:
            x0[s, n:] = dpad
    t0 = time.time()
    x1, logdet = flow.flow(x0.to(dev), cage_x[None].expand(NSAMP, -1, -1).to(dev),
                           spb.to(dev), sp_cage[None].expand(NSAMP, -1).to(dev), R=R)
    gen_wall = time.time() - t0
    Xg = x1[:, :n].cpu().double().numpy()
    # exact log q: uniform-in-sphere base density + flow logdet
    logq_base = -n * math.log(4.0 / 3.0 * math.pi * R ** 3)
    logq = logq_base - logdet.cpu().numpy()
    alls = np.concatenate([sin.numpy(), bs.numpy()]).astype(np.int64)
    bnd_np = bx.numpy().astype(np.float64); n_tot = n + bnd_np.shape[0]
    Uraw = [mobile_U(np.concatenate([Xg[s], bnd_np]), alls, n, n_tot) / n for s in range(NSAMP)]
    Uref = mobile_U(np.concatenate([xin.numpy().astype(np.float64), bnd_np]), alls, n, n_tot) / n
    all_Uraw += Uraw
    print(f"=== cav {ci} (n={n}) | gen {gen_wall:.1f}s ===", flush=True)
    print(f"  GATE1 CLASH: raw U/n med {st.median(Uraw):+.3f} [min {min(Uraw):+.3f}, max {max(Uraw):+.3f}] "
          f"| data ref {Uref:+.3f} | AR baseline was +36..+1e11", flush=True)
    print(f"  exact log q: med {float(np.median(logq)):+.1f} (base {logq_base:+.1f}, logdet med "
          f"{float(np.median(logdet.cpu().numpy())):+.1f})", flush=True)
    # ---- relax (identical protocol to relax_sweep) ----
    W = 2 + NSAMP
    X0r = np.zeros((W, n, 3))
    X0r[0] = xin.numpy(); X0r[1] = xin.numpy()
    for s in range(NSAMP):
        X0r[2 + s] = Xg[s]
    seeds = np.array([abs(hash((ci, s))) % 2**31 for s in range(W)], dtype=np.int64)
    nu = SW // UREC
    Utr = np.zeros((W, nu)); Xfin = np.zeros((W, n, 3))
    t0 = time.time()
    relax_all(X0r, alls, n, n_tot, bnd_np, BETA, R, STEP, SW, UREC, seeds, Utr, Xfin)
    ref_tail = Utr[:2, nu // 2:]
    Ueq = float(ref_tail.mean()); band = Ueq + max(TOL, 2 * float(ref_tail.std()))
    tail = Utr[:, 3 * nu // 4:].mean(axis=1)
    passed = [w for w in range(2, W) if tail[w] <= band]
    qcs = [bcy_qc(Xfin[w], xin.numpy(), gq) for w in passed]
    all_pass.append(len(passed) / NSAMP); all_qc += qcs
    print(f"  GATE2 p_success: {len(passed)}/{NSAMP} = {len(passed)/NSAMP:.2f}  (random baseline 0.04 @R=2.0) "
          f"| Ueq {Ueq:+.3f} band {band:+.3f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  GATE3 q_c vs ref (survivors): {st.mean(qcs) if qcs else float('nan'):+.3f}  (BCY ~0.66)", flush=True)
print(f"\n=== SUMMARY R={R}: raw U/n med {st.median(all_Uraw):+.3f} | p_success {st.mean(all_pass) if all_pass else 0:.3f} "
      f"(random 0.04) | q_c {st.mean(all_qc) if all_qc else float('nan'):+.3f} (BCY 0.66) ===", flush=True)
