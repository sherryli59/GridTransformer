"""RELAX RACE: are AR cavity-conditioned proposals RELAXABLE to equilibrium much faster than random?
The missing measurement (never run in the 3D cavity PTS setting; closest priors point OPPOSITE ways:
2D SMC seed head-start ~270-510 sweeps POSITIVE vs 3D frozen-cage K=8 block AR-start stuck +8.6/p NEGATIVE).

Setup: R=2.0 cavity from the N512 T=0.5 rho=1.2 box (known truth: single basin, q_c(indep)~0.64,
cold E/n ~ -9.4). Three init types, relaxed by IDENTICAL plain local MC at T=0.5 (exact, no learned parts
after t=0), tracking U/n(t) and q_c(x(t), ref):
  REF    : the carved reference itself (control; starts equilibrated)
  AR     : full-cavity fillings from the EBM/AR generator, cavity-conditioned (GPU, then relax on CPU)
  RANDOM : uniform random positions in the cavity, same species counts (the honest 'no model' baseline)
Metrics: sweeps to reach ref-basin energy band; q_c(t) approach to ~0.64; AR vs RANDOM = the answer.
If AR relaxes fast -> independent equilibrations WITHOUT PT round-trips (basin crossing by generation).
numba MC (~5k sweeps/sec) so the whole race is minutes."""
import sys, os, time, statistics as st
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from numba import njit
from liquid_coupling_flow.ka_energy import SIGMA, EPS
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
torch.set_grad_enabled(False)

R = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
SW = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
NCAV = int(sys.argv[3]) if len(sys.argv) > 3 else 2
NPROP = 8; T = 0.50; BETA = 1.0 / T; RCTX = 2.5; STEP = 0.3; REC = 250
B_OV = 0.2; RC_CORE = 0.5; P_MC = 1500; K_NN = 6
OUT = f"reports/logs-2026-07-15/relax_race_R{R}.pt"
dev = "cuda"
sig_t = np.array(SIGMA, dtype=np.float64); eps_t = np.array(EPS, dtype=np.float64)

D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location="cpu", weights_only=False)
Xds, Sds, L = D["x"].float(), D["s"].long(), float(D["L"])
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False


@njit(cache=True, fastmath=True)
def row_e(allx, alls, i, xi, n_tot):
    e = 0.0
    for j in range(n_tot):
        if j == i:
            continue
        dx = allx[j, 0] - xi[0]; dy = allx[j, 1] - xi[1]; dz = allx[j, 2] - xi[2]
        r2 = dx * dx + dy * dy + dz * dz
        s = sig_t[alls[i], alls[j]]
        rc = 2.5 * s
        if r2 >= rc * rc:
            continue
        ep = eps_t[alls[i], alls[j]]
        sr6 = (s * s / r2) ** 3
        sc6 = (1.0 / 2.5) ** 6
        e += 4.0 * ep * ((sr6 * sr6 - sr6) - (sc6 * sc6 - sc6))
    return e


@njit(cache=True, fastmath=True)
def mobile_U(allx, alls, n, n_tot):
    u = 0.0
    for i in range(n):
        u += 0.5 * row_e(allx, alls, i, allx[i], n_tot)
    # mobile-boundary counted half above; add the other half of the mobile-boundary part
    for i in range(n):
        eb = 0.0
        for j in range(n, n_tot):
            dx = allx[j, 0] - allx[i, 0]; dy = allx[j, 1] - allx[i, 1]; dz = allx[j, 2] - allx[i, 2]
            r2 = dx * dx + dy * dy + dz * dz
            s = sig_t[alls[i], alls[j]]
            rc = 2.5 * s
            if r2 >= rc * rc:
                continue
            ep = eps_t[alls[i], alls[j]]
            sr6 = (s * s / r2) ** 3
            sc6 = (1.0 / 2.5) ** 6
            eb += 4.0 * ep * ((sr6 * sr6 - sr6) - (sc6 * sc6 - sc6))
        u += 0.5 * eb
    return u


@njit(cache=True, fastmath=True)
def relax(allx, alls, n, n_tot, beta, R, step, nsw, rec, traj_x, seed):
    np.random.seed(seed)
    k = 0
    for sw in range(nsw):
        for i in range(n):
            xo0, xo1, xo2 = allx[i, 0], allx[i, 1], allx[i, 2]
            l = step * np.random.random()
            v0 = np.random.randn(); v1 = np.random.randn(); v2 = np.random.randn()
            vn = (v0 * v0 + v1 * v1 + v2 * v2) ** 0.5
            xn0 = xo0 + l * v0 / vn; xn1 = xo1 + l * v1 / vn; xn2 = xo2 + l * v2 / vn
            if xn0 * xn0 + xn1 * xn1 + xn2 * xn2 >= R * R:
                continue
            xi_old = np.array([xo0, xo1, xo2])
            e0 = row_e(allx, alls, i, xi_old, n_tot)
            xi_new = np.array([xn0, xn1, xn2])
            e1 = row_e(allx, alls, i, xi_new, n_tot)
            if np.random.random() < np.exp(-beta * (e1 - e0)):
                allx[i, 0], allx[i, 1], allx[i, 2] = xn0, xn1, xn2
        if (sw + 1) % rec == 0:
            for i in range(n):
                traj_x[k, i, 0] = allx[i, 0]; traj_x[k, i, 1] = allx[i, 1]; traj_x[k, i, 2] = allx[i, 2]
            k += 1
    return k


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


results = {}
gsel = torch.Generator().manual_seed(320)
gq = torch.Generator().manual_seed(7)
print(f"RELAX RACE: R={R} T={T} SW={SW} | inits: REF x2, AR x{NPROP}, RANDOM x{NPROP} | plain local MC (numba)", flush=True)
for cav in range(NCAV):
    ci = int(torch.randperm(Xds.shape[0], generator=gsel)[0])
    c = torch.rand(3, generator=gsel) * L
    p = carve(Xds[ci], Sds[ci], c, R, L)
    if p["n_in"] < 14:
        continue
    xin = _mic(p["x_in"], c, L); sin = p["s_in"].long(); n = xin.shape[0]
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX)
    bnd, sb = xout[bm], p["s_out"][bm].long()
    nb = bnd.shape[0]; n_tot = n + nb
    # --- AR proposals (GPU) ---
    xo, so, _ = label_to_scaffold(xin.to(dev), sin.to(dev), R)
    allm = torch.ones(n, dtype=torch.bool, device=dev)
    gg = torch.Generator(device=dev).manual_seed(4000 + cav)
    Xa, Sa, _ = m.sample_block_b(xo[None].expand(NPROP, n, 3).clone().to(dev),
                                 so[None].expand(NPROP, n).clone().to(dev),
                                 allm, bnd.to(dev), sb.to(dev), R, gen=gg)
    Xa = Xa.cpu(); Sa = Sa.cpu()
    # --- inits ---
    inits = []
    inits.append(("REF", xin.numpy().astype(np.float64), sin.numpy()))
    for w in range(NPROP):
        inits.append((f"AR{w}", Xa[w].numpy().astype(np.float64), Sa[w].numpy()))
    grnd = torch.Generator().manual_seed(900 + cav)
    for w in range(NPROP):
        u = torch.randn(n, 3, generator=grnd); u = u / u.norm(dim=-1, keepdim=True)
        rr = R * torch.rand(n, 1, generator=grnd) ** (1.0 / 3.0)
        inits.append((f"RND{w}", (u * rr).numpy().astype(np.float64), sin.numpy()))
    nrec = SW // REC
    xin_np = xin.numpy()
    print(f"=== cav {ci} (n={n}, nb={nb}) ===", flush=True)
    for tag, x0, s0 in inits:
        allx = np.concatenate([x0, bnd.numpy().astype(np.float64)])
        alls = np.concatenate([s0, sb.numpy()]).astype(np.int64)
        U0 = mobile_U(allx, alls, n, n_tot) / n
        traj = np.zeros((nrec, n, 3))
        t0 = time.time()
        relax(allx, alls, n, n_tot, BETA, R, STEP, SW, REC, traj, abs(hash(tag + str(ci))) % 2**31)
        wall = time.time() - t0
        Us = [mobile_U(np.concatenate([traj[k], bnd.numpy()]), alls, n, n_tot) / n for k in range(nrec)]
        qcs = [bcy_qc(traj[k], xin_np, gq) for k in range(0, nrec, 4)]
        results[(ci, tag)] = {"U0": U0, "Us": Us, "qcs": qcs, "rec": REC, "wall": wall}
        torch.save(results, OUT)
        print(f"  {tag:>5}: U0/n {U0:+9.2f} -> U/n [{Us[0]:+7.3f} @{REC}, {Us[nrec//4]:+7.3f}, {Us[nrec//2]:+7.3f}, "
              f"{Us[-1]:+7.3f} @{SW}] | q_c(t) [{qcs[0]:.2f} -> {qcs[len(qcs)//2]:.2f} -> {qcs[-1]:.2f}] ({wall:.0f}s)", flush=True)
print(f"saved -> {OUT}", flush=True)
