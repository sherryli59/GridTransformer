"""RANDOM-RESTART RELAXATION R-SWEEP: measure where brute-force independent equilibration stops working,
and read G_PTS(R) off the survivors. Basin crossing by independent restarts -- no PT ladder, no FL weights,
no learned model. Motivated by relax_race verdict: 2/8 uniform-random walkers found the R=2.0 ref basin in
30k sweeps (4s numba); AR proposals gave no advantage.

Per (R, cavity): 2 REF walkers (control, define the equilibrium energy band) + NW random walkers, all relaxed
by identical plain local MC at T=0.5 (exact). Gate: walker passes if last-quarter mean U/n is within TOL of
the REF band. Outputs per R: p_success (the trapping crossover), tau-to-band distribution, G_PTS = <q_c>
over PASSING pairs + q_c(pass, ref) (ref = genuine equilibrium draw -> one-leg-unbiased estimator).
HONEST CAVEAT (recorded in the .pt): restart-relaxation weights basins by REACHABILITY, not Boltzmann
(quench bias). Below xi (single basin) irrelevant; at larger R uncontrolled -- compare vs BCY Fig 2a.
Saves FULL data: U traces, final configs, snapshots, per-pair q_c (record-simulation-data directive)."""
import sys, os, time, statistics as st
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from numba import njit, prange
from liquid_coupling_flow.ka_energy import SIGMA, EPS
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_cavity_ar import _mic
torch.set_grad_enabled(False)

RADII = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["2.0", "2.3", "2.6", "2.9", "3.2"])]
SW = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
NCAV = int(sys.argv[3]) if len(sys.argv) > 3 else 3
NW = 16; NREF = 2; T = 0.50; BETA = 1.0 / T; RCTX = 2.5; STEP = 0.3
UREC = 1000                      # U recorded every UREC sweeps
SNAP = 20_000                    # config snapshot every SNAP sweeps (for q_c(t) post-hoc)
TOL = 0.06                       # pass if last-quarter mean U/n <= REF band mean + TOL
B_OV = 0.2; RC_CORE = 0.5; P_MC = 1500; K_NN = 6
OUT = "reports/logs-2026-07-15/relax_sweep.pt"
sig_t = np.array(SIGMA, dtype=np.float64); eps_t = np.array(EPS, dtype=np.float64)
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location="cpu", weights_only=False)
Xds, Sds, L = D["x"].float(), D["s"].long(), float(D["L"])
print(f"RELAX SWEEP: R={RADII} SW={SW} NCAV={NCAV} NW={NW}+{NREF}ref T={T} TOL={TOL} | plain local MC (numba, parallel walkers)", flush=True)


@njit(cache=True, fastmath=True)
def row_e(allx, alls, i, x0, x1, x2, n_tot):
    e = 0.0
    for j in range(n_tot):
        if j == i:
            continue
        dx = allx[j, 0] - x0; dy = allx[j, 1] - x1; dz = allx[j, 2] - x2
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
        u += 0.5 * row_e(allx, alls, i, allx[i, 0], allx[i, 1], allx[i, 2], n_tot)
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


@njit(cache=True, fastmath=True, parallel=True)
def relax_all(X0, alls, n, n_tot, bndx, beta, R, step, nsw, urec, snap, seeds, Utr, snaps, Xfin):
    W = X0.shape[0]
    nu = nsw // urec
    nsn = nsw // snap
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
                if xn0 * xn0 + xn1 * xn1 + xn2 * xn2 >= R * R:
                    continue
                e0 = row_e(allx, alls, i, allx[i, 0], allx[i, 1], allx[i, 2], n_tot)
                e1 = row_e(allx, alls, i, xn0, xn1, xn2, n_tot)
                if np.random.random() < np.exp(-beta * (e1 - e0)):
                    allx[i, 0] = xn0; allx[i, 1] = xn1; allx[i, 2] = xn2
            if (sw + 1) % urec == 0:
                Utr[w, (sw + 1) // urec - 1] = mobile_U(allx, alls, n, n_tot) / n
            if (sw + 1) % snap == 0:
                k = (sw + 1) // snap - 1
                if k < nsn:
                    for i in range(n):
                        snaps[w, k, i, 0] = allx[i, 0]; snaps[w, k, i, 1] = allx[i, 1]; snaps[w, k, i, 2] = allx[i, 2]
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


results = {"meta": {"RADII": RADII, "SW": SW, "NW": NW, "TOL": TOL, "T": T,
                    "caveat": "restart-relaxation weights basins by reachability, not Boltzmann (quench bias); "
                              "below xi irrelevant, above uncontrolled -- compare vs BCY Fig2a"}}
gsel = torch.Generator().manual_seed(321)
gq = torch.Generator().manual_seed(7)
for R in RADII:
    r_pass = []; r_qc_pairs = []; r_qc_ref = []
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
        W = NREF + NW
        X0 = np.zeros((W, n, 3))
        for w in range(NREF):
            X0[w] = xin.numpy()
        grnd = torch.Generator().manual_seed(910 + cav + int(R * 100))
        for w in range(NW):
            u = torch.randn(n, 3, generator=grnd); u = u / u.norm(dim=-1, keepdim=True)
            rr = R * torch.rand(n, 1, generator=grnd) ** (1.0 / 3.0)
            X0[NREF + w] = (u * rr).numpy()
        alls = np.concatenate([sin.numpy(), sb.numpy()]).astype(np.int64)
        seeds = np.array([abs(hash((ci, R, w))) % 2**31 for w in range(W)], dtype=np.int64)
        nu = SW // UREC; nsn = SW // SNAP
        Utr = np.zeros((W, nu)); snaps = np.zeros((W, nsn, n, 3)); Xfin = np.zeros((W, n, 3))
        t0 = time.time()
        relax_all(X0, alls, n, n_tot, bnd.numpy().astype(np.float64), BETA, R, STEP, SW, UREC, SNAP, seeds, Utr, snaps, Xfin)
        wall = time.time() - t0
        # gate: REF band from last half of REF walkers
        ref_tail = Utr[:NREF, nu // 2:]
        Ueq = float(ref_tail.mean()); Usd = float(ref_tail.std())
        band = Ueq + max(TOL, 2 * Usd)
        tail = Utr[:, 3 * nu // 4:].mean(axis=1)
        passed = [w for w in range(NREF, W) if tail[w] <= band]
        # tau-to-band: first record where running U crosses band (walker-level)
        taus = []
        for w in range(NREF, W):
            idx = np.argmax(Utr[w] <= band) if (Utr[w] <= band).any() else -1
            taus.append(int(idx * UREC) if idx >= 0 else -1)
        # q_c between passing pairs + vs ref
        qc_pairs = []
        for a in range(len(passed)):
            for b in range(a + 1, len(passed)):
                qc_pairs.append(bcy_qc(Xfin[passed[a]], Xfin[passed[b]], gq))
        qc_ref = [bcy_qc(Xfin[w], xin.numpy(), gq) for w in passed]
        r_pass.append(len(passed) / NW); r_qc_pairs += qc_pairs; r_qc_ref += qc_ref
        results[(R, ci)] = {"n": n, "nb": nb, "Ueq": Ueq, "band": band, "Utr": Utr, "Xfin": Xfin,
                            "snaps": snaps[:, -1], "passed": passed, "taus": taus,
                            "qc_pairs": qc_pairs, "qc_ref": qc_ref, "wall": wall}
        torch.save(results, OUT)
        tp = [t for t in taus if t >= 0]
        print(f"  R={R} cav {ci} (n={n}): pass {len(passed)}/{NW} | Ueq {Ueq:+.3f} band {band:+.3f} | "
              f"tau-to-band med {int(st.median(tp)) if tp else -1} | q_c pairs "
              f"{st.mean(qc_pairs) if qc_pairs else float('nan'):+.3f} (n={len(qc_pairs)}) | "
              f"q_c vs ref {st.mean(qc_ref) if qc_ref else float('nan'):+.3f} ({wall:.0f}s)", flush=True)
    gp = st.mean(r_qc_pairs) if r_qc_pairs else float("nan")
    gr = st.mean(r_qc_ref) if r_qc_ref else float("nan")
    results[(R, "summary")] = {"p_success": st.mean(r_pass) if r_pass else 0.0, "G_PTS_pairs": gp, "G_PTS_ref": gr,
                               "n_pairs": len(r_qc_pairs)}
    torch.save(results, OUT)
    print(f"=== R={R}: p_success={st.mean(r_pass) if r_pass else 0:.2f} | G_PTS(pairs)={gp:+.3f} "
          f"G_PTS(vs ref)={gr:+.3f} (n_pairs {len(r_qc_pairs)}) ===", flush=True)
print(f"saved -> {OUT}", flush=True)
