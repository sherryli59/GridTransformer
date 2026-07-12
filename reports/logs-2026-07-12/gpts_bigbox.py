"""G_PTS(R) on the N=4096 big box (T=0.5, rho=1.1486, L=15.278) — first measurement through the radius
range the N=512 box could never reach (R up to 6.4; xi_PTS expected ~3-4ish scale).

Sampler: ORACLE per-site heat-bath (true-KA-energy candidate softmax; validated in
reports/logs-2026-07-11/test_fullcage_oracle_gate.py — holds equilibrium, equilibrates from any seed at
pinned radii). Positions-only, composition fixed (paper protocol: particles can't leave the cavity; KA
species swaps are hopeless anyway). Per cavity: M=8 chains = 4 DATA-START + 4 ALIEN-START (interior carved
from a DIFFERENT equilibrated reference config at the same center = liquid-like packing, wrong basin for
this boundary; uniform-random starts REFUTED: they land in clash-free but collectively-strained states that
single-site moves cannot anneal — measured, see gpts run 1-2 logs). Alien chains are annealed by a geometric
BETA LADDER (0.8 -> 2.0) over the burn phase, then measured at beta=2 (the paper needed PT for the same
reason; the ladder is the measurement tool, not a learned-sampler claim).

Estimator (Berthier-Charbonneau-Yaida 2016): G_PTS = <q_c> over pairs of INDEPENDENT equilibrium samples;
q_c = core overlap (|r|<0.5 window at cavity center, same-species NN, Gaussian b=0.2). We use CROSS pairs
(data-start x uniform-start, 16/cavity) as the estimator; WITHIN-group pairs (dd, uu) are convergence
diagnostics: healthy equilibration => dd ~ uu ~ cross. U/N(uniform) - U/N(data) at end = equilibration gate;
non-converged (R,cav) are FLAGGED, not silently pooled.

Full data saved per (R, cavity): final configs, species, boundary, qc matrices, energy traces (user
directive: never just summary scalars). Incremental save after every (R, cavity)."""
import time
from pathlib import Path
import torch

from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA, EPS, RCUT_FACTOR

DEV = "cuda"; T = 0.5; BETA = 1.0 / T; RCTX = 2.5; BIGL = 100.0
M_DATA = 4; M_ALIEN = 4; M = M_DATA + M_ALIEN
NCAND = 192; TEL_FRAC = 0.5; SIGLOC = 0.2; SNAP_EVERY = 6
RGRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
SWEEPS = {1.5: 40, 2.0: 40, 2.5: 40, 3.0: 60, 3.5: 60, 4.0: 80}
BURN_FRAC = 0.6                      # fraction of sweeps on the beta ladder (0.8 -> BETA geometric);
                                     # rest at BETA for measurement. Gates stay the convergence arbiter.
BETA_HOT = 0.8
NCAV = 6
OUT = Path("liquid_coupling_flow/artifacts/gpts_bigbox_N4096.pt")
T_SIG = torch.tensor(SIGMA, device=DEV); T_EPS = torch.tensor(EPS, device=DEV)


def row_E(cand, s_i, ox, os_):
    """Shifted/cutoff LJ energies of candidate positions vs all others (ka_energy convention, no MIC —
    center-relative frame, extent << BIGL). cand [M,C,3], s_i [M], ox [M,P,3], os_ [M,P] -> [M,C]."""
    sig = T_SIG[s_i[:, None], os_]; eps = T_EPS[s_i[:, None], os_]
    rc2 = (RCUT_FACTOR * sig) ** 2
    r2 = (cand[:, :, None, :] - ox[:, None, :, :]).square().sum(-1)
    inv6 = (sig[:, None] ** 2 / r2) ** 3
    e = 4 * eps[:, None] * (inv6 ** 2 - inv6)
    src6 = (1.0 / RCUT_FACTOR) ** 6
    e = torch.where(r2 < rc2[:, None], e - 4 * eps[:, None] * (src6 ** 2 - src6), torch.zeros_like(e))
    return e.sum(-1)


def full_E(x_int, s_int, bnd, sb):
    Mn = x_int.shape[0]
    allx = torch.cat([x_int, bnd[None].expand(Mn, bnd.shape[0], 3)], 1)
    alls = torch.cat([s_int, sb[None].expand(Mn, sb.shape[0])], 1)
    return ka_energy(allx, alls.long(), BIGL)


def heatbath_sweep(x_int, s_int, bnd, sb, Rr, gen, beta=BETA):
    Mn, Nn = x_int.shape[0], x_int.shape[1]
    n_tel = int(TEL_FRAC * NCAND)
    for i in torch.randperm(Nn, generator=gen, device=DEV).tolist():
        oth = [j for j in range(Nn) if j != i]
        ox = torch.cat([x_int[:, oth], bnd[None].expand(Mn, bnd.shape[0], 3)], 1)
        os_ = torch.cat([s_int[:, oth], sb[None].expand(Mn, sb.shape[0])], 1)
        u = torch.randn(Mn, n_tel, 3, generator=gen, device=DEV)
        u = u / u.norm(dim=-1, keepdim=True) * (torch.rand(Mn, n_tel, 1, generator=gen, device=DEV) ** (1 / 3)) * Rr
        loc = x_int[:, i:i + 1] + SIGLOC * torch.randn(Mn, NCAND - n_tel, 3, generator=gen, device=DEV)
        cand = torch.cat([u, loc, x_int[:, i:i + 1]], 1)
        logits = -beta * row_E(cand, s_int[:, i], ox, os_)
        pick = torch.multinomial(torch.softmax(logits, 1), 1, generator=gen).squeeze(1)
        x_int = x_int.clone(); x_int[:, i] = cand[torch.arange(Mn, device=DEV), pick]
    return x_int


def core_qc(X, sX, Y, sY, gen, rc=0.5, b=0.2, n_mc=3000):
    """Paper core overlap between configs X and Y (center-relative frames): Gaussian same-species-NN field of
    X vs Y, MC-averaged over the |r|<rc window (window points assigned to nearest X particle)."""
    qX = torch.zeros(X.shape[0], device=DEV)
    for sp in (0, 1):
        xm, ym = (sX == sp), (sY == sp)
        if int(xm.sum()) > 0 and int(ym.sum()) > 0:
            qX[xm] = torch.exp(-torch.cdist(X[xm], Y[ym]).min(1).values ** 2 / (2 * b * b))
    u = torch.randn(n_mc, 3, generator=gen, device=DEV); u = u / u.norm(dim=-1, keepdim=True)
    pts = u * (torch.rand(n_mc, 1, generator=gen, device=DEV) ** (1.0 / 3.0)) * rc
    return float(qX[torch.cdist(pts, X).argmin(1)].mean())


def qc_groups(xf, s_int, gen):
    """qc over chain pairs: cross (data x unif) = estimator; dd/uu = diagnostics."""
    def q(i, j):
        return core_qc(xf[i], s_int[i], xf[j], s_int[j], gen)
    cross = [q(i, j) for i in range(M_DATA) for j in range(M_DATA, M)]  # data x alien = estimator
    dd = [q(i, j) for i in range(M_DATA) for j in range(i + 1, M_DATA)]
    uu = [q(i, j) for i in range(M_DATA, M) for j in range(i + 1, M)]
    return cross, dd, uu


def main():
    t0 = time.time()
    d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt",
                   map_location=DEV, weights_only=False)
    X, S, L = d["x"].to(DEV).float(), d["s"].to(DEV).long(), float(d["L"])
    gen = torch.Generator(device=DEV).manual_seed(0)
    results = []
    for Rr in RGRID:
        for cav in range(NCAV):
            ci = (cav * len(RGRID) + RGRID.index(Rr)) % X.shape[0]                    # distinct ref configs
            c = torch.rand(3, generator=gen, device=DEV) * L
            p = carve(X[ci], S[ci], c, Rr, L)
            if p["n_in"] < 8:
                continue
            x_in = _mic(p["x_in"], c, L); s_in = p["s_in"]
            xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (Rr + RCTX)
            bnd, sb = xout[bm], p["s_out"][bm]
            n = x_in.shape[0]
            x0 = torch.empty(M, n, 3, device=DEV)
            x0[:M_DATA] = x_in[None]                                                  # data starts
            # ALIEN starts: same center, different reference configs (different underlying chain id:
            # configs are 3 snapshots x 16 chains stacked, chain id = index % 16)
            n_chain_ids = 16
            k = 0
            for cj in range(X.shape[0]):
                if k >= M_ALIEN:
                    break
                if cj % n_chain_ids == ci % n_chain_ids:
                    continue
                pa = carve(X[cj], S[cj], c, Rr, L)
                xa = _mic(pa["x_in"], c, L)
                na = xa.shape[0]
                if na >= n:
                    sel = torch.argsort(xa.norm(dim=-1))[:n]                          # innermost n
                    x0[M_DATA + k] = xa[sel]
                else:
                    u = torch.randn(n - na, 3, generator=gen, device=DEV)
                    pad = u / u.norm(dim=-1, keepdim=True) * (torch.rand(n - na, 1, generator=gen, device=DEV) ** (1 / 3)) * Rr
                    x0[M_DATA + k] = torch.cat([xa, pad], 0)
                k += 1
            if k < M_ALIEN:                                                           # fallback (shouldn't happen)
                for kk in range(k, M_ALIEN):
                    x0[M_DATA + kk] = x0[M_DATA + max(0, k - 1)]
            s_int = s_in[None].expand(M, n).contiguous()
            e_ref = float(full_E(x_in[None], s_in[None], bnd, sb)[0])
            traces = {"sweep": [], "U_data": [], "U_unif": [], "qc_cross": []}
            xf = x0
            n_burn = int(BURN_FRAC * SWEEPS[Rr])
            for sw in range(1, SWEEPS[Rr] + 1):
                if sw <= n_burn:
                    b_now = BETA_HOT * (BETA / BETA_HOT) ** ((sw - 1) / max(1, n_burn - 1))   # geometric ladder
                else:
                    b_now = BETA
                xf = heatbath_sweep(xf, s_int, bnd, sb, Rr, gen, beta=b_now)
                if sw % SNAP_EVERY == 0 or sw == SWEEPS[Rr]:
                    e = full_E(xf, s_int, bnd, sb)
                    cross, _, _ = qc_groups(xf, s_int, gen)
                    traces["sweep"].append(sw)
                    traces["U_data"].append(float((e[:M_DATA].mean() - e_ref) / n))
                    traces["U_unif"].append(float((e[M_DATA:].mean() - e_ref) / n))
                    traces["qc_cross"].append(sum(cross) / len(cross))
            cross, dd, uu = qc_groups(xf, s_int, gen)
            e = full_E(xf, s_int, bnd, sb)
            gap = float((e[M_DATA:].mean() - e[:M_DATA].mean()) / n)
            conv = abs(gap) < 0.3
            rec = {"R": Rr, "cav": cav, "ref_ci": ci, "n_in": n, "center": c.cpu(),
                   "qc_cross": cross, "qc_dd": dd, "qc_aa": uu, "traces": traces,
                   "unif_minus_data_gap": gap, "converged": conv,
                   "x_final": xf.cpu(), "s_int": s_int[0].cpu(), "bnd": bnd.cpu(), "sb": sb.cpu(),
                   "x_data": x_in.cpu(), "e_ref": e_ref}
            results.append(rec)
            torch.save({"results": results, "L": L, "T": T, "rho": float(d["rho"]),
                        "protocol": "oracle heat-bath 4data+4alien, geometric beta-ladder burn, qc cross-pair estimator"}, OUT)
            qm = sum(cross) / len(cross)
            print(f"[gpts] R={Rr} cav={cav} n_in={n:4d}  qc_cross={qm:.3f}  dd={sum(dd)/len(dd):.3f} "
                  f"aa={sum(uu)/len(uu):.3f}  gapU={gap:+.3f} {'OK' if conv else 'NOT-CONVERGED'} "
                  f"({(time.time()-t0)/60:.0f} min)", flush=True)
    # summary curve
    print("\n[gpts] G_PTS(R) (cross-pair median over converged cavities):", flush=True)
    for Rr in RGRID:
        qs = [sum(r["qc_cross"]) / len(r["qc_cross"]) for r in results if r["R"] == Rr and r["converged"]]
        nc = sum(1 for r in results if r["R"] == Rr and not r["converged"])
        if qs:
            qs.sort()
            print(f"  R={Rr}:  G_PTS={qs[len(qs)//2]:.3f}   ({len(qs)} conv, {nc} flagged)", flush=True)
        else:
            print(f"  R={Rr}:  NO converged cavities ({nc} flagged)", flush=True)
    print(f"[gpts] DONE -> {OUT}  ({(time.time()-t0)/60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
