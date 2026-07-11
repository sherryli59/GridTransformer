"""Batched-over-M SMC PTS: same algorithm as ka3d_pts_full but the M particles are advanced through the
generator IN PARALLEL (KA3DScaffoldEBMBatched), so each annealing step is a few big GPU calls instead of
M*(1+nmut) sequential ones. Mutation uses a SHARED random blob per move (valid: blob is random each move).
Saves populations; computes G_PTS(core), q_whole, chi_T. Matched to Berthier-Charbonneau-Yaida 2016 T=0.51."""
import argparse, math, statistics as st, time
from pathlib import Path
import torch
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

ART = "liquid_coupling_flow/artifacts"; dev = "cuda"


def energy_b(Xb, Sb, bnd, s_bnd):
    M, mm = Xb.shape[0], bnd.shape[0]
    x = torch.cat([Xb, bnd[None].expand(M, mm, 3)], 1); s = torch.cat([Sb, s_bnd[None].expand(M, mm)], 1)
    return ka_energy(x, s.long(), 100.0)                                            # [M]


def overlap_whole(C, C0, a=0.3):
    return float((torch.cdist(C0, C).min(1).values < a).float().mean())


def core_qc(X, sX, Y, sY, gen, rc=0.5, b=0.2, n_mc=3000):
    qX = torch.zeros(X.shape[0], device=dev)
    for sp in (0, 1):
        xm, ym = (sX == sp), (sY == sp)
        if int(xm.sum()) > 0 and int(ym.sum()) > 0:
            qX[xm] = torch.exp(-torch.cdist(X[xm], Y[ym]).min(1).values ** 2 / (2 * b * b))
    u = torch.randn(n_mc, 3, generator=gen, device=dev); u = u / u.norm(dim=-1, keepdim=True)
    pts = u * (torch.rand(n_mc, 1, generator=gen, device=dev) ** (1.0 / 3.0)) * rc
    return float(qX[torch.cdist(pts, X).argmin(1)].mean())


def blob(n, R, K, gen):
    a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    mm = torch.zeros(n, dtype=torch.bool, device=dev)
    mm[(a - a[seed]).norm(dim=-1).topk(min(K, n), largest=False).indices] = True
    return mm


def ess_frac(logw):
    w = torch.softmax(logw, 0); return float(1.0 / (w ** 2).sum() / w.shape[0])


def smc_batched(m, xo, so, bnd, s_bnd, R, beta, M, T, n_mut, K, resamp, gen):
    n = xo.shape[0]; allmask = torch.ones(n, dtype=torch.bool, device=dev)
    Xb, Sb, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                 allmask, bnd, s_bnd, R, gen=gen)                    # M independent full-regens

    def G():
        return -beta * energy_b(Xb, Sb, bnd, s_bnd) - m.block_log_prob_b(Xb, Sb, allmask, bnd, s_bnd, R)

    lam = torch.linspace(0, 1, T + 1, device=dev); logw = torch.zeros(M, device=dev)
    for t in range(1, T + 1):
        logw = logw + float(lam[t] - lam[t - 1]) * G()
        if ess_frac(logw) < resamp:
            w = torch.softmax(logw, 0)
            pos = (torch.arange(M, device=dev) + torch.rand((), generator=gen, device=dev)) / M
            idx = torch.searchsorted(w.cumsum(0), pos).clamp(max=M - 1)
            Xb, Sb = Xb[idx].clone(), Sb[idx].clone(); logw = torch.zeros(M, device=dev)
        lt = float(lam[t])
        for _ in range(n_mut):
            blk = blob(n, R, K, gen)
            u0 = energy_b(Xb, Sb, bnd, s_bnd); lqr = m.block_log_prob_b(Xb, Sb, blk, bnd, s_bnd, R)
            Xn, Sn, lqf = m.sample_block_b(Xb, Sb, blk, bnd, s_bnd, R, gen=gen)
            la = lt * (-beta * (energy_b(Xn, Sn, bnd, s_bnd) - u0) + lqr - lqf)
            acc = torch.rand(M, device=dev, generator=gen).log() < la
            Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb)
    return Xb, Sb, torch.softmax(logw, 0), ess_frac(logw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[1.4, 1.7, 2.0, 2.3])
    ap.add_argument("--ncav", type=int, default=6); ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--T", type=int, default=40); ap.add_argument("--nmut", type=int, default=4)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--resamp", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0); ap.add_argument("--r-ctx", type=float, default=2.5)
    ap.add_argument("--ckpt", default=f"{ART}/ka3d_cavity_ebm3ax.pt")
    ap.add_argument("--out", default="reports/logs-2026-07-11/pts_batched.pt")
    ap.add_argument("--pop-out", default="reports/logs-2026-07-11/pts_pop_batched.pt")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    d = torch.load(f"{ART}/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
    X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
    gen = torch.Generator(device=dev).manual_seed(0); results = {}; pops_all = {}
    for R in a.radii:
        cavs = []
        for ci in range(900, 900 + 12 * a.ncav):
            if len(cavs) >= a.ncav + 1:
                break
            c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
            if p["n_in"] < 10:
                continue
            xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
            xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + a.r_ctx)
            cavs.append((xo, so, xout[bm], p["s_out"][bm]))
        gbulk = st.mean([core_qc(cavs[i][0], cavs[i][1], cavs[i + 1][0], cavs[i + 1][1], gen) for i in range(len(cavs) - 1)])
        t0 = time.time(); gc, qw, qp, ess, pops = [], [], [], [], []
        for (xo, so, bnd, sb) in cavs[:a.ncav]:
            Xb, Sb, w, e = smc_batched(m, xo, so, bnd, sb, R, a.beta, a.M, a.T, a.nmut, a.K, a.resamp, gen)
            gc.append(sum(float(w[k]) * core_qc(Xb[k], Sb[k], xo, so, gen) for k in range(a.M)))
            qw.append(sum(float(w[k]) * overlap_whole(Xb[k], xo) for k in range(a.M)))
            ii = torch.multinomial(w, 150, replacement=True, generator=gen); jj = torch.multinomial(w, 150, replacement=True, generator=gen)
            qp.append(st.mean([core_qc(Xb[int(i)], Sb[int(i)], Xb[int(j)], Sb[int(j)], gen) for i, j in zip(ii, jj) if int(i) != int(j)]))
            ess.append(e); pops.append({"Xb": Xb.cpu(), "Sb": Sb.cpu(), "w": w.cpu(), "C0": xo.cpu(), "sC0": so.cpu()})
        pops_all[R] = pops
        results[R] = {"g_core": st.mean(gc), "q_whole": st.mean(qw), "qc_pair": st.mean(qp),
                      "ess": 100 * st.mean(ess), "gbulk": gbulk, "ncav": a.ncav}
        r = results[R]
        print(f"R={R} ({time.time()-t0:.0f}s, ESS {r['ess']:.0f}%): G_PTS(core)={r['g_core']:.3f} "
              f"qc_pair={r['qc_pair']:.3f} q_whole={r['q_whole']:.3f} G_bulk={gbulk:.3f}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); torch.save(results, a.out); torch.save(pops_all, a.pop_out)
    print("\n=== batched matched G_PTS(R) vs Berthier-Charbonneau-Yaida 2016 (T~0.51) ===", flush=True)
    for R in a.radii:
        r = results[R]; print(f"  R={R}: G_PTS={r['g_core']:.3f} (paper est ~{ {1.4:0.81,1.7:0.74,2.0:0.65,2.3:0.55}.get(R,0):.2f}) "
                               f"qc_pair={r['qc_pair']:.3f} ESS {r['ess']:.0f}%", flush=True)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
