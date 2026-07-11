"""Matched-observable PTS: recompute the SMC cavity overlap as Berthier-Charbonneau-Yaida (2016)'s CORE
correlation G_PTS(R) = <q_c>, for a genuine comparison to their J.Chem.Phys.144,024501 T=0.51 curve.

Their overlap (paper eq 6-8): for each particle x_i, nearest SAME-species y in the other config, overlap
w(|x_i-y|), w(z)=exp(-z^2/2b^2), b=0.2; interpolate to a field q(r); core q_c = (1/vol) int_{|r|<rc} q(r),
rc=0.5 at the cavity CENTER (MC-integrated, NN interpolation here). G_PTS = disorder-avg of q_c between two
independent equilibrium configs. Fit form G_PTS = A exp[-(R/xi)^3] + G_bulk (eta=3). Radii 1.4/1.7/2.0/2.3
(their small-R regime; my box caps R~2.5). Reuses the SMC sampler (ka3d_pts_smc machinery, inlined)."""
import argparse, math, statistics as st, time
from pathlib import Path
import torch
from liquid_coupling_flow.ka3d_scaffold_ebm import KA3DScaffoldEBM
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

ART = "liquid_coupling_flow/artifacts"; dev = "cuda"


def energy(x, s, bnd, s_bnd):
    xx = torch.cat([x, bnd], 0); ss = torch.cat([s, s_bnd], 0)
    return float(ka_energy(xx[None], ss.long()[None], 100.0)[0])


def core_gpts(X, sX, Y, sY, gen, rc=0.5, b=0.2, n_mc=3000):
    """Paper's core overlap between config X (species sX) and Y (sY), cavity center = origin."""
    qX = torch.zeros(X.shape[0], device=dev)
    for sp in (0, 1):
        xm, ym = (sX == sp), (sY == sp)
        if int(xm.sum()) > 0 and int(ym.sum()) > 0:
            qX[xm] = torch.exp(-torch.cdist(X[xm], Y[ym]).min(1).values ** 2 / (2 * b * b))
    u = torch.randn(n_mc, 3, generator=gen, device=dev); u = u / u.norm(dim=-1, keepdim=True)
    pts = u * (torch.rand(n_mc, 1, generator=gen, device=dev) ** (1.0 / 3.0)) * rc      # uniform in ball rc
    return float(qX[torch.cdist(pts, X).argmin(1)].mean())                              # NN-interpolated field


def blob(n, R, K, gen):
    a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    mm = torch.zeros(n, dtype=torch.bool, device=dev)
    mm[(a - a[seed]).norm(dim=-1).topk(min(K, n), largest=False).indices] = True
    return mm


def systematic_resample(logw, gen):
    w = torch.softmax(logw, 0); M = w.shape[0]
    pos = (torch.arange(M, device=dev) + torch.rand((), generator=gen, device=dev)) / M
    return torch.searchsorted(w.cumsum(0), pos).clamp(max=M - 1)


def ess_frac(logw):
    w = torch.softmax(logw, 0); return float(1.0 / (w ** 2).sum() / w.shape[0])


def smc_cavity(m, xo, so, bnd, s_bnd, R, beta, M, T, n_mut, K, resamp, gen):
    n = xo.shape[0]; allmask = torch.ones(n, dtype=torch.bool, device=dev)

    def full_G(x, s):
        return -beta * energy(x, s, bnd, s_bnd) - float(m.block_log_prob(x, s, allmask, bnd, s_bnd, R))

    xs, ss = [], []
    for _ in range(M):
        fx, fs, _ = m.sample_block(xo, so, allmask, bnd, s_bnd, R, gen=gen); xs.append(fx); ss.append(fs)
    lam = torch.linspace(0, 1, T + 1, device=dev); logw = torch.zeros(M, device=dev)
    for t in range(1, T + 1):
        dlam = float(lam[t] - lam[t - 1])
        logw = logw + dlam * torch.tensor([full_G(xs[k], ss[k]) for k in range(M)], device=dev)
        if ess_frac(logw) < resamp:
            idx = systematic_resample(logw, gen)
            xs = [xs[int(i)].clone() for i in idx]; ss = [ss[int(i)].clone() for i in idx]
            logw = torch.zeros(M, device=dev)
        lt = float(lam[t])
        for k in range(M):
            for _ in range(n_mut):
                blk = blob(n, R, K, gen)
                u0 = energy(xs[k], ss[k], bnd, s_bnd); lqr = float(m.block_log_prob(xs[k], ss[k], blk, bnd, s_bnd, R))
                xn, sn, lqf = m.sample_block(xs[k], ss[k], blk, bnd, s_bnd, R, gen=gen)
                la = lt * (-beta * (energy(xn, sn, bnd, s_bnd) - u0) + lqr - float(lqf))
                if torch.rand((), device=dev, generator=gen).log() < la:
                    xs[k], ss[k] = xn, sn
    w = torch.softmax(logw, 0)
    gpts = float(sum(float(w[k]) * core_gpts(xs[k], ss[k], xo, so, gen) for k in range(M)))
    return gpts, ess_frac(logw), xs, ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[1.4, 1.7, 2.0, 2.3])
    ap.add_argument("--ncav", type=int, default=4); ap.add_argument("--M", type=int, default=48)
    ap.add_argument("--T", type=int, default=40); ap.add_argument("--nmut", type=int, default=4)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--resamp", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0); ap.add_argument("--r-ctx", type=float, default=2.5)
    ap.add_argument("--ckpt", default=f"{ART}/ka3d_cavity_ebm3ax.pt")
    ap.add_argument("--out", default="reports/logs-2026-07-11/pts_gpts.pt")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    m = KA3DScaffoldEBM(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    d = torch.load(f"{ART}/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
    X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
    gen = torch.Generator(device=dev).manual_seed(0); results = {}
    # bulk floor G_bulk: core overlap between interiors of DIFFERENT (uncorrelated) configs, per R
    for R in a.radii:
        cavs = []
        for ci in range(900, 900 + 8 * a.ncav):
            if len(cavs) >= a.ncav + 1:
                break
            c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
            if p["n_in"] < 10:
                continue
            xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
            xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + a.r_ctx)
            cavs.append((xo, so, xout[bm], p["s_out"][bm]))
        gbulk = st.mean([core_gpts(cavs[i][0], cavs[i][1], cavs[i + 1][0], cavs[i + 1][1], gen)
                         for i in range(len(cavs) - 1)])
        t0 = time.time(); rs = [smc_cavity(m, xo, so, bnd, sb, R, a.beta, a.M, a.T, a.nmut, a.K, a.resamp, gen)
                                for (xo, so, bnd, sb) in cavs[:a.ncav]]
        g = st.mean([r[0] for r in rs]); e = 100 * st.mean([r[1] for r in rs])
        results[R] = {"gpts": g, "ess": e, "gbulk": gbulk, "ncav": a.ncav}
        print(f"R={R}: G_PTS(core)={g:.3f}  [ESS {e:.0f}%]   G_bulk~{gbulk:.3f}   ({time.time()-t0:.0f}s)", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); torch.save(results, a.out)
    print("\n=== matched-observable G_PTS(R) vs Berthier-Charbonneau-Yaida 2016 (T~0.51) ===", flush=True)
    for R in a.radii:
        r = results[R]; print(f"  R={R}: G_PTS={r['gpts']:.3f}  (excess over bulk {r['gpts']-r['gbulk']:+.3f})", flush=True)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
