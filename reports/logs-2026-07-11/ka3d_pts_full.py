"""ONE SMC per (R, cavity) -> save the weighted Boltzmann population {(C_k, w_k)} -> compute ALL observables
offline (the observable never enters the sampler). Demonstrates the correct design: the SMC targets
P(interior|boundary)~e^{-bU}; whole-interior overlap, Berthier-Charbonneau-Yaida core G_PTS, and their
susceptibility chi_T = <q_c^2>-<q_c>^2 are all just weighted averages over the same samples.

Saves the populations to disk so any FUTURE observable is free (no re-run). Radii 1.4/1.7/2.0/2.3, T=0.51."""
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


def overlap_whole(C, C0, a=0.3):                                       # my earlier whole-interior overlap
    return float((torch.cdist(C0, C).min(1).values < a).float().mean())


def core_qc(X, sX, Y, sY, gen, rc=0.5, b=0.2, n_mc=3000):             # paper core overlap q_c between X,Y
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


def resample(logw, gen):
    w = torch.softmax(logw, 0); M = w.shape[0]
    pos = (torch.arange(M, device=dev) + torch.rand((), generator=gen, device=dev)) / M
    return torch.searchsorted(w.cumsum(0), pos).clamp(max=M - 1)


def ess_frac(logw):
    w = torch.softmax(logw, 0); return float(1.0 / (w ** 2).sum() / w.shape[0])


def smc_population(m, xo, so, bnd, s_bnd, R, beta, M, T, n_mut, K, resamp, gen):
    """Run SMC; RETURN the weighted Boltzmann population (xs, ss, w) — no observable computed here."""
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
            idx = resample(logw, gen); xs = [xs[int(i)].clone() for i in idx]; ss = [ss[int(i)].clone() for i in idx]
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
    return xs, ss, torch.softmax(logw, 0), ess_frac(logw)


def observables(pop, gen, n_pair=200):
    """ALL PTS observables from ONE saved population (the point: observable = post-processing)."""
    xs, ss, w, C0, sC0 = pop["xs"], pop["ss"], pop["w"], pop["C0"], pop["sC0"]
    M = len(xs)
    q_whole = float(sum(float(w[k]) * overlap_whole(xs[k], C0) for k in range(M)))
    g_core = float(sum(float(w[k]) * core_qc(xs[k], ss[k], C0, sC0, gen) for k in range(M)))
    # susceptibility chi_T = Var(q_c) over PAIRS of independent samples (their eq 10), weighted by w_k w_l
    ii = torch.multinomial(w, n_pair, replacement=True, generator=gen)
    jj = torch.multinomial(w, n_pair, replacement=True, generator=gen)
    qc = torch.tensor([core_qc(xs[int(i)], ss[int(i)], xs[int(j)], ss[int(j)], gen)
                       for i, j in zip(ii, jj) if int(i) != int(j)], device=dev)
    return {"q_whole": q_whole, "g_core": g_core, "qc_mean": float(qc.mean()),
            "chi_T": float(qc.var(unbiased=True))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[1.4, 1.7, 2.0, 2.3])
    ap.add_argument("--ncav", type=int, default=4); ap.add_argument("--M", type=int, default=48)
    ap.add_argument("--T", type=int, default=40); ap.add_argument("--nmut", type=int, default=4)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--resamp", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0); ap.add_argument("--r-ctx", type=float, default=2.5)
    ap.add_argument("--ckpt", default=f"{ART}/ka3d_cavity_ebm3ax.pt")
    ap.add_argument("--pop-out", default="reports/logs-2026-07-11/pts_populations.pt")
    ap.add_argument("--out", default="reports/logs-2026-07-11/pts_full.pt")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    m = KA3DScaffoldEBM(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    d = torch.load(f"{ART}/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
    X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
    gen = torch.Generator(device=dev).manual_seed(0); results = {}; pops_all = {}
    for R in a.radii:
        cavs = []
        for ci in range(900, 900 + 10 * a.ncav):
            if len(cavs) >= a.ncav + 1:
                break
            c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
            if p["n_in"] < 10:
                continue
            xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
            xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + a.r_ctx)
            cavs.append((xo, so, xout[bm], p["s_out"][bm]))
        gbulk = st.mean([core_qc(cavs[i][0], cavs[i][1], cavs[i + 1][0], cavs[i + 1][1], gen)
                         for i in range(len(cavs) - 1)])
        t0 = time.time(); pops, obs = [], []
        for (xo, so, bnd, sb) in cavs[:a.ncav]:
            xs, ss, w, ess = smc_population(m, xo, so, bnd, sb, R, a.beta, a.M, a.T, a.nmut, a.K, a.resamp, gen)
            pop = {"xs": [x.cpu() for x in xs], "ss": [s.cpu() for s in ss], "w": w.cpu(),
                   "C0": xo.cpu(), "sC0": so.cpu(), "ess": ess}
            pops.append(pop)
            ob = observables({"xs": xs, "ss": ss, "w": w, "C0": xo, "sC0": so}, gen); ob["ess"] = 100 * ess
            obs.append(ob)
        pops_all[R] = pops
        agg = {k: st.mean([o[k] for o in obs]) for k in obs[0]}
        agg["gbulk"] = gbulk; results[R] = agg
        print(f"R={R} ({time.time()-t0:.0f}s, ESS {agg['ess']:.0f}%):  G_PTS(core)={agg['g_core']:.3f}  "
              f"qc_pair={agg['qc_mean']:.3f}  chi_T={agg['chi_T']:.4f}  q_whole={agg['q_whole']:.3f}  "
              f"G_bulk={gbulk:.3f}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(results, a.out); torch.save(pops_all, a.pop_out)
    print("\n=== matched to Berthier-Charbonneau-Yaida 2016 (T~0.51), ONE SMC per R, all observables ===", flush=True)
    for R in a.radii:
        r = results[R]
        print(f"  R={R}: G_PTS(core)={r['g_core']:.3f}  chi_T={r['chi_T']:.4f} (peak=xi_PTS)  "
              f"[q_whole {r['q_whole']:.3f}, bulk {r['gbulk']:.3f}]", flush=True)
    print(f"saved observables -> {a.out}; populations -> {a.pop_out} (future observables are free)", flush=True)


if __name__ == "__main__":
    main()
