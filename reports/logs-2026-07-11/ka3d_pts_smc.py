"""SMC/AIS PTS estimate: bridge the proposal q -> Boltzmann e^{-bU} that one-shot IS cannot swallow and
local MCMC cannot cross. Geometric path pi_t ~ q^{1-lam} (e^{-bU})^{lam}, lam: 0->1.

Two exact ingredients (both reuse block_log_prob / sample_block / energy):
  * incremental AIS weight  dlog w = dlam * G(x),  G = -bU - logq_full  (full IS weight split into T pieces)
  * mutation leaving pi_t invariant = the block move with MH log-accept SCALED by lam:
        log alpha = lam * [ -b dU + logq(b|ret) - logq(b'|ret) ]   (= exact block-MTM at lam=1)
Reports ESS(lam) (should stay UP vs naive-IS ~1/M collapse) and the converged q_PTS(R). Also prints the
naive one-shot IS estimate from the SAME initial particles for direct comparison."""
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


def overlap(C, C0, a=0.3):
    return float((torch.cdist(C0, C).min(1).values < a).float().mean())


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


def smc_cavity(m, xo, so, bnd, s_bnd, R, C0, beta, M, T, n_mut, K, resamp, gen):
    n = xo.shape[0]; allmask = torch.ones(n, dtype=torch.bool, device=dev)

    def full_G(x, s):
        return -beta * energy(x, s, bnd, s_bnd) - float(m.block_log_prob(x, s, allmask, bnd, s_bnd, R))

    # init M particles from q + their overlaps + naive-IS weights
    xs, ss, G0, ov0 = [], [], [], []
    for _ in range(M):
        fx, fs, _ = m.sample_block(xo, so, allmask, bnd, s_bnd, R, gen=gen)
        xs.append(fx); ss.append(fs); G0.append(full_G(fx, fs)); ov0.append(overlap(fx, C0))
    G0 = torch.tensor(G0, device=dev); ov0 = torch.tensor(ov0, device=dev)
    q_naive = float((torch.softmax(G0, 0) * ov0).sum()); ess_naive = ess_frac(G0)

    lam = torch.linspace(0, 1, T + 1, device=dev)
    logw = torch.zeros(M, device=dev); ess_trace = []
    for t in range(1, T + 1):
        dlam = float(lam[t] - lam[t - 1])
        G = torch.tensor([full_G(xs[k], ss[k]) for k in range(M)], device=dev)
        logw = logw + dlam * G
        ess_trace.append((float(lam[t]), ess_frac(logw)))
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
    q_smc = float((w * torch.tensor([overlap(xs[k], C0) for k in range(M)], device=dev)).sum())
    return {"q_smc": q_smc, "ess_smc": ess_frac(logw), "q_naive": q_naive, "ess_naive": ess_naive,
            "ess_trace": ess_trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[1.6, 2.0, 2.4])
    ap.add_argument("--ncav", type=int, default=4); ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--T", type=int, default=24); ap.add_argument("--nmut", type=int, default=4)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--resamp", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0); ap.add_argument("--r-ctx", type=float, default=2.5)
    ap.add_argument("--ckpt", default=f"{ART}/ka3d_cavity_ebm3ax.pt")
    ap.add_argument("--out", default="reports/logs-2026-07-11/pts_smc.pt")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    m = KA3DScaffoldEBM(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    d = torch.load(f"{ART}/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
    X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
    gen = torch.Generator(device=dev).manual_seed(0); results = {}
    for R in a.radii:
        cavs = []; t0 = time.time()
        for ci in range(900, 900 + 6 * a.ncav):
            if len(cavs) >= a.ncav:
                break
            c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
            if p["n_in"] < 10:
                continue
            xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
            xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + a.r_ctx)
            cavs.append((xo, so, xout[bm], p["s_out"][bm]))
        rs = [smc_cavity(m, xo, so, bnd, sb, R, xo.clone(), a.beta, a.M, a.T, a.nmut, a.K, a.resamp, gen)
              for (xo, so, bnd, sb) in cavs]
        q_smc = st.mean([r["q_smc"] for r in rs]); e_smc = 100 * st.mean([r["ess_smc"] for r in rs])
        q_nv = st.mean([r["q_naive"] for r in rs]); e_nv = 100 * st.mean([r["ess_naive"] for r in rs])
        # ESS(lam) averaged across cavities
        trace = [(l, 100 * st.mean([r["ess_trace"][i][1] for r in rs])) for i, (l, _) in enumerate(rs[0]["ess_trace"])]
        results[R] = {"q_smc": q_smc, "ess_smc": e_smc, "q_naive": q_nv, "ess_naive": e_nv,
                      "trace": trace, "ncav": len(cavs)}
        print(f"R={R} (ncav={len(cavs)}, {time.time()-t0:.0f}s):  SMC q={q_smc:.3f} ESS={e_smc:.1f}%   "
              f"vs naive-IS q={q_nv:.3f} ESS={e_nv:.1f}%", flush=True)
        print("    ESS(lam):  " + "  ".join(f"{l:.2f}:{e:.0f}%" for l, e in trace[::max(1, len(trace)//8)]), flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); torch.save(results, a.out)
    print("\n=== SMC PTS overlap vs R ===", flush=True)
    for R in a.radii:
        r = results[R]
        print(f"  R={R}: q_PTS={r['q_smc']:.3f} (ESS {r['ess_smc']:.0f}%)  [naive-IS {r['q_naive']:.3f} @ ESS {r['ess_naive']:.1f}%]", flush=True)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
