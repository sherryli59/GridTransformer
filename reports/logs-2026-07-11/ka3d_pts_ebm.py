"""First-pass POINT-TO-SET measurement using the K=4 multi-axis block-MTM (ka3d_cavity_ebm3ax) as the
equilibration kernel for a cavity interior under a FROZEN boundary.

Static PTS (cavity method): carve a cavity from an equilibrium config (the reference C0), freeze the
exterior, and sample the constrained interior C ~ P(interior | boundary) ~ e^{-bU}. The block-MTM
(independence-MTM within a K=4 blob, exact e^{-bU} target) is that sampler. The cavity overlap
q_inf(R) = <overlap(C, C0)> measures the amorphous order the boundary imposes: high (few states) at small
R, decaying to the random floor at large R -> the point-to-set length.

This run: start the chain FROM the reference and watch q(t) decay to its plateau (a convergence diagnostic
AND the estimate of q_inf). Reports plateau, acceptance, uniform floor, and a full-regen independent-sample
overlap. Deliberately instrumented to expose gaps (mixing, K ceiling, R range)."""
import argparse, math, statistics as st, time
from pathlib import Path
import torch
from liquid_coupling_flow.ka3d_scaffold_ebm import KA3DScaffoldEBM
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

ART = "liquid_coupling_flow/artifacts"


def energy(xo, so, bnd, s_bnd):
    x = torch.cat([xo, bnd], 0); s = torch.cat([so, s_bnd], 0)
    return float(ka_energy(x[None], s.long()[None], 100.0)[0])       # big box: isolated cavity


def overlap(C, C0, a=0.3):
    """Collective (permutation-invariant) density overlap: fraction of reference sites with a sampled
    particle within a."""
    return float((torch.cdist(C0, C).min(1).values < a).float().mean())


def imtm_step(m, xo, so, blk, bnd, s_bnd, R, beta, Nt, dev, gen):
    """One exact independence-MTM block move (Liu-Liang-Wong reusable batch). Returns (xo, so, accepted)."""
    u0 = energy(xo, so, bnd, s_bnd); lqx = float(m.block_log_prob(xo, so, blk, bnd, s_bnd, R))
    lus = [-beta * u0 - lqx]; props = []
    for _ in range(Nt):
        xn, sn, lq = m.sample_block(xo, so, blk, bnd, s_bnd, R, gen=gen)
        props.append((xn, sn)); lus.append(-beta * energy(xn, sn, bnd, s_bnd) - float(lq))
    lu = torch.tensor(lus[1:], device=dev); sfwd = torch.logsumexp(lu, 0)
    J = int(torch.multinomial(torch.softmax(lu, 0), 1, generator=gen))
    lr = lu.clone(); lr[J] = lus[0]
    if torch.rand((), device=dev, generator=gen).log() < (sfwd - torch.logsumexp(lr, 0)):
        return props[J][0], props[J][1], True
    return xo, so, False


def blob(n, R, K, dev, gen):
    a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    m = torch.zeros(n, dtype=torch.bool, device=dev)
    m[(a - a[seed]).norm(dim=-1).topk(min(K, n), largest=False).indices] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[1.6, 2.0, 2.4])
    ap.add_argument("--ncav", type=int, default=6); ap.add_argument("--sweeps", type=int, default=20)
    ap.add_argument("--ntrials", type=int, default=8); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--M", type=int, default=32, help="importance-sampling full-regen draws per cavity")
    ap.add_argument("--beta", type=float, default=2.0); ap.add_argument("--r-ctx", type=float, default=2.5)
    ap.add_argument("--ckpt", default=f"{ART}/ka3d_cavity_ebm3ax.pt")
    ap.add_argument("--out", default="reports/logs-2026-07-11/pts_ebm.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = a.device
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    m = KA3DScaffoldEBM(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    d = torch.load(f"{ART}/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
    X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
    gen = torch.Generator(device=dev).manual_seed(0)
    results = {}
    for R in a.radii:
        qt = torch.zeros(a.sweeps + 1); q_unif, q_regen, q_is, accs, ess_list, ncav = 0.0, 0.0, 0.0, [], [], 0
        t0 = time.time()
        for ci in range(900, 900 + 4 * a.ncav):
            if ncav >= a.ncav:
                break
            c = torch.rand(3, generator=gen, device=dev) * L
            p = carve(X[ci], S[ci], c, R, L)
            if p["n_in"] < a.K + 4:
                continue
            xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
            xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + a.r_ctx)
            bnd, s_bnd, n = xout[bm], p["s_out"][bm], xo.shape[0]
            C0 = xo.clone(); state_x, state_s = xo.clone(), so.clone()
            qt[0] += overlap(state_x, C0)
            mv_per = max(1, n // a.K)
            for t in range(a.sweeps):
                for _ in range(mv_per):
                    blk = blob(n, R, a.K, dev, gen)
                    state_x, state_s, acc = imtm_step(m, state_x, state_s, blk, bnd, s_bnd, R, a.beta, a.ntrials, dev, gen)
                    accs.append(acc)
                qt[t + 1] += overlap(state_x, C0)
            # baselines
            unif = torch.randn(n, 3, device=dev, generator=gen); unif = unif / unif.norm(dim=-1, keepdim=True)
            unif = unif * (torch.rand(n, 1, device=dev, generator=gen) ** (1 / 3)) * R
            q_unif += overlap(unif, C0)
            allmask = torch.ones(n, dtype=torch.bool, device=dev)
            rx, rs, _ = m.sample_block(xo, so, allmask, bnd, s_bnd, R, gen=gen)
            q_regen += overlap(rx, C0)
            # IMPORTANCE SAMPLING: full-regen proposal (exact logq) reweighted to Boltzmann -> no mixing.
            logw, overs = [], []
            for _ in range(a.M):
                fx, fs, flq = m.sample_block(xo, so, allmask, bnd, s_bnd, R, gen=gen)
                logw.append(-a.beta * energy(fx, fs, bnd, s_bnd) - float(flq)); overs.append(overlap(fx, C0))
            lw = torch.tensor(logw, device=dev); w = torch.softmax(lw, 0)
            ess_list.append(float(1.0 / (w ** 2).sum() / a.M))       # normalized ESS in [0,1]
            q_is += float((w * torch.tensor(overs, device=dev)).sum())
            ncav += 1
        qt /= max(ncav, 1); q_unif /= max(ncav, 1); q_regen /= max(ncav, 1); q_is /= max(ncav, 1)
        acc = 100 * sum(accs) / max(len(accs), 1); ess = 100 * st.mean(ess_list) if ess_list else 0.0
        results[R] = {"qt": qt.tolist(), "q_unif": q_unif, "q_regen": q_regen, "q_is": q_is,
                      "accept": acc, "ess": ess, "ncav": ncav}
        print(f"R={R}: ncav={ncav} accept={acc:.0f}% ESS={ess:.0f}%  q_mcmc(0->t/2->inf)={qt[0]:.2f}->"
              f"{qt[a.sweeps//2]:.2f}->{qt[-1]:.2f}  q_IS={q_is:.2f} regen={q_regen:.2f} unif={q_unif:.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(results, a.out)
    print(f"saved {a.out}", flush=True)
    # PTS summary: plateau vs R (drop toward the floors = point-to-set decay)
    print("\n=== PTS overlap vs R (excess over uniform floor = amorphous order) ===", flush=True)
    for R in a.radii:
        r = results[R]
        print(f"  R={R}: q_IS={r['q_is']:.3f} (ESS {r['ess']:.0f}%)  q_mcmc_inf={r['qt'][-1]:.3f} "
              f"(accept {r['accept']:.0f}%)  unif {r['q_unif']:.3f}  excess_IS {r['q_is']-r['q_unif']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
