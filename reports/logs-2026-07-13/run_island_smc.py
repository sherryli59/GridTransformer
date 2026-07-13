"""ISLAND-SMC arm of the PT-vs-AR-SMC head-to-head (user proposal: 'with AR assist no shrinkage-PT
needed'). J independent annealed-SMC islands per cavity, geometric path q_AR -> Boltzmann, lambda-scheduled
block-MTM mutations, EVIDENCE (logZ_j) accumulated per island (log-mean-exp of incremental weights at every
resample + final). Islands may each commit to one basin; recombining islands weighted by exp(logZ_j)
recovers cross-basin statistics in expectation -- the one-pass substitute for PT's round trips.

Cavities MATCH the running 24-replica G1 exactly (frame=-(c+1), particle_id=(37c+11)%N, same N=512
dataset), so P(q) can be compared island-reweighted vs the G1 gold-standard ref_samples when it lands.
Compare later with: python run_island_smc.py --compare liquid_coupling_flow/artifacts/ka3d_pt_g1_24rep.pt
"""
import argparse, time
import torch, statistics as st
import sys
sys.path.insert(0, "reports/logs-2026-07-11")
from ka3d_pts_batched import energy_b, ess_frac, blob, dev  # noqa: E402  (their dev == cuda)
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic

RCTX = 2.5; BETA = 2.0


def smc_island(m, xo, so, bnd, s_bnd, R, M, T, n_mut, K, kbig, resamp, gen):
    """One island: annealed SMC with EVIDENCE accounting. Returns (X, S, logw, logZ)."""
    n = xo.shape[0]; allmask = torch.ones(n, dtype=torch.bool, device=dev)
    Xb, Sb, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                 allmask, bnd, s_bnd, R, gen=gen)
    Kbig = min(n - 1, kbig) if kbig > 0 else K

    def G():
        return -BETA * energy_b(Xb, Sb, bnd, s_bnd) - m.block_log_prob_b(Xb, Sb, allmask, bnd, s_bnd, R)

    # WARMUP schedule (t/T)^4, not linear: the first linear increment multiplies clash-blown AR full-regen
    # energies (U~1e7 at r->0) by dlam~1/T and poisons logZ additively (smoke measured logZ -5.5e6). The
    # quartic start makes early increments ~1e-6*U while resampling+mutation declash the population; the
    # estimator stays exact (any monotone schedule is a valid AIS path).
    lam = torch.linspace(0, 1, T + 1, device=dev) ** 4
    logw = torch.zeros(M, device=dev); logZ = 0.0
    for t in range(1, T + 1):
        logw = logw + float(lam[t] - lam[t - 1]) * G()
        if ess_frac(logw) < resamp:
            logZ += float(torch.logsumexp(logw, 0)) - torch.log(torch.tensor(float(M)))
            w = torch.softmax(logw, 0)
            pos = (torch.arange(M, device=dev) + torch.rand((), generator=gen, device=dev)) / M
            idx = torch.searchsorted(w.cumsum(0), pos).clamp(max=M - 1)
            Xb, Sb = Xb[idx].clone(), Sb[idx].clone(); logw = torch.zeros(M, device=dev)
        lt = float(lam[t])
        Kmove = max(K, int(round(K + (Kbig - K) * (1.0 - lt))))
        for _ in range(n_mut):
            blk = blob(n, R, Kmove, gen)
            u0 = energy_b(Xb, Sb, bnd, s_bnd); lqr = m.block_log_prob_b(Xb, Sb, blk, bnd, s_bnd, R)
            Xn, Sn, lqf = m.sample_block_b(Xb, Sb, blk, bnd, s_bnd, R, gen=gen)
            la = lt * (-BETA * (energy_b(Xn, Sn, bnd, s_bnd) - u0) + lqr - lqf)
            acc = torch.rand(M, device=dev, generator=gen).log() < la
            Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb)
        # K=1 single-site AR heat-bath pass per rung: same exact tempered-MH ratio, but single-site
        # proposals accept readily and DECLASH the population (the frozen-cage-relaxation lesson) --
        # without this, full-regen clash energies (U~1e4-1e7) survive into late rungs and blow the
        # island evidence (smoke: logZ spread 20k nats even with the quartic warmup).
        for i in torch.randperm(n, generator=gen, device=dev)[:n].tolist():
            blk1 = torch.zeros(n, dtype=torch.bool, device=dev); blk1[i] = True
            u0 = energy_b(Xb, Sb, bnd, s_bnd); lqr = m.block_log_prob_b(Xb, Sb, blk1, bnd, s_bnd, R)
            Xn, Sn, lqf = m.sample_block_b(Xb, Sb, blk1, bnd, s_bnd, R, gen=gen)
            la = lt * (-BETA * (energy_b(Xn, Sn, bnd, s_bnd) - u0) + lqr - lqf)
            acc = torch.rand(M, device=dev, generator=gen).log() < la
            Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb)
    logZ += float(torch.logsumexp(logw, 0)) - torch.log(torch.tensor(float(M)))
    return Xb, Sb, logw, logZ


def build_cavity(X, S, L, R, c):
    """EXACTLY the G1 driver's deterministic cavity (pt_gate_g1.py lines 52-60)."""
    frame = -(c + 1)
    x = X[frame]; s = S[frame]
    particle_id = (37 * c + 11) % x.shape[0]
    center = x[particle_id].clone()
    delta = x - center; delta = delta - L * torch.round(delta / L)
    mobile = delta.square().sum(-1) < R * R
    xin = _mic(x[mobile], center, L); sin = s[mobile]
    xout = _mic(x[~mobile], center, L); bm = xout.norm(dim=-1) < (R + RCTX)
    xo, so, _ = label_to_scaffold(xin, sin, R)
    return xo, so, xout[bm], s[~mobile][bm], xin, sin


def overlap_pop(Xp, Sp, xref, sref, a=0.3):
    out = []
    for k in range(Xp.shape[0]):
        d = torch.cdist(Xp[k], xref).masked_fill(Sp[k][:, None] != sref[None, :], 9.0)
        out.append(float((d.min(1).values < a).float().mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[2.2, 2.4])
    ap.add_argument("--centers", type=int, default=2)
    ap.add_argument("--islands", type=int, default=8)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--n-mut", type=int, default=4)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--kbig", type=int, default=12)
    ap.add_argument("--resamp", type=float, default=0.5)
    ap.add_argument("--out", default="reports/logs-2026-07-13/island_smc.pt")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.radii = [2.2]; a.centers = 1; a.islands = 2; a.m = 8; a.T = 12; a.n_mut = 2

    ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax.pt", map_location=dev, weights_only=False)
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
    X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
    results = {}
    for R in a.radii:
        for c in range(a.centers):
            t0 = time.time()
            xo, so, bnd, sb, xin, sin = build_cavity(X, S, L, R, c)
            isl = []
            for j in range(a.islands):
                gen = torch.Generator(device=dev).manual_seed(10_000 + 97 * j + int(R * 10) + c)
                Xp, Sp, logw, logZ = smc_island(m, xo, so, bnd, sb, R, a.m, a.T, a.n_mut, a.K, a.kbig, a.resamp, gen)
                q = overlap_pop(Xp, Sp, xin.to(dev), sin.to(dev))
                isl.append({"logZ": logZ, "logw": logw.cpu(), "q": q, "X": Xp.cpu(), "S": Sp.cpu()})
                print(f"  R={R} c={c} island {j}: logZ {logZ:+.1f}  q med {st.median(q):.2f}", flush=True)
            lz = torch.tensor([i["logZ"] for i in isl])
            wj = torch.softmax(lz, 0)
            # island-reweighted mean overlap (within-island weights logw ~ uniform post-resample)
            qbar = float(sum(wj[j] * st.mean(isl[j]["q"]) for j in range(len(isl))))
            results[(R, c)] = {"islands": isl, "logZ_j": lz, "w_j": wj, "q_island_reweighted": qbar}
            torch.save(results, a.out)                                              # incremental
            print(f"R={R} c={c}: logZ_j spread {float(lz.max()-lz.min()):.1f}  "
                  f"island-reweighted q {qbar:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
