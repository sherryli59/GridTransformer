"""Island-SMC for PTS on N=4096 rho=1.15 cavities, with the RL+demand base+mutation model. Sweeps:
  SIZES     : cavity R in {2.0, 2.5, 3.0}
  MOVE      : '1blob' (one K-blob) vs '2blob' (two well-separated K/2 blobs, union mask -> shared species
              budget => non-local species redistribution)
  SWAP      : for 2blob, optionally add batched swap-MC sweeps (exchange 2 particles' species, MH at pi_lam)
Tempered path pi_lam ~ q0^(1-lam) e^(-lam beta U) (quartic schedule); exact geometric-bridge MH mutations
(full-q0 rescore at lam<1 -- ka3d_smc_bridge). J islands recombined by exp(logZ_j). Metrics per (R,config):
overlap q(R) (species-min-dist<0.3 vs data interior), logZ_j spread, mean ESS, wall-time. INCREMENTAL save
per (config, cavity) -- durability. Compares which mutation kernel samples PTS best/cheapest."""
import argparse, time, sys, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept

dev = "cuda"; RCTX = 2.5; BIGL = 100.0; BETA = 2.0
SIG = torch.tensor(SIGMA, device=dev)
ART = "liquid_coupling_flow/artifacts"


def energy_b(Xb, Sb, bnd, sb):
    Mn, mm = Xb.shape[0], bnd.shape[0]
    x = torch.cat([Xb, bnd[None].expand(Mn, mm, 3)], 1); s = torch.cat([Sb, sb[None].expand(Mn, mm)], 1)
    return ka_energy(x.double(), s.long(), BIGL).float()


def ess(logw):
    w = torch.softmax(logw, 0); return float(1.0 / (w * w).sum() / len(logw))


def blob_mask(n, K, a, gen, two=False):
    mk = torch.zeros(n, dtype=torch.bool, device=dev)
    if not two:
        seed = int(torch.randint(n, (), generator=gen, device=dev))
        mk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    else:
        s1 = int(torch.randint(n, (), generator=gen, device=dev))
        far = (a - a[s1]).norm(dim=-1)
        s2 = int(far.topk(min(n, n), largest=True).indices[torch.randint(min(8, n), (), generator=gen, device=dev)])
        mk[(a - a[s1]).norm(dim=-1).topk(K // 2, largest=False).indices] = True
        mk[(a - a[s2]).norm(dim=-1).topk(K // 2, largest=False).indices] = True
    return mk


@torch.no_grad()
def swap_sweep(m, Xb, Sb, bnd, sb, R, lam, q0, Ecur, allm, gen, n_sw=3):
    """Batched swap-MC: each of M configs proposes swapping 2 random diff-species interior particles; MH at
    pi_lam (species change alters both U and q0 -> full q0 rescore). Returns updated Xb,Sb,q0,Ecur."""
    M, n, _ = Xb.shape
    for _ in range(n_sw):
        Sp = Sb.clone(); ok = torch.zeros(M, dtype=torch.bool, device=dev)
        ii = torch.zeros(M, dtype=torch.long, device=dev); jj = torch.zeros(M, dtype=torch.long, device=dev)
        for k in range(M):
            A = (Sb[k] == 0).nonzero().squeeze(1); B = (Sb[k] == 1).nonzero().squeeze(1)
            if len(A) == 0 or len(B) == 0:
                continue
            a_ = int(A[torch.randint(len(A), (), generator=gen, device=dev)])
            b_ = int(B[torch.randint(len(B), (), generator=gen, device=dev)])
            Sp[k, a_], Sp[k, b_] = Sb[k, b_], Sb[k, a_]; ok[k] = True; ii[k], jj[k] = a_, b_
        if not ok.any():
            continue
        En = energy_b(Xb, Sp, bnd, sb); q0n = m.block_log_prob_b(Xb, Sp, allm, bnd, sb, R)
        la = (1 - lam) * (q0n - q0) - lam * BETA * (En - Ecur)
        acc = (torch.rand(M, device=dev, generator=gen).log() < la) & ok
        Sb = torch.where(acc[:, None], Sp, Sb); Ecur = torch.where(acc, En, Ecur); q0 = torch.where(acc, q0n, q0)
    return Sb, q0, Ecur


@torch.no_grad()
def island(m, xo, so, bnd, sb, R, cfg, M, T, n_mut, gen):
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    Xb, Sb, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(), allm, bnd, sb, R, gen=gen)
    Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
    lam = torch.linspace(0, 1, T + 1, device=dev) ** 4
    a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    logw = torch.zeros(M, device=dev); logZ = 0.0; esses = []
    for t in range(1, T + 1):
        logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur - q0)
        esses.append(ess(logw))
        if ess(logw) < 0.5:
            logZ += float(torch.logsumexp(logw, 0)) - torch.log(torch.tensor(float(M)))
            w = torch.softmax(logw, 0); idx = torch.multinomial(w, M, replacement=True, generator=gen)
            Xb, Sb, Ecur, q0 = Xb[idx].clone(), Sb[idx].clone(), Ecur[idx].clone(), q0[idx].clone(); logw = torch.zeros(M, device=dev)
        lt = float(lam[t])
        for _ in range(n_mut):
            blk = blob_mask(n, cfg["K"], a, gen, two=(cfg["move"] == "2blob"))
            lqr = m.block_log_prob_b(Xb, Sb, blk, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xb, Sb, blk, bnd, sb, R, gen=gen)
            En = energy_b(Xn, Sn, bnd, sb); q0n = None if lt == 1.0 else m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
            la = geometric_bridge_log_accept(log_q0_current=q0, log_q0_proposed=q0n, energy_current=Ecur,
                                             energy_proposed=En, log_r_reverse=lqr, log_r_forward=lqf, lam=lt, beta=BETA)
            acc = torch.rand(M, device=dev, generator=gen).log() < la
            Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb)
            Ecur = torch.where(acc, En, Ecur)
            if q0n is not None:
                q0 = torch.where(acc, q0n, q0)
        if cfg["swap"]:
            Sb, q0, Ecur = swap_sweep(m, Xb, Sb, bnd, sb, R, lt, q0, Ecur, allm, gen)
    logZ += float(torch.logsumexp(logw, 0)) - torch.log(torch.tensor(float(M)))
    return Xb, Sb, logZ, esses


def overlap(Xp, Sp, xin, sin, aov=0.3):
    out = []
    for k in range(Xp.shape[0]):
        d = torch.cdist(Xp[k], xin).masked_fill(Sp[k][:, None] != sin[None, :], 9.0)
        out.append(float((d.min(1).values < aov).float().mean()))
    return st.mean(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[2.0, 2.5, 3.0])
    ap.add_argument("--islands", type=int, default=4); ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--T", type=int, default=14); ap.add_argument("--n-mut", type=int, default=3)
    ap.add_argument("--ncav", type=int, default=6); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--ckpt", default=f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_demand_knn24_best.pt")
    ap.add_argument("--out", default="reports/logs-2026-07-14/smc_sweep.pt")
    a = ap.parse_args()
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
    m.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["state_dict"], strict=False)
    m.eval(); m.use_frame = False
    D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
    X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
    configs = [{"move": "1blob", "K": a.K, "swap": False, "name": f"1blob-K{a.K}"},
               {"move": "2blob", "K": a.K, "swap": False, "name": f"2blob-K{a.K}"},
               {"move": "2blob", "K": a.K, "swap": True, "name": f"2blob-K{a.K}+swap"}]
    print(f"SMC sweep: radii={tuple(a.radii)} configs={[c['name'] for c in configs]} J={a.islands} M={a.m} "
          f"T={a.T} ncav={a.ncav} ckpt={a.ckpt.split('/')[-1]}", flush=True)
    results = {}
    for R in a.radii:
        for cfg in configs:
            key = (R, cfg["name"]); t0 = time.time(); rows = []
            gen = torch.Generator(device=dev).manual_seed(0); ncav = 0
            for ci in range(16):
                c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
                if p["n_in"] < a.K + 6:
                    continue
                xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
                xin = _mic(p["x_in"], c, L); sin = p["s_in"]
                xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
                lz, qs, es = [], [], []
                for j in range(a.islands):
                    g = torch.Generator(device=dev).manual_seed(1000 + 31 * ci + j)
                    Xp, Sp, logZ, esses = island(m, xo, so, bnd, sb, R, cfg, a.m, a.T, a.n_mut, g)
                    lz.append(logZ); qs.append(overlap(Xp, Sp, xin, sin)); es.append(st.mean(esses))
                lzt = torch.tensor(lz); wj = torch.softmax(lzt, 0)
                q_rw = float(sum(wj[j] * qs[j] for j in range(len(qs))))
                rows.append({"q_rw": q_rw, "logZ_spread": float(lzt.max() - lzt.min()), "ess": st.mean(es), "n_in": int(p["n_in"])})
                results[key] = rows; torch.save(results, a.out)                    # incremental per-cavity
                ncav += 1
                if ncav >= a.ncav:
                    break
            q = st.mean([r["q_rw"] for r in rows]); spr = st.mean([r["logZ_spread"] for r in rows]); es = st.mean([r["ess"] for r in rows])
            print(f"  R={R} {cfg['name']:>16}: q(R)={q:.3f}  logZ_spread={spr:6.1f}  ESS={es:.2f}  ({time.time()-t0:.0f}s, {ncav} cav)", flush=True)
    torch.save(results, a.out)
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
