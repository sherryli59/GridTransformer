"""BASIN-STRATIFIED SMC -- the learned-sampler PRODUCT (PT = baseline only). Exact decomposition:
  E_q0[W f] = sum_s mass_s E_{q0|s}[W f],   W = AIS weight over the tempered path pi_lam ~ q0^(1-lam) e^(-lam beta U)
Each stratum s (deterministic box-q~ bins of the pool: cand/mid/bulk -- presence of the rare basin family
GUARANTEED by construction) runs its OWN SMC from a uniform subsample of its members: reweight dlam*(-beta*U
- logq0), ESS<0.5 -> resample WITHIN the stratum (controls weight variance WITHOUT cross-stratum culling --
the two failure modes of 2026-07-14 both removed), mutation = suffix + local AR geometric-bridge moves;
at lam=1: displacement polish sweeps (q0 term gone) + averaging. Recombine:
  q(R) = sum_s mass_s Zhat_s qbar_s / sum_s mass_s Zhat_s
(Zhat_s = stratum SMC normalizer, qbar_s = endpoint weighted mean). Risk stated: Zhat_s precision at O(1)
nat across strata -> validated against the PT baseline brackets + Hocky anchors. Usage: [R] argv."""
import sys, functools, time, statistics as st, math
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b, blob_mask, ess
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; BETA = 2.0; RHO = 1.149
R = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
T_ARG = int(sys.argv[3]) if len(sys.argv) > 3 else 24
POOL = 4096; BATCH = 512; K = 8; ART = "liquid_coupling_flow/artifacts"
L_HOCKY = (0.06 / RHO) ** (1.0 / 3.0); BULK_HOCKY = 0.06
THR_CAND, THR_MID = 0.30, 0.15; M_CAND, M_MID, M_BULK = 24, 16, 24
NMUT = 3; POLISH = 20; AVG = 15; DISP = 0.10
T = T_ARG
OUT = f"reports/logs-2026-07-14/smc_stratified_R{R}_s{SEED}_T{T_ARG}.pt"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


@functools.lru_cache(maxsize=16)
def n_boxes(l, R):
    g = torch.arange(-int(R / l) - 1, int(R / l) + 2) * l + l / 2
    gx, gy, gz = torch.meshgrid(g, g, g, indexing="ij")
    return int((gx ** 2 + gy ** 2 + gz ** 2 < R * R).sum())


def box_set(x, l, R):
    mm = x.norm(dim=-1) < R; ijk = torch.floor(x[mm] / l).long()
    h = int(R / l) + 2; side = 2 * h + 1
    return set(((ijk[:, 0] + h) * side * side + (ijk[:, 1] + h) * side + (ijk[:, 2] + h)).tolist())


def q_hocky(a, b, R):
    return len(box_set(a, L_HOCKY, R) & box_set(b, L_HOCKY, R)) / (L_HOCKY ** 3 * n_boxes(L_HOCKY, R))


@torch.no_grad()
def stratum_smc(Xb, Sb, bnd, sb, R, n, a, xin, gen):
    """One stratum's tempered SMC. Returns (logZ_s, qbar_s, q_sd, ess_min)."""
    Mw = Xb.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
    lam = torch.linspace(0, 1, T + 1, device=dev) ** 4
    logw = torch.zeros(Mw, device=dev, dtype=torch.float64); logZ = 0.0; ess_min = 1.0
    for t in range(1, T + 1):
        logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur.double() - q0.double())
        ef = ess(logw) / Mw; ess_min = min(ess_min, ef)
        if ef < 0.5:
            logZ += float(torch.logsumexp(logw, 0)) - math.log(Mw)
            w = torch.softmax(logw, 0).float()
            idx = torch.multinomial(w, Mw, replacement=True, generator=gen)
            Xb, Sb, Ecur, q0 = Xb[idx].clone(), Sb[idx].clone(), Ecur[idx].clone(), q0[idx].clone()
            logw = torch.zeros(Mw, device=dev, dtype=torch.float64)
        lt = float(lam[t])
        for mv in range(NMUT):
            if mv == 0:
                ks = int(torch.randint(n // 2, 3 * n // 4 + 1, (), generator=gen, device=dev))
                mask = torch.zeros(n, dtype=torch.bool, device=dev); mask[n - ks:] = True
            else:
                mask = blob_mask(n, K, a, gen)
            lqr = m.block_log_prob_b(Xb, Sb, mask, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xb, Sb, mask, bnd, sb, R, gen=gen)
            En = energy_b(Xn, Sn, bnd, sb); q0n = None if lt == 1.0 else m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
            la = geometric_bridge_log_accept(log_q0_current=q0, log_q0_proposed=q0n, energy_current=Ecur,
                                             energy_proposed=En, log_r_reverse=lqr, log_r_forward=lqf, lam=lt, beta=BETA)
            acc = torch.rand(Mw, device=dev, generator=gen).log() < la
            Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb); Ecur = torch.where(acc, En, Ecur)
            if q0n is not None:
                q0 = torch.where(acc, q0n, q0)
    logZ += float(torch.logsumexp(logw, 0)) - math.log(Mw)
    wend = torch.softmax(logw, 0)                                        # endpoint weights within stratum
    # lam=1 polish + averaging with displacement sweeps (pi_1 = e^{-beta U}; q0 gone)
    qs_acc = torch.zeros(Mw, dtype=torch.float64); n_acc = 0
    for sw in range(POLISH + AVG):
        for i in torch.randperm(n, generator=gen, device=dev).tolist():
            prop = Xb.clone()
            prop[:, i] = Xb[:, i] + DISP * torch.randn(Mw, 3, device=dev, generator=gen)
            ok = prop[:, i].norm(dim=-1) < R
            En = energy_b(prop, Sb, bnd, sb)
            acc = ok & (torch.rand(Mw, device=dev, generator=gen).log() < -BETA * (En - Ecur))
            Xb = torch.where(acc[:, None, None], prop, Xb); Ecur = torch.where(acc, En, Ecur)
        if sw >= POLISH:
            qs_acc += torch.tensor([q_hocky(Xb[k].cpu(), xin.cpu(), R) - BULK_HOCKY for k in range(Mw)]).double()
            n_acc += 1
    qf = qs_acc / n_acc
    qbar = float((wend.double().cpu() * qf).sum())
    return logZ, qbar, float(qf.std()), ess_min


results = {}
gen = torch.Generator(device=dev).manual_seed(0); ncav = 0
for ci in range(24):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < 14:
        continue
    t0 = time.time()
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xin = _mic(p["x_in"], c, L)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    g = torch.Generator(device=dev).manual_seed(1300 + ci + 7777 * SEED)
    allX, allS, allq = [], [], []
    for b0 in range(0, POOL, BATCH):
        nb = min(BATCH, POOL - b0)
        Xa, Sa, _ = m.sample_block_b(xo[None].expand(nb, n, 3).clone(), so[None].expand(nb, n).clone(),
                                     allm, bnd, sb, R, gen=g)
        Xc = Xa.cpu(); allq += [q_hocky(Xc[k], xin.cpu(), R) - BULK_HOCKY for k in range(nb)]
        allX.append(Xc); allS.append(Sa.cpu())
    allX = torch.cat(allX); allS = torch.cat(allS); allq = torch.tensor(allq)
    strata = {"cand": (allq > THR_CAND).nonzero().squeeze(1),
              "mid": ((allq > THR_MID) & (allq <= THR_CAND)).nonzero().squeeze(1),
              "bulk": (allq <= THR_MID).nonzero().squeeze(1)}
    caps = {"cand": M_CAND, "mid": M_MID, "bulk": M_BULK}
    print(f"=== cav {ci} (n={n}, R={R}) STRATIFIED SMC: sizes "
          f"{ {k: int(len(v)) for k, v in strata.items()} } ===", flush=True)
    srows = {}
    for sname, sidx in strata.items():
        if len(sidx) == 0:
            continue
        Ms = min(caps[sname], len(sidx))
        pick = sidx[torch.randperm(len(sidx), generator=torch.Generator().manual_seed(3 + SEED))[:Ms]]
        lz, qb, qsd, emin = stratum_smc(allX[pick].to(dev), allS[pick].to(dev), bnd, sb, R, n, a, xin, g)
        srows[sname] = {"mass": len(sidx) / POOL, "M": Ms, "logZ": lz, "qbar": qb, "q_sd": qsd, "ess_min": emin}
        print(f"  {sname:>5}: mass {len(sidx)/POOL:.4f}  M {Ms:>3}  logZ {lz:+9.2f}  qbar {qb:+.3f}+-{qsd:.3f}  "
              f"ess_min {emin:.2f}", flush=True)
    lw = torch.tensor([math.log(r["mass"]) + r["logZ"] for r in srows.values()]).double()
    wgt = torch.softmax(lw - lw.max(), 0)
    q_est = float(sum(float(wgt[i]) * r["qbar"] for i, r in enumerate(srows.values())))
    print(f"  q(R={R}) STRATIFIED-SMC = {q_est:+.3f}   stratum weights "
          f"{ {k: round(float(wgt[i]), 3) for i, k in enumerate(srows)} }  ({time.time()-t0:.0f}s)", flush=True)
    results[ci] = {"strata": srows, "q_est": q_est, "weights": {k: float(wgt[i]) for i, k in enumerate(srows)},
                   "wall_s": time.time() - t0}
    torch.save(results, OUT)
    ncav += 1
    if ncav >= 2:
        break
print(f"saved -> {OUT}", flush=True)
print("VALIDATE vs PT baseline brackets (R2.0 cav0 [0.393,0.797], cav1 [0.403,0.515]; R2.2 cav0 [0.278,0.649])"
      " and Hocky 0.33@R2.2 / 0.29@R2.4.", flush=True)
