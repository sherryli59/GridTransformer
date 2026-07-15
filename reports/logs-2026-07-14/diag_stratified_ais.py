"""STRATIFIED-SEED AIS: the basin-weighted PTS estimator. Exact scheme, no q~ in any weight:
  1. N=4096 iid q0 draws; strata by box-q~ vs the (known) reference -- deterministic f(draw) => exact
     stratified bookkeeping, NOT conditioning. Keep ALL candidates (q~>0.35), subsample mid/bulk.
     Initial log-weight of walker in stratum s: log[(N_s/N)/M_s].
  2. Tempered path pi_lam ~ q0^(1-lam) e^(-lam beta U), lam 0->1 quartic T=24; mutation = 2 suffix + 2 local
     per rung (the best within-basin relaxers); NO resampling (weights carried; removes the premature-cull
     failure). Weight increments dlam*(-beta*U - logq0) with q0 tracked through accepted moves.
  3. Estimate q(R) = sum w_i q~_i / sum w_i at lam=1; report ESS, per-stratum weight shares, and the
     data-seeded relaxed arm as the upper bracket (agreement = convergence certificate from below).
Ref anchors: PT-converged ~0.63 (different observable); data-seed suffix plateau 0.584. lam05 Rext, R=2.0,
cavities 0/1 (known p_true~1e-3)."""
import sys, functools, time
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b, blob_mask, ess
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; BETA = 2.0; L_BOX = 0.368; RHO = 1.149; BULK = RHO * L_BOX ** 3
K = 8; R = 2.0; POOL = 4096; BATCH = 512; T = 24; ART = "liquid_coupling_flow/artifacts"
THR_CAND, THR_MID = 0.35, 0.20; M_MID, M_BULK = 16, 32
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


def box_id_set(x, l, R):
    mm = x.norm(dim=-1) < R; ijk = torch.floor(x[mm] / l).long(); off = ijk - ijk.min(0).values
    span = off.max(0).values + 1
    return set((off[:, 0] * span[1] * span[2] + off[:, 1] * span[2] + off[:, 2]).tolist())


def qtil_one(a, xin, R):
    return len(box_id_set(a, L_BOX, R) & box_id_set(xin, L_BOX, R)) / (L_BOX ** 3 * n_boxes(L_BOX, R)) - BULK


def mutate(Xb, Sb, bnd, sb, R, lt, q0, Ecur, allm, a, gen, n):
    """2 suffix + 2 local geometric-bridge moves; returns updated state (q0 tracked)."""
    for kind in ("suf", "suf", "loc", "loc"):
        if kind == "suf":
            ks = int(torch.randint(n // 2, 3 * n // 4 + 1, (), generator=gen, device=dev))
            mask = torch.zeros(n, dtype=torch.bool, device=dev); mask[n - ks:] = True
        else:
            mask = blob_mask(n, K, a, gen)
        lqr = m.block_log_prob_b(Xb, Sb, mask, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xb, Sb, mask, bnd, sb, R, gen=gen)
        En = energy_b(Xn, Sn, bnd, sb); q0n = None if lt == 1.0 else m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
        la = geometric_bridge_log_accept(log_q0_current=q0, log_q0_proposed=q0n, energy_current=Ecur,
                                         energy_proposed=En, log_r_reverse=lqr, log_r_forward=lqf, lam=lt, beta=BETA)
        acc = torch.rand(Xb.shape[0], device=dev, generator=gen).log() < la
        Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb); Ecur = torch.where(acc, En, Ecur)
        if q0n is not None:
            q0 = torch.where(acc, q0n, q0)
        else:
            # lam=1 rung: q0 of accepted movers must still be tracked for consistency of later increments
            with torch.no_grad():
                q0u = m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
            q0 = torch.where(acc, q0u, q0)
    return Xb, Sb, q0, Ecur


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
    # --- 1. pool + strata ---
    g = torch.Generator(device=dev).manual_seed(500 + ci)
    allX, allS, allq = [], [], []
    for b0 in range(0, POOL, BATCH):
        nb = min(BATCH, POOL - b0)
        Xa, Sa, _ = m.sample_block_b(xo[None].expand(nb, n, 3).clone(), so[None].expand(nb, n).clone(),
                                     allm, bnd, sb, R, gen=g)
        Xc = Xa.cpu()
        allq += [qtil_one(Xc[k], xin.cpu(), R) for k in range(nb)]
        allX.append(Xc); allS.append(Sa.cpu())
    allX = torch.cat(allX); allS = torch.cat(allS); allq = torch.tensor(allq)
    s_cand = (allq > THR_CAND).nonzero().squeeze(1)
    s_mid = ((allq > THR_MID) & (allq <= THR_CAND)).nonzero().squeeze(1)
    s_bulk = (allq <= THR_MID).nonzero().squeeze(1)
    perm = torch.randperm(len(s_mid), generator=torch.Generator().manual_seed(1)); s_mid_k = s_mid[perm[:M_MID]]
    perm = torch.randperm(len(s_bulk), generator=torch.Generator().manual_seed(2)); s_bulk_k = s_bulk[perm[:M_BULK]]
    sel = torch.cat([s_cand, s_mid_k, s_bulk_k])
    lw0 = torch.cat([torch.full((len(s_cand),), float(torch.log(torch.tensor(len(s_cand) / POOL / max(len(s_cand), 1))))),
                     torch.full((len(s_mid_k),), float(torch.log(torch.tensor(len(s_mid) / POOL / max(len(s_mid_k), 1))))),
                     torch.full((len(s_bulk_k),), float(torch.log(torch.tensor(len(s_bulk) / POOL / max(len(s_bulk_k), 1)))))]).to(dev)
    strat = torch.cat([torch.zeros(len(s_cand)), torch.ones(len(s_mid_k)), 2 * torch.ones(len(s_bulk_k))]).long()
    Xb = allX[sel].to(dev); Sb = allS[sel].to(dev); Mw = len(sel)
    print(f"=== cav {ci} (n={n}) strata: cand {len(s_cand)}/{POOL}  mid {len(s_mid)}->{len(s_mid_k)}  "
          f"bulk {len(s_bulk)}->{len(s_bulk_k)}  walkers {Mw} ===", flush=True)
    # --- 2. AIS, no resampling ---
    Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
    lam = torch.linspace(0, 1, T + 1, device=dev) ** 4
    logw = lw0.double().clone()
    for t in range(1, T + 1):
        logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur.double() - q0.double())
        Xb, Sb, q0, Ecur = mutate(Xb, Sb, bnd, sb, R, float(lam[t]), q0, Ecur, allm, a, g, n)
    qf = torch.tensor([qtil_one(Xb[k].cpu(), xin.cpu(), R) for k in range(Mw)]).double()
    w = torch.softmax(logw.cpu() - logw.cpu().max(), 0)
    q_est = float((w * qf).sum()); es = float(1.0 / (w * w).sum())
    shares = [float(w[strat == s].sum()) for s in (0, 1, 2)]
    # --- 3. data-seed bracket (same mutation budget) ---
    Xd = xo[None].expand(32, n, 3).clone(); Sd = so[None].expand(32, n).clone()
    Ed = energy_b(Xd, Sd, bnd, sb); q0d = m.block_log_prob_b(Xd, Sd, allm, bnd, sb, R)
    for t in range(1, T + 1):
        Xd, Sd, q0d, Ed = mutate(Xd, Sd, bnd, sb, R, float(lam[t]), q0d, Ed, allm, a, g, n)
    qd = torch.tensor([qtil_one(Xd[k].cpu(), xin.cpu(), R) for k in range(32)])
    print(f"  q(R) STRATIFIED-AIS = {q_est:+.3f}   (ESS {es:.1f}/{Mw}; weight shares cand/mid/bulk = "
          f"{shares[0]:.2f}/{shares[1]:.2f}/{shares[2]:.2f})", flush=True)
    print(f"  data-seed bracket  = {float(qd.mean()):+.3f} +- {float(qd.std()):.3f}   [PT ref ~0.63 core-metric]", flush=True)
    print(f"  walker q~ range: min {float(qf.min()):+.3f} max {float(qf.max()):+.3f}  ({time.time()-t0:.0f}s)", flush=True)
    ncav += 1
    if ncav >= 2:
        break
