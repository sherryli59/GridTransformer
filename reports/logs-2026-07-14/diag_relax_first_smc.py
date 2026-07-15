"""RELAX-BEFORE-SELECT SMC (the redemption experiment). Basin mass p=3-9% => M=192 alien seeds contain
~6-17 reference-basin members; the old schedule discarded them because raw in-basin draws are CLASHY and
early raw-energy resampling prefers relaxed wrong-basin walkers. New schedule:
  1. draw M alien seeds (full q0 regens)
  2. RELAX each walker independently: n_relax local blob-MTM sweeps at a small fixed lambda_relax
     (energy pressure enough to declash, NO resampling, NO reweighting -- pure per-walker relaxation)
  3. THEN run the standard quartic anneal with reweight/resample/mutate.
Exactness note: step 2 changes the initial distribution (it is a fixed stochastic map applied to q0 draws);
the SMC afterwards is still a valid annealer to e^{-beta U} for the POPULATION (endpoint healed by
pi-invariant mutation + selection); we JUDGE by q~ vs the data-seed bracket, as everywhere in this campaign.
Compare endpoint alien q~: baseline (old schedule) vs relax-first, + track the q~ of the top-energy-ranked
walkers after relaxation (are ref-basin members now the deepest?). Also per-seed diagnostic: corr(q~,U) at
draw time vs after relaxation. lam05 Rext, R=2.0, l=0.368, M=192, cavities = the SAME first 4 as
diag_basin_mass (cav 2 = the hard one)."""
import sys, statistics as st, functools, time
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
K = 8; R = 2.0; M = 192; T = 20; NMUT = 2; ART = "liquid_coupling_flow/artifacts"
LAM_RELAX = 0.08; N_RELAX = 12
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


def qtil(Xb, xin, R):
    ref = box_id_set(xin, L_BOX, R); Nb = n_boxes(L_BOX, R)
    Xc = Xb.cpu()
    return torch.tensor([len(ref & box_id_set(Xc[k], L_BOX, R)) / (L_BOX ** 3 * Nb) - BULK
                         for k in range(Xb.shape[0])])


def local_move(Xb, Sb, bnd, sb, R, lt, q0, Ecur, allm, a, gen, n):
    blk = blob_mask(n, K, a, gen)
    lqr = m.block_log_prob_b(Xb, Sb, blk, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xb, Sb, blk, bnd, sb, R, gen=gen)
    En = energy_b(Xn, Sn, bnd, sb); q0n = None if lt == 1.0 else m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
    la = geometric_bridge_log_accept(log_q0_current=q0, log_q0_proposed=q0n, energy_current=Ecur,
                                     energy_proposed=En, log_r_reverse=lqr, log_r_forward=lqf, lam=lt, beta=BETA)
    acc = torch.rand(Xb.shape[0], device=dev, generator=gen).log() < la
    Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb); Ecur = torch.where(acc, En, Ecur)
    if q0n is not None:
        q0 = torch.where(acc, q0n, q0)
    return Xb, Sb, q0, Ecur


def run(ci_target, mode, gen_master):
    gen = torch.Generator(device=dev).manual_seed(0); found = 0
    for ci in range(24):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < 14:
            continue
        if found == ci_target:
            break
        found += 1
    else:
        return None
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xin = _mic(p["x_in"], c, L)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    g = torch.Generator(device=dev).manual_seed(1000 + ci_target)
    Xb, Sb, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(), allm, bnd, sb, R, gen=g)
    Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
    a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    diag = {}
    q_seed = qtil(Xb, xin, R)
    diag["corr_draw"] = float(torch.corrcoef(torch.stack([q_seed.to(dev), Ecur.float()]))[0, 1])
    diag["hits_seed"] = int((q_seed > 0.3).sum())
    if mode == "relax_first":
        for _ in range(N_RELAX):
            Xb, Sb, q0, Ecur = local_move(Xb, Sb, bnd, sb, R, LAM_RELAX, q0, Ecur, allm, a, g, n)
        q_rel = qtil(Xb, xin, R)
        diag["corr_relaxed"] = float(torch.corrcoef(torch.stack([q_rel.to(dev), Ecur.float()]))[0, 1])
        diag["hits_relaxed"] = int((q_rel > 0.3).sum())
        # are the deepest-energy walkers now ref-basin? mean q~ of the best-32 by energy
        best = Ecur.topk(32, largest=False).indices
        diag["q_of_deepest32"] = float(q_rel[best.cpu()].mean())
    lam = torch.linspace(0, 1, T + 1, device=dev) ** 4
    logw = torch.zeros(M, device=dev)
    for t in range(1, T + 1):
        logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur - q0)
        if ess(logw) < 0.5:
            wt = torch.softmax(logw, 0); idx = torch.multinomial(wt, M, replacement=True, generator=g)
            Xb, Sb, Ecur, q0 = Xb[idx].clone(), Sb[idx].clone(), Ecur[idx].clone(), q0[idx].clone(); logw = torch.zeros(M, device=dev)
        lt = float(lam[t])
        for _ in range(NMUT):
            Xb, Sb, q0, Ecur = local_move(Xb, Sb, bnd, sb, R, lt, q0, Ecur, allm, a, g, n)
    q_end = qtil(Xb, xin, R)
    diag["q_end"] = float(q_end.mean()); diag["q_end_max"] = float(q_end.max())
    return diag


print(f"=== RELAX-BEFORE-SELECT SMC (M={M}, N_relax={N_RELAX}@lam={LAM_RELAX}, R={R}) ===", flush=True)
print(f"basin mass p per cavity: 0.089/0.050/0.0004/0.030 -> seeds contain ~17/10/0/6 hits", flush=True)
print(f"{'cav':>4} {'mode':>12} | {'hits_seed':>9} {'hits_relax':>10} {'corr_draw':>9} {'corr_relax':>10} "
      f"{'q_deep32':>8} | {'q_end':>7} {'q_max':>7}", flush=True)
gm = torch.Generator(device=dev).manual_seed(7)
for ci in range(4):
    for mode in ("baseline", "relax_first"):
        t0 = time.time(); d = run(ci, mode, gm)
        if d is None:
            continue
        print(f"{ci:>4} {mode:>12} | {d['hits_seed']:>9} {d.get('hits_relaxed','-'):>10} "
              f"{d['corr_draw']:>9.2f} {d.get('corr_relaxed', float('nan')):>10.2f} "
              f"{d.get('q_of_deepest32', float('nan')):>8.3f} | {d['q_end']:>7.3f} {d['q_end_max']:>7.3f}  ({time.time()-t0:.0f}s)", flush=True)
