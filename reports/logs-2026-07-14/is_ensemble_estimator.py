"""INHERENT-STRUCTURE ENSEMBLE ESTIMATOR (v4) -- the NO-PT estimator; the learned model IS the engine.
Z = sum_b Z_b exactly (configuration space partitions under the quench map). Per cavity:
  1. DISCOVER: pool of energy-free AR draws (+ the reference), prescreen (top box-q~ + deepest raw U +
     random), QUENCH candidates (displacement-capped descent) -> basin labels by U_IS + IS-overlap clustering.
  2. WEIGH classically per basin: log w_b = -beta*U_IS,b - 0.5*sum_i ln(lambda_i)  (harmonic F_vib from ONE
     autodiff Hessian at the IS; temperature-only factors are basin-independent at fixed n and cancel).
  3. MEASURE q~_b: short within-basin displacement MC at beta (thermalize 30 sweeps, average 20).
  4. COMBINE: q(R) = sum_b w_b q~_b / sum w_b. No PT, no exchange, no annealing path, no extensive-weight
     curse (free energies are LOCAL computations). Honest caveats: basin completeness (report discovery
     saturation), harmonic approximation (TI check later), basin count grows with R (fine at R<xi = the PTS
     regime). Validation: v3 PT brackets + Hocky anchors (0.33@R2.2, 0.29@R2.4, rho1.2 T0.55).
Usage: python is_ensemble_estimator.py [R]. Observable: exact Hocky convention (fixed grid, matched l)."""
import sys, functools, time, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; BETA = 2.0; RHO = 1.149; BIGL = 100.0
R = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
POOL = 4096; BATCH = 512; NQ = 96; QSTEPS = 800; ART = "liquid_coupling_flow/artifacts"
L_HOCKY = (0.06 / RHO) ** (1.0 / 3.0); BULK_HOCKY = 0.06
MERGE_Q = 0.55; MERGE_DU = 0.08          # basin identity: IS-overlap>MERGE_Q OR |dU_IS|/n<MERGE_DU joins
NW = 8; SW_TH, SW_AV = 30, 20; DISP = 0.10
OUT = f"reports/logs-2026-07-14/is_ensemble_R{R}.pt"
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


def quench(Xb, Sb, bnd, sb, steps=QSTEPS, dmax=0.05):
    Mn, n, _ = Xb.shape; mm = bnd.shape[0]
    xi = Xb.double().clone(); bndd = bnd.double()
    sfull = torch.cat([Sb, sb[None].expand(Mn, mm)], 1).long()
    with torch.enable_grad():
        for t in range(steps):
            xi = xi.detach().requires_grad_(True)
            U = ka_energy(torch.cat([xi, bndd[None].expand(Mn, mm, 3)], 1), sfull, BIGL)
            (gr,) = torch.autograd.grad(U.sum(), xi)
            step = dmax * (0.2 + 0.8 * (1 - t / steps))
            gn = gr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            xi = (xi - gr / gn * torch.minimum(gn * 1e-3, torch.full_like(gn, step))).detach()
    gfin = gr.norm(dim=-1).mean(-1)
    return (ka_energy(torch.cat([xi, bndd[None].expand(Mn, mm, 3)], 1), sfull, BIGL).float(),
            xi.float(), gfin.float())


def hessian_logdet(x_is, s_in, bnd, sb):
    """0.5*sum ln lambda_i of the interior Hessian at the IS (f64). Returns (val, n_neg)."""
    n = x_is.shape[0]; mm = bnd.shape[0]
    bndd = bnd.double(); sfull = torch.cat([s_in[None], sb[None].expand(1, mm)], 1).long()
    def en(flat):
        xi = flat.view(1, n, 3)
        return ka_energy(torch.cat([xi, bndd[None]], 1), sfull, BIGL)[0]
    with torch.enable_grad():
        H = torch.autograd.functional.hessian(en, x_is.double().flatten(), vectorize=True)
    ev = torch.linalg.eigvalsh(0.5 * (H + H.T))
    n_neg = int((ev < 1e-8).sum())
    ev = ev.clamp_min(1e-8)
    return 0.5 * float(torch.log(ev).sum()), n_neg


def within_basin_q(x_is, s_in, bnd, sb, xin, gen):
    """NW walkers from the IS: SW_TH thermalize + SW_AV averaging displacement sweeps at beta."""
    n = x_is.shape[0]
    Xb = x_is[None].expand(NW, n, 3).clone().float(); Sb = s_in[None].expand(NW, n).clone()
    U = energy_b(Xb, Sb, bnd, sb)
    qs = []
    for sw in range(SW_TH + SW_AV):
        for i in torch.randperm(n, generator=gen, device=dev).tolist():
            prop = Xb.clone()
            prop[:, i] = Xb[:, i] + DISP * torch.randn(NW, 3, device=dev, generator=gen)
            ok = prop[:, i].norm(dim=-1) < R
            En = energy_b(prop, Sb, bnd, sb)
            acc = ok & (torch.rand(NW, device=dev, generator=gen).log() < -BETA * (En - U))
            Xb = torch.where(acc[:, None, None], prop, Xb); U = torch.where(acc, En, U)
        if sw >= SW_TH:
            qs += [q_hocky(Xb[k].cpu(), xin.cpu(), R) - BULK_HOCKY for k in range(NW)]
    return st.mean(qs), st.pstdev(qs)


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
    n = xo.shape[0]
    g = torch.Generator(device=dev).manual_seed(1100 + ci)
    # 1. DISCOVER
    allX, allS, allq, allU = [], [], [], []
    allm = torch.ones(n, dtype=torch.bool, device=dev)
    for b0 in range(0, POOL, BATCH):
        nb = min(BATCH, POOL - b0)
        Xa, Sa, _ = m.sample_block_b(xo[None].expand(nb, n, 3).clone(), so[None].expand(nb, n).clone(),
                                     allm, bnd, sb, R, gen=g)
        allU.append(energy_b(Xa, Sa, bnd, sb).cpu())
        Xc = Xa.cpu(); allq += [q_hocky(Xc[k], xin.cpu(), R) - BULK_HOCKY for k in range(nb)]
        allX.append(Xc); allS.append(Sa.cpu())
    allX = torch.cat(allX); allS = torch.cat(allS); allU = torch.cat(allU); allq = torch.tensor(allq)
    n_top, n_deep = NQ // 3, NQ // 3
    sel = torch.unique(torch.cat([allq.topk(n_top).indices, (-allU).topk(n_deep).indices,
                                  torch.randperm(POOL)[:NQ - n_top - n_deep]]))
    Xq = torch.cat([xo[None].cpu(), allX[sel]]); Sq = torch.cat([so[None].cpu(), allS[sel]])   # ref first
    Uis, Xis, gfin = quench(Xq.to(dev), Sq.to(dev), bnd, sb)
    # 2. CLUSTER basins (union-find on IS-overlap / dU)
    NQm = len(Xq); par = list(range(NQm))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    Xis_c = Xis.cpu()
    for i in range(NQm):
        for j in range(i + 1, NQm):
            if abs(float(Uis[i] - Uis[j])) / n < MERGE_DU and \
               q_hocky(Xis_c[i], Xis_c[j], R) - BULK_HOCKY > MERGE_Q:
                par[find(i)] = find(j)
    roots = {}
    for i in range(NQm):
        roots.setdefault(find(i), []).append(i)
    # 3+4. per-basin weight + observable
    rows = []
    for root, members in roots.items():
        rep = members[int(torch.tensor([float(Uis[k]) for k in members]).argmin())]  # deepest member = rep
        hld, n_neg = hessian_logdet(Xis[rep].to(dev), Sq[rep].to(dev), bnd, sb)
        logw = -BETA * float(Uis[rep]) - hld
        qb, qb_sd = within_basin_q(Xis[rep].to(dev), Sq[rep].to(dev), bnd, sb, xin, g)
        rows.append({"U_is_n": float(Uis[rep]) / n, "logw": logw, "q_b": qb, "q_b_sd": qb_sd,
                     "n_members": len(members), "has_ref": 0 in members, "n_neg_ev": n_neg,
                     "grad_fin": float(gfin[rep])})
    lws = torch.tensor([r["logw"] for r in rows]).double()
    w = torch.softmax(lws - lws.max(), 0)
    q_est = float(sum(float(w[i]) * rows[i]["q_b"] for i in range(len(rows))))
    rows_sorted = sorted(range(len(rows)), key=lambda i: -float(w[i]))
    print(f"=== cav {ci} (n={n}, R={R}) IS-ENSEMBLE: {len(rows)} basins from {NQm} quenches ===", flush=True)
    print(f"{'w_b':>7} {'U_IS/n':>8} {'q_b':>7} {'members':>7} {'ref?':>5} {'negEV':>5}", flush=True)
    for i in rows_sorted[:8]:
        r_ = rows[i]
        print(f"{float(w[i]):>7.3f} {r_['U_is_n']:>8.3f} {r_['q_b']:>+7.3f} {r_['n_members']:>7} "
              f"{'REF' if r_['has_ref'] else '':>5} {r_['n_neg_ev']:>5}", flush=True)
    print(f"  q(R={R}) IS-ENSEMBLE = {q_est:+.3f}   (ref-basin weight {sum(float(w[i]) for i in range(len(rows)) if rows[i]['has_ref']):.3f}; "
          f"quench |grad| {float(gfin.mean()):.2e}; {time.time()-t0:.0f}s)", flush=True)
    results[ci] = {"rows": rows, "q_est": q_est, "pool_q": allq, "sel": sel,
                   "Uis": Uis.cpu(), "wall_s": time.time() - t0}
    torch.save(results, OUT)
    ncav += 1
    if ncav >= 2:
        break
print(f"saved -> {OUT}", flush=True)
print("VALIDATE vs v3 PT brackets (R2.0: [0.393,0.797] cav0, [0.403,0.515] cav1) + Hocky anchors.", flush=True)
