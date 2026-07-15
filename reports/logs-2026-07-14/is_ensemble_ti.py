"""BASIN-LOCAL FREE-ENERGY ESTIMATOR (the fix): discovery by the AR model, weighing by LOCAL thermodynamics.
No q0 in any weight, no tempered path, no resampling, no exchange. Per cavity:
  1. DISCOVER: AR pool (4096) -> prescreen (top box-q~ + deepest raw U + random) + reference -> capped quench
     -> union-find basins (IS-overlap>0.55 AND |dU_IS|/n<0.08).
  2. WEIGH each basin b LOCALLY: beta*F_b = beta*U_IS + 0.5*sum ln(lambda_i)  [harmonic, one autodiff Hessian]
     + beta*F_anh  [ANHARMONIC TI: H_k = U_IS + (1-k)*harmonic + k*(U-U_IS); F_anh = int_0^1 <U - U_IS -
     harm>_k dk, 8-point trapezoid x (50 burn + 100 avg) displacement sweeps, warm-started in k -- WITHIN a
     basin there are no barriers so this converges like a liquid TI; sub-nat noise replaces the ~250-nat
     AIS-normalizer lottery].
  3. OBSERVE q~_b at the k=1 window (true-U within-basin MC; Hocky fixed grid, matched l).
  4. COMBINE q(R) = sum_b w_b q~_b / sum_b w_b, w_b = exp(-beta*U_IS - 0.5*sum ln lambda - beta*F_anh).
  Diagnostics: discovery saturation (pool members/basin), TI integrand values (smoothness), quench |grad|,
  negative-eigenvalue count. Referees: BCY-certified 0.49+-0.04 (cav1), overnight verbatim run (cav0).
Usage: [R] argv."""
import sys, functools, time, statistics as st, math
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
MERGE_Q = 0.55; MERGE_DU = 0.08
NW = 8; NK = 8; TI_BURN, TI_AVG = 50, 100; DISP = 0.08
OUT = f"reports/logs-2026-07-14/is_ensemble_ti_R{R}.pt"
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
    return (ka_energy(torch.cat([xi, bndd[None].expand(Mn, mm, 3)], 1), sfull, BIGL).float(),
            xi.float(), gr.norm(dim=-1).mean(-1).float())


def hessian_logdet(x_is, s_in, bnd, sb):
    n = x_is.shape[0]; mm = bnd.shape[0]
    bndd = bnd.double(); sfull = torch.cat([s_in[None], sb[None].expand(1, mm)], 1).long()
    def en(flat):
        return ka_energy(torch.cat([flat.view(1, n, 3), bndd[None]], 1), sfull, BIGL)[0]
    with torch.enable_grad():
        H = torch.autograd.functional.hessian(en, x_is.double().flatten(), vectorize=True)
    H = 0.5 * (H + H.T)
    ev = torch.linalg.eigvalsh(H)
    n_neg = int((ev < 1e-8).sum())
    return 0.5 * float(torch.log(ev.clamp_min(1e-8)).sum()), n_neg, H


def ti_and_observe(x_is, s_in, bnd, sb, H, U_is, xin, gen):
    """Anharmonic TI + k=1 observable. H_k = U_IS + (1-k)*0.5 d^T H d + k*(U - U_IS).
    Returns (beta*F_anh, integrand list, q~_b, q_sd)."""
    n = x_is.shape[0]
    Xb = x_is[None].expand(NW, n, 3).clone().float()
    Sb = s_in[None].expand(NW, n).clone()
    Hf = H.float()
    def harm(Xc):
        d = (Xc - x_is[None]).reshape(NW, -1)
        return 0.5 * torch.einsum('bi,ij,bj->b', d, Hf, d)
    U = energy_b(Xb, Sb, bnd, sb); Hm = harm(Xb)
    integrand = []; qs = []
    ks = torch.linspace(0, 1, NK)
    for ki, k in enumerate(ks.tolist()):
        acc_int = []
        for sw in range(TI_BURN + TI_AVG):
            for i in torch.randperm(n, generator=gen, device=dev).tolist():
                prop = Xb.clone()
                prop[:, i] = Xb[:, i] + DISP * torch.randn(NW, 3, device=dev, generator=gen)
                ok = prop[:, i].norm(dim=-1) < R
                Up = energy_b(prop, Sb, bnd, sb); Hp = harm(prop)
                dE = (1 - k) * (Hp - Hm) + k * (Up - U)
                acc = ok & (torch.rand(NW, device=dev, generator=gen).log() < -BETA * dE)
                Xb = torch.where(acc[:, None, None], prop, Xb)
                U = torch.where(acc, Up, U); Hm = torch.where(acc, Hp, Hm)
            if sw >= TI_BURN:
                acc_int.append(float((U - U_is - Hm).mean()))
                if ki == NK - 1:
                    qs += [q_hocky(Xb[w].cpu(), xin.cpu(), R) - BULK_HOCKY for w in range(NW)]
        integrand.append(st.mean(acc_int))
    F_anh = float(torch.trapz(torch.tensor(integrand), ks))                    # <U - U_IS - harm>_k dk
    return BETA * F_anh, integrand, (st.mean(qs) if qs else float("nan")), (st.pstdev(qs) if qs else 0.0)


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
    g = torch.Generator(device=dev).manual_seed(2100 + ci)
    # 1. DISCOVER
    allX, allS, allq, allU = [], [], [], []
    for b0 in range(0, POOL, BATCH):
        nb = min(BATCH, POOL - b0)
        Xa, Sa, _ = m.sample_block_b(xo[None].expand(nb, n, 3).clone(), so[None].expand(nb, n).clone(),
                                     allm, bnd, sb, R, gen=g)
        allU.append(energy_b(Xa, Sa, bnd, sb).cpu())
        Xc = Xa.cpu(); allq += [q_hocky(Xc[k], xin.cpu(), R) - BULK_HOCKY for k in range(nb)]
        allX.append(Xc); allS.append(Sa.cpu())
    allX = torch.cat(allX); allS = torch.cat(allS); allU = torch.cat(allU); allq = torch.tensor(allq)
    nt = NQ // 3
    sel = torch.unique(torch.cat([allq.topk(nt).indices, (-allU).topk(nt).indices,
                                  torch.randperm(POOL)[:NQ - 2 * nt]]))
    Xq = torch.cat([xo[None].cpu(), allX[sel]]); Sq = torch.cat([so[None].cpu(), allS[sel]])
    Uis, Xis, gfin = quench(Xq.to(dev), Sq.to(dev), bnd, sb)
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
    print(f"=== cav {ci} (n={n}, R={R}) TI-ENSEMBLE: {len(roots)} basins from {NQm} quenches "
          f"(quench |grad| {float(gfin.mean()):.2e}) ===", flush=True)
    # 2-3. per-basin F_b + q~_b
    rows = []
    for root, members in roots.items():
        rep = members[int(torch.tensor([float(Uis[k]) for k in members]).argmin())]
        hld, n_neg, H = hessian_logdet(Xis[rep].to(dev), Sq[rep].to(dev), bnd, sb)
        bFanh, integrand, qb, qsd = ti_and_observe(Xis[rep].to(dev), Sq[rep].to(dev), bnd, sb, H,
                                                   float(Uis[rep]), xin, g)
        logw = -BETA * float(Uis[rep]) - hld - bFanh
        rows.append({"U_is_n": float(Uis[rep]) / n, "hld": hld, "bFanh": bFanh, "logw": logw,
                     "q_b": qb, "q_b_sd": qsd, "n_members": len(members), "has_ref": 0 in members,
                     "n_neg": n_neg, "ti_integrand": integrand})
        print(f"   basin U_IS/n {float(Uis[rep])/n:+.3f} members {len(members):>3} {'REF' if 0 in members else '   '} "
              f"negEV {n_neg} bF_anh {bFanh:+8.2f} q_b {qb:+.3f}+-{qsd:.3f} "
              f"TI[{integrand[0]:+.2f}..{integrand[-1]:+.2f}]", flush=True)
    lws = torch.tensor([r["logw"] for r in rows]).double()
    w = torch.softmax(lws - lws.max(), 0)
    q_est = float(sum(float(w[i]) * rows[i]["q_b"] for i in range(len(rows))))
    wref = sum(float(w[i]) for i in range(len(rows)) if rows[i]["has_ref"])
    print(f"  q(R={R}) TI-ENSEMBLE = {q_est:+.3f}   (ref-basin weight {wref:.3f}; "
          f"basin weights {[round(float(x),3) for x in w.tolist()]}; {time.time()-t0:.0f}s)", flush=True)
    results[ci] = {"rows": rows, "q_est": q_est, "w": w.tolist(), "wall_s": time.time() - t0}
    torch.save(results, OUT)
    ncav += 1
    if ncav >= 2:
        break
print(f"saved -> {OUT}", flush=True)
print("REFEREES: cav1 certified 0.49+-0.04 (BCY 20k); cav0 overnight verbatim BCY in flight.", flush=True)
