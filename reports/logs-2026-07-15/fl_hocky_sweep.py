"""FRENKEL-LADD (tether) METABASIN ESTIMATOR -- one unbiased q(R) from local thermodynamics.
Per cavity: discovery (AR pool -> prescreen -> quench) -> cluster into FAMILIES (loose: dU_IS/n<0.5 AND
IS-overlap>0.30 -- metabasin granularity, fixing v1's 70-micro-IS shattering) -> for each family rep x_b
(quenched, so gradient ~0; NO Hessian anywhere -- saddle-immune):
    U_k(x) = U(x) + k * sum_i |x_i - x_b,i|^2 ,  k on a ln-grid k_min..k_max
    beta*F_b = beta*U(x_b) + (3n/2) ln(beta k_max / pi) - beta * int k*Lambda(k) dln k + [O(trH/4beta k_max)]
    with Lambda(k) = <sum d^2>_k. Einstein constants and the k_max correction largely CANCEL in Delta-F
    between same-n families; both reported. Integrand k*Lambda is positive/smooth/sigmoidal (v1's wild
    Hessian-mixing integrand eliminated); equipartition check at k_max: k*Lambda -> 3n/(2 beta).
q~_b measured in the k_min window (free within-family MC). Combine: w_b = softmax(-beta F_b);
q = sum w_b q~_b. Gates: (G1) k*Lambda(k_max)/(3n/2beta) in [0.9,1.1]; (G2) cross-family k_min-ensemble
overlap LOW (else merged -> re-cluster); (G3) lower-tail truncation k_min*Lambda(k_min) small. Per-family
incremental saves. Usage: [R] argv. Referee: cav1 dual-init bracket [0.45,0.53]."""
import sys, functools, time, statistics as st, math
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
sys.path.insert(0, "/mnt/ssd/GridTransformer/reports/logs-2026-07-13")
sys.path.insert(0, "/mnt/ssd/GridTransformer/reports/logs-2026-07-14")
from ka3d_smc_sweep import energy_b
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; BETA = 1.0 / 0.55; RHO = 1.200; BIGL = 100.0   # HOCKY state point
R = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
POOL = 2048; BATCH = 512; NQ = 48; QSTEPS = 800
NCAV = int(sys.argv[2]) if len(sys.argv) > 2 else 4
L_HOCKY = (0.06 / RHO) ** (1.0 / 3.0); BULK_HOCKY = 0.06
FAM_DU, FAM_Q = 0.5, 0.45                     # member/look-alike boundary (measured ~0.45)
MAX_FAM = 3                                    # ref + top-2 alien families
NW = 8; KMIN, KMAX, NK = 0.05, 1e6, 18        # ln-grid; KMAX=1e6 dominates LJ stiff modes (G1)
BURN, AVG = 40, 80; DISP0 = 0.10
ART = "liquid_coupling_flow/artifacts"
OUT = f"reports/logs-2026-07-15/fl_hocky_R{R}.pt"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
BANK = torch.load("reports/logs-2026-07-15/bulk_ka12_T055_N4096.pt", map_location="cpu", weights_only=False)
frames = BANK["bank"]; L = float(BANK["L"])
print(f"HOCKY bank: {len(frames)} frames x {frames[0]['x'].shape[0]} replicas, rho={BANK['rho']:.4f} T={BANK['T']}", flush=True)
# one config per (replica, spread over latest frames) -> independent cavities
cfgs = []
for k in range(64):
    fr = frames[-1 - (k // frames[0]["x"].shape[0]) % len(frames)]
    b = k % frames[0]["x"].shape[0]
    cfgs.append((fr["x"][b], fr["s"][b]))
X = torch.stack([c[0] for c in cfgs]).to(dev).float()
S = torch.stack([c[1] for c in cfgs]).to(dev).long()


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


CTR_H = 2.5 * L_HOCKY   # Hocky SI: overlap probed in a 5x5x5 cube of side-l boxes AT THE CAVITY CENTER (125 boxes),
                        # NOT the whole cavity (wall shell = 84% of volume at R=2.2, trivially localized)


def q_center(a, b):
    """Paper-exact center overlap: shared occupied boxes in the central 5x5x5 cube / (rho l^3 125).
    Normalization rho*l^3*Nbox => self-overlap -> 1, independent-bulk -> rho l^3 = 0.06 (both paper-stated)."""
    def cset(x):
        mm = (x.abs() < CTR_H).all(-1)
        ijk = torch.floor((x[mm] + CTR_H) / L_HOCKY).long().clamp(0, 4)
        return set((ijk[:, 0] * 25 + ijk[:, 1] * 5 + ijk[:, 2]).tolist())
    return len(cset(a) & cset(b)) / (RHO * L_HOCKY ** 3 * 125)


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
    return (ka_energy(torch.cat([xi, bndd[None].expand(Mn, mm, 3)], 1), sfull, BIGL).float(), xi.float())


def tether_family(x_b, s_in, bnd, sb, xin, gen):
    """FL ladder for the family anchored at x_b. Returns dict with beta*F pieces, q~_b, gates, ensemble."""
    n = x_b.shape[0]
    U_b = float(energy_b(x_b[None], s_in[None], bnd, sb)[0])
    kappas = torch.logspace(math.log10(KMIN), math.log10(KMAX), NK)
    Xb = x_b[None].expand(NW, n, 3).clone(); Sb = s_in[None].expand(NW, n).clone()
    U = energy_b(Xb, Sb, bnd, sb)
    lam_vals = []; q_min = None; q_sd = 0.0; ens_min = None; qc_min = None; qc_sd = 0.0
    for ki in range(NK - 1, -1, -1):                              # start at k_max (pinned), anneal DOWN
        k = float(kappas[ki])
        step = min(0.25, max(2e-4, 0.6 / math.sqrt(1 + BETA * k)))  # step ~ tether width (floor 2e-4:
        # the old 0.02 floor was 40x the width at k=1e6 -> total rejection -> frozen rungs -> G1=0)
        qs_here = []; qc_here = []; lam_here = []
        for sw in range(BURN + AVG):
            for i in torch.randperm(n, generator=gen, device=dev).tolist():
                prop = Xb.clone()
                prop[:, i] = Xb[:, i] + step * torch.randn(NW, 3, device=dev, generator=gen)
                ok = prop[:, i].norm(dim=-1) < R
                Up = energy_b(prop, Sb, bnd, sb)
                dteth = ((prop[:, i] - x_b[i]) ** 2).sum(-1) - ((Xb[:, i] - x_b[i]) ** 2).sum(-1)
                dE = (Up - U) + k * dteth
                acc = ok & (torch.rand(NW, device=dev, generator=gen).log() < -BETA * dE)
                Xb = torch.where(acc[:, None, None], prop, Xb); U = torch.where(acc, Up, U)
            if sw >= BURN:
                lam_here.append(float(((Xb - x_b[None]) ** 2).sum((1, 2)).mean()))
                if ki == 0:
                    qs_here += [q_hocky(Xb[w].cpu(), xin.cpu(), R) - BULK_HOCKY for w in range(NW)]
                    qc_here += [q_center(Xb[w].cpu(), xin.cpu()) - BULK_HOCKY for w in range(NW)]
        lam_vals.append((k, st.mean(lam_here), st.pstdev(lam_here) / max(len(lam_here) ** .5, 1)))
        if ki == 0:
            q_min = st.mean(qs_here); q_sd = st.pstdev(qs_here); ens_min = Xb.clone()
            qc_min = st.mean(qc_here); qc_sd = st.pstdev(qc_here)
    lam_vals = lam_vals[::-1]                                     # ascending k
    lnk = torch.tensor([math.log(k) for k, _, _ in lam_vals])
    integ = torch.tensor([k * lam for k, lam, _ in lam_vals])     # k*Lambda(k) in dln k
    bF_teth = -BETA * float(torch.trapz(integ, lnk))              # -beta int k*Lambda dln k
    bF = BETA * U_b + 1.5 * n * math.log(BETA * KMAX / math.pi) + bF_teth
    equi = float(integ[-1]) / (1.5 * n / BETA)                    # gate G1 -> 1
    tail = float(integ[0])                                        # gate G3 -> ~0
    return {"U_b": U_b, "bF": bF, "bF_teth": bF_teth, "q_b": q_min, "q_sd": q_sd,
            "q_c": qc_min, "qc_sd": qc_sd, "x_b": x_b.cpu(),
            "equi": equi, "tail": tail, "integrand": [(k, l) for k, l, _ in lam_vals],
            "ens_min": ens_min.cpu()}


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
    g = torch.Generator(device=dev).manual_seed(3100 + ci)
    # discovery
    allX, allS, allU = [], [], []
    for b0 in range(0, POOL, BATCH):
        nb = min(BATCH, POOL - b0)
        Xa, Sa, _ = m.sample_block_b(xo[None].expand(nb, n, 3).clone(), so[None].expand(nb, n).clone(),
                                     allm, bnd, sb, R, gen=g)
        allU.append(energy_b(Xa, Sa, bnd, sb).cpu()); allX.append(Xa.cpu()); allS.append(Sa.cpu())
    allX = torch.cat(allX); allS = torch.cat(allS); allU = torch.cat(allU)
    sel = torch.unique(torch.cat([(-allU).topk(NQ // 2).indices, torch.randperm(POOL)[:NQ // 2]]))
    Xq = torch.cat([xo[None].cpu(), allX[sel]]); Sq = torch.cat([so[None].cpu(), allS[sel]])
    Uis, Xis = quench(Xq.to(dev), Sq.to(dev), bnd, sb)
    # loose family clustering
    NQm = len(Xq); par = list(range(NQm))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    Xis_c = Xis.cpu()
    for i in range(NQm):
        for j in range(i + 1, NQm):
            if abs(float(Uis[i] - Uis[j])) / n < FAM_DU and \
               q_hocky(Xis_c[i], Xis_c[j], R) - BULK_HOCKY > FAM_Q:
                par[find(i)] = find(j)
    fams = {}
    for i in range(NQm):
        fams.setdefault(find(i), []).append(i)
    fam_list = sorted(fams.values(), key=lambda ms: -len(ms))
    keep = [ms for ms in fam_list if 0 in ms]                                  # ref family first
    keep += [ms for ms in fam_list if 0 not in ms][:MAX_FAM - len(keep)]
    if len(keep) < 2:                                              # FORCE an alien anchor: deepest quench
        ref_ov = torch.tensor([q_hocky(Xis_c[i], Xis_c[0], R) - BULK_HOCKY for i in range(NQm)])
        alien_cand = [(float(Uis[i]), i) for i in range(1, NQm) if float(ref_ov[i]) < 0.20]
        if alien_cand:
            alien_rep = min(alien_cand)[1]
            keep.append([alien_rep])
            print(f"  forced alien anchor: idx {alien_rep} U_IS/n {float(Uis[alien_rep])/n:+.3f} "
                  f"ref-overlap {float(ref_ov[alien_rep]):+.2f}", flush=True)
        else:
            print("  NO alien candidate with ref-overlap<0.20 among quenches (single-family cavity?)", flush=True)
    print(f"=== cav {ci} (n={n}, R={R}) FRENKEL-LADD: {len(fams)} raw clusters -> keeping {len(keep)} "
          f"families sizes {[len(ms) for ms in keep]} ===", flush=True)
    rows = []
    for fi, ms in enumerate(keep):
        rep = ms[int(torch.tensor([float(Uis[k]) for k in ms]).argmin())]
        r_ = tether_family(Xis[rep].to(dev), Sq[rep].to(dev), bnd, sb, xin, g)
        r_.update({"n_members": len(ms), "has_ref": 0 in ms, "rep_Uis_n": float(Uis[rep]) / n})
        rows.append(r_)
        print(f"  fam{fi} {'REF' if r_['has_ref'] else '   '} U_IS/n {r_['rep_Uis_n']:+.3f} members {len(ms):>3} | "
              f"bF {r_['bF']:+.1f} (teth {r_['bF_teth']:+.1f}) qCENTER {r_['q_c']:+.3f}+-{r_['qc_sd']:.3f} "
              f"(whole-cav {r_['q_b']:+.3f}+-{r_['q_sd']:.3f}) | "
              f"G1 equi {r_['equi']:.2f} G3 tail {r_['tail']:.2f}", flush=True)
        results[(ci, fi)] = r_                                                 # FULL data incl. window ensemble+anchor
        torch.save(results, OUT)                                               # per-family incremental save
    # G2: cross-family k_min ensemble overlap (double-count check)
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            xov = st.mean([q_hocky(rows[a]["ens_min"][w].cpu(), rows[b]["ens_min"][v].cpu(), R) - BULK_HOCKY
                           for w in range(0, NW, 3) for v in range(0, NW, 3)])
            print(f"  G2 cross-overlap fam{a}-fam{b}: {xov:+.3f} (LOW => distinct families)", flush=True)
    bFs = torch.tensor([r_["bF"] for r_ in rows]).double()
    w = torch.softmax(-(bFs - bFs.min()), 0)
    q_est = float(sum(float(w[i]) * rows[i]["q_b"] for i in range(len(rows))))
    q_ctr = float(sum(float(w[i]) * rows[i]["q_c"] for i in range(len(rows))))
    wref = sum(float(w[i]) for i in range(len(rows)) if rows[i]["has_ref"])
    print(f"  q~CENTER(R={R}) FRENKEL-LADD = {q_ctr:+.3f}  [paper fit T=0.55: 0.5*exp(-((R-1)/2.055)^3.119) = "
          f"{0.5 * math.exp(-((R - 1) / 2.055) ** 3.119):+.3f}]  (whole-cav legacy {q_est:+.3f}; "
          f"w = {[round(float(x),3) for x in w.tolist()]}, w_ref {wref:.3f}; "
          f"dbF vs best: {[round(float(b - bFs.min()),1) for b in bFs.tolist()]}) ({time.time()-t0:.0f}s)", flush=True)
    results[(ci, "final")] = {"q_est": q_est, "q_ctr": q_ctr, "w": w.tolist(), "wref": wref,
                              "xin": xin.cpu(), "n": n}
    torch.save(results, OUT)
    ncav += 1
    if ncav >= NCAV:
        break
print(f"saved -> {OUT}", flush=True)
print("Referee: cav1 dual-init bracket [0.45,0.53]; fixed-ladder PT in flight for cav0.", flush=True)
