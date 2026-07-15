"""DUAL-SEED PT v3 -- HOCKY EXACT CONVENTION recompute (same sampler as v2, corrected observable):
  C1 FIXED spatial grid (v2's box ids were anchored to each config's min cell -> occasional one-box shifts
     corrupted intersections; Hocky's n_i are indicators on a fixed grid).
  C2 MATCHED l: the paper chose l^3=0.05 at rho=1.2 so that rho*l^3=0.06; convention-matched at our
     rho=1.1486 => l=(0.06/rho)^(1/3)=0.3738; bulk subtraction EXACTLY rho*l^3=0.06 (+ empirical Cavagna
     bulk as cross-check). Non-species indicators; N_box = cells with center inside the cavity.
  C3 SAVE cold-replica configs every REPORT (record-simulation-data directive; v2 saved scalars only).
Reports BOTH conventions on the same states to quantify the grid-shift bug. Anchors: Hocky Fig2c (T=0.55,
rho=1.2): q~ ~ 0.33 @R=2.2, 0.29 @R=2.4. Ours: T=0.5, rho=1.1486, R=2.0.
v2 base (fixes over v1):
  F1 DISPLACEMENT SWEEP per replica per PT sweep: n single-particle Gaussian moves (step 0.10, reject
     |x|>R) -- the within-basin THERMALIZER the AR proposal can't provide at cold beta (its blocks are
     +1.4/particle hot => always rejected from a deep config; the reference could never relax in v1).
  F2 DENSER COLD LADDER: betas = linspace(0.4,1.4,5) + linspace(1.5,2.0,11) (16 rungs, dbeta_cold=0.05):
     basin-scale dU~33 x 0.05 => exchange acc ~20% (v1: 0.178 x 33 => e^-6, ladder impassable).
  F3 QUENCH-VERIFIED B SEEDS: top-16 pool draws quenched vs quenched reference; cold-B chains seeded with
     TRUE members (z<3 AND IS-overlap>0.4) when found (v1 seeded raw look-alikes that collapsed).
Same certificate: post-burn stack A (data-seeded) == stack B (alien/verified-seeded) at beta=2. Incremental
saves. lam05 Rext proposal, R=2.0, cavities 0/1."""
import sys, functools, time, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b, blob_mask
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; L_BOX = 0.368; RHO = 1.149; BULK = RHO * L_BOX ** 3; BIGL = 100.0  # L_BOX = legacy config-anchored metric, kept for comparison
import sys as _sys
K = 8; R = float(_sys.argv[1]) if len(_sys.argv) > 1 else 2.0; POOL = 4096; BATCH = 512; ART = "liquid_coupling_flow/artifacts"
NCH = 2; SW = 250; REPORT = 25; BURN_FRAC = 0.4; DISP = 0.10; QSTEPS = 400
BETAS = torch.cat([torch.linspace(0.4, 1.4, 5), torch.linspace(1.5, 2.0, 11)]).to(dev)   # F2
NR = len(BETAS)
L_HOCKY = (0.06 / RHO) ** (1.0 / 3.0); BULK_HOCKY = 0.06
OUT = f"reports/logs-2026-07-14/pt_dualseed_v3_hocky_R{R}.pt"
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


def box_id_set_fixed(x, l, R):
    """C1: indicators on a FIXED grid (constant offset), not per-config min-anchored."""
    mm = x.norm(dim=-1) < R
    ijk = torch.floor(x[mm] / l).long()
    h = int(R / l) + 2; side = 2 * h + 1
    return set(((ijk[:, 0] + h) * side * side + (ijk[:, 1] + h) * side + (ijk[:, 2] + h)).tolist())


def q_hocky(a, b, R):
    """Hocky q = (l^3 N_box)^-1 * shared occupied cells, fixed grid, matched l; subtract rho*l^3 outside."""
    return len(box_id_set_fixed(a, L_HOCKY, R) & box_id_set_fixed(b, L_HOCKY, R)) / (L_HOCKY ** 3 * n_boxes(L_HOCKY, R))


def quench(Xb, Sb, bnd, sb, steps=QSTEPS, dmax=0.05):
    Mn, n, _ = Xb.shape; mm = bnd.shape[0]
    xi = Xb.double().clone(); bndd = bnd.double()
    sfull = torch.cat([Sb, sb[None].expand(Mn, mm)], 1).long()
    grad_on = torch.enable_grad()
    with grad_on:
        for t in range(steps):
            xi = xi.detach().requires_grad_(True)
            U = ka_energy(torch.cat([xi, bndd[None].expand(Mn, mm, 3)], 1), sfull, BIGL)
            (gr,) = torch.autograd.grad(U.sum(), xi)
            step = dmax * (0.3 + 0.7 * (1 - t / steps))
            gn = gr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            xi = (xi - gr / gn * torch.minimum(gn * 1e-3, torch.full_like(gn, step))).detach()
    return ka_energy(torch.cat([xi, bndd[None].expand(Mn, mm, 3)], 1), sfull, BIGL).float(), xi.float()


def displacement_sweep(Xf, Sf, bnd, sb, U, beta_f, gen, n):
    """F1: n single-particle Gaussian MH micro-moves per replica (reject outside the R-ball)."""
    for i in torch.randperm(n, generator=gen, device=dev).tolist():
        prop = Xf.clone()
        prop[:, i] = Xf[:, i] + DISP * torch.randn(len(Xf), 3, device=dev, generator=gen)
        ok = prop[:, i].norm(dim=-1) < R
        En = energy_b(prop, Sf, bnd, sb)
        la = -beta_f * (En - U)
        acc = ok & (torch.rand(len(Xf), device=dev, generator=gen).log() < la)
        Xf = torch.where(acc[:, None, None], prop, Xf); U = torch.where(acc, En, U)
    return Xf, U


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
    g = torch.Generator(device=dev).manual_seed(900 + ci)
    # --- pool + F3 quench-verified seeds ---
    allX, allS, allq = [], [], []
    for b0 in range(0, POOL, BATCH):
        nb = min(BATCH, POOL - b0)
        Xa, Sa, _ = m.sample_block_b(xo[None].expand(nb, n, 3).clone(), so[None].expand(nb, n).clone(),
                                     allm, bnd, sb, R, gen=g)
        Xc = Xa.cpu(); allq += [q_hocky(Xc[k], xin.cpu(), R) - BULK_HOCKY for k in range(nb)]
        allX.append(Xc); allS.append(Sa.cpu())
    allX = torch.cat(allX); allS = torch.cat(allS); allq = torch.tensor(allq)
    top = allq.topk(16).indices
    Uref_is, Xref_is = quench(xo[None], so[None], bnd, sb)
    Utop_is, Xtop_is = quench(allX[top].to(dev), allS[top].to(dev), bnd, sb)
    uref = float(Uref_is[0]) / n; utop = Utop_is / n
    sig = max(float(utop.std()), 0.15)
    memb = []
    for k in range(16):
        z = (float(utop[k]) - uref) / sig
        qis = q_hocky(Xtop_is[k].cpu(), Xref_is[0].cpu(), R) - BULK_HOCKY
        if z < 3.0 and qis > 0.4:
            memb.append(int(top[k]))
    seed_idx = (memb + [int(t) for t in top])[:NCH]                        # members first, fill with best
    print(f"=== cav {ci} (n={n}) PT-v2: {NR} rungs, members found {len(memb)}, B cold seeds q~ "
          f"{[round(float(allq[j]), 2) for j in seed_idx]} verified={len(memb) >= 1} ===", flush=True)
    # Cavagna empirical bulk cross-check: ref vs OTHER data configs' interiors at the SAME center
    qb_emp = []
    for cj in range(2, 12):
        pj = carve(X[cj], S[cj], c, R, L)
        if pj["n_in"] < 1 or cj == ci:
            continue
        qb_emp.append(q_hocky(_mic(pj["x_in"], c, L).cpu(), xin.cpu(), R))
        if len(qb_emp) >= 6:
            break
    print(f"  bulk: convention rho*l^3 = {BULK_HOCKY:.3f}; empirical (Cavagna) = {st.mean(qb_emp):.3f}; "
          f"q(self) = {q_hocky(xin.cpu(), xin.cpu(), R):.3f} (~rho)", flush=True)
    rnd = torch.randperm(POOL)[:NR * NCH]
    Xrep = torch.empty(2, NR, NCH, n, 3, device=dev); Srep = torch.empty(2, NR, NCH, n, dtype=torch.long, device=dev)
    Xrep[0] = xo[None, None]; Srep[0] = so[None, None]
    for r in range(NR):
        for ch in range(NCH):
            idx = seed_idx[ch] if r == NR - 1 else int(rnd[r * NCH + ch])
            Xrep[1, r, ch] = allX[idx].to(dev); Srep[1, r, ch] = allS[idx].to(dev)
    Xf = Xrep.reshape(-1, n, 3); Sf = Srep.reshape(-1, n)
    U = energy_b(Xf, Sf, bnd, sb)
    beta_f = BETAS[None, :, None].expand(2, NR, NCH).reshape(-1).clone()
    ex_acc = torch.zeros(NR - 1); ex_try = torch.zeros(NR - 1); traj = []
    for sw in range(1, SW + 1):
        for kind in ("suf", "loc"):                                        # AR moves (collective)
            if kind == "suf":
                ks = int(torch.randint(n // 2, 3 * n // 4 + 1, (), generator=g, device=dev))
                mask = torch.zeros(n, dtype=torch.bool, device=dev); mask[n - ks:] = True
            else:
                mask = blob_mask(n, K, a, g)
            lqr = m.block_log_prob_b(Xf, Sf, mask, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xf, Sf, mask, bnd, sb, R, gen=g)
            En = energy_b(Xn, Sn, bnd, sb)
            la = -beta_f * (En - U) + lqr - lqf
            acc = torch.rand(len(Xf), device=dev, generator=g).log() < la
            Xf = torch.where(acc[:, None, None], Xn, Xf); Sf = torch.where(acc[:, None], Sn, Sf); U = torch.where(acc, En, U)
        Xf, U = displacement_sweep(Xf, Sf, bnd, sb, U, beta_f, g, n)        # F1 thermalizer
        Xrep = Xf.view(2, NR, NCH, n, 3); Srep = Sf.view(2, NR, NCH, n); Urep = U.view(2, NR, NCH)
        par = sw % 2
        for r in range(par, NR - 1, 2):
            dlog = (BETAS[r] - BETAS[r + 1]) * (Urep[:, r] - Urep[:, r + 1])
            swp = torch.rand(2, NCH, device=dev, generator=g).log() < dlog
            ex_acc[r] += float(swp.float().sum()); ex_try[r] += swp.numel()
            for stk in range(2):
                for ch in range(NCH):
                    if swp[stk, ch]:
                        Xrep[stk, [r, r + 1], ch] = Xrep[stk, [r + 1, r], ch]
                        Srep[stk, [r, r + 1], ch] = Srep[stk, [r + 1, r], ch]
                        Urep[stk, [r, r + 1], ch] = Urep[stk, [r + 1, r], ch]
        Xf = Xrep.reshape(-1, n, 3); Sf = Srep.reshape(-1, n); U = Urep.reshape(-1)
        if sw % REPORT == 0:
            qA = [q_hocky(Xrep[0, -1, ch].cpu(), xin.cpu(), R) - BULK_HOCKY for ch in range(NCH)]
            qB = [q_hocky(Xrep[1, -1, ch].cpu(), xin.cpu(), R) - BULK_HOCKY for ch in range(NCH)]
            qA_old = [qtil_one(Xrep[0, -1, ch].cpu(), xin.cpu(), R) for ch in range(NCH)]
            qB_old = [qtil_one(Xrep[1, -1, ch].cpu(), xin.cpu(), R) for ch in range(NCH)]
            traj.append({"sweep": sw, "qA": qA, "qB": qB, "qA_old": qA_old, "qB_old": qB_old,
                         "U_cold_A": float(Urep[0, -1].mean()), "U_cold_B": float(Urep[1, -1].mean()),
                         "X_cold_A": Xrep[0, -1].cpu().clone(), "X_cold_B": Xrep[1, -1].cpu().clone(),
                         "S_cold_A": Srep[0, -1].cpu().clone(), "S_cold_B": Srep[1, -1].cpu().clone()})
            print(f"  sw {sw:>4}: HOCKY q~ A {[f'{q:+.2f}' for q in qA]}  B {[f'{q:+.2f}' for q in qB]}  "
                  f"(old-metric A {[f'{q:+.2f}' for q in qA_old]})", flush=True)
            results[(ci, "traj")] = traj; torch.save(results, OUT)
    nb_ = int(len(traj) * BURN_FRAC)
    qAf = [q for row in traj[nb_:] for q in row["qA"]]; qBf = [q for row in traj[nb_:] for q in row["qB"]]
    agree = abs(st.mean(qAf) - st.mean(qBf))
    results[(ci, "final")] = {"qA": st.mean(qAf), "qA_sd": st.pstdev(qAf), "qB": st.mean(qBf),
                              "qB_sd": st.pstdev(qBf), "agree_gap": agree, "members_seeded": len(memb),
                              "exchange_acc": (ex_acc / ex_try.clamp(min=1)).tolist(), "wall_s": time.time() - t0}
    torch.save(results, OUT)
    print(f"  FINAL: A {st.mean(qAf):+.3f}+-{st.pstdev(qAf):.3f}  B {st.mean(qBf):+.3f}+-{st.pstdev(qBf):.3f}"
          f"  gap {agree:.3f}  exch cold-end {[f'{x:.2f}' for x in (ex_acc/ex_try.clamp(min=1)).tolist()[-4:]]}"
          f"  ({time.time()-t0:.0f}s)", flush=True)
    ncav += 1
    if ncav >= 2:
        break
print(f"saved -> {OUT}", flush=True)
