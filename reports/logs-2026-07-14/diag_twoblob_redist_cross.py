"""PAYOFF TEST: does two-blob SPECIES REDISTRIBUTION (natural joint regen, model reassigns species across two
separated regions -- proven real, ~80% of ceiling at K=12-16) enable BASIN-CROSSING? Add a two-blob-redist
move (regenerate the union of two well-separated K_r-blobs NORMALLY; model redistributes positions+species
under the union budget) to the tempered SMC, MH'd via the exact geometric bridge (permissive at low lambda
where the big clashy move is accepted). Compare ALIEN-seed q~ with vs without the redist move -- does alien
rise toward the data plateau (cross to the reference basin)? Local single-blob moves were non-ergodic (alien
flat ~0.06-0.16). data-seed shown as the upper bracket. lam05, l=0.368, R in {1.6,2.0}, K_r=12, n_redist=2."""
import sys, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-14")
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b, blob_mask, ess
from smc_pts_sweep_hocky import box_overlap
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept

dev = "cuda"; RCTX = 2.5; BETA = 2.0; ART = "liquid_coupling_flow/artifacts"; L_BOX = 0.368
RHO = 1.149; BULK = RHO * L_BOX ** 3; K = 8; K_R = 12
CK = f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(CK, map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
J, M, T, NMUT, NRD, NCAV = 4, 16, 20, 2, 2, 3


def two_blob_mask(n, a, gen, Kr):
    s1 = int(torch.randint(n, (), generator=gen, device=dev))
    far = (a - a[s1]).norm(dim=-1); s2 = int(far.topk(min(8, n), largest=True).indices[torch.randint(min(8, n), (), generator=gen, device=dev)])
    mk = torch.zeros(n, dtype=torch.bool, device=dev)
    mk[(a - a[s1]).norm(dim=-1).topk(min(Kr, n), largest=False).indices] = True
    mk[(a - a[s2]).norm(dim=-1).topk(min(Kr, n), largest=False).indices] = True
    return mk


@torch.no_grad()
def island(xo, so, bnd, sb, R, seed_mode, gen, use_redist):
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    if seed_mode == "alien":
        Xb, Sb, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(), allm, bnd, sb, R, gen=gen)
    else:
        Xb = xo[None].expand(M, n, 3).clone(); Sb = so[None].expand(M, n).clone()
    Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
    lam = torch.linspace(0, 1, T + 1, device=dev) ** 4
    a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    logw = torch.zeros(M, device=dev); rd_acc = []

    def mtm(blk, lt):
        nonlocal Xb, Sb, Ecur, q0
        lqr = m.block_log_prob_b(Xb, Sb, blk, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xb, Sb, blk, bnd, sb, R, gen=gen)
        En = energy_b(Xn, Sn, bnd, sb); q0n = None if lt == 1.0 else m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
        la = geometric_bridge_log_accept(log_q0_current=q0, log_q0_proposed=q0n, energy_current=Ecur,
                                         energy_proposed=En, log_r_reverse=lqr, log_r_forward=lqf, lam=lt, beta=BETA)
        acc = torch.rand(M, device=dev, generator=gen).log() < la
        Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb)
        Ecur = torch.where(acc, En, Ecur)
        if q0n is not None:
            q0 = torch.where(acc, q0n, q0)
        return float(acc.float().mean())

    for t in range(1, T + 1):
        logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur - q0)
        if ess(logw) < 0.5:
            w = torch.softmax(logw, 0); idx = torch.multinomial(w, M, replacement=True, generator=gen)
            Xb, Sb, Ecur, q0 = Xb[idx].clone(), Sb[idx].clone(), Ecur[idx].clone(), q0[idx].clone(); logw = torch.zeros(M, device=dev)
        lt = float(lam[t])
        Kr = min(K_R, (n - 2) // 2)                              # adaptive: two disjoint blobs must fit in n
        for _ in range(NMUT):
            mtm(blob_mask(n, K, a, gen), lt)                     # local single-blob position/species moves
        if use_redist and Kr >= 4:
            for _ in range(NRD):
                rd_acc.append(mtm(two_blob_mask(n, a, gen, Kr), lt))   # two-blob REDISTRIBUTION move
    return Xb, Sb, (st.mean(rd_acc) if rd_acc else 0.0)


def run(R, seed_mode, use_redist):
    gen = torch.Generator(device=dev).manual_seed(0); ncav = 0; qs = []; accs = []
    for ci in range(24):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xin = _mic(p["x_in"], c, L)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        qc = []
        for j in range(J):
            g = torch.Generator(device=dev).manual_seed((1000 if seed_mode == "alien" else 8000) + 31 * ci + j)
            Xd, Sd, af = island(xo, so, bnd, sb, R, seed_mode, g, use_redist); accs.append(af)
            qc.append(st.mean([box_overlap(xin, Xd[k], R, L_BOX) for k in range(Xd.shape[0])]) - BULK)
        qs.append(st.mean(qc)); ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(qs), (st.mean(accs) if accs else 0.0)


print(f"=== two-blob REDISTRIBUTION as basin-crossing move (lam05, l={L_BOX}, K_r={K_R}) ===", flush=True)
print(f"K={K} T={T} J={J} M={M} n_mut={NMUT} n_redist={NRD} ncav={NCAV}. Does alien-seed q~ RISE?", flush=True)
print(f"{'R':>4} {'seed':>6} | {'base q~':>8} {'+redist q~':>11} {'rd_acc':>7} | verdict", flush=True)
for R in (2.0,):
    for seed_mode in ("alien", "data"):
        qb, _ = run(R, seed_mode, False)
        qr, af = run(R, seed_mode, True)
        v = "CROSSES (alien rises)" if (seed_mode == "alien" and qr > qb + 0.05) else \
            ("holds" if seed_mode == "data" else "no cross")
        print(f"{R:>4} {seed_mode:>6} | {qb:>8.3f} {qr:>11.3f} {af:>7.3f} | {v}", flush=True)
