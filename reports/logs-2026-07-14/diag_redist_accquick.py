"""QUICK acceptance-only probe (no full anneal, no q~). Measures the redistribution I-MTM acceptance vs lambda
on (a) the FRESH/loose alien state and (b) after 4 local moves (tightened) -- to see where and how much the
reorder ('redist first, while loose') buys. alien MB=32, 3 cavities, N_try=8, R=2.0."""
import sys, statistics as st, functools
import torch
sys.path.insert(0, "reports/logs-2026-07-14"); sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b, blob_mask
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; BETA = 2.0; K = 8; K_R = 12; R = 2.0; N_TRY = 8; MB = 32; NCAV = 3
ART = "liquid_coupling_flow/artifacts"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
LAMS = [0.01, 0.02, 0.04, 0.0625, 0.09, 0.15, 0.30]


def two_blob(n, a, gen, Kr):
    s1 = int(torch.randint(n, (), generator=gen, device=dev))
    far = (a - a[s1]).norm(dim=-1); s2 = int(far.topk(min(8, n), largest=True).indices[0])
    mk = torch.zeros(n, dtype=torch.bool, device=dev)
    mk[(a - a[s1]).norm(dim=-1).topk(min(Kr, n), largest=False).indices] = True
    mk[(a - a[s2]).norm(dim=-1).topk(min(Kr, n), largest=False).indices] = True
    return mk


def redist_acc(Xb, Sb, bnd, sb, R, lam, q0, Ecur, allm, U, gen, n_try):
    B, n = Xb.shape[0], Xb.shape[1]
    def w(Xc, Sc, lqu):
        return (1 - lam) * m.block_log_prob_b(Xc, Sc, allm, bnd, sb, R) - lam * BETA * energy_b(Xc, Sc, bnd, sb) - lqu
    Xr = Xb.repeat_interleave(n_try, 0); Sr = Sb.repeat_interleave(n_try, 0)
    Yf, Sf, lqf = m.sample_block_b(Xr, Sr, U, bnd, sb, R, gen=gen); lwf = w(Yf, Sf, lqf).view(B, n_try)
    Xr2 = Xb.repeat_interleave(n_try - 1, 0); Sr2 = Sb.repeat_interleave(n_try - 1, 0)
    Yr, Sr2b, lqr2 = m.sample_block_b(Xr2, Sr2, U, bnd, sb, R, gen=gen); lwr = w(Yr, Sr2b, lqr2).view(B, n_try - 1)
    lw_cur = (1 - lam) * q0 - lam * BETA * Ecur - m.block_log_prob_b(Xb, Sb, U, bnd, sb, R)
    la = torch.logsumexp(lwf, 1) - torch.logsumexp(torch.cat([lwr, lw_cur[:, None]], 1), 1)
    return float((torch.rand(B, device=dev, generator=gen).log() < la).float().mean())


def local_move(Xb, Sb, bnd, sb, R, lt, q0, Ecur, allm, a, gen, n):
    blk = blob_mask(n, K, a, gen)
    lqr = m.block_log_prob_b(Xb, Sb, blk, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xb, Sb, blk, bnd, sb, R, gen=gen)
    En = energy_b(Xn, Sn, bnd, sb); q0n = m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
    la = geometric_bridge_log_accept(log_q0_current=q0, log_q0_proposed=q0n, energy_current=Ecur,
                                     energy_proposed=En, log_r_reverse=lqr, log_r_forward=lqf, lam=lt, beta=BETA)
    acc = torch.rand(Xb.shape[0], device=dev, generator=gen).log() < la
    return (torch.where(acc[:, None, None], Xn, Xb), torch.where(acc[:, None], Sn, Sb),
            torch.where(acc, q0n, q0), torch.where(acc, En, Ecur))


gen = torch.Generator(device=dev).manual_seed(0)
loose = {l: [] for l in LAMS}; tight = {l: [] for l in LAMS}; ncav = 0
for ci in range(24):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < K + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev); Kr = min(K_R, (n - 2) // 2)
    a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    g = torch.Generator(device=dev).manual_seed(100 + ci)
    X0, S0, _ = m.sample_block_b(xo[None].expand(MB, n, 3).clone(), so[None].expand(MB, n).clone(), allm, bnd, sb, R, gen=g)
    E0 = energy_b(X0, S0, bnd, sb); Q0 = m.block_log_prob_b(X0, S0, allm, bnd, sb, R)
    for lam in LAMS:
        U = two_blob(n, a, g, Kr)
        loose[lam].append(redist_acc(X0, S0, bnd, sb, R, lam, Q0, E0, allm, U, g, N_TRY))    # FRESH state
        Xb, Sb, q0, Ec = X0.clone(), S0.clone(), Q0.clone(), E0.clone()                       # tighten: 4 local moves at this lam
        for _ in range(4):
            Xb, Sb, q0, Ec = local_move(Xb, Sb, bnd, sb, R, lam, q0, Ec, allm, a, g, n)
        U2 = two_blob(n, a, g, Kr)
        tight[lam].append(redist_acc(Xb, Sb, bnd, sb, R, lam, q0, Ec, allm, U2, g, N_TRY))
    ncav += 1
    if ncav >= NCAV:
        break

print(f"=== QUICK redist acceptance: FRESH/loose vs after-4-local (tightened), R={R}, N_try={N_TRY} ===", flush=True)
print(f"{'lambda':>7} | {'loose_acc':>9} {'tight_acc':>9}", flush=True)
for lam in LAMS:
    print(f"{lam:>7.4f} | {st.mean(loose[lam]):>9.3f} {st.mean(tight[lam]):>9.3f}", flush=True)
