"""FAST sweet-spot soak. Speedups vs diag_redist_soak.py: (1) ONE big-batch population of MB=48 walkers
(folds the 3 islands -> the autoregressive sample_block_b Python loop runs once over 48 configs, amortizing
kernel-launch overhead that made it CPU-bound); (2) n_boxes_in_sphere CACHED (was a fresh meshgrid every
box_overlap call); (3) redist_imtm reimplemented with DYNAMIC batch (imported one hardcodes M=16). Sweep the
lever that matters -- n_extra window rungs (more distinct sweet-spot lambda visited once each) -- at low
n_soak (hammering one rung saturates). alien seed, R=2.0, N_try=8, window lam in [0.04,0.12]."""
import sys, statistics as st, functools
import torch
sys.path.insert(0, "reports/logs-2026-07-14")
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b, blob_mask, ess
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept

dev = "cuda"; RCTX = 2.5; BETA = 2.0; L_BOX = 0.368; RHO = 1.149; BULK = RHO * L_BOX ** 3
K = 8; K_R = 12; R = 2.0; N_TRY = 8; SOAK_LO, SOAK_HI = 0.04, 0.12
MB = 48; T = 20; NCAV = 3; NMUT = 2
ART = "liquid_coupling_flow/artifacts"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


@functools.lru_cache(maxsize=64)
def n_boxes(l, R):
    g = torch.arange(-int(R / l) - 1, int(R / l) + 2) * l + l / 2
    gx, gy, gz = torch.meshgrid(g, g, g, indexing="ij")
    return int((gx ** 2 + gy ** 2 + gz ** 2 < R * R).sum())


def box_ids(x, l, R):
    mm = x.norm(dim=-1) < R; ijk = torch.floor(x[mm] / l).long(); off = ijk - ijk.min(0).values
    span = off.max(0).values + 1
    return (off[:, 0] * span[1] * span[2] + off[:, 1] * span[2] + off[:, 2]).tolist()


def box_overlap(xa, xb, R, l):
    return len(set(box_ids(xa, l, R)) & set(box_ids(xb, l, R))) / (l ** 3 * n_boxes(l, R))


def two_blob_mask(n, a, gen, Kr):
    s1 = int(torch.randint(n, (), generator=gen, device=dev))
    far = (a - a[s1]).norm(dim=-1); s2 = int(far.topk(min(8, n), largest=True).indices[torch.randint(min(8, n), (), generator=gen, device=dev)])
    mk = torch.zeros(n, dtype=torch.bool, device=dev)
    mk[(a - a[s1]).norm(dim=-1).topk(min(Kr, n), largest=False).indices] = True
    mk[(a - a[s2]).norm(dim=-1).topk(min(Kr, n), largest=False).indices] = True
    return mk


@torch.no_grad()
def redist_imtm(Xb, Sb, bnd, sb, R, lam, q0, Ecur, allm, U, gen, n_try):
    """multi-try I-MTM redistribution, DYNAMIC batch B=Xb.shape[0]."""
    B, n = Xb.shape[0], Xb.shape[1]
    def w(Xc, Sc, lqu):
        q0c = m.block_log_prob_b(Xc, Sc, allm, bnd, sb, R); Ec = energy_b(Xc, Sc, bnd, sb)
        return (1 - lam) * q0c - lam * BETA * Ec - lqu, q0c, Ec
    Xr = Xb.repeat_interleave(n_try, 0); Sr = Sb.repeat_interleave(n_try, 0)
    Yf, Sf, lqf = m.sample_block_b(Xr, Sr, U, bnd, sb, R, gen=gen)
    lwf, q0f, Ef = w(Yf, Sf, lqf)
    lwf = lwf.view(B, n_try); q0f = q0f.view(B, n_try); Ef = Ef.view(B, n_try)
    Yf = Yf.view(B, n_try, n, 3); Sf = Sf.view(B, n_try, n); ar = torch.arange(B, device=dev)
    Js = torch.multinomial(torch.softmax(lwf, 1), 1, generator=gen).squeeze(1)
    Xsel, Ssel, q0sel, Esel = Yf[ar, Js], Sf[ar, Js], q0f[ar, Js], Ef[ar, Js]
    if n_try > 1:
        Xr2 = Xb.repeat_interleave(n_try - 1, 0); Sr2 = Sb.repeat_interleave(n_try - 1, 0)
        Yr, Sr2b, lqr2 = m.sample_block_b(Xr2, Sr2, U, bnd, sb, R, gen=gen)
        lwr, _, _ = w(Yr, Sr2b, lqr2); lwr = lwr.view(B, n_try - 1)
    lq_cur = m.block_log_prob_b(Xb, Sb, U, bnd, sb, R)
    lw_cur = (1 - lam) * q0 - lam * BETA * Ecur - lq_cur
    lwr_all = torch.cat([lwr, lw_cur[:, None]], 1) if n_try > 1 else lw_cur[:, None]
    la = torch.logsumexp(lwf, 1) - torch.logsumexp(lwr_all, 1)
    acc = torch.rand(B, device=dev, generator=gen).log() < la
    Xb = torch.where(acc[:, None, None], Xsel, Xb); Sb = torch.where(acc[:, None], Ssel, Sb)
    q0 = torch.where(acc, q0sel, q0); Ecur = torch.where(acc, Esel, Ecur)
    return Xb, Sb, q0, Ecur, float(acc.float().mean())


def schedule(n_extra):
    base = (torch.linspace(0, 1, T + 1, device=dev) ** 4)
    if n_extra > 0:
        base = torch.cat([base, torch.linspace(SOAK_LO, SOAK_HI, n_extra + 2, device=dev)[1:-1]])
    return torch.sort(torch.unique(base)).values


@torch.no_grad()
def run(n_soak, n_extra):
    gen = torch.Generator(device=dev).manual_seed(0); ncav = 0; qs = []; accs = []
    for ci in range(24):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xin = _mic(p["x_in"], c, L)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
        g = torch.Generator(device=dev).manual_seed(1000 + ci)
        Xb, Sb, _ = m.sample_block_b(xo[None].expand(MB, n, 3).clone(), so[None].expand(MB, n).clone(), allm, bnd, sb, R, gen=g)
        Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
        lam = schedule(n_extra); a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
        logw = torch.zeros(MB, device=dev); Kr = min(K_R, (n - 2) // 2); rd_win = []
        for t in range(1, len(lam)):
            logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur - q0)
            if ess(logw) < 0.5:
                wt = torch.softmax(logw, 0); idx = torch.multinomial(wt, MB, replacement=True, generator=g)
                Xb, Sb, Ecur, q0 = Xb[idx].clone(), Sb[idx].clone(), Ecur[idx].clone(), q0[idx].clone(); logw = torch.zeros(MB, device=dev)
            lt = float(lam[t])
            for _ in range(NMUT):
                blk = blob_mask(n, K, a, g)
                lqr = m.block_log_prob_b(Xb, Sb, blk, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xb, Sb, blk, bnd, sb, R, gen=g)
                En = energy_b(Xn, Sn, bnd, sb); q0n = None if lt == 1.0 else m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
                la = geometric_bridge_log_accept(log_q0_current=q0, log_q0_proposed=q0n, energy_current=Ecur,
                                                 energy_proposed=En, log_r_reverse=lqr, log_r_forward=lqf, lam=lt, beta=BETA)
                acc = torch.rand(MB, device=dev, generator=g).log() < la
                Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb); Ecur = torch.where(acc, En, Ecur)
                if q0n is not None:
                    q0 = torch.where(acc, q0n, q0)
            if Kr >= 4 and SOAK_LO <= lt <= SOAK_HI:
                for _ in range(n_soak):
                    U = two_blob_mask(n, a, g, Kr)
                    Xb, Sb, q0, Ecur, af = redist_imtm(Xb, Sb, bnd, sb, R, lt, q0, Ecur, allm, U, g, N_TRY)
                    rd_win.append(af)
        qs.append(st.mean([box_overlap(xin, Xb[k], R, L_BOX) for k in range(MB)]) - BULK)
        accs.append(st.mean(rd_win) if rd_win else 0.0); ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(qs), st.mean(accs)


import time
print(f"=== FAST soak: lever = extra window rungs (MB={MB} big-batch, R={R}, N_try={N_TRY}) ===", flush=True)
print(f"{'n_soak':>7} {'n_extra':>8} | {'alien q~':>9} {'win_acc':>8} {'sec':>6}", flush=True)
for n_soak, n_extra in [(3, 0), (3, 6), (3, 14), (6, 14)]:
    t0 = time.time(); q, af = run(n_soak, n_extra)
    print(f"{n_soak:>7} {n_extra:>8} | {q:>9.3f} {af:>8.3f} {time.time()-t0:>6.0f}", flush=True)
