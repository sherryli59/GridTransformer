"""LEVERS 1+2 on the two-blob redistribution move. Prior single-try redist lifted alien only 0.03->0.13 at
R=2.0 with 0.5% acceptance (clashy big-block regen rarely accepted). Fix:
  LEVER 1  low-lambda CONCENTRATION: fire redistribution only in permissive early rungs (lam<LAM_HI), more
           attempts there (acceptance is highest where the tempered target is soft).
  LEVER 2  multi-try I-MTM: draw N_TRY union regens, SELECT the best by the tempered weight, exact Liu I-MTM
           acceptance (union regen is an INDEPENDENCE proposal given the fixed surroundings, so w=pi_lam/g).
           log w = (1-lam)*q0_full - lam*beta*U - g_union ; accept = logsumexp(w_fwd) - logsumexp(w_rev),
           reverse = (N_TRY-1) fresh regens + the current union.
Compare ALIEN-seed q~: baseline (no redist) / single-try low-lam / multi-try low-lam. Does alien rise toward
the data bracket + acceptance climb? lam05, l=0.368, R=2.0 (the case that showed a lift), J=4 M=16 T=20."""
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
J, M, T, NMUT, NCAV, R = 4, 16, 20, 2, 3, 2.0
N_TRY = 8; LAM_HI = 0.6; NRD_LOW = 3          # levers: multi-try count, low-lambda cutoff, attempts/low-rung


def two_blob_mask(n, a, gen, Kr):
    s1 = int(torch.randint(n, (), generator=gen, device=dev))
    far = (a - a[s1]).norm(dim=-1); s2 = int(far.topk(min(8, n), largest=True).indices[torch.randint(min(8, n), (), generator=gen, device=dev)])
    mk = torch.zeros(n, dtype=torch.bool, device=dev)
    mk[(a - a[s1]).norm(dim=-1).topk(min(Kr, n), largest=False).indices] = True
    mk[(a - a[s2]).norm(dim=-1).topk(min(Kr, n), largest=False).indices] = True
    return mk


@torch.no_grad()
def redist_imtm(Xb, Sb, bnd, sb, R, lam, q0, Ecur, allm, U, gen, n_try):
    """Multi-try I-MTM redistribution on union U. Independence proposal g = union regen (pos+species).
    log w = (1-lam)*q0_full - lam*beta*U_energy - g_union. Returns updated Xb,Sb,q0,Ecur, accept-frac."""
    n = Xb.shape[1]
    def weights(Xc, Sc, lq_union):                                   # [.,n,3],[.,n],[.] -> log w [.]
        q0c = m.block_log_prob_b(Xc, Sc, allm, bnd, sb, R)
        Ec = energy_b(Xc, Sc, bnd, sb)
        return (1 - lam) * q0c - lam * BETA * Ec - lq_union, q0c, Ec
    # forward: n_try union regens per config
    Xrep = Xb.repeat_interleave(n_try, 0); Srep = Sb.repeat_interleave(n_try, 0)
    Yf, Sf, lqf = m.sample_block_b(Xrep, Srep, U, bnd, sb, R, gen=gen)
    lwf, q0f, Ef = weights(Yf, Sf, lqf)
    lwf = lwf.view(M, n_try); q0f = q0f.view(M, n_try); Ef = Ef.view(M, n_try)
    Yf = Yf.view(M, n_try, n, 3); Sf = Sf.view(M, n_try, n)
    Jsel = torch.multinomial(torch.softmax(lwf, 1), 1, generator=gen).squeeze(1)     # [M]
    ar = torch.arange(M, device=dev)
    Xsel = Yf[ar, Jsel]; Ssel = Sf[ar, Jsel]; q0sel = q0f[ar, Jsel]; Esel = Ef[ar, Jsel]
    # reverse: (n_try-1) fresh regens from CURRENT (independence proposal) + the current union
    if n_try > 1:
        Xr2 = Xb.repeat_interleave(n_try - 1, 0); Sr2 = Sb.repeat_interleave(n_try - 1, 0)
        Yr, Sr, lqr2 = m.sample_block_b(Xr2, Sr2, U, bnd, sb, R, gen=gen)
        lwr, _, _ = weights(Yr, Sr, lqr2); lwr = lwr.view(M, n_try - 1)
    lq_cur = m.block_log_prob_b(Xb, Sb, U, bnd, sb, R)                                # current union density
    lw_cur = (1 - lam) * q0 - lam * BETA * Ecur - lq_cur                              # [M]
    lwr_all = torch.cat([lwr, lw_cur[:, None]], 1) if n_try > 1 else lw_cur[:, None]
    la = torch.logsumexp(lwf, 1) - torch.logsumexp(lwr_all, 1)
    acc = torch.rand(M, device=dev, generator=gen).log() < la
    Xb = torch.where(acc[:, None, None], Xsel, Xb); Sb = torch.where(acc[:, None], Ssel, Sb)
    q0 = torch.where(acc, q0sel, q0); Ecur = torch.where(acc, Esel, Ecur)
    return Xb, Sb, q0, Ecur, float(acc.float().mean())


@torch.no_grad()
def island(xo, so, bnd, sb, R, seed_mode, gen, mode):
    """mode: 'base' (no redist), 'single' (lever1 low-lam single-try), 'multi' (levers1+2 low-lam I-MTM)."""
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    if seed_mode == "alien":
        Xb, Sb, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(), allm, bnd, sb, R, gen=gen)
    else:
        Xb = xo[None].expand(M, n, 3).clone(); Sb = so[None].expand(M, n).clone()
    Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
    lam = torch.linspace(0, 1, T + 1, device=dev) ** 4
    a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    logw = torch.zeros(M, device=dev); rd_acc = []
    Kr = min(K_R, (n - 2) // 2)
    for t in range(1, T + 1):
        logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur - q0)
        if ess(logw) < 0.5:
            w = torch.softmax(logw, 0); idx = torch.multinomial(w, M, replacement=True, generator=gen)
            Xb, Sb, Ecur, q0 = Xb[idx].clone(), Sb[idx].clone(), Ecur[idx].clone(), q0[idx].clone(); logw = torch.zeros(M, device=dev)
        lt = float(lam[t])
        for _ in range(NMUT):                                        # local single-blob moves (all rungs)
            blk = blob_mask(n, K, a, gen)
            lqr = m.block_log_prob_b(Xb, Sb, blk, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xb, Sb, blk, bnd, sb, R, gen=gen)
            En = energy_b(Xn, Sn, bnd, sb); q0n = None if lt == 1.0 else m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
            la = geometric_bridge_log_accept(log_q0_current=q0, log_q0_proposed=q0n, energy_current=Ecur,
                                             energy_proposed=En, log_r_reverse=lqr, log_r_forward=lqf, lam=lt, beta=BETA)
            acc = torch.rand(M, device=dev, generator=gen).log() < la
            Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb)
            Ecur = torch.where(acc, En, Ecur)
            if q0n is not None:
                q0 = torch.where(acc, q0n, q0)
        if mode != "base" and Kr >= 4 and lt < LAM_HI:               # LEVER 1: redist only at low lambda
            for _ in range(NRD_LOW):
                U = two_blob_mask(n, a, gen, Kr)
                nt = N_TRY if mode == "multi" else 1                 # LEVER 2: multi-try vs single
                Xb, Sb, q0, Ecur, af = redist_imtm(Xb, Sb, bnd, sb, R, lt, q0, Ecur, allm, U, gen, nt)
                rd_acc.append(af)
    return Xb, Sb, (st.mean(rd_acc) if rd_acc else 0.0)


def run(seed_mode, mode):
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
            Xd, Sd, af = island(xo, so, bnd, sb, R, seed_mode, g, mode); accs.append(af)
            qc.append(st.mean([box_overlap(xin, Xd[k], R, L_BOX) for k in range(Xd.shape[0])]) - BULK)
        qs.append(st.mean(qc)); ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(qs), (st.mean(accs) if accs else 0.0)


if __name__ == "__main__":                       # guard so importing (two_blob_mask/redist_imtm) does NOT re-run
    print(f"=== two-blob redist LEVERS 1+2 (low-lam concentration + multi-try I-MTM), R={R} lam05 ===", flush=True)
    print(f"N_try={N_TRY} lam_hi={LAM_HI} n_rd_low={NRD_LOW} | K={K} Kr<={K_R} T={T} J={J} M={M} ncav={NCAV}", flush=True)
    print(f"prior single-try uniform: alien 0.033->0.128 @0.5%acc; data ~0.96 (gap 0.83)", flush=True)
    print(f"{'mode':>8} {'seed':>6} | {'q~':>7} {'rd_acc':>7}", flush=True)
    res = {}
    for mode in ("base", "single", "multi"):
        for seed_mode in ("alien", "data"):
            q, af = run(seed_mode, mode); res[(mode, seed_mode)] = q
            print(f"{mode:>8} {seed_mode:>6} | {q:>7.3f} {af:>7.3f}", flush=True)
    print(f"\nGAP (data-alien): base {res[('base','data')]-res[('base','alien')]:.3f}  "
          f"single {res[('single','data')]-res[('single','alien')]:.3f}  "
          f"multi {res[('multi','data')]-res[('multi','alien')]:.3f}  (smaller => closer to convergence)", flush=True)
