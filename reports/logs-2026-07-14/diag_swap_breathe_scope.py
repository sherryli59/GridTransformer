"""REFINE the swap-breathe move: the full-2-blob breathe (16 positions) gave 0.4% acceptance (clashy large-
block regen). Test whether a MINIMAL breathe -- only the swapped particles + k_b nearest interior neighbours
each -- raises acceptance to a usable level (the 2D swap-breathe used k=7 for ~1.9%). Sweep breathe scope
k_b in {1,2,3, full-blob}; measure MH acceptance + whether alien-seed q~ RISES (crosses). Selection: uniform
over (species-0, species-1) interior pairs (n0*n1 transposition-invariant => cancels); breathe mask = anchor-
KNN around the two swapped slots (position-independent => reverse mask identical => exact). lam05, l=0.368,
R=2.0, alien seed, J=3, M=16, T=16, n_sb=3."""
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
RHO = 1.149; BULK = RHO * L_BOX ** 3; K = 8; R = 2.0
CK = f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(CK, map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
J, M, T, NSB, NCAV = 3, 16, 16, 3, 3


@torch.no_grad()
def swap_breathe(Xb, Sb, bnd, sb, R, lam, q0, Ecur, allm, a, gen, k_b):
    """Swap one unlike interior pair; breathe a minimal anchor-KNN mask around the two swapped slots (k_b
    neighbours each; k_b=None => the two full K-blobs the pair belongs to). Exact Hastings at pi_lam."""
    n = Xb.shape[1]; ar = torch.arange(M, device=dev)
    isA = (Sb == 0).float(); isB = (Sb == 1).float()
    ok = (isA.sum(1) > 0) & (isB.sum(1) > 0)
    pA = torch.where(ok[:, None], isA, torch.ones_like(isA)); pB = torch.where(ok[:, None], isB, torch.ones_like(isB))
    iA = torch.multinomial(pA, 1, generator=gen).squeeze(1); iB = torch.multinomial(pB, 1, generator=gen).squeeze(1)
    Sp = Sb.clone(); Sp[ar, iA] = 1; Sp[ar, iB] = 0
    # breathe mask per config: anchor-KNN around iA and iB (position-independent => reverse-identical)
    dA = (a[None] - a[iA][:, None]).norm(dim=-1); dB = (a[None] - a[iB][:, None]).norm(dim=-1)   # [M,n]
    kk = k_b if k_b is not None else K
    mA = dA <= dA.topk(kk, largest=False).values[:, -1:]; mB = dB <= dB.topk(kk, largest=False).values[:, -1:]
    Umask = mA | mB                                                                              # [M,n] per-config
    # process configs in one batch requires a shared mask; instead loop-light over unique mask is costly ->
    # use a SHARED union across M is wrong. Do per-config via a python loop (M small).
    Xn = Xb.clone(); Sn = Sp.clone(); lqf = Xb.new_zeros(M); lqr = Xb.new_zeros(M)
    for cfgi in range(M):
        um = Umask[cfgi]
        x1, s1, lf = m.sample_block_b(Xb[cfgi:cfgi + 1], Sp[cfgi:cfgi + 1], um, bnd, sb, R, gen=gen, fix_species=True)
        lr = m.block_log_prob_b(Xb[cfgi:cfgi + 1], Sb[cfgi:cfgi + 1], um, bnd, sb, R, pos_only=True)
        Xn[cfgi] = x1[0]; lqf[cfgi] = lf[0]; lqr[cfgi] = lr[0]
    En = energy_b(Xn, Sn, bnd, sb); q0n = m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
    la = (1 - lam) * (q0n - q0) - lam * BETA * (En - Ecur) + lqr - lqf
    acc = (torch.rand(M, device=dev, generator=gen).log() < la) & ok
    Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb)
    Ecur = torch.where(acc, En, Ecur); q0 = torch.where(acc, q0n, q0)
    return Xb, Sb, q0, Ecur, float(acc.float().mean())


@torch.no_grad()
def island(xo, so, bnd, sb, R, gen, k_b):
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    Xb, Sb, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(), allm, bnd, sb, R, gen=gen)
    Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
    lam = torch.linspace(0, 1, T + 1, device=dev) ** 4
    a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    logw = torch.zeros(M, device=dev); accs = []
    for t in range(1, T + 1):
        logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur - q0)
        if ess(logw) < 0.5:
            w = torch.softmax(logw, 0); idx = torch.multinomial(w, M, replacement=True, generator=gen)
            Xb, Sb, Ecur, q0 = Xb[idx].clone(), Sb[idx].clone(), Ecur[idx].clone(), q0[idx].clone(); logw = torch.zeros(M, device=dev)
        lt = float(lam[t])
        for _ in range(2):                                             # a couple local position moves per rung
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
        for _ in range(NSB):
            Xb, Sb, q0, Ecur, af = swap_breathe(Xb, Sb, bnd, sb, R, lt, q0, Ecur, allm, a, gen, k_b)
            accs.append(af)
    return Xb, Sb, st.mean(accs)


def run(k_b):
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
            g = torch.Generator(device=dev).manual_seed(1000 + 31 * ci + j)
            Xd, Sd, af = island(xo, so, bnd, sb, R, g, k_b)
            qc.append(st.mean([box_overlap(xin, Xd[k], R, L_BOX) for k in range(Xd.shape[0])]) - BULK); accs.append(af)
        qs.append(st.mean(qc)); ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(qs), st.mean(accs)


print(f"=== swap-breathe SCOPE sweep (alien seed, R={R}, lam05, l={L_BOX}) ===", flush=True)
print(f"alien 1blob baseline ~0.06; does smaller breathe raise acceptance + make alien q~ rise?", flush=True)
print(f"{'k_breathe':>10} | {'sb_acc':>7} {'alien q~':>9}", flush=True)
for k_b in (1, 2, 3, None):
    q, af = run(k_b)
    print(f"{str(k_b if k_b is not None else 'full-blob'):>10} | {af:>7.3f} {q:>9.3f}", flush=True)
