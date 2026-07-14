"""TWO-BLOB SPECIES-EXCHANGE + BREATHE MTM (user idea) as a BASIN-CROSSING mutation. Local single-blob moves
are non-ergodic (stack-agreement: alien-seed never rises). Here: treat the union U of two well-separated
K-blobs as one cluster; transpose an unlike (0<->1) pair within U (straddling pairs => NON-LOCAL species
exchange between the two regions); BREATHE all U positions under the swapped labels via sample_block_b(
fix_species=True). Exact Hastings at pi_lam: selection uniform over unlike pairs in U (n0*n1 transposition-
invariant => selection cancels); q_pos_fwd (breathe) / q_pos_rev (pos_only score of original) exact; energy +
full-q0 rescore. TEST: does adding this move make ALIEN-seed q~ RISE toward the data plateau (cross into the
reference basin)? Compare 1blob-only vs 1blob + two-blob-breathe-swap, alien seed. lam05, l=0.368, R in
{1.6,2.0}. J=4, M=16, T=20, n_mut=3, n_sb=2 (breathe-swaps/rung)."""
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
RHO = 1.149; BULK = RHO * L_BOX ** 3; K = 8
CK = f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(CK, map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
J, M, T, NMUT, NSB, NCAV = 4, 16, 20, 3, 2, 3


def two_blob_union_mask(n, a, gen):
    """union of two well-separated K-blobs (geometric on anchors; same for all M configs)."""
    s1 = int(torch.randint(n, (), generator=gen, device=dev))
    far = (a - a[s1]).norm(dim=-1)
    s2 = int(far.topk(min(8, n), largest=True).indices[torch.randint(min(8, n), (), generator=gen, device=dev)])
    mk = torch.zeros(n, dtype=torch.bool, device=dev)
    mk[(a - a[s1]).norm(dim=-1).topk(K, largest=False).indices] = True
    mk[(a - a[s2]).norm(dim=-1).topk(K, largest=False).indices] = True
    return mk


@torch.no_grad()
def two_blob_breathe_swap(Xb, Sb, bnd, sb, R, lam, q0, Ecur, allm, a, gen):
    """One exact breathe-swap move for all M configs at a random two-blob union U."""
    n = Xb.shape[0] if Xb.dim() == 2 else Xb.shape[1]
    U = two_blob_union_mask(n, a, gen)
    Uidx = U.nonzero().squeeze(1)                                     # [2K] slots in U
    Su = Sb[:, Uidx]                                                  # [M, 2K]
    isA = (Su == 0).float(); isB = (Su == 1).float()
    ok = (isA.sum(1) > 0) & (isB.sum(1) > 0)                          # an unlike pair exists in U
    pA = torch.where(ok[:, None], isA, torch.ones_like(isA)); pB = torch.where(ok[:, None], isB, torch.ones_like(isB))
    ar = torch.arange(M, device=dev)
    iA = torch.multinomial(pA, 1, generator=gen).squeeze(1)          # uniform species-0 slot in U
    iB = torch.multinomial(pB, 1, generator=gen).squeeze(1)          # uniform species-1 slot in U
    Sp = Sb.clone()                                                   # transpose the pair's labels (count-preserving)
    Sp[ar, Uidx[iA]] = 1; Sp[ar, Uidx[iB]] = 0
    # BREATHE positions of U under the swapped labels s'
    Xn, Sn, lqf = m.sample_block_b(Xb, Sp, U, bnd, sb, R, gen=gen, fix_species=True)
    lqr = m.block_log_prob_b(Xb, Sb, U, bnd, sb, R, pos_only=True)    # reverse breathe: original positions under s
    En = energy_b(Xn, Sn, bnd, sb)
    q0n = m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)                # full boundary logq for the tempered target
    la = (1 - lam) * (q0n - q0) - lam * BETA * (En - Ecur) + lqr - lqf
    acc = (torch.rand(M, device=dev, generator=gen).log() < la) & ok
    Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb)
    Ecur = torch.where(acc, En, Ecur); q0 = torch.where(acc, q0n, q0)
    return Xb, Sb, q0, Ecur, float(acc.float().mean())


@torch.no_grad()
def island(xo, so, bnd, sb, R, seed_mode, gen, use_sb):
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    if seed_mode == "alien":
        Xb, Sb, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(), allm, bnd, sb, R, gen=gen)
    else:
        Xb = xo[None].expand(M, n, 3).clone(); Sb = so[None].expand(M, n).clone()
    Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
    lam = torch.linspace(0, 1, T + 1, device=dev) ** 4
    a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    logw = torch.zeros(M, device=dev); sb_acc = []
    for t in range(1, T + 1):
        logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur - q0)
        if ess(logw) < 0.5:
            w = torch.softmax(logw, 0); idx = torch.multinomial(w, M, replacement=True, generator=gen)
            Xb, Sb, Ecur, q0 = Xb[idx].clone(), Sb[idx].clone(), Ecur[idx].clone(), q0[idx].clone(); logw = torch.zeros(M, device=dev)
        lt = float(lam[t])
        for _ in range(NMUT):
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
        if use_sb:
            for _ in range(NSB):
                Xb, Sb, q0, Ecur, af = two_blob_breathe_swap(Xb, Sb, bnd, sb, R, lt, q0, Ecur, allm, a, gen)
                sb_acc.append(af)
    return Xb, Sb, (st.mean(sb_acc) if sb_acc else 0.0)


def run(R, use_sb, seed_mode):
    gen = torch.Generator(device=dev).manual_seed(0); ncav = 0; qs = []; accs = []
    for ci in range(24):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < 2 * K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xin = _mic(p["x_in"], c, L)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        qc = []
        for j in range(J):
            g = torch.Generator(device=dev).manual_seed((1000 if seed_mode == "alien" else 8000) + 31 * ci + j)
            Xd, Sd, af = island(xo, so, bnd, sb, R, seed_mode, g, use_sb)
            qc.append(st.mean([box_overlap(xin, Xd[k], R, L_BOX) for k in range(Xd.shape[0])]) - BULK); accs.append(af)
        qs.append(st.mean(qc)); ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(qs), (st.mean(accs) if accs else 0.0)


print(f"=== TWO-BLOB breathe-swap as basin-crossing move (lam05, l={L_BOX}) ===", flush=True)
print(f"K={K} T={T} J={J} M={M} n_mut={NMUT} n_sb={NSB} ncav={NCAV}. baseline alien(1blob) was ~0.06-0.16", flush=True)
print(f"{'R':>4} {'seed':>6} | {'1blob q~':>9} {'+2blobSwap q~':>13} {'sb_acc':>7} | verdict", flush=True)
for R in (1.6, 2.0):
    for seed_mode in ("alien", "data"):
        q_base, _ = run(R, False, seed_mode)
        q_sb, af = run(R, True, seed_mode)
        v = "CROSSES (alien rises)" if (seed_mode == "alien" and q_sb > q_base + 0.05) else \
            ("holds" if seed_mode == "data" else "no cross")
        print(f"{R:>4} {seed_mode:>6} | {q_base:>9.3f} {q_sb:>13.3f} {af:>7.3f} | {v}", flush=True)
