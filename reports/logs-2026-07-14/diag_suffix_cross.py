"""LARGE MORTON-SUFFIX resample as the SMC mutation / basin-crossing move. The suffix move is a CLEAN q0-Gibbs
move (acc 0.95 @lam=0, vs interior blob 0). A large suffix resample draws a whole tail from q0's FULL marginal
-> can propose a different basin; energy MH (lam>0) keeps low-energy ones. Test: does alien-seed q~ CROSS toward
the data bracket when the mutation is large-suffix resamples, vs interior blobs (baseline)? Suffix size random
in [n/2, 3n/4] each move. alien + data seeds, R=2.0, MB=48, quartic T=20, n_mut=4."""
import sys, statistics as st, functools, time
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b, blob_mask, ess
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; BETA = 2.0; L_BOX = 0.368; RHO = 1.149; BULK = RHO * L_BOX ** 3
K = 8; R = 2.0; MB = 48; T = 20; NCAV = 3; NMUT = 4; ART = "liquid_coupling_flow/artifacts"
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


def move(Xb, Sb, bnd, sb, R, lt, q0, Ecur, allm, mask, gen):
    lqr = m.block_log_prob_b(Xb, Sb, mask, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xb, Sb, mask, bnd, sb, R, gen=gen)
    En = energy_b(Xn, Sn, bnd, sb); q0n = None if lt == 1.0 else m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
    la = geometric_bridge_log_accept(log_q0_current=q0, log_q0_proposed=q0n, energy_current=Ecur,
                                     energy_proposed=En, log_r_reverse=lqr, log_r_forward=lqf, lam=lt, beta=BETA)
    acc = torch.rand(Xb.shape[0], device=dev, generator=gen).log() < la
    Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb); Ecur = torch.where(acc, En, Ecur)
    if q0n is not None:
        q0 = torch.where(acc, q0n, q0)
    return Xb, Sb, q0, Ecur, float(acc.float().mean())


def run(mode, seed_mode):
    gen = torch.Generator(device=dev).manual_seed(0); ncav = 0; qs = []; accs = []
    for ci in range(24):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xin = _mic(p["x_in"], c, L)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
        g = torch.Generator(device=dev).manual_seed((1000 if seed_mode == "alien" else 8000) + ci)
        if seed_mode == "alien":
            Xb, Sb, _ = m.sample_block_b(xo[None].expand(MB, n, 3).clone(), so[None].expand(MB, n).clone(), allm, bnd, sb, R, gen=g)
        else:
            Xb = xo[None].expand(MB, n, 3).clone(); Sb = so[None].expand(MB, n).clone()
        Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
        lam = torch.linspace(0, 1, T + 1, device=dev) ** 4
        a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order); logw = torch.zeros(MB, device=dev)
        for t in range(1, T + 1):
            logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur - q0)
            if ess(logw) < 0.5:
                wt = torch.softmax(logw, 0); idx = torch.multinomial(wt, MB, replacement=True, generator=g)
                Xb, Sb, Ecur, q0 = Xb[idx].clone(), Sb[idx].clone(), Ecur[idx].clone(), q0[idx].clone(); logw = torch.zeros(MB, device=dev)
            lt = float(lam[t])
            for _ in range(NMUT):
                if mode == "suffix":
                    ks = int(torch.randint(n // 2, 3 * n // 4 + 1, (), generator=g, device=dev))  # LARGE suffix
                    mask = torch.zeros(n, dtype=torch.bool, device=dev); mask[n - ks:] = True
                else:
                    mask = blob_mask(n, K, a, g)                                                    # interior blob
                Xb, Sb, q0, Ecur, af = move(Xb, Sb, bnd, sb, R, lt, q0, Ecur, allm, mask, g); accs.append(af)
        qs.append(st.mean([box_overlap(xin, Xb[k], R, L_BOX) for k in range(MB)]) - BULK); ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(qs), st.mean(accs)


print(f"=== LARGE-SUFFIX resample as crossing move vs interior BLOB (R={R}, MB={MB}) ===", flush=True)
print(f"suffix size random [n/2, 3n/4]; NMUT={NMUT}. data bracket = upper", flush=True)
print(f"{'mode':>8} {'seed':>6} | {'q~':>7} {'acc':>6} {'sec':>5}", flush=True)
res = {}
for mode in ("blob", "suffix"):
    for seed_mode in ("alien", "data"):
        t0 = time.time(); q, af = run(mode, seed_mode); res[(mode, seed_mode)] = q
        print(f"{mode:>8} {seed_mode:>6} | {q:>7.3f} {af:>6.3f} {time.time()-t0:>5.0f}", flush=True)
print(f"\nALIEN: blob {res[('blob','alien')]:.3f} -> suffix {res[('suffix','alien')]:.3f}  "
      f"(data bracket blob {res[('blob','data')]:.3f} / suffix {res[('suffix','data')]:.3f})", flush=True)
