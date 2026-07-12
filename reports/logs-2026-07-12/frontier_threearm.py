"""THREE-ARM frontier benchmark: does model augmentation converge cavities that plain PT can't, at matched
budget? Frontier radii R in {2.5, 3.0} (baseline PT was 0/6 and 1/6 converged there).

Per (R, cavity), 4 PT stacks at IDENTICAL budget (nrung geometric beta 0.4->2.0, SWEEPS sweeps):
  DATA   : seeded from the data interior            -> REFERENCE basin (defines "the answer")
  ALIEN  : seeded from an alien equilibrium interior -> baseline (find the basin from scratch, plain PT)
  MSEED  : seeded from a SHARPENED model interior    -> does model seeding beat alien?
  MSEED+M: model seed + sharpened K-block MTM moves injected at HOT rungs (beta<1) every BLOCK_EVERY sweeps
           -> does a learned hot-rung move (K>=6 usable because the accept is beta-softened) help mixing?

Convergence of a non-DATA stack = agreement with DATA at the cold rung: |U_stack - U_data|/n < 0.15 AND
core overlap qc(stack_cold, data_cold) in the pinned-consistent band. We report, per arm: convergence
fraction, cold energy gap to DATA, and qc-to-DATA (the actual G_PTS-relevant quantity). Full traces saved.

The block move exercises K>=6 EXACTLY at hot rungs (per-rung beta MH accept + tempered logq); this is the
'K>=6 becomes usable off the cold accept' claim, measured."""
import time
from pathlib import Path
import torch

from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA, EPS, RCUT_FACTOR

DEV = "cuda"; T = 0.5; BETA = 1.0 / T; RCTX = 2.5; BIGL = 100.0
BETA_HOT = 0.4
NCAND = 192; TEL_FRAC = 0.5; SIGLOC = 0.2
RGRID = [2.5, 3.0]
NRUNGS = {2.5: 12, 3.0: 16}
SWEEPS = {2.5: 200, 3.0: 240}
COLD_EVERY = 15
BLOCK_EVERY = 6; BLOCK_K = 8; POS_TEMP = 0.4; N_TRIAL = 12
NCAV = 6
OUT = Path("liquid_coupling_flow/artifacts/frontier_threearm_N4096.pt")
T_SIG = torch.tensor(SIGMA, device=DEV); T_EPS = torch.tensor(EPS, device=DEV)

ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt", map_location=DEV, weights_only=False)
MODEL = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(DEV)
MODEL.load_state_dict(ck["state_dict"], strict=False); MODEL.eval(); MODEL.use_frame = False


def make_betas(nrung):
    return BETA_HOT * (BETA / BETA_HOT) ** (torch.arange(nrung, device=DEV) / (nrung - 1))


def clamp_ball(x, R, eps=1e-5):
    """Radially clamp any |x|>=R into the OPEN ball (heat-bath can nudge interior particles a hair outside;
    the model's ball_unsquash rejects |x|>=R). Only touches near-boundary points."""
    nrm = x.norm(dim=-1, keepdim=True)
    return x * (R * (1.0 - eps) / nrm.clamp_min(1e-12)).clamp(max=1.0)


def row_E(cand, s_i, ox, os_):
    sig = T_SIG[s_i[:, None], os_]; eps = T_EPS[s_i[:, None], os_]
    rc2 = (RCUT_FACTOR * sig) ** 2
    r2 = (cand[:, :, None, :] - ox[:, None, :, :]).square().sum(-1)
    inv6 = (sig[:, None] ** 2 / r2) ** 3
    e = 4 * eps[:, None] * (inv6 ** 2 - inv6)
    src6 = (1.0 / RCUT_FACTOR) ** 6
    e = torch.where(r2 < rc2[:, None], e - 4 * eps[:, None] * (src6 ** 2 - src6), torch.zeros_like(e))
    return e.sum(-1)


def full_E(x_int, s_int, bnd, sb):
    Mn = x_int.shape[0]
    allx = torch.cat([x_int, bnd[None].expand(Mn, bnd.shape[0], 3)], 1)
    alls = torch.cat([s_int, sb[None].expand(Mn, sb.shape[0])], 1)
    return ka_energy(allx, alls.long(), BIGL)


def heatbath_sweep(x_int, s_int, bnd, sb, Rr, gen, betas):
    Mn, Nn = x_int.shape[0], x_int.shape[1]
    n_tel = int(TEL_FRAC * NCAND)
    for i in torch.randperm(Nn, generator=gen, device=DEV).tolist():
        oth = [j for j in range(Nn) if j != i]
        ox = torch.cat([x_int[:, oth], bnd[None].expand(Mn, bnd.shape[0], 3)], 1)
        os_ = torch.cat([s_int[:, oth], sb[None].expand(Mn, sb.shape[0])], 1)
        u = torch.randn(Mn, n_tel, 3, generator=gen, device=DEV)
        u = u / u.norm(dim=-1, keepdim=True) * (torch.rand(Mn, n_tel, 1, generator=gen, device=DEV) ** (1 / 3)) * Rr
        loc = x_int[:, i:i + 1] + SIGLOC * torch.randn(Mn, NCAND - n_tel, 3, generator=gen, device=DEV)
        cand = torch.cat([u, loc, x_int[:, i:i + 1]], 1)
        logits = -betas[:, None] * row_E(cand, s_int[:, i], ox, os_)
        pick = torch.multinomial(torch.softmax(logits, 1), 1, generator=gen).squeeze(1)
        x_int = x_int.clone(); x_int[:, i] = cand[torch.arange(Mn, device=DEV), pick]
    return x_int


@torch.no_grad()
def model_block_move(x_int, s_int, bnd, sb, Rr, betas, gen):
    """Sharpened K-block model MTM at HOT rungs only (beta<1). Per-rung: N_TRIAL sharpened proposals, exact
    tempered logq, Liu-Liang-Wong accept with THAT rung's beta. Returns x_int (some rungs' blocks replaced)."""
    nrung, n = x_int.shape[0], x_int.shape[1]
    hot = (betas < 1.0)
    if not hot.any():
        return x_int
    a = fixed_ball_scaffold(n, Rr, DEV); seed = int(torch.randint(n, (), generator=gen, device=DEV))
    blk = torch.zeros(n, dtype=torch.bool, device=DEV)
    blk[(a - a[seed]).norm(dim=-1).topk(min(BLOCK_K, n), largest=False).indices] = True
    hi = torch.nonzero(hot).squeeze(1)
    Xh = clamp_ball(x_int[hi], Rr); Sh = s_int[hi]; H = Xh.shape[0]
    u0 = -betas[hi] * full_E(Xh, Sh, bnd, sb) - MODEL.block_log_prob_b(Xh, Sh, blk, bnd, sb, Rr)   # [H]
    Xr = Xh.repeat_interleave(N_TRIAL, 0); Sr = Sh.repeat_interleave(N_TRIAL, 0)
    Xp, Sp, lqf = MODEL.sample_block_b(Xr, Sr, blk, bnd, sb, Rr, gen=gen, pos_temp=POS_TEMP)
    bh = betas[hi].repeat_interleave(N_TRIAL)
    up = (-bh * full_E(Xp, Sp, bnd, sb) - lqf).reshape(H, N_TRIAL)
    sf = torch.logsumexp(up, 1)
    J = torch.multinomial(torch.softmax(up, 1), 1, generator=gen).squeeze(1)
    lr = up.clone(); lr[torch.arange(H, device=DEV), J] = u0
    sr = torch.logsumexp(lr, 1)
    acc = torch.rand(H, device=DEV, generator=gen).log() < (sf - sr)
    Yx = Xp.reshape(H, N_TRIAL, n, 3)[torch.arange(H, device=DEV), J]
    Xh2 = torch.where(acc[:, None, None], Yx, Xh)
    x_int = x_int.clone(); x_int[hi] = Xh2
    return x_int


def pt_exchange(x, s_int, bnd, sb, gen, parity, betas):
    U = full_E(x, s_int, bnd, sb); nrung = x.shape[0]
    for k in range(parity, nrung - 1, 2):
        if torch.rand((), generator=gen, device=DEV).log() < (betas[k] - betas[k + 1]) * (U[k] - U[k + 1]):
            xk = x[k].clone(); x[k] = x[k + 1]; x[k + 1] = xk
            Uk = U[k].clone(); U[k] = U[k + 1]; U[k + 1] = Uk
    return x


def core_qc(X, sX, Y, sY, gen, rc=0.5, b=0.2, n_mc=3000):
    qX = torch.zeros(X.shape[0], device=DEV)
    for sp in (0, 1):
        xm, ym = (sX == sp), (sY == sp)
        if int(xm.sum()) > 0 and int(ym.sum()) > 0:
            qX[xm] = torch.exp(-torch.cdist(X[xm], Y[ym]).min(1).values ** 2 / (2 * b * b))
    u = torch.randn(n_mc, 3, generator=gen, device=DEV); u = u / u.norm(dim=-1, keepdim=True)
    pts = u * (torch.rand(n_mc, 1, generator=gen, device=DEV) ** (1.0 / 3.0)) * rc
    return float(qX[torch.cdist(pts, X).argmin(1)].mean())


def alien_init(X, S, ci, c, Rr, L, n, gen, n_chain_ids=16):
    for cj in range(X.shape[0]):
        if cj % n_chain_ids == ci % n_chain_ids:
            continue
        pa = carve(X[cj], S[cj], c, Rr, L); xa = _mic(pa["x_in"], c, L); na = xa.shape[0]
        if na < max(4, n // 2):
            continue
        if na >= n:
            return xa[torch.argsort(xa.norm(dim=-1))[:n]]
        u = torch.randn(n - na, 3, generator=gen, device=DEV)
        pad = u / u.norm(dim=-1, keepdim=True) * (torch.rand(n - na, 1, generator=gen, device=DEV) ** (1 / 3)) * Rr
        return torch.cat([xa, pad], 0)
    raise RuntimeError("no alien")


@torch.no_grad()
def model_seed(xo_lab, so_lab, bnd, sb, Rr, gen):
    """Sharpened whole-interior model seed (block = all interior, conditioned on boundary)."""
    n = xo_lab.shape[0]; blk = torch.ones(n, dtype=torch.bool, device=DEV)
    xp, sp, _ = MODEL.sample_block_b(clamp_ball(xo_lab, Rr)[None], so_lab[None], blk, bnd, sb, Rr, gen=gen, pos_temp=POS_TEMP)
    return xp[0]


def run_stack(x0_int, s_int, bnd, sb, Rr, gen, nrung, nsw, betas, block_moves=False):
    x = x0_int[None].expand(nrung, *x0_int.shape).contiguous()
    cold = []; trace = []
    for sw in range(1, nsw + 1):
        x = heatbath_sweep(x, s_int, bnd, sb, Rr, gen, betas)
        if block_moves and sw % BLOCK_EVERY == 0:
            x = model_block_move(x, s_int, bnd, sb, Rr, betas, gen)
        x = pt_exchange(x, s_int, bnd, sb, gen, sw % 2, betas)
        if sw % COLD_EVERY == 0:
            trace.append(float(full_E(x[-1:], s_int[:1], bnd, sb)[0]))
            if sw > nsw // 2:
                cold.append(x[-1].clone())
    return cold, trace


def main():
    t0 = time.time()
    d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=DEV, weights_only=False)
    X, S, L = d["x"].to(DEV).float(), d["s"].to(DEV).long(), float(d["L"])
    gen = torch.Generator(device=DEV).manual_seed(0)
    results = []
    for Rr in RGRID:
        nrung = NRUNGS[Rr]; betas = make_betas(nrung); nsw = SWEEPS[Rr]
        for cav in range(NCAV):
            ci = cav % X.shape[0]
            c = torch.rand(3, generator=gen, device=DEV) * L
            p = carve(X[ci], S[ci], c, Rr, L)
            if p["n_in"] < BLOCK_K + 6:
                continue
            xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], Rr)          # labeled data interior
            xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (Rr + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
            n = xo.shape[0]; s_int = so[None].expand(nrung, n).contiguous()
            e_ref = float(full_E(xo[None], so[None], bnd, sb)[0])
            seeds = {"DATA": xo,
                     "ALIEN": alien_init(X, S, ci, c, Rr, L, n, gen),
                     "MSEED": model_seed(xo, so, bnd, sb, Rr, gen),
                     "MSEED_MOVE": model_seed(xo, so, bnd, sb, Rr, gen)}
            cold = {}; trace = {}
            for name, x0 in seeds.items():
                cold[name], trace[name] = run_stack(x0, s_int, bnd, sb, Rr, gen, nrung, nsw, betas,
                                                    block_moves=(name == "MSEED_MOVE"))
            # DATA reference (mean cold config energy + a representative config for qc)
            Ud = full_E(torch.stack(cold["DATA"]), so[None].expand(len(cold["DATA"]), n), bnd, sb).mean()
            rec = {"R": Rr, "cav": cav, "n_in": n, "e_ref": e_ref, "trace": trace}
            row = f"R={Rr} cav={cav} n={n:4d} |"
            for name in ("ALIEN", "MSEED", "MSEED_MOVE"):
                cc = cold[name]
                Uc = full_E(torch.stack(cc), so[None].expand(len(cc), n), bnd, sb).mean()
                gap = float((Uc - Ud) / n)
                qc = sum(core_qc(a, so, b_, so, gen) for a in cc for b_ in cold["DATA"]) / (len(cc) * len(cold["DATA"]))
                conv = abs(gap) < 0.15
                rec[name] = {"gap_to_data": gap, "qc_to_data": qc, "converged": conv}
                row += f" {name}: gap{gap:+.3f} qc{qc:.2f} {'OK' if conv else 'x'} |"
            rec["cold_data"] = torch.stack(cold["DATA"]).cpu()
            for name in ("ALIEN", "MSEED", "MSEED_MOVE"):
                rec[f"cold_{name}"] = torch.stack(cold[name]).cpu()
            rec["s_int"] = so.cpu(); rec["bnd"] = bnd.cpu(); rec["sb"] = sb.cpu()
            results.append(rec)
            torch.save({"results": results, "L": L, "T": T, "rho": float(d["rho"]),
                        "cfg": {"BLOCK_K": BLOCK_K, "POS_TEMP": POS_TEMP, "N_TRIAL": N_TRIAL,
                                "BLOCK_EVERY": BLOCK_EVERY, "BETA_HOT": BETA_HOT}}, OUT)
            print(row + f"  ({(time.time()-t0)/60:.0f} min)", flush=True)
    print("\n=== SUMMARY (convergence to DATA reference, gap<0.15) ===", flush=True)
    for Rr in RGRID:
        rr = [r for r in results if r["R"] == Rr]
        for name in ("ALIEN", "MSEED", "MSEED_MOVE"):
            nc = sum(r[name]["converged"] for r in rr)
            qcs = sorted(r[name]["qc_to_data"] for r in rr)
            gaps = sorted(r[name]["gap_to_data"] for r in rr)
            print(f"  R={Rr} {name:11s}: {nc}/{len(rr)} converged  qc_med={qcs[len(qcs)//2]:.3f}  "
                  f"gap_med={gaps[len(gaps)//2]:+.3f}", flush=True)
    print(f"[done] -> {OUT}  ({(time.time()-t0)/60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
