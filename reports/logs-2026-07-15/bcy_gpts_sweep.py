"""BCY Fig.2a REPRODUCTION: radial decay of the cavity PTS correlation G_PTS(R)=<q_c> at T=0.51,
via faithful shrinkage-PT (Berthier-Charbonneau-Yaida JCP 144 024501, arXiv:1510.06320). NO learned parts.

Reuses the validated PT machinery from reports/logs-2026-07-14/bcy_shrinkage_pt.py (deformed lambda-ladder,
Metropolis exchange, dual-init certificate) with THREE changes for the G_PTS curve:
  (1) boundaries from our rho=1.2 bank (BCY use canonical KA rho=1.2; paper line 114) -- the correct density.
  (2) T_BOT = 0.51 (BCY Fig 2a target isotherm).
  (3) observable = BCY CORE overlap q_c (Eqs 5-8: Gaussian-NN w(z)=exp(-(z/0.2)^2), field integrated over
      rc=0.5 core ball, symmetrised) between the TWO INDEPENDENT stacks at the BOTTOM replica (stack A init
      from reference, stack B init from randomized-at-top -> two independent re-equilibrations, exactly BCY's
      convention). G_PTS(R) = <q_c> over production snapshots, averaged over CONVERGED cavities (dual-init
      gap < q_tol=0.1). Whole-cav Hocky q~ kept only as the convergence certificate.
Usage: bcy_gpts_sweep.py "2.0,2.5,3.0" SW NCAV_TARGET   (defaults 2.0 / 8000 / 2 for a MICRO validation)."""
import sys, functools, time, statistics as st, math
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA, EPS
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; RHO = 1.200
RADII = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["2.0"])]
SW = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
NCAV_TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 2
# BCY Table IV (T=0.51) ladder, midpoint-densified for our state point (their caveat endorses local retuning)
_BASE = [1.0000, 0.9825, 0.9640, 0.9450, 0.9250, 0.9050, 0.8850, 0.8640, 0.8423, 0.8200, 0.7960]
LAM_LADDER = []
for _i, _v in enumerate(_BASE):
    LAM_LADDER.append(_v)
    if _i + 1 < len(_BASE):
        LAM_LADDER.append(0.5 * (_v + _BASE[_i + 1]))
T_BOT = 0.51; T_DEC = 1.0; LAM_DEC = 0.8
NR = len(LAM_LADDER); NCH = 2                                          # 2 chains/replica -> 2 indep A/B pairs/snapshot
MAX_ATTEMPT = 5                                                        # cap cavities attempted per R (no runaway)
EXCH_MEAN = 10; REC = 1000; RAND_SW = 6000; Q_TOL = 0.1; STEP_MAX = 0.3   # small cavity randomizes fast at top
L_HOCKY = (0.06 / RHO) ** (1.0 / 3.0); BULK_HOCKY = 0.06; RCUT_F = 2.5
B_OV = 0.2; RC_CORE = 0.5; P_MC = 1500; K_NN = 6                       # BCY core-overlap params
OUT = "reports/logs-2026-07-15/bcy_gpts_sweep.pt"
BANK = torch.load("reports/logs-2026-07-15/bulk_ka12_T055_N4096.pt", map_location=dev, weights_only=False)
frames = BANK["bank"]; L = float(BANK["L"]); nrep = frames[0]["x"].shape[0]
LAMs = torch.tensor(LAM_LADDER, device=dev)
Ts = T_BOT + (T_DEC - T_BOT) * (1.0 - LAMs) / (1.0 - LAM_DEC)
BETAs = 1.0 / Ts
print(f"BCY G_PTS sweep: rho={RHO} T_BOT={T_BOT} | bank {len(frames)}fr x {nrep}rep | R={RADII} SW={SW} "
      f"target {NCAV_TARGET} converged/R | NR={NR} replicas", flush=True)


@functools.lru_cache(maxsize=16)
def n_boxes(l, R):
    g = torch.arange(-int(R / l) - 1, int(R / l) + 2) * l + l / 2
    gx, gy, gz = torch.meshgrid(g, g, g, indexing="ij")
    return int((gx ** 2 + gy ** 2 + gz ** 2 < R * R).sum())


def box_set(x, l, R):
    mm = x.norm(dim=-1) < R; ijk = torch.floor(x[mm] / l).long()
    h = int(R / l) + 2; side = 2 * h + 1
    return set(((ijk[:, 0] + h) * side * side + (ijk[:, 1] + h) * side + (ijk[:, 2] + h)).tolist())


def q_hocky(a, b, R):                                                  # convergence certificate only
    return len(box_set(a, L_HOCKY, R) & box_set(b, L_HOCKY, R)) / (L_HOCKY ** 3 * n_boxes(L_HOCKY, R))


def bcy_qc(X, Y, gen):
    """BCY core overlap (Eqs 5-8) between two centered interior configs, symmetrised. CPU."""
    X = X.cpu(); Y = Y.cpu()
    def field_core(vpos, vq):
        u = torch.randn(P_MC, 3, generator=gen); u = u / u.norm(dim=-1, keepdim=True)
        r = RC_CORE * torch.rand(P_MC, generator=gen) ** (1.0 / 3.0)
        pmc = u * r[:, None]
        d2 = ((pmc[:, None] - vpos[None]) ** 2).sum(-1)
        w = 1.0 / (d2 + 1e-6)
        kk = min(K_NN, vpos.shape[0]); topw, idx = w.topk(kk, dim=1)
        return float(((topw * vq[idx]).sum(1) / topw.sum(1)).mean())
    qX = torch.exp(-(torch.cdist(X, Y).min(1).values / B_OV) ** 2)
    qY = torch.exp(-(torch.cdist(Y, X).min(1).values / B_OV) ** 2)
    return 0.5 * (field_core(X, qX) + field_core(Y, qY))


def pair_e(r2, sig, eps):
    inv6 = (sig ** 2 / r2.clamp_min(1e-12)) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (1.0 / RCUT_F) ** 6; eshift = 4 * eps * (src6 ** 2 - src6)
    return torch.where(r2 < (RCUT_F * sig) ** 2, e - eshift, torch.zeros_like(e))


class Cavity:
    def __init__(self, xin, sin, bnd, sb, n):
        self.n = n; self.m = bnd.shape[0]; self.Nt = n + self.m; self.bnd = bnd
        s_all = torch.cat([sin, sb]).long()
        t_sig = torch.tensor(SIGMA, device=dev); t_eps = torch.tensor(EPS, device=dev)
        self.sig0 = t_sig[s_all[:, None], s_all[None, :]]; self.eps0 = t_eps[s_all[:, None], s_all[None, :]]
        B = 2 * NR * NCH
        lam = LAMs[None, :, None].expand(2, NR, NCH).reshape(-1)
        lam_pair = torch.empty(B, self.Nt, device=dev)
        lam_pair[:, :n] = lam[:, None]; lam_pair[:, n:] = (1 + lam[:, None]) / 2
        self.lam_pair = lam_pair; self.beta = BETAs[None, :, None].expand(2, NR, NCH).reshape(-1).clone(); self.lam = lam

    def full_U(self, Xm, lam_vec=None):
        B = Xm.shape[0]
        xa = torch.cat([Xm, self.bnd[None].expand(B, self.m, 3)], 1)
        d2 = torch.cdist(xa, xa) ** 2
        eye = torch.eye(self.Nt, dtype=torch.bool, device=Xm.device)
        d2 = d2.masked_fill(eye[None], 1e12)
        lam_pair = self.lam_pair if lam_vec is None else self._pair(lam_vec)
        sig = self.sig0[None] * lam_pair[:, None, :]
        e = pair_e(d2, sig, self.eps0[None])
        e = e - torch.diag_embed(torch.diagonal(e, dim1=1, dim2=2))
        return e[:, :self.n, :self.n].sum((1, 2)) * 0.5 + e[:, :self.n, self.n:].sum((1, 2))

    def _pair(self, lam_vec):
        B = lam_vec.shape[0]; lp = torch.empty(B, self.Nt, device=dev)
        lp[:, :self.n] = lam_vec[:, None]; lp[:, self.n:] = (1 + lam_vec[:, None]) / 2
        return lp

    def row_U(self, Xm, i, xi):
        B = Xm.shape[0]
        xa = torch.cat([Xm, self.bnd[None].expand(B, self.m, 3)], 1)
        d2 = ((xa - xi[:, None]) ** 2).sum(-1); d2[:, i] = 1e12
        sig_row = self.sig0[i][None] * self.lam_pair
        return pair_e(d2, sig_row, self.eps0[i][None]).sum(1)


def run_cavity(xin, sin, bnd, sb, n, ci, R, gcpu):
    cav = Cavity(xin, sin, bnd, sb, n); B = 2 * NR * NCH
    Xm = xin[None].expand(B, n, 3).clone()
    g = torch.Generator(device=dev).manual_seed(1500 + ci)
    # init-B randomization at TOP (T=1.0, lam=0.6)
    lam_pair_save = cav.lam_pair; beta_save = cav.beta
    cav.lam_pair = cav._pair(torch.full((B,), 0.60, device=dev)); cav.beta = torch.full((B,), 1.0, device=dev)
    Bmask = torch.zeros(B, dtype=torch.bool, device=dev); Bmask.view(2, NR, NCH)[1] = True
    U = cav.full_U(Xm)
    for sw in range(RAND_SW):
        for i in torch.randperm(n, generator=g, device=dev).tolist():
            l = STEP_MAX * torch.rand(B, 1, device=dev, generator=g)
            nh = torch.randn(B, 3, device=dev, generator=g); nh = nh / nh.norm(dim=-1, keepdim=True)
            xi_new = Xm[:, i] + l * nh
            ok = (xi_new.norm(dim=-1) < R) & Bmask
            dU = cav.row_U(Xm, i, xi_new) - cav.row_U(Xm, i, Xm[:, i])
            acc = ok & (torch.rand(B, device=dev, generator=g).log() < -cav.beta * dU)
            Xm[acc, i] = xi_new[acc]
    cav.lam_pair = lam_pair_save; cav.beta = beta_save; U = cav.full_U(Xm)
    qA_run, qB_run, qc_run = [], [], []
    ex_acc = torch.zeros(NR - 1); ex_try = torch.zeros(NR - 1)
    for sw in range(1, SW + 1):
        for i in torch.randperm(n, generator=g, device=dev).tolist():
            l = STEP_MAX * torch.rand(B, 1, device=dev, generator=g)
            nh = torch.randn(B, 3, device=dev, generator=g); nh = nh / nh.norm(dim=-1, keepdim=True)
            xi_new = Xm[:, i] + l * nh
            ok = xi_new.norm(dim=-1) < R
            dU = cav.row_U(Xm, i, xi_new) - cav.row_U(Xm, i, Xm[:, i])
            acc = ok & (torch.rand(B, device=dev, generator=g).log() < -cav.beta * dU)
            Xm[acc, i] = xi_new[acc]
        att = (torch.rand(NR - 1, generator=g, device=dev) < 1.0 / EXCH_MEAN).nonzero().squeeze(1).tolist()
        if att:
            Xr = Xm.view(2, NR, NCH, n, 3)
            for r in att:
                xa = Xr[:, r].reshape(-1, n, 3); xb = Xr[:, r + 1].reshape(-1, n, 3)
                la_v = torch.full((xa.shape[0],), float(LAMs[r]), device=dev)
                lb_v = torch.full((xa.shape[0],), float(LAMs[r + 1]), device=dev)
                dlog = -BETAs[r] * (cav.full_U(xb, la_v) - cav.full_U(xa, la_v)) \
                       - BETAs[r + 1] * (cav.full_U(xa, lb_v) - cav.full_U(xb, lb_v))
                swp = (torch.rand(xa.shape[0], device=dev, generator=g).log() < dlog).view(2, NCH)
                ex_acc[r] += float(swp.float().sum()); ex_try[r] += swp.numel()
                for stk in range(2):
                    for ch in range(NCH):
                        if swp[stk, ch]:
                            tmp = Xr[stk, r, ch].clone(); Xr[stk, r, ch] = Xr[stk, r + 1, ch]; Xr[stk, r + 1, ch] = tmp
            Xm = Xr.reshape(B, n, 3)
        if sw % REC == 0:
            Xr = Xm.view(2, NR, NCH, n, 3)
            qA = st.mean([q_hocky(Xr[0, 0, ch].cpu(), xin.cpu(), R) - BULK_HOCKY for ch in range(NCH)])
            qB = st.mean([q_hocky(Xr[1, 0, ch].cpu(), xin.cpu(), R) - BULK_HOCKY for ch in range(NCH)])
            qA_run.append(qA); qB_run.append(qB)
            if sw > SW // 2:                                            # production: BCY q_c between indep stacks
                qc_run += [bcy_qc(Xr[0, 0, ch], Xr[1, 0, ch], gcpu) for ch in range(NCH)]
    half = len(qA_run) // 2
    rA, rB = st.mean(qA_run[half:]), st.mean(qB_run[half:]); gap = abs(rA - rB)
    qc = st.mean(qc_run) if qc_run else float("nan"); qc_sd = st.pstdev(qc_run) if len(qc_run) > 1 else 0.0
    return {"qA": rA, "qB": rB, "gap": gap, "converged": gap < Q_TOL, "qc": qc, "qc_sd": qc_sd,
            "n_qc": len(qc_run), "exch_med": float((ex_acc / ex_try.clamp(min=1)).median())}


results = {}; gcpu = torch.Generator().manual_seed(7)
for R in RADII:
    gsel = torch.Generator(device=dev).manual_seed(300 + int(R * 10))
    per_cav = []; nconv = 0; nattempt = 0
    for ci in range(40):
        fr = frames[-1 - ci % len(frames)]; b = ci % nrep
        c = torch.rand(3, generator=gsel, device=dev) * L
        p = carve(fr["x"][b], fr["s"][b].long(), c, R, L)
        if p["n_in"] < 14:
            continue
        xin = _mic(p["x_in"], c, L); sin = p["s_in"]
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        t0 = time.time()
        res = run_cavity(xin, sin, bnd, sb, xin.shape[0], ci, R, gcpu)
        per_cav.append(res); results[(R, ci)] = res; torch.save({"results": results, "RADII": RADII}, OUT)
        nconv += int(res["converged"]); nattempt += 1
        print(f"  R={R} cav {ci} (n={xin.shape[0]}): q_c={res['qc']:+.3f}+-{res['qc_sd']:.3f} (n_qc {res['n_qc']}) "
              f"| certif gap {res['gap']:.3f} {'CONV' if res['converged'] else 'NOT-CONV'} "
              f"| exch-med {res['exch_med']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        if nconv >= NCAV_TARGET or nattempt >= MAX_ATTEMPT:
            break
    conv_qc = [r["qc"] for r in per_cav if r["converged"] and r["qc"] == r["qc"]]
    all_qc = [r["qc"] for r in per_cav if r["qc"] == r["qc"]]
    use = conv_qc if len(conv_qc) >= 2 else all_qc                     # converged-preferred, fall back to all
    gpts = st.mean(use) if use else float("nan")
    gsd = st.pstdev(use) / max(len(use) ** .5, 1) if len(use) > 1 else 0.0
    results[(R, "G_PTS")] = {"G_PTS": gpts, "n_conv": len(conv_qc), "n_cav": len(per_cav), "used_all": len(conv_qc) < 2}
    torch.save({"results": results, "RADII": RADII}, OUT)
    print(f"=== R={R}: G_PTS = <q_c> = {gpts:+.3f} +- {gsd:.3f} over {len(use)} cavities "
          f"({len(conv_qc)} converged{'; FELL BACK to all' if len(conv_qc) < 2 else ''}) ===", flush=True)
print(f"saved -> {OUT}", flush=True)
