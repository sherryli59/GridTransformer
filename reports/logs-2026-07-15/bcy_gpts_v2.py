"""BCY Fig.2a G_PTS(R) -- CLEAN BCY-ONLY implementation. Hocky box-occupation REMOVED entirely.
Everything is BCY core overlap q_c (Eqs 5-8): Gaussian-NN w(z)=exp(-(z/0.2)^2), field integrated over
rc=0.5 core ball at cavity center, symmetrised.

Fixes vs the mosaic-bank attempt (which gave q_c collapsing 0.38->0.16 = stacks diverging into different
states => our R=2.0 looked multi-state while BCY's is single-state):
  - BOUNDARIES from the PROPERLY-EQUILIBRATED single box ka3d_dataset_N512_T0.5.pt (rho=1.2, T=0.5 ~ BCY's
    0.51), NOT the 2x2x2 mosaic-tiled T=0.55 bank (unphysical seams + hotter T weakened the pinning).
  - CONVERGENCE CERTIFICATE is BCY's own dual-init on q_c: qcA(t)=q_c(A_bottom, reference) decreasing from ~1,
    qcB(t)=q_c(B_bottom, reference) increasing from ~0; converged when running means agree within q_tol=0.1
    (BCY Sect III). NO whole-cavity Hocky metric anywhere.
  - OBSERVABLE G_PTS = q_c(A_bottom, B_bottom) between the two independent re-equilibrated stacks.
Usage: bcy_gpts_v2.py "2.0" SW NCAV   (CONV_TRACE=1 env prints q_c dual-init trajectory)."""
import sys, os, functools, time, statistics as st, math
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA, EPS
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5
RADII = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["2.0"])]
SW = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
NCAV_TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 1
CONV_TRACE = bool(int(os.environ.get("CONV_TRACE", "0")))
_BASE = [1.0000, 0.9825, 0.9640, 0.9450, 0.9250, 0.9050, 0.8850, 0.8640, 0.8423, 0.8200, 0.7960]
LAM_LADDER = []
for _i, _v in enumerate(_BASE):
    LAM_LADDER.append(_v)
    if _i + 1 < len(_BASE):
        LAM_LADDER.append(0.5 * (_v + _BASE[_i + 1]))
T_BOT = 0.50; T_DEC = 1.0; LAM_DEC = 0.8                               # match the T=0.5 boundary dataset (~BCY 0.51)
NR = len(LAM_LADDER); NCH = 2; MAX_ATTEMPT = 5
EXCH_MEAN = 10; REC = 1000; RAND_SW = 6000; Q_TOL = 0.1; STEP_MAX = 0.3
B_OV = 0.2; RC_CORE = 0.5; P_MC = 1500; K_NN = 6
OUT = os.environ.get("GPTS_OUT", "reports/logs-2026-07-15/bcy_gpts_v2.pt")
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
Xds, Sds, L = D["x"].float(), D["s"].long(), float(D["L"])
LAMs = torch.tensor(LAM_LADDER, device=dev)
Ts = T_BOT + (T_DEC - T_BOT) * (1.0 - LAMs) / (1.0 - LAM_DEC)
BETAs = 1.0 / Ts
print(f"BCY G_PTS v2 (BCY-only, no Hocky): rho={D.get('rho',1.2)} T_BOT={T_BOT} boundary=N512-T0.5 equilibrated | "
      f"R={RADII} SW={SW} NR={NR} | q_c: b={B_OV} rc={RC_CORE}", flush=True)


def bcy_qc(X, Y, gen):
    """BCY core overlap (Eqs 5-8) between two centered interior configs, symmetrised. CPU."""
    X = X.cpu(); Y = Y.cpu()
    def field_core(vpos, vq):
        u = torch.randn(P_MC, 3, generator=gen); u = u / u.norm(dim=-1, keepdim=True)
        r = RC_CORE * torch.rand(P_MC, generator=gen) ** (1.0 / 3.0); pmc = u * r[:, None]
        d2 = ((pmc[:, None] - vpos[None]) ** 2).sum(-1); w = 1.0 / (d2 + 1e-6)
        kk = min(K_NN, vpos.shape[0]); topw, idx = w.topk(kk, dim=1)
        return float(((topw * vq[idx]).sum(1) / topw.sum(1)).mean())
    qX = torch.exp(-(torch.cdist(X, Y).min(1).values / B_OV) ** 2)
    qY = torch.exp(-(torch.cdist(Y, X).min(1).values / B_OV) ** 2)
    return 0.5 * (field_core(X, qX) + field_core(Y, qY))


def pair_e(r2, sig, eps):
    inv6 = (sig ** 2 / r2.clamp_min(1e-12)) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (1.0 / 2.5) ** 6; eshift = 4 * eps * (src6 ** 2 - src6)
    return torch.where(r2 < (2.5 * sig) ** 2, e - eshift, torch.zeros_like(e))


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
        self.lam_pair = lam_pair; self.beta = BETAs[None, :, None].expand(2, NR, NCH).reshape(-1).clone()

    def full_U(self, Xm, lam_vec=None):
        B = Xm.shape[0]
        xa = torch.cat([Xm, self.bnd[None].expand(B, self.m, 3)], 1)
        d2 = torch.cdist(xa, xa) ** 2
        eye = torch.eye(self.Nt, dtype=torch.bool, device=Xm.device)
        d2 = d2.masked_fill(eye[None], 1e12)
        lam_pair = self.lam_pair if lam_vec is None else self._pair(lam_vec)
        sig = self.sig0[None] * lam_pair[:, None, :]
        e = pair_e(d2, sig, self.eps0[None]); e = e - torch.diag_embed(torch.diagonal(e, dim1=1, dim2=2))
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
    lam_pair_save = cav.lam_pair; beta_save = cav.beta
    cav.lam_pair = cav._pair(torch.full((B,), 0.60, device=dev)); cav.beta = torch.full((B,), 1.0, device=dev)
    Bmask = torch.zeros(B, dtype=torch.bool, device=dev); Bmask.view(2, NR, NCH)[1] = True
    for sw in range(RAND_SW):                                          # randomize stack B at top (T=1, lam=0.6)
        for i in torch.randperm(n, generator=g, device=dev).tolist():
            l = STEP_MAX * torch.rand(B, 1, device=dev, generator=g)
            nh = torch.randn(B, 3, device=dev, generator=g); nh = nh / nh.norm(dim=-1, keepdim=True)
            xi_new = Xm[:, i] + l * nh; ok = (xi_new.norm(dim=-1) < R) & Bmask
            dU = cav.row_U(Xm, i, xi_new) - cav.row_U(Xm, i, Xm[:, i])
            acc = ok & (torch.rand(B, device=dev, generator=g).log() < -cav.beta * dU)
            Xm[acc, i] = xi_new[acc]
    cav.lam_pair = lam_pair_save; cav.beta = beta_save
    xin_c = xin.cpu()
    qcA_run, qcB_run, qc_run = [], [], []                             # BCY dual-init on q_c + observable
    for sw in range(1, SW + 1):
        for i in torch.randperm(n, generator=g, device=dev).tolist():
            l = STEP_MAX * torch.rand(B, 1, device=dev, generator=g)
            nh = torch.randn(B, 3, device=dev, generator=g); nh = nh / nh.norm(dim=-1, keepdim=True)
            xi_new = Xm[:, i] + l * nh; ok = xi_new.norm(dim=-1) < R
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
                for stk in range(2):
                    for ch in range(NCH):
                        if swp[stk, ch]:
                            tmp = Xr[stk, r, ch].clone(); Xr[stk, r, ch] = Xr[stk, r + 1, ch]; Xr[stk, r + 1, ch] = tmp
            Xm = Xr.reshape(B, n, 3)
        if sw % REC == 0:
            Xr = Xm.view(2, NR, NCH, n, 3)
            qcA = st.mean([bcy_qc(Xr[0, 0, ch], xin_c, gcpu) for ch in range(NCH)])      # A vs ref (decreasing)
            qcB = st.mean([bcy_qc(Xr[1, 0, ch], xin_c, gcpu) for ch in range(NCH)])      # B vs ref (increasing)
            qAB = st.mean([bcy_qc(Xr[0, 0, ch], Xr[1, 0, ch], gcpu) for ch in range(NCH)])  # observable
            qcA_run.append(qcA); qcB_run.append(qcB)
            if sw > SW // 2:
                qc_run.append(qAB)
            if CONV_TRACE and sw % (REC * 3) == 0:
                hh = len(qcA_run) // 2
                rA, rB = st.mean(qcA_run[hh:]), st.mean(qcB_run[hh:])
                print(f"    [conv] sw {sw:>6}: qc(A,ref)={rA:.3f} qc(B,ref)={rB:.3f} gap={abs(rA-rB):.3f} "
                      f"| q_c(A,B)={qAB:.3f}", flush=True)
    hh = len(qcA_run) // 2
    rA, rB = st.mean(qcA_run[hh:]), st.mean(qcB_run[hh:]); gap = abs(rA - rB)
    qc = st.mean(qc_run) if qc_run else float("nan"); qc_sd = st.pstdev(qc_run) if len(qc_run) > 1 else 0.0
    return {"qcA_ref": rA, "qcB_ref": rB, "gap": gap, "converged": gap < Q_TOL,
            "G_PTS": qc, "G_PTS_sd": qc_sd, "n_qc": len(qc_run)}


results = {}; gcpu = torch.Generator().manual_seed(7)
for R in RADII:
    gsel = torch.Generator(device=dev).manual_seed(300 + int(R * 10))
    per_cav = []; nconv = 0; nattempt = 0
    for ci in torch.randperm(Xds.shape[0], generator=gsel, device=dev).tolist():
        c = torch.rand(3, generator=gsel, device=dev) * L
        p = carve(Xds[ci], Sds[ci], c, R, L)
        if p["n_in"] < 14:
            continue
        xin = _mic(p["x_in"], c, L); sin = p["s_in"]
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        t0 = time.time()
        res = run_cavity(xin, sin, bnd, sb, xin.shape[0], ci, R, gcpu)
        per_cav.append(res); results[(R, ci)] = res; torch.save({"results": results, "RADII": RADII}, OUT)
        nconv += int(res["converged"]); nattempt += 1
        print(f"  R={R} cav {ci} (n={xin.shape[0]}): G_PTS=q_c(A,B)={res['G_PTS']:+.3f}+-{res['G_PTS_sd']:.3f} | "
              f"dual-init qc(A,ref)={res['qcA_ref']:.3f} qc(B,ref)={res['qcB_ref']:.3f} gap {res['gap']:.3f} "
              f"{'CONV' if res['converged'] else 'NOT-CONV'} ({time.time()-t0:.0f}s)", flush=True)
        if nconv >= NCAV_TARGET or nattempt >= MAX_ATTEMPT:
            break
    use = [r["G_PTS"] for r in per_cav if r["converged"] and r["G_PTS"] == r["G_PTS"]]
    if len(use) < 2:
        use = [r["G_PTS"] for r in per_cav if r["G_PTS"] == r["G_PTS"]]
    gpts = st.mean(use) if use else float("nan")
    print(f"=== R={R}: G_PTS = {gpts:+.3f} over {len(use)} cavities ({nconv} converged) ===", flush=True)
    results[(R, "G_PTS")] = {"G_PTS": gpts, "n": len(use), "n_conv": nconv}
    torch.save({"results": results, "RADII": RADII}, OUT)
print(f"saved -> {OUT}", flush=True)
