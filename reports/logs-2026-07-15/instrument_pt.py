"""INSTRUMENTED shrinkage-PT: measure WHY stack B can't reach the reference basin.
Tracks (over one R=2.0 cavity, N512 T=0.5 boundary): ROUND-TRIP count (configs traversing top<->bottom),
per-rung EXCHANGE acceptance (locate any bottleneck), and COLD-replica ENERGY of stack A vs B (jammed-high
=> ladder can't feed it a good config; low-but-different => genuine 2nd basin). Also qc(A,ref), qc(B,ref).
If round-trips ~ 0 => the cold replica never decorrelates => mixing starvation (the diagnosis), not a formula bug."""
import sys, time, statistics as st
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA, EPS
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; R = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
SW = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
EXCH_MEAN = int(sys.argv[3]) if len(sys.argv) > 3 else 10               # sweeps between exchange attempts per pair
_BASE = [1.0000, 0.9825, 0.9640, 0.9450, 0.9250, 0.9050, 0.8850, 0.8640, 0.8423, 0.8200, 0.7960]
LAM_LADDER = []
for _i, _v in enumerate(_BASE):
    LAM_LADDER.append(_v)
    if _i + 1 < len(_BASE):
        LAM_LADDER.append(0.5 * (_v + _BASE[_i + 1]))
T_BOT = 0.50; T_DEC = 1.0; LAM_DEC = 0.8
NR = len(LAM_LADDER); NCH = 2; RAND_SW = 6000; STEP_MAX = 0.3; REC = 1500
B_OV = 0.2; RC_CORE = 0.5; P_MC = 1200; K_NN = 6
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
Xds, Sds, L = D["x"].float(), D["s"].long(), float(D["L"])
LAMs = torch.tensor(LAM_LADDER, device=dev)
Ts = T_BOT + (T_DEC - T_BOT) * (1.0 - LAMs) / (1.0 - LAM_DEC); BETAs = 1.0 / Ts
gcpu = torch.Generator().manual_seed(7)
print(f"INSTRUMENT PT: R={R} NR={NR} EXCH_MEAN={EXCH_MEAN} SW={SW} | ladder lam {LAM_LADDER[0]:.2f}->{LAM_LADDER[-1]:.3f} "
      f"T {float(Ts[0]):.2f}->{float(Ts[-1]):.2f}", flush=True)


def bcy_qc(X, Y):
    X = X.cpu(); Y = Y.cpu()
    def fc(vpos, vq):
        u = torch.randn(P_MC, 3, generator=gcpu); u = u / u.norm(dim=-1, keepdim=True)
        r = RC_CORE * torch.rand(P_MC, generator=gcpu) ** (1.0 / 3.0); pmc = u * r[:, None]
        d2 = ((pmc[:, None] - vpos[None]) ** 2).sum(-1); w = 1.0 / (d2 + 1e-6)
        kk = min(K_NN, vpos.shape[0]); tw, idx = w.topk(kk, dim=1)
        return float(((tw * vq[idx]).sum(1) / tw.sum(1)).mean())
    qX = torch.exp(-(torch.cdist(X, Y).min(1).values / B_OV) ** 2)
    qY = torch.exp(-(torch.cdist(Y, X).min(1).values / B_OV) ** 2)
    return 0.5 * (fc(X, qX) + fc(Y, qY))


def pair_e(r2, sig, eps):
    inv6 = (sig ** 2 / r2.clamp_min(1e-12)) ** 3; e = 4 * eps * (inv6 ** 2 - inv6)
    s6 = (1.0 / 2.5) ** 6; return torch.where(r2 < (2.5 * sig) ** 2, e - 4 * eps * (s6 ** 2 - s6), torch.zeros_like(e))


gsel = torch.Generator(device=dev).manual_seed(300 + int(R * 10))
ci = int(torch.randperm(Xds.shape[0], generator=gsel, device=dev)[0])
c = torch.rand(3, generator=gsel, device=dev) * L
p = carve(Xds[ci], Sds[ci], c, R, L)
xin = _mic(p["x_in"], c, L); sin = p["s_in"].long(); n = xin.shape[0]
xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm].long()
m = bnd.shape[0]; Nt = n + m
s_all = torch.cat([sin, sb]).long()
t_sig = torch.tensor(SIGMA, device=dev); t_eps = torch.tensor(EPS, device=dev)
sig0 = t_sig[s_all[:, None], s_all[None, :]]; eps0 = t_eps[s_all[:, None], s_all[None, :]]
B = 2 * NR * NCH
lam0 = LAMs[None, :, None].expand(2, NR, NCH).reshape(-1)
beta0 = BETAs[None, :, None].expand(2, NR, NCH).reshape(-1).clone()


def lam_pair_of(lam):
    lp = torch.empty(lam.shape[0], Nt, device=dev); lp[:, :n] = lam[:, None]; lp[:, n:] = (1 + lam[:, None]) / 2
    return lp


def full_U(Xm, lam):
    Bb = Xm.shape[0]
    xa = torch.cat([Xm, bnd[None].expand(Bb, m, 3)], 1); d2 = torch.cdist(xa, xa) ** 2
    d2 = d2.masked_fill(torch.eye(Nt, dtype=torch.bool, device=dev)[None], 1e12)
    sig = sig0[None] * lam_pair_of(lam)[:, None, :]
    e = pair_e(d2, sig, eps0[None]); e = e - torch.diag_embed(torch.diagonal(e, dim1=1, dim2=2))
    return e[:, :n, :n].sum((1, 2)) * 0.5 + e[:, :n, n:].sum((1, 2))


def row_U(Xm, i, xi, lp):
    Bb = Xm.shape[0]
    xa = torch.cat([Xm, bnd[None].expand(Bb, m, 3)], 1); d2 = ((xa - xi[:, None]) ** 2).sum(-1); d2[:, i] = 1e12
    return pair_e(d2, sig0[i][None] * lp, eps0[i][None]).sum(1)


Xm = xin[None].expand(B, n, 3).clone(); xin_c = xin.cpu()
g = torch.Generator(device=dev).manual_seed(1500)
# randomize stack B at (T=1, lam=0.6)
lam_r = torch.full((B,), 0.60, device=dev); beta_r = torch.full((B,), 1.0, device=dev); lp_r = lam_pair_of(lam_r)
Bmask = torch.zeros(B, dtype=torch.bool, device=dev); Bmask.view(2, NR, NCH)[1] = True
for sw in range(RAND_SW):
    for i in torch.randperm(n, generator=g, device=dev).tolist():
        l = STEP_MAX * torch.rand(B, 1, device=dev, generator=g); nh = torch.randn(B, 3, device=dev, generator=g)
        nh = nh / nh.norm(dim=-1, keepdim=True); xi_new = Xm[:, i] + l * nh; ok = (xi_new.norm(dim=-1) < R) & Bmask
        dU = row_U(Xm, i, xi_new, lp_r) - row_U(Xm, i, Xm[:, i], lp_r)
        acc = ok & (torch.rand(B, device=dev, generator=g).log() < -beta_r * dU); Xm[acc, i] = xi_new[acc]
lp0 = lam_pair_of(lam0)
# round-trip bookkeeping (per slot, swapped with configs)
seen_top = torch.zeros(2, NR, NCH, dtype=torch.bool, device=dev); trips = 0
ex_acc = torch.zeros(NR - 1); ex_try = torch.zeros(NR - 1); t0 = time.time()
for sw in range(1, SW + 1):
    for i in torch.randperm(n, generator=g, device=dev).tolist():
        l = STEP_MAX * torch.rand(B, 1, device=dev, generator=g); nh = torch.randn(B, 3, device=dev, generator=g)
        nh = nh / nh.norm(dim=-1, keepdim=True); xi_new = Xm[:, i] + l * nh; ok = xi_new.norm(dim=-1) < R
        dU = row_U(Xm, i, xi_new, lp0) - row_U(Xm, i, Xm[:, i], lp0)
        acc = ok & (torch.rand(B, device=dev, generator=g).log() < -beta0 * dU); Xm[acc, i] = xi_new[acc]
    att = (torch.rand(NR - 1, generator=g, device=dev) < 1.0 / EXCH_MEAN).nonzero().squeeze(1).tolist()
    if att:
        Xr = Xm.view(2, NR, NCH, n, 3)
        for r in att:
            xa = Xr[:, r].reshape(-1, n, 3); xb = Xr[:, r + 1].reshape(-1, n, 3)
            la = torch.full((xa.shape[0],), float(LAMs[r]), device=dev); lb = torch.full((xa.shape[0],), float(LAMs[r + 1]), device=dev)
            dlog = -BETAs[r] * (full_U(xb, la) - full_U(xa, la)) - BETAs[r + 1] * (full_U(xa, lb) - full_U(xb, lb))
            swp = (torch.rand(xa.shape[0], device=dev, generator=g).log() < dlog).view(2, NCH)
            ex_acc[r] += float(swp.float().sum()); ex_try[r] += swp.numel()
            for stk in range(2):
                for ch in range(NCH):
                    if swp[stk, ch]:
                        tmp = Xr[stk, r, ch].clone(); Xr[stk, r, ch] = Xr[stk, r + 1, ch]; Xr[stk, r + 1, ch] = tmp
                        s1 = bool(seen_top[stk, r, ch]); seen_top[stk, r, ch] = seen_top[stk, r + 1, ch].clone(); seen_top[stk, r + 1, ch] = s1
        seen_top[:, NR - 1, :] = True
        trips += int(seen_top[:, 0, :].sum()); seen_top[:, 0, :] = False
        Xm = Xr.reshape(B, n, 3)
    if sw % REC == 0:
        Xr = Xm.view(2, NR, NCH, n, 3)
        eA = float(full_U(Xr[0, 0], torch.full((NCH,), 1.0, device=dev)).mean()) / n
        eB = float(full_U(Xr[1, 0], torch.full((NCH,), 1.0, device=dev)).mean()) / n
        eref = float(full_U(xin[None], torch.full((1,), 1.0, device=dev))[0]) / n
        qcA = st.mean([bcy_qc(Xr[0, 0, ch], xin_c) for ch in range(NCH)])
        qcB = st.mean([bcy_qc(Xr[1, 0, ch], xin_c) for ch in range(NCH)])
        ea = ex_acc / ex_try.clamp(min=1)
        print(f"  sw {sw:>6}: TRIPS={trips:>4} | exch min/med {float(ea.min()):.2f}/{float(ea.median()):.2f} "
              f"| cold-E/n A {eA:+.3f} B {eB:+.3f} (ref {eref:+.3f}) | qc(A,ref) {qcA:.3f} qc(B,ref) {qcB:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
print(f"per-rung exch acc: {[round(float(x),2) for x in (ex_acc/ex_try.clamp(min=1)).tolist()]}", flush=True)
print("DONE", flush=True)
