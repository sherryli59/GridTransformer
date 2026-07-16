"""PT DECORRELATION METRICS + LADDER TUNING.
Measures the three canonical diagnostics on the shrinkage-PT cavity sampler, and uses ROUND-TRIPS/SECOND
as the objective to tune the rung count (acceptance is a means; round-trips/sec is the end).

  (1) tau_RT / N_indep: a config is an independent sample only after bottom->top->bottom. Reports
      round-trips, tau_RT (sweeps), N_indep = TRIPS/n_bottom_slots, and ROUND-TRIPS PER SECOND.
  (2) Katzgraber replica FLOW f(r): label each config by whether it last touched bottom (1) or top (0);
      f(r) = <label> at rung r. Ideal = LINEAR ramp 0->1. Plateau/kink = bottleneck that uniform
      acceptance hides. Reports f(r) and its max deviation from linear.
  (3) per-rung exchange acceptance (for reference).

Ladder tuning: tau_RT ~ NR^2 / rate, but fewer rungs -> bigger lambda jumps -> lower acceptance (rate).
These fight => a measurable optimum. Sweeps NR by SUBSAMPLING the densified BCY Table-IV ladder.
Usage: pt_decorrelation_metrics.py R SW "stride1,stride2,..."   (stride subsamples the 21-rung ladder)"""
import sys, time, statistics as st
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA, EPS
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5
R = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
SW = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
STRIDES = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["1", "2", "4"])]
_BASE = [1.0000, 0.9825, 0.9640, 0.9450, 0.9250, 0.9050, 0.8850, 0.8640, 0.8423, 0.8200, 0.7960]
_DENSE = []
for _i, _v in enumerate(_BASE):
    _DENSE.append(_v)
    if _i + 1 < len(_BASE):
        _DENSE.append(0.5 * (_v + _BASE[_i + 1]))
T_BOT = 0.50; T_DEC = 1.0; LAM_DEC = 0.8
NCH = 2; RAND_SW = 2000; STEP_MAX = 0.3; EXCH_MEAN = 2
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
Xds, Sds, L = D["x"].float(), D["s"].long(), float(D["L"])


def pair_e(r2, sig, eps):
    inv6 = (sig ** 2 / r2.clamp_min(1e-12)) ** 3; e = 4 * eps * (inv6 ** 2 - inv6)
    s6 = (1.0 / 2.5) ** 6
    return torch.where(r2 < (2.5 * sig) ** 2, e - 4 * eps * (s6 ** 2 - s6), torch.zeros_like(e))


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
print(f"PT DECORRELATION METRICS: R={R} cav {ci} n={n} | SW={SW} EXCH_MEAN={EXCH_MEAN} | strides {STRIDES}", flush=True)


def run(ladder):
    NR = len(ladder); B = NR * NCH                                    # single stack (metrics only)
    LAMs = torch.tensor(ladder, device=dev)
    Ts = T_BOT + (T_DEC - T_BOT) * (1.0 - LAMs) / (1.0 - LAM_DEC); BETAs = 1.0 / Ts
    lam0 = LAMs[:, None].expand(NR, NCH).reshape(-1)
    beta0 = BETAs[:, None].expand(NR, NCH).reshape(-1).clone()

    def lp_of(lam):
        lp = torch.empty(lam.shape[0], Nt, device=dev); lp[:, :n] = lam[:, None]; lp[:, n:] = (1 + lam[:, None]) / 2
        return lp

    def full_U(Xm, lam):
        Bb = Xm.shape[0]
        xa = torch.cat([Xm, bnd[None].expand(Bb, m, 3)], 1); d2 = torch.cdist(xa, xa) ** 2
        d2 = d2.masked_fill(torch.eye(Nt, dtype=torch.bool, device=dev)[None], 1e12)
        e = pair_e(d2, sig0[None] * lp_of(lam)[:, None, :], eps0[None])
        e = e - torch.diag_embed(torch.diagonal(e, dim1=1, dim2=2))
        return e[:, :n, :n].sum((1, 2)) * 0.5 + e[:, :n, n:].sum((1, 2))

    def row_U(Xm, i, xi, lp):
        Bb = Xm.shape[0]
        xa = torch.cat([Xm, bnd[None].expand(Bb, m, 3)], 1); d2 = ((xa - xi[:, None]) ** 2).sum(-1); d2[:, i] = 1e12
        return pair_e(d2, sig0[i][None] * lp, eps0[i][None]).sum(1)

    lp0 = lp_of(lam0)
    Xm = xin[None].expand(B, n, 3).clone()
    g = torch.Generator(device=dev).manual_seed(1500)
    for sw in range(RAND_SW):                                          # brief burn-in
        for i in torch.randperm(n, generator=g, device=dev).tolist():
            l = STEP_MAX * torch.rand(B, 1, device=dev, generator=g); nh = torch.randn(B, 3, device=dev, generator=g)
            nh = nh / nh.norm(dim=-1, keepdim=True); xi_new = Xm[:, i] + l * nh; ok = xi_new.norm(dim=-1) < R
            dU = row_U(Xm, i, xi_new, lp0) - row_U(Xm, i, Xm[:, i], lp0)
            acc = ok & (torch.rand(B, device=dev, generator=g).log() < -beta0 * dU); Xm[acc, i] = xi_new[acc]
    # metric state: seen_top (round trips) + flow label (1=last touched bottom, 0=last touched top)
    seen_top = torch.zeros(NR, NCH, dtype=torch.bool, device=dev)
    flow = torch.full((NR, NCH), 0.5, device=dev)                      # 0.5 = undetermined
    flow_sum = torch.zeros(NR, device=dev); flow_cnt = torch.zeros(NR, device=dev)
    ex_acc = torch.zeros(NR - 1); ex_try = torch.zeros(NR - 1); trips = 0
    t0 = time.time()
    for sw in range(1, SW + 1):
        for i in torch.randperm(n, generator=g, device=dev).tolist():
            l = STEP_MAX * torch.rand(B, 1, device=dev, generator=g); nh = torch.randn(B, 3, device=dev, generator=g)
            nh = nh / nh.norm(dim=-1, keepdim=True); xi_new = Xm[:, i] + l * nh; ok = xi_new.norm(dim=-1) < R
            dU = row_U(Xm, i, xi_new, lp0) - row_U(Xm, i, Xm[:, i], lp0)
            acc = ok & (torch.rand(B, device=dev, generator=g).log() < -beta0 * dU); Xm[acc, i] = xi_new[acc]
        att = (torch.rand(NR - 1, generator=g, device=dev) < 1.0 / EXCH_MEAN).nonzero().squeeze(1).tolist()
        if att:
            Xr = Xm.view(NR, NCH, n, 3)
            for r in att:
                xa = Xr[r]; xb = Xr[r + 1]
                la = torch.full((NCH,), float(LAMs[r]), device=dev); lb = torch.full((NCH,), float(LAMs[r + 1]), device=dev)
                dlog = -BETAs[r] * (full_U(xb, la) - full_U(xa, la)) - BETAs[r + 1] * (full_U(xa, lb) - full_U(xb, lb))
                swp = torch.rand(NCH, device=dev, generator=g).log() < dlog
                ex_acc[r] += float(swp.float().sum()); ex_try[r] += NCH
                for ch in range(NCH):
                    if swp[ch]:
                        tmp = Xr[r, ch].clone(); Xr[r, ch] = Xr[r + 1, ch]; Xr[r + 1, ch] = tmp
                        s1 = bool(seen_top[r, ch]); seen_top[r, ch] = seen_top[r + 1, ch].clone(); seen_top[r + 1, ch] = s1
                        f1 = float(flow[r, ch]); flow[r, ch] = flow[r + 1, ch].clone(); flow[r + 1, ch] = f1
            Xm = Xr.reshape(B, n, 3)
            seen_top[NR - 1, :] = True; flow[NR - 1, :] = 0.0          # touched top
            trips += int(seen_top[0, :].sum()); seen_top[0, :] = False
            flow[0, :] = 1.0                                           # touched bottom
            det = flow != 0.5
            flow_sum += (flow * det).sum(1); flow_cnt += det.sum(1)
    wall = time.time() - t0
    fr = (flow_sum / flow_cnt.clamp(min=1)).cpu()
    lin = torch.linspace(1, 0, NR)                                     # ideal: 1 at bottom -> 0 at top
    dev_lin = float((fr - lin).abs().max())
    tau_rt = SW * NCH / max(trips, 1e-9)
    return {"NR": NR, "trips": trips, "tau_RT": tau_rt, "N_indep": trips / NCH, "rt_per_sec": trips / wall,
            "exch_min": float((ex_acc / ex_try.clamp(min=1)).min()), "exch_med": float((ex_acc / ex_try.clamp(min=1)).median()),
            "flow": [round(float(x), 2) for x in fr.tolist()], "flow_dev": dev_lin, "wall": wall}


for stride in STRIDES:
    ladder = _DENSE[::stride]
    if ladder[-1] != _DENSE[-1]:
        ladder = ladder + [_DENSE[-1]]                                 # always keep the decorrelating top
    r_ = run(ladder)
    print(f"  NR={r_['NR']:>2} (stride {stride}): TRIPS={r_['trips']:>3} tau_RT={r_['tau_RT']:>8.0f} sw "
          f"N_indep={r_['N_indep']:.1f} | ROUND-TRIPS/SEC={r_['rt_per_sec']:.4f} | exch min/med "
          f"{r_['exch_min']:.2f}/{r_['exch_med']:.2f} | flow-dev-from-linear {r_['flow_dev']:.2f} ({r_['wall']:.0f}s)", flush=True)
    print(f"      flow f(r) [bottom->top, ideal linear 1->0]: {r_['flow']}", flush=True)
print("DONE", flush=True)
