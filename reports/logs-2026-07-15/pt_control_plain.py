"""PT-SHRINKAGE CONTROL: plain single-T (T=0.5) local MC from the reference, NO shrinkage, NO exchange.
Local moves cannot cross barriers, so the config MUST stay in the reference basin -> q_c(evolved, ref) should
stay HIGH (~0.5-0.6, same-basin). Decisive test vs the shrinkage-PT (which collapses q_c to ~0.16):
  - plain-MC stays HIGH but shrinkage-PT collapses -> the SHRINKAGE-PT is over-decorrelating (bug/too aggressive).
  - plain-MC ALSO collapses -> the reference basin is unstable under local dynamics => data/state-point issue.
Same boundary data as v2 (N512 T=0.5, rho=1.2). ML-FREE."""
import sys, time, statistics as st
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA, EPS
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; T = 0.50; BETA = 1.0 / T; R = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
SW = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
NW = 8; STEP_MAX = 0.30; REC = 1000
B_OV = 0.2; RC_CORE = 0.5; P_MC = 1500; K_NN = 6
t_sig = torch.tensor(SIGMA, device=dev); t_eps = torch.tensor(EPS, device=dev)
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
Xds, Sds, L = D["x"].float(), D["s"].long(), float(D["L"])
gcpu = torch.Generator().manual_seed(7)


def bcy_qc(X, Y):
    X = X.cpu(); Y = Y.cpu()
    def field_core(vpos, vq):
        u = torch.randn(P_MC, 3, generator=gcpu); u = u / u.norm(dim=-1, keepdim=True)
        r = RC_CORE * torch.rand(P_MC, generator=gcpu) ** (1.0 / 3.0); pmc = u * r[:, None]
        d2 = ((pmc[:, None] - vpos[None]) ** 2).sum(-1); w = 1.0 / (d2 + 1e-6)
        kk = min(K_NN, vpos.shape[0]); topw, idx = w.topk(kk, dim=1)
        return float(((topw * vq[idx]).sum(1) / topw.sum(1)).mean())
    qX = torch.exp(-(torch.cdist(X, Y).min(1).values / B_OV) ** 2)
    qY = torch.exp(-(torch.cdist(Y, X).min(1).values / B_OV) ** 2)
    return 0.5 * (field_core(X, qX) + field_core(Y, qY))


def pair_row(Xf, Sf, i, xi, bnd, sb):
    """energy of mobile i at xi vs all (mobile + boundary), shifted-LJ rc=2.5 sig. [NW]"""
    xa = torch.cat([Xf, bnd[None].expand(NW, bnd.shape[0], 3)], 1)
    sa = torch.cat([Sf, sb[None].expand(NW, sb.shape[0])], 1)
    d = xa - xi[:, None]; r2 = (d ** 2).sum(-1); r2[:, i] = 1e12
    si = Sf[:, i]; sig = t_sig[si[:, None], sa]; eps = t_eps[si[:, None], sa]
    inv6 = (sig ** 2 / r2.clamp_min(1e-12)) ** 3; e = 4 * eps * (inv6 ** 2 - inv6)
    s6 = (1.0 / 2.5) ** 6; e = torch.where(r2 < (2.5 * sig) ** 2, e - 4 * eps * (s6 ** 2 - s6), torch.zeros_like(e))
    return e.sum(1)


gsel = torch.Generator(device=dev).manual_seed(300 + int(R * 10))
ci = int(torch.randperm(Xds.shape[0], generator=gsel, device=dev)[0])
c = torch.rand(3, generator=gsel, device=dev) * L
p = carve(Xds[ci], Sds[ci], c, R, L)
xin = _mic(p["x_in"], c, L); sin = p["s_in"].long(); n = xin.shape[0]
xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm].long()
print(f"PT CONTROL (plain T={T} MC, no shrinkage): R={R} cav {ci} n={n} boundary={bnd.shape[0]} | SW={SW}", flush=True)
Xf = xin[None].expand(NW, n, 3).clone(); Sf = sin[None].expand(NW, n).clone()
xin_c = xin.cpu()
g = torch.Generator(device=dev).manual_seed(1500)
for sw in range(1, SW + 1):
    for i in torch.randperm(n, generator=g, device=dev).tolist():
        l = STEP_MAX * torch.rand(NW, 1, device=dev, generator=g)
        nh = torch.randn(NW, 3, device=dev, generator=g); nh = nh / nh.norm(dim=-1, keepdim=True)
        xi_new = Xf[:, i] + l * nh; ok = xi_new.norm(dim=-1) < R
        dU = pair_row(Xf, Sf, i, xi_new, bnd, sb) - pair_row(Xf, Sf, i, Xf[:, i], bnd, sb)
        acc = ok & (torch.rand(NW, device=dev, generator=g).log() < -BETA * dU)
        Xf[acc, i] = xi_new[acc]
    if sw % (REC * 3) == 0:
        qc = st.mean([bcy_qc(Xf[w], xin_c) for w in range(NW)])
        print(f"    plain sw {sw:>6}: q_c(evolved, ref) = {qc:.3f}", flush=True)
print("DONE", flush=True)
