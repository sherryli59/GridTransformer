"""Does the flow's logq reflect the Boltzmann weight? Direct test of the calibration claim + the
one-sample-collapse root cause. For held cages, generate M diverse flow trials; per trial record
(logq_flow, -beta*U). Boltzmann-calibrated => within-cage corr(logq, -bU) ~ +1 (reweight -> uniform).
Collapsed map => logq tracks distance-from-map-center, NOT energy => corr ~ 0 or negative.
Also measure VARIANCE COLLAPSE: per-cage RMS spread of flow trials vs the gated-AR base trials they
came from (flow spread << base spread confirms contraction to a near-point)."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow
from liquid_coupling_flow.ka3d_gated_base import GatedARBase
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; K = 8; R = 2.0; PT = 0.4; BETA = 2.0; BIGL = 100.0; N_CAGE = 48; DUMMY_R = 50.0; M = 24; GATE = 0.55
ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_egnn_flow_repprior_best.pt", map_location=dev, weights_only=False)
a = ck["args"]
arb = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
arb.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt", map_location=dev, weights_only=False)["state_dict"], strict=False)
arb.eval(); arb.use_frame = False
ar = GatedARBase(arb, cut=GATE)
flow = CavityBlockFlow(k=a["k"], n_cage=a["n_cage"], r_c=a.get("r_c", 2.5), hidden_nf=a["hidden_nf"], n_layers=a["n_layers"],
                       n_species=2, max_neighbors=a["max_neighbors"], rep_prior=a.get("rep_prior", False),
                       ode_rtol=1e-5, ode_atol=1e-5, max_steps=1000).to(dev)
flow.load_state_dict(ck["state_dict"]); flow.eval()
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
gen = torch.Generator(device=dev).manual_seed(0)


def fsc(cx, cs, cen, ncg):
    Mn, m = cx.shape[0], cx.shape[1]
    if m >= ncg:
        idx = torch.topk((cx[0] - cen).norm(dim=-1), ncg, largest=False).indices
        return cx[:, idx], cs[:, idx]
    pad = ncg - m
    dp = torch.tensor([DUMMY_R, 0., 0.], device=dev, dtype=cx.dtype)[None, None].expand(Mn, pad, 3)
    return torch.cat([cx, dp], 1), torch.cat([cs, torch.zeros(Mn, pad, dtype=cs.dtype, device=dev)], 1)


def wcorr(a_, b_):
    a_, b_ = torch.tensor(a_), torch.tensor(b_)
    a_, b_ = a_ - a_.mean(), b_ - b_.mean()
    return float((a_ * b_).sum() / (a_.norm() * b_.norm() + 1e-9))


percage_corr, flow_rms, base_rms = [], [], []
allq, allu = [], []
ncav = 0
for ci in range(16):
    while True:
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] >= K + 6:
            break
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; anch = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev)
    blk[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    xo_b = xo[None].expand(M, n, 3).contiguous(); so_b = so[None].expand(M, n).contiguous()
    xa, sa, lqf = ar.sample_block_b(xo_b, so_b, blk, bnd, sb, R, gen=gen, pos_temp=PT)
    ar_blk, sp_blk = xa[:, blk], sa[:, blk]
    cage_x = torch.cat([bnd[None].expand(M, -1, -1), xa[:, ~blk]], 1); cage_s = torch.cat([sb[None].expand(M, -1), sa[:, ~blk]], 1)
    cfx, cfs = fsc(cage_x, cage_s, xo[blk].mean(0), N_CAGE)
    y, ld = flow.flow(ar_blk, cfx, sp_blk, cfs, reverse=False)
    logq = (lqf + ld)
    cage_full = torch.cat([bnd[None].expand(M, -1, -1), xa[:, ~blk]], 1); cs_full = torch.cat([sb[None].expand(M, -1), sa[:, ~blk]], 1)
    U = ka_energy(torch.cat([y, cage_full], 1).double(), torch.cat([sp_blk, cs_full], 1).long(), BIGL).float()
    keep = torch.isfinite(logq) & torch.isfinite(U) & (U < 1e4)         # drop clash-blown trials
    if keep.sum() < 4:
        continue
    lq, mu = logq[keep].tolist(), (-BETA * U[keep]).tolist()
    percage_corr.append(wcorr(lq, mu))
    allq += [x - st.mean(lq) for x in lq]; allu += [x - st.mean(mu) for x in mu]   # within-cage centered
    # variance collapse: RMS spread about the per-cage mean position
    flow_rms.append(float((y[keep] - y[keep].mean(0)).norm(dim=-1).mean()))
    base_rms.append(float((ar_blk[keep] - ar_blk[keep].mean(0)).norm(dim=-1).mean()))
    ncav += 1

print(f"=== FLOW BOLTZMANN-CALIBRATION ({ncav} held cages, M={M} trials) ===", flush=True)
print(f"corr(logq, -bU): per-cage median {st.median(percage_corr):+.2f}  pooled(within-cage centered) {wcorr(allq, allu):+.2f}", flush=True)
print(f"   Boltzmann-calibrated => ~+1.0 ; collapsed map (logq = map Jacobian, not energy) => ~0", flush=True)
print(f"VARIANCE COLLAPSE: per-cage RMS trial spread  flow {st.median(flow_rms):.3f}  vs gated-AR base {st.median(base_rms):.3f}  "
      f"ratio {st.median(flow_rms)/st.median(base_rms):.2f}  (<<1 = contraction to a near-point)", flush=True)
torch.save({"percage_corr": percage_corr, "allq": allq, "allu": allu, "flow_rms": flow_rms, "base_rms": base_rms}, "reports/logs-2026-07-13/diag_flow_boltzmann_calib.pt")
print("saved -> reports/logs-2026-07-13/diag_flow_boltzmann_calib.pt", flush=True)
