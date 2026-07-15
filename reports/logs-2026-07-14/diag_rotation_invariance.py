"""Is q0 invariant under JOINT rotation of interior+boundary? (Enabler for the rotation-tail scheme: rotate
both by R, resample the now-different Morton-tail, rotate back -> refreshes a different interior region, clean
IFF q0 is joint-rotation-invariant.) The cavity was trained with rand_rot augmentation on interior+boundary,
so it SHOULD be ~invariant. Measure per-particle |logq0(Rx|Rb) - logq0(x|b)| over random rotations R. If ~=
the exactness-leak level (~0.02/particle), the model is as invariant as it is exact -> rotation-tail is
deployable. Data interiors, R=2.5, a few configs x random rotations."""
import sys, statistics as st
import torch
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; R = 2.5; ART = "liquid_coupling_flow/artifacts"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def rand_rot(gen):
    A = torch.randn(3, 3, generator=gen, device=dev); Q, r = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(r))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def logq0(xin_c, sin, xout_c, sout):
    xo, so, _ = label_to_scaffold(xin_c, sin, R)
    bm = xout_c.norm(dim=-1) < (R + RCTX); bnd, sb = xout_c[bm], sout[bm]
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    return float(m.block_log_prob_b(xo[None], so[None], allm, bnd, sb, R)[0]), n


gen = torch.Generator(device=dev).manual_seed(0); devs = []; ncav = 0
for ci in range(12):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < 14:
        continue
    xin = _mic(p["x_in"], c, L); sin = p["s_in"]; xout = _mic(p["x_out"], c, L); sout = p["s_out"]
    q_orig, n = logq0(xin, sin, xout, sout)
    for _ in range(4):
        Q = rand_rot(gen)
        q_rot, _ = logq0(xin @ Q.T, sin, xout @ Q.T, sout)         # JOINT rotation of interior+boundary
        devs.append(abs(q_rot - q_orig) / n)
    ncav += 1
    if ncav >= 6:
        break

print(f"=== joint-rotation invariance of q0 (per-particle |dlogq0|) ===", flush=True)
print(f"  mean {st.mean(devs):.4f}  median {st.median(devs):.4f}  max {max(devs):.4f}  nats/particle", flush=True)
print(f"  (exactness-leak floor ~0.02/particle; if ~ that => model is joint-rotation-invariant => rotation-tail deployable)", flush=True)
print(f"  => {'INVARIANT (rotation-tail deployable, clean high-acc whole-cavity moves)' if st.median(devs) < 0.06 else 'NOT invariant enough -- rotation-tail would be low-acceptance (MH still exact); needs order-agnostic retrain'}", flush=True)
