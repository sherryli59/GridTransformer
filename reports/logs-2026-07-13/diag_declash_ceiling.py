"""CEILING TEST: would perfect declashing even help MTM acceptance? Decides whether to build the
(substantial) nested within-cell mask. For coarse-head trials, split by whether the trial is CLASH-FREE
(min sigma-gap>=0.9 vs ALL block+cage) and report the MTM weight component -bU/particle above data.
If clash-free trials STILL have a large deficit (structural energy gap, not clashes), declashing alone
cannot reach acceptance -> nested mask not worth building. If clash-free trials are near data energy,
declashing is the lever -> build it."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_coarse_head import KA3DScaffoldEBMCoarse
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA

dev = "cuda"; RCTX = 2.5; K = 8; R = 2.0; PT = 0.4; BETA = 2.0; BIGL = 100.0; M = 32; NCAV = 12
sig = torch.tensor(SIGMA, device=dev)
m = KA3DScaffoldEBMCoarse(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_coarse_best.pt", map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
gen = torch.Generator(device=dev).manual_seed(0)

clashfree_dE, clashy_dE, cf_frac = [], [], []
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
    xa, sa, _ = m.sample_block_b(xo_b, so_b, blk, bnd, sb, R, gen=gen, pos_temp=PT, min_sep=0.9)
    blk_x, blk_s = xa[:, blk], sa[:, blk]
    cage_x = torch.cat([bnd[None].expand(M, -1, -1), xa[:, ~blk]], 1); cage_s = torch.cat([sb[None].expand(M, -1), sa[:, ~blk]], 1)
    data_blk = xo[None, blk]; data_s = so[None, blk]
    Ed = float(ka_energy(torch.cat([data_blk, cage_x[:1]], 1).double(), torch.cat([data_s, cage_s[:1]], 1).long(), BIGL)[0]) / K
    U = ka_energy(torch.cat([blk_x, cage_x], 1).double(), torch.cat([blk_s, cage_s], 1).long(), BIGL).float() / K - Ed  # dE/particle [M]
    # clash-free flag per trial: min sigma-gap vs (block+cage) >= 0.9
    nbr_x = torch.cat([blk_x, cage_x], 1); nbr_s = torch.cat([blk_s, cage_s], 1)
    cf = torch.ones(M, dtype=torch.bool, device=dev)
    for mm in range(M):
        d = torch.cdist(blk_x[mm], nbr_x[mm]); d[torch.arange(K), torch.arange(K)] = float("inf")
        cf[mm] = (d / sig[blk_s[mm][:, None], nbr_s[mm][None, :]]).min() >= 0.9
    fin = torch.isfinite(U)
    clashfree_dE += U[cf & fin].tolist(); clashy_dE += U[(~cf) & fin].tolist(); cf_frac.append(float(cf.float().mean()))

print("=== DECLASH CEILING (coarse head, min_sep=0.9, 12 cages x M=32) ===", flush=True)
print(f"clash-free trials: {100*st.mean(cf_frac):.0f}% of samples", flush=True)
print(f"dE/particle above data:  clash-free median {st.median(clashfree_dE) if clashfree_dE else float('nan'):+6.2f}  "
      f"(n={len(clashfree_dE)})   clashy median {st.median(clashy_dE) if clashy_dE else float('nan'):+8.1f}", flush=True)
if clashfree_dE:
    cfe = st.median(clashfree_dE)
    print(f"-> a clash-free block's MTM energy deficit ~ beta*K*dE = {BETA*K*cfe:+.0f} nats/block", flush=True)
    print(f"   ACCEPTANCE-VIABLE if this is within ~ln(N_trials)=~3 nats of 0; DECLASH INSUFFICIENT if <<-10", flush=True)
torch.save({"clashfree_dE": clashfree_dE, "clashy_dE": clashy_dE, "cf_frac": cf_frac}, "reports/logs-2026-07-13/diag_declash_ceiling.pt")
print("saved", flush=True)
