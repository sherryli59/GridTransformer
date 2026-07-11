"""Exactness gate for the batched-over-M block methods: batched == stacked-unbatched to fp precision,
and batched sample logq == batched scorer. If green, the SMC can run M configs per GPU call (M x fewer
transformer launches -> fixes the ~10% GPU utilisation)."""
import torch
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; R = 2.0; RCTX = 2.5; M = 8
ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax.pt", map_location=dev, weights_only=False)
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(1)

c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[900], S[900], c, R, L)
xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, s_bnd = xout[bm], p["s_out"][bm]
n = xo.shape[0]; allmask = torch.ones(n, dtype=torch.bool, device=dev)

# M independent interiors (same boundary) via unbatched full-regen
cfgs = [m.sample_block(xo, so, allmask, bnd, s_bnd, R, gen=gen) for _ in range(M)]
Xb = torch.stack([c[0] for c in cfgs]); Sb = torch.stack([c[1] for c in cfgs])
a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
blk = torch.zeros(n, dtype=torch.bool, device=dev); blk[(a - a[seed]).norm(dim=-1).topk(4, largest=False).indices] = True

# (1) batched block_log_prob == unbatched per config
lp_b = m.block_log_prob_b(Xb, Sb, blk, bnd, s_bnd, R)
lp_u = torch.tensor([float(m.block_log_prob(Xb[i], Sb[i], blk, bnd, s_bnd, R)) for i in range(M)], device=dev)
e1 = (lp_b - lp_u).abs().max().item()
print(f"batched vs unbatched block_log_prob: max |d| = {e1:.2e} -> {'PASS' if e1 < 1e-3 else 'FAIL'}", flush=True)

# (2) batched sample logq == batched scorer on the sampled configs
xn, sn, lq = m.sample_block_b(Xb, Sb, blk, bnd, s_bnd, R, gen=gen)
lq_score = m.block_log_prob_b(xn, sn, blk, bnd, s_bnd, R)
e2 = (lq - lq_score).abs().max().item()
print(f"batched sample logq vs scorer:       max |d| = {e2:.2e} -> {'PASS' if e2 < 1e-2 else 'FAIL (note base 8e-3)'}", flush=True)
