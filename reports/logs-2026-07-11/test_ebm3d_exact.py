"""3D EBM exactness gate: sample_block logq == block_log_prob scorer on a boundary-conditioned cavity
block (validates ball-map + frame + c-tilt geometry). Warm-loads the frameless free-cluster cat model
(phi~0, R_embed~0) into KA3DScaffoldEBM; with the tilt neutral it must (a) be exact and (b) reproduce
the base's block_log_prob (ka3d_block) to fp precision."""
import torch
from liquid_coupling_flow.ka3d_scaffold_ebm import KA3DScaffoldEBM
from liquid_coupling_flow.ka3d_scaffold_ar import KA3DScaffoldCatAR, label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow import ka3d_block

dev = "cuda"; R = 2.0; R_CTX = 2.5
ck = torch.load("liquid_coupling_flow/artifacts/ka3d_blob_noframe.pt", map_location=dev, weights_only=False)
m = KA3DScaffoldEBM(cat_bins=128, cat_range=2.5).to(dev)
miss, unexp = m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
print(f"loaded blob_noframe: {len(miss)} new (phi/pair_emb/R_embed), {len(unexp)} unused", flush=True)
mb = KA3DScaffoldCatAR(cat_bins=128, cat_range=2.5).to(dev)     # base (no tilt) for cross-check
mb.load_state_dict(ck["state_dict"], strict=False); mb.eval(); mb.use_frame = False
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(0)


def make_cavity(ci, center):
    p = carve(X[ci], S[ci], center, R, L)
    x_in = _mic(p["x_in"], center, L); s_in = p["s_in"]
    xo, so, _ = label_to_scaffold(x_in, s_in, R)
    xout = _mic(p["x_out"], center, L); bm = xout.norm(dim=-1) < (R + R_CTX)
    return xo, so, xout[bm], p["s_out"][bm], p["n_in"]


def blob(n, K):
    a = fixed_ball_scaffold(n, R, dev)
    seed = int(torch.randint(n, (), generator=gen, device=dev))
    mask = torch.zeros(n, dtype=torch.bool, device=dev)
    mask[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    return mask


errs, base_diffs = [], []
for ci in range(900, 912):
    center = torch.rand(3, generator=gen, device=dev) * L
    xo, so, bnd, s_bnd, n_in = make_cavity(ci, center)
    if n_in < 8:
        continue
    blk = blob(xo.shape[0], 4)
    xn, sn, lq_fwd = m.sample_block(xo, so, blk, bnd, s_bnd, R, gen=gen)
    lq_score = m.block_log_prob(xn, sn, blk, bnd, s_bnd, R)
    errs.append(abs(float(lq_fwd) - float(lq_score)))
    # cross-check: with phi~0 / R_embed~0 the EBM score == base ka3d_block score on the SAME config
    lp_base = ka3d_block.block_log_prob(mb, xn, sn, blk, bnd, s_bnd, R)
    base_diffs.append(abs(float(lq_score) - float(lp_base)))

print(f"EXACTNESS |sample logq - score|: max {max(errs):.2e} mean {sum(errs)/len(errs):.2e} "
      f"-> {'PASS' if max(errs) < 1e-3 else 'FAIL'}", flush=True)
print(f"EBM(phi~0) vs base ka3d_block:   max {max(base_diffs):.2e} "
      f"-> {'PASS (reproduces base)' if max(base_diffs) < 1e-2 else 'FAIL'}", flush=True)
