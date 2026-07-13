"""Round-trip exactness of block_log_prob_b(pos_temp=T): scoring a pos_temp=T sample under the tempered
density must reproduce sample_block_b's returned logq row-by-row. Known leak: the base cat-head's ~8e-3
sample-vs-score bin flip (documented in pts_results.md 'exactness leak') -> assert on the MEDIAN and the
match FRACTION, not the max. Also checks pos_temp=1.0 is unchanged, and T-mismatch scoring clearly differs."""
import torch
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cpu"; RCTX = 2.5; K = 8; M = 16; R = 2.0

ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt", map_location=dev, weights_only=False)
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(0)

c = torch.rand(3, generator=gen, device=dev) * L
p = carve(X[0], S[0], c, R, L)
xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
assert bnd.shape[0] < 600
n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev)
blk = torch.zeros(n, dtype=torch.bool, device=dev)
blk[(a - a[0]).norm(dim=-1).topk(K, largest=False).indices] = True
Xb = xo[None].expand(M, n, 3).contiguous(); Sb = so[None].expand(M, n).contiguous()

ok = True
for T in (1.0, 0.4):
    xs, ss, lq_sample = m.sample_block_b(Xb, Sb, blk, bnd, sb, R, gen=gen, pos_temp=T)
    lq_score = m.block_log_prob_b(xs, ss, blk, bnd, sb, R, pos_temp=T)
    diff = (lq_score - lq_sample).abs()
    frac = float((diff < 1e-3).float().mean())
    print(f"T={T}: median|score-sample| {diff.median():.2e}  match(<1e-3) {100*frac:.1f}%  max {diff.max():.2e}")
    ok &= diff.median().item() < 1e-3 and frac > 0.9
    if T != 1.0:   # scoring at the WRONG temperature must clearly differ (guards against a no-op arg)
        lq_wrong = m.block_log_prob_b(xs, ss, blk, bnd, sb, R, pos_temp=1.0)
        gap = (lq_wrong - lq_sample).abs().median()
        print(f"      T-mismatch control: median|score(T=1)-sample(T={T})| {gap:.2f} (must be >> 1e-3)")
        ok &= gap.item() > 0.1
print("PASS" if ok else "FAIL")
assert ok
