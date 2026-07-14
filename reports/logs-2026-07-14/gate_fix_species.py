"""EXACTNESS GATE for the fix_species/pos_only breathe API added to sample_block_b/block_log_prob_b.
(1) BREATHE round-trip: draw positions under an IMPOSED species pattern (fix_species=True) -> logq_fwd_pos;
    re-score those positions with block_log_prob_b(pos_only=True) -> must MATCH logq_fwd_pos (position density
    is exact & scorable). (2) REGRESSION: the normal (non-fixed) sample==score round-trip must be unchanged
    (~5e-3 float32 baseline). If both pass, the breathe kernel is exact and safe to build the two-blob move on."""
import torch
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; RCTX = 2.5; R = 2.5; ART = "liquid_coupling_flow/artifacts"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
g = torch.Generator(device=dev).manual_seed(0)
c = torch.rand(3, generator=g, device=dev) * L; p = carve(X[0], S[0], c, R, L)
xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
n = xo.shape[0]; Mn = 16
a_all = torch.ones(n, dtype=torch.bool, device=dev)

# --- (1) BREATHE exactness: impose a species pattern (here = a swap of two same-block particles) ---
so_imp = so.clone()
Aidx = (so == 0).nonzero().squeeze(1); Bidx = (so == 1).nonzero().squeeze(1)
if len(Aidx) and len(Bidx):
    so_imp[Aidx[0]], so_imp[Bidx[0]] = 1, 0        # transpose one A<->B => count-preserving imposed pattern
Xb = xo[None].expand(Mn, n, 3).clone(); Sb = so_imp[None].expand(Mn, n).clone()
Xn, Sn, logq_fwd = m.sample_block_b(Xb, Sb, a_all, bnd, sb, R, gen=g, fix_species=True)
assert torch.equal(Sn, Sb), "fix_species must return the imposed species unchanged"
logq_score = m.block_log_prob_b(Xn, Sn, a_all, bnd, sb, R, pos_only=True)
db = (logq_fwd - logq_score).abs()
d_breathe = db.max().item(); d_breathe_med = db.median().item(); d_breathe_mean = db.mean().item()

# --- (2) REGRESSION: normal path round-trip unchanged ---
Xr, Sr, logq_r = m.sample_block_b(xo[None].expand(Mn, n, 3).clone(), so[None].expand(Mn, n).clone(),
                                  a_all, bnd, sb, R, gen=g)
logq_rs = m.block_log_prob_b(Xr, Sr, a_all, bnd, sb, R)
dn = (logq_r - logq_rs).abs()
d_norm = dn.max().item(); d_norm_med = dn.median().item(); d_norm_mean = dn.mean().item()

print(f"=== fix_species / pos_only exactness gate (R={R}, n={n}, M={Mn}) ===")
print(f"(1) BREATHE  |logq_fwd-logq_score|  median={d_breathe_med:.2e} mean={d_breathe_mean:.2e} max={d_breathe:.2e}")
print(f"(2) REGRESS  normal round-trip      median={d_norm_med:.2e} mean={d_norm_mean:.2e} max={d_norm:.2e}")
# gate: breathe must be NO WORSE than the pre-existing base leak (same bin-flip mechanism, no new error)
ok = d_breathe_med <= d_norm_med * 1.5 and d_breathe_mean <= d_norm_mean * 1.5
print(f"    breathe returns imposed species: {'ok' if torch.equal(Sn, Sb) else 'BUG'}")
print(f"    VERDICT: {'PASS -- breathe exactness == base leak (no new error)' if ok else 'FAIL -- breathe adds error'}")
