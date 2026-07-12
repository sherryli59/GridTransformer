"""Task 4 gate: fm_batch's minibatch-OT coupling (AR-block base -> data-block target) is a valid
permutation, is species-preserving, and its rectified-flow interpolant hits the base/target exactly
at t=0/t=1. See docs/superpowers/specs/2026-07-12-egnn-cavity-block-flow-corrector-design.md
("Training objective") and reports/logs-2026-07-12/fm_data.py's module docstring for the coupling
design decision (per-mover Hungarian on the K-block, not cross-cavity / not SO(3)-augmented).

Uses a REAL carved cavity from the TRAINING split (ka3d_train_N4096_T0.5_rho1.15.pt, 112 chains --
NEVER the 16-chain held/reference set) and the REAL trained AR checkpoint
(ka3d_cavity_ebm3ax_rho115_best.pt), same load pattern as reports/logs-2026-07-12/mtm_early_read.py.
"""
import torch

from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from fm_data import carve_cavity_block, fm_batch

dev = "cuda" if torch.cuda.is_available() else "cpu"
Rr, K, M = 2.0, 6, 8
POS_TEMP = 0.7

ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt", map_location=dev, weights_only=False)
ar = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
ar.load_state_dict(ck["state_dict"], strict=False)
ar.eval(); ar.use_frame = False

d = torch.load("liquid_coupling_flow/artifacts/ka3d_train_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
assert X.shape[0] == 112, f"expected the 112-chain TRAINING split, got {X.shape[0]} chains"
gen = torch.Generator(device=dev).manual_seed(0)

cav = None
for ci in range(X.shape[0]):
    c = torch.rand(3, generator=gen, device=dev) * L
    cav = carve_cavity_block(X, S, L, ci, c, Rr, K, gen)
    if cav is not None:
        break
assert cav is not None, "could not carve a cavity with n_in >= K+1 in the first 112 chains -- unexpected"
print(f"[cavity] ci={ci} R={Rr} n_in={cav['n']} K={K} n_bnd={cav['bnd'].shape[0]}", flush=True)

out = fm_batch(ar, cav, gen, pos_temp=POS_TEMP, M=M)
x0, x1, perm = out["x0_block"], out["x1_block"], out["perm"]
sp_block = out["sp_block"]

print(f"[fm_batch] x0_block {tuple(x0.shape)} x1_block {tuple(x1.shape)} cage_x {tuple(out['cage_x'].shape)} "
      f"sp_cage {tuple(out['sp_cage'].shape)} R={out['R']} identity_fallback={out['n_identity_fallback']}/{M}",
      flush=True)

# ---- (4) shapes: k movers, n_cage cage ----
n_ret = cav["n"] - K
n_cage = cav["bnd"].shape[0] + n_ret
assert x0.shape == (M, K, 3) and x1.shape == (M, K, 3), f"bad block shape {x0.shape}"
assert out["cage_x"].shape == (M, n_cage, 3), f"bad cage shape {out['cage_x'].shape} != (M,{n_cage},3)"
assert out["sp_cage"].shape == (M, n_cage)
assert sp_block.shape == (M, K)
assert out["x_t"].shape == (M, K, 3) and out["target_v"].shape == (M, K, 3) and out["t"].shape == (M,)
print("[shapes] PASS", flush=True)

# ---- (1) OT coupling is a valid permutation (bijection) for every one of the M draws ----
xo_all, so_all = cav["xo"], cav["so"]
x1_data = xo_all[cav["block_mask"]]
sp_data = so_all[cav["block_mask"]]
for mi in range(M):
    p = perm[mi]
    assert p.numel() == K
    assert torch.equal(torch.sort(p).values, torch.arange(K, device=dev)), \
        f"draw {mi}: perm is not a bijection over the K block slots: {p.tolist()}"
    assert torch.equal(x1[mi], x1_data[p]), f"draw {mi}: x1_block != x1_data[perm]"
print(f"[permutation] all {M} draws: valid bijection over K={K} slots -- PASS", flush=True)

# ---- (2) species preserved along the pairing: a mover of species s maps to a target slot of species s ----
for mi in range(M):
    matched_species = sp_data[perm[mi]]
    assert torch.equal(matched_species, sp_block[mi]), \
        f"draw {mi}: species NOT preserved along the OT pairing: AR={sp_block[mi].tolist()} " \
        f"matched-data={matched_species.tolist()}"
assert out["n_identity_fallback"] == 0, \
    "expected exact species-count matching (same cavity, sample_block_b budget) -- got identity fallback(s)"
print(f"[species] all {M} draws: species preserved exactly along the OT pairing "
      f"(0 identity fallbacks) -- PASS", flush=True)

# ---- (3) x_t at t=0 == x0, t=1 == x1, to 1e-6 (force t explicitly; independent AR draw from a fresh gen) ----
gen0 = torch.Generator(device=dev).manual_seed(7)
out0 = fm_batch(ar, cav, gen0, pos_temp=POS_TEMP, M=M, t=0.0)
err0 = (out0["x_t"] - out0["x0_block"]).abs().max().item()
assert torch.equal(out0["t"], torch.zeros(M, device=dev, dtype=out0["t"].dtype))

gen1 = torch.Generator(device=dev).manual_seed(7)
out1 = fm_batch(ar, cav, gen1, pos_temp=POS_TEMP, M=M, t=1.0)
err1 = (out1["x_t"] - out1["x1_block"]).abs().max().item()
assert torch.equal(out1["t"], torch.ones(M, device=dev, dtype=out1["t"].dtype))

# same gen seed -> same AR draw at t=0 and t=1 -> target_v == x1_block - x0_block consistent across both calls
tv_err = (out0["target_v"] - out1["target_v"]).abs().max().item()
print(f"[interpolant] max|x_t(t=0) - x0_block| = {err0:.3e}   max|x_t(t=1) - x1_block| = {err1:.3e}   "
      f"target_v consistency (same gen, t=0 vs t=1 draws) = {tv_err:.3e}", flush=True)
assert err0 < 1e-6, f"x_t at t=0 != x0_block: {err0:.3e}"
assert err1 < 1e-6, f"x_t at t=1 != x1_block: {err1:.3e}"
assert tv_err < 1e-6, f"target_v not consistent between the t=0 and t=1 calls (same AR draw expected): {tv_err:.3e}"
print("[interpolant boundary] PASS", flush=True)

print("\nALL PASS")
