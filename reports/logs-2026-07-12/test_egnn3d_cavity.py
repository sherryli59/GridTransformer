"""Task 3 gate: sample_corrected_block wires the frozen AR base (KA3DScaffoldEBMBatched) to the Task-2
CavityBlockFlow, composing an AR block sample with an exact flow correction into a single composed log-q.
See docs/superpowers/specs/2026-07-12-egnn-cavity-block-flow-corrector-design.md and
reports/logs-2026-07-12/test_egnn3d_flow.py (Task 2 gate, which this reuses the identity-init idea from).

MAKE-OR-BREAK: with an UNTRAINED (zero-init velocity) CavityBlockFlow, sample_corrected_block must reproduce
the AR block EXACTLY (x1==x0, since the identity flow is x1==x0, logdet==0) and logq_composed == logq_ar to
~1e-5. This proves the wiring composes the AR base and the flow without corrupting either -- no bug in the
cage assembly / block extraction / write-back can hide behind a trained (nonzero) flow.

Uses a REAL carved cavity (ka3d_dataset_N4096_T0.5_rho1.15.pt) and the REAL trained AR checkpoint
(ka3d_cavity_ebm3ax_rho115_best.pt), same load pattern as reports/logs-2026-07-12/mtm_early_read.py, at a
small (R=2.0, K=4) cell for speed -- the AR model itself is not under test here (Task 2 already gates the
flow in isolation; this gates only the NEW wiring).
"""
import torch

from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow, sample_corrected_block

dev = "cuda" if torch.cuda.is_available() else "cpu"
RCTX = 2.5
Rr, K = 2.0, 4
M = 4          # configs run in parallel (same cavity replicated -- wiring test, not a diversity test)
POS_TEMP = 0.7

ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt", map_location=dev, weights_only=False)
ar = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
ar.load_state_dict(ck["state_dict"], strict=False)
ar.eval(); ar.use_frame = False

d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(0)

# ---- carve one held cavity, build a K-block (movers) via the fixed_ball_scaffold nearest-neighbour rule,
# exactly as frontier_threearm.py / mtm_early_read.py do ----
ci = 0
c = torch.rand(3, generator=gen, device=dev) * L
p = carve(X[ci], S[ci], c, Rr, L)
assert p["n_in"] >= K + 6, "carved cavity too small for this K -- pick a different ci/center"
xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], Rr)
xout = _mic(p["x_out"], c, L)
bm = xout.norm(dim=-1) < (Rr + RCTX)
bnd, sb = xout[bm], p["s_out"][bm]
n = xo.shape[0]
a = fixed_ball_scaffold(n, Rr, dev)
seed = int(torch.randint(n, (), generator=gen, device=dev))
block_mask = torch.zeros(n, dtype=torch.bool, device=dev)
block_mask[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
n_ret = n - K
n_cage = bnd.shape[0] + n_ret
print(f"[cavity] R={Rr} n_in={n} K={K} n_ret={n_ret} n_bnd={bnd.shape[0]} n_cage={n_cage}", flush=True)

xo_b = xo[None].expand(M, n, 3).contiguous()
so_b = so[None].expand(M, n).contiguous()

# ---- untrained (zero-init) flow: CavityBlockFlow.__init__ zeros pot_model[-1] -> v==0 everywhere -> the
# identity flow (x1==x0, logdet==0), per CavityBlockFlow's own docstring and the Task-2 gate. ----
flow = CavityBlockFlow(n_cage=n_cage, k=K, r_c=RCTX, hidden_nf=32, n_layers=3, n_species=2,
                        max_neighbors=16, ode_rtol=1e-6, ode_atol=1e-6).to(dev).double()

SEED = 123
gen_direct = torch.Generator(device=dev).manual_seed(SEED)
xo_ar, so_ar, logq_ar = ar.sample_block_b(xo_b, so_b, block_mask, bnd, sb, Rr, gen=gen_direct, pos_temp=POS_TEMP)

gen_corr = torch.Generator(device=dev).manual_seed(SEED)
x_full, s_full, logq_composed = sample_corrected_block(ar, flow, xo_b, so_b, block_mask, bnd, sb, Rr,
                                                        gen=gen_corr, pos_temp=POS_TEMP)

block_err = (x_full[:, block_mask] - xo_ar[:, block_mask]).abs().max().item()
retained_err = (x_full[:, ~block_mask] - xo_ar[:, ~block_mask]).abs().max().item()
species_match = torch.equal(s_full, so_ar)
logq_err = (logq_composed - logq_ar).abs().max().item()

print(f"[identity: block pos]     max|x_full_blk - xo_ar_blk|     = {block_err:.3e}", flush=True)
print(f"[identity: retained pos]  max|x_full_ret - xo_ar_ret|     = {retained_err:.3e}", flush=True)
print(f"[identity: species]       s_full == so_ar (all M,n)       = {species_match}", flush=True)
print(f"[identity: logq]          max|logq_composed - logq_ar|    = {logq_err:.3e}", flush=True)
print(f"  logq_ar        = {logq_ar.tolist()}", flush=True)
print(f"  logq_composed  = {logq_composed.tolist()}", flush=True)

assert block_err < 1e-5, f"corrected block != AR block under identity flow: {block_err:.3e} >= 1e-5"
assert retained_err < 1e-9, f"retained slots must be byte-identical (never touched by the flow): {retained_err:.3e}"
assert species_match, "species must be exactly the AR species (flow never touches species)"
assert logq_err < 1e-5, f"logq_composed != logq_ar under identity flow: {logq_err:.3e} >= 1e-5"

print("PASS")
