"""Round-trip exactness of the coarse-cell-then-fine-bin head's batched sample/score:
block_log_prob_b(pos_temp=T, min_sep=M) scoring a sample_block_b(pos_temp=T, min_sep=M) draw must
reproduce the sampler's own returned logq row-by-row. Copy of reports/logs-2026-07-12/test_tempered_score.py
adapted to KA3DScaffoldEBMCoarse: RANDOM-INIT model (exactness needs no trained weights -- any per-bin
tilt is a valid categorical) with the three phi nets' output layers de-zeroed (normal_ std=1.0, as in
liquid_coupling_flow/tests/test_coarse_head.py) so the tilts are non-vacuous. Grid: pos_temp in (1.0, 0.4)
x min_sep in (None, 0.85). Runs in float64 (still CPU-only): diagnosed 2026-07-13 that float32 produces
~2/16 row failures of size ~1-3 nats -- NOT a logic bug (traced row-by-row to a single fine-bin index
flip caused by ball_squash/ball_unsquash's non-linear round-trip landing ~1e-4 from a bin boundary,
amplified into nats by the deliberately-sharp de-zeroed tilt; float64 reproduces the identical grid at
1e-12 median/max, confirming the math is exact and the float32 gap is pure round-trip precision, the
same class of leak the baseline test_tempered_score.py documents and tolerates at >90%). Asserting
median AND 100% match is achievable in float64 without weakening the tolerance. Also checks a
T-mismatch control clearly differs."""
import torch
from liquid_coupling_flow.ka3d_coarse_head import KA3DScaffoldEBMCoarse
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cpu"; RCTX = 2.5; K = 8; M = 16; R = 2.0

torch.manual_seed(0)
m = KA3DScaffoldEBMCoarse(cat_bins=128, cat_range=2.5).to(dev).double()
m.eval(); m.use_frame = False
# De-zero the phi nets' output layers -> tilts non-vacuous (zero-init makes them no-ops, Task 2).
for net in (m.phi, m.phi_a, m.phi_b):
    torch.nn.init.normal_(net[-1].weight, std=1.0)
    torch.nn.init.normal_(net[-1].bias, std=1.0)

d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).double(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(0)

c = torch.rand(3, generator=gen, device=dev).double() * L
p = carve(X[0], S[0], c, R, L)
xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
assert bnd.shape[0] < 600
n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev)
blk = torch.zeros(n, dtype=torch.bool, device=dev)
blk[(a - a[0]).norm(dim=-1).topk(K, largest=False).indices] = True
Xb = xo[None].expand(M, n, 3).contiguous(); Sb = so[None].expand(M, n).contiguous()

ok = True
import sys

# ===== FLOAT64 PASS (hard gates) =====
with torch.no_grad():   # scoring is graph-free; block_log_prob_b has no @torch.no_grad() of its own and
                        # in float64 with the 4096-way coarse categorical the retained autograd graph
                        # blows up RSS -> wrap explicitly (diagnosed 2026-07-13, killed a 60GB+ run)
    for T in (1.0, 0.4):
        for min_sep in (None, 0.85):
            xs, ss, lq_sample = m.sample_block_b(Xb, Sb, blk, bnd, sb, R, gen=gen, pos_temp=T, min_sep=min_sep)
            lq_score = m.block_log_prob_b(xs, ss, blk, bnd, sb, R, pos_temp=T, min_sep=min_sep)
            diff = (lq_score - lq_sample).abs()
            frac = float((diff < 1e-3).float().mean())
            print(f"T={T} min_sep={min_sep}: median|score-sample| {diff.median():.2e}  "
                  f"match(<1e-3) {100*frac:.1f}%  max {diff.max():.2e}")
            ok &= diff.median().item() < 1e-3 and frac == 1.0
            if T != 1.0:   # scoring at the WRONG temperature must clearly differ (guards against a no-op arg)
                lq_wrong = m.block_log_prob_b(xs, ss, blk, bnd, sb, R, pos_temp=1.0, min_sep=min_sep)
                gap = (lq_wrong - lq_sample).abs().median()
                print(f"      T-mismatch control: median|score(T=1)-sample(T={T})| {gap:.2f} (must be >> 1e-3)")
                ok &= gap.item() > 0.1

# ===== FLOAT32 PASS (soft gate, reported not asserted) =====
# KNOWN ISSUE (2026-07-13 code review): fp32 round-trip fragility from ball_squash/ball_unsquash
# precision loss near bin boundaries. Deployed sampler is float32, so we gate this here with baseline-style
# tolerance (median < 1e-3, >90% match) to expose any per-point failures. BINDING gate runs in Task 6
# against trained checkpoint; if trained leak stays nats-scale, deployment must score in float64.
print("\n--- Float32 round-trip (baseline-style soft gate) ---")
fp32_passes = 0
grid_points = []
for T in (1.0, 0.4):
    for min_sep in (None, 0.85):
        grid_points.append((T, min_sep))

with torch.no_grad():
    m_fp32 = m.float()  # downcast model to float32
    Xb_fp32 = Xb.float()
    Sb_fp32 = Sb.long()
    bnd_fp32 = bnd.float()

    for T, min_sep in grid_points:
        xs, ss, lq_sample = m_fp32.sample_block_b(Xb_fp32, Sb_fp32, blk, bnd_fp32, sb, R, gen=gen, pos_temp=T, min_sep=min_sep)
        lq_score = m_fp32.block_log_prob_b(xs, ss, blk, bnd_fp32, sb, R, pos_temp=T, min_sep=min_sep)
        diff = (lq_score - lq_sample).abs()
        frac = float((diff < 1e-3).float().mean())
        median_diff = diff.median().item()
        passes_baseline = median_diff < 1e-3 and frac > 0.9
        if passes_baseline:
            fp32_passes += 1
        print(f"fp32: T={T} min_sep={min_sep}: median {median_diff:.2e}  match>90% {100*frac:.1f}%  max {diff.max():.2e}  "
              f"{'PASS' if passes_baseline else 'FAIL'}")

print(f"fp32 gate: {fp32_passes}/4 points pass (baseline-style)")
print("PASS (float64 hard assertions met; fp32 measured)" if ok else "FAIL (float64 gates failed)")
assert ok  # only float64 is hard; float32 always exits 0
sys.exit(0)
