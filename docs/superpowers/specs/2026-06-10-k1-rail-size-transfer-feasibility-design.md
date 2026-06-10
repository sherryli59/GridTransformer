# K=1 Rail Size-Transfer Feasibility — Design

**Date:** 2026-06-10
**Status:** Approved (design), pending implementation plan
**Owner:** Sherry Li

## Goal

Determine — cheaply and **without any training** — whether a K=1 fixed-template rail
model trained at N=27 (L=3, ρ=1, T=1) can transfer to larger systems N=64 (L=4) and
N=125 (L=5) at the same density. Produce a go/no-go signal before committing GPU time
to size-transfer training or fine-tuning.

## Background / Key Facts

- **Model under test:** `lj_ckpts_lj27_pbc_fixedrail_k1_ablation/lj27_pbc_continput_fixedrail_k1_fullcov/best.ckpt`
  (continuous head, full covariance, `continuous_input=True`, `use_curve_rail=True`,
  `curve_rail_k=1`, `curve_rail_mode=fixed_template`, `curve_rail_reference=absolute`,
  **`curve_rail_residual_target=False`**, `cell_size=None`, `hilbert_resolution=64`).
- **In-distribution N=27 quality (reference point):** OTgap ≈ 0.51, g(r) L1 ≈ 0.19
  (vs K=8 rail OTgap 0.20 / g(r) L1 0.10, and uniform-noise OTgap = 1.0).
- **Reconstruction is cell-independent.** Because `residual_target=False`, the model's
  prediction target is the raw cartesian displacement `pos_{t+1} − pos_t`, and the
  sampler rebuilds positions by pure accumulation (`raw_pos = x_base[:,t,:] + delta`).
  There is **no Hilbert decode and no cell_size in the output path**. The cartesian-delta
  target is scale-invariant at constant density regardless of R.
- **cell_size enters in exactly one place:** the K=1 rail lookahead waypoint
  `ŵ₁ = decode((j+1)·X) − decode(j·X)` fed to the model as conditioning context
  (`X = R³ // N`). The R/cell choice only perturbs how closely this rail **input**
  distribution at N=64/125 matches what the model saw at N=27.
- **Prior evidence:** `tests/test_size_transfer_parity.py` already shows the K=1 rail
  waypoint distribution is approximately scale-invariant (KS ≈ 0.28 < 0.30) between
  N=27/R=128 and N=125/R=256 — using **power-of-two R** with cell only approximately matched.
- **Reference data available:** `/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L{3,4,5}_rho1.0_N{27,64,125}_T1.0.h5`.
- **Benchmark is size-agnostic:** `benchmark_lj27.py` reads N and L from `--target` h5 and
  the sample npz, so it is reused at N=64/125 by pointing `--target` at the matching file.

## Resolution-strategy note (why this is subtle)

The training cell size is `cell_train = L/R = 3/64 = 0.046875`. To hold it constant at the
larger boxes we need `R = L / cell_train`, which is **not** a power of two:

| L | exact R (non-pow2) | cell @ exact R | pow2 R | cell @ pow2 R |
|---|--------------------|----------------|--------|---------------|
| 3 | 64                 | 0.0469         | 64     | 0.0469        |
| 4 | 85                 | 0.0471         | 128    | 0.0313        |
| 5 | 107                | 0.0467         | 128    | 0.0391        |

The Hilbert curve is fundamentally power-of-two: with R=85, `bits = ceil(log2(85)) = 7`,
the curve actually fills a 128³ cube while `X = R³//N` uses 85³ as the curve length — a
mismatch that lets the rail waypoint land on cells outside the [0,84] physical region.
The recent `_hilbert_bits` fix prevents a crash but does not fix the waypoint *semantics*.

**Decision:** Do not pre-commit a strategy. The audit (Stage 1) measures the rail-input
KS under **both** strategies; power-of-two R is the expected default (it is the validated
path), and exact-constant non-pow2 R is an experimental arm included only to see whether it
measurably tightens the rail-input KS **and** keeps waypoints in-box.

## Stage 1 — Distribution Audit (no model)

New script `analyze_k1_rail_transfer.py` (mirrors `analyze_arc_repr_transfer.py`).

For each size (N=27/L=3, N=64/L=4, N=125/L=5), load ~500 real MCMC configs and, under
each R strategy (pow2 R and exact non-pow2 R, `cell_train = 0.046875`):

1. Hilbert-sort positions; compute Hilbert codes.
2. **Rail input** ŵ₁ = `min_image(decode((j+1)·X) − decode(j·X))`, reported normalized by
   inter-particle spacing `a = ρ^(-1/3) = 1` and by cell_size. This is the conditioning
   feature whose invariance gates transfer.
3. **Cartesian-delta target** `min_image(pos_{t+1} − pos_t)` — reported for completeness
   (expected invariant regardless of R).
4. **In-box fraction** of exact-R waypoints (fraction with any decoded coord > L) — a
   disqualifier for the non-pow2 arm if large.

**Outputs:** KS table (N27-vs-N64, N27-vs-N125 for ŵ₁ per axis and |ŵ₁|, plus the target
deltas) and overlaid histograms → `reports/k1_rail_transfer/`.

**Go/no-go (Stage 1):** rail-input KS < 0.30 for both size pairs under the chosen strategy.
Exact-R is chosen over pow2-R only if it both lowers the max rail-input KS **and** keeps the
waypoint in-box fraction ≥ 0.95; otherwise pow2-R is used.

## Stage 2 — Zero-Shot Generation (existing checkpoint, no training)

Gated on Stage 1 passing. Sample the trained K=1 best.ckpt at N=64/L=4 and N=125/L=5.

- **Sampler invocation:** `sample_lj.py` with `--num_particles {64,125}`, `--Lx/Ly/Lz {4,5}`,
  reusing the same continuous-head / full-cov flags as the N=27 run.
- **cell_size injection:** the checkpoint carries `cell_size=None`, so `_rail_resolution_for_box`
  would otherwise return R=64 at every box (cell grows). A thin wrapper (à la
  `sample_arc_legacy.py`) sets the constant `cell_train` on the loaded model so R scales
  with the box per the Stage-1-winning strategy. For pow2 R this works through the existing
  `_resolution_for_box`; for exact non-pow2 R the wrapper overrides `_rail_resolution_for_box`
  to return `round(L/cell_train)`.
- **Scale:** start with nsamples=256 (CPU; N=125 sequences are long) — feasibility, not a
  final benchmark. GPU may be used opportunistically if free.
- **Evaluation:** `benchmark_lj27.py --target <size-matched h5> --pattern <samples>` →
  g(r) L1 (gr_L1), OTgap, clamped energy, min-dist, composite.

**Go/no-go (Stage 2), reading OTgap** (in-dist N=27 K=1 ≈ 0.51; uniform = 1.0):

| Band | OTgap | Meaning |
|------|-------|---------|
| Green  | ≲ 0.6      | Zero-shot transfer works; g(r) first peak near r≈1.0 |
| Yellow | 0.6 – 0.8  | Marginal; fine-tuning likely needed |
| Red    | ≳ 0.8      | No meaningful transfer (≈ noise) |

## Deliverables

1. `analyze_k1_rail_transfer.py` + `reports/k1_rail_transfer/` (KS table, histograms).
2. Zero-shot sample wrapper + `samples_*` npz at N=64/125.
3. `benchmark_lj27.py` numbers at both sizes.
4. A short findings summary with the explicit go/no-go call and recommended next step
   (proceed to multi-size training, fine-tune, or stop).

## Non-Goals (YAGNI)

- No training or fine-tuning in this feasibility phase.
- No new model architecture; reuse the existing K=1 checkpoint and sampler.
- No densities/temperatures other than ρ=1, T=1.
- Not a publication-grade benchmark (small nsamples acceptable for the go/no-go).

## Risks

- **Non-pow2 R waypoint leakage** (exact-R arm): waypoints may decode out-of-box; mitigated
  by the Stage-1 in-box-fraction check disqualifying the arm.
- **CPU sampling cost at N=125:** mitigated by reduced nsamples; GPU opportunistic.
- **Checkpoint still training:** use the latest `best.ckpt`; numbers may nudge as the K=1
  run finishes (epoch ~390/400), but the gap to K=8 is large enough that the call is stable.
