# mW generator v5 — prefix-derived local frames (design addendum)

**Date:** 2026-07-10 · **Status:** approved (user-authored diagnosis, reviewed+verified) · **Parent:**
2026-07-09-mw-annealed-smc-design.md §4 · **Scope:** bounded bet — ONE retrain on the same bank with a
teacher-forced structural gate as go/no-go. Does not change the SMC design; Phase 2 may run with the
tempered v4 base independently.

## Verified evidence (2026-07-10, 2000 reference configs)

The v1–v4 output coordinate system (offset from the FIXED slot anchor t_j = scaffold cell center) is
misaligned with the liquid:
- **72.0%** of canonically-ordered particles lie outside their slot's cell;
- median particle→anchor distance **1.333σ** > first-shell radius 1.19σ > half-cell 0.649σ;
- 13.3 empty + 12.9 multi-occupied cells per config (sub-Poisson: liquid regularity).

Consequence: the head must model box-scale, occupancy-dependent conditionals — mode-width inflation is
STRUCTURAL. Confirmed downstream: TF ≈ FR g(r) (drift innocent), first shell 1.19–1.47 vs data 2.12,
excluded volume blurred (g(1.0) = 0.61 vs 0.12), temperature scan inert on structure, val NLL 0.51
co-existing with wrong TF structure (mass-covering hedging across candidate sites ~1.3σ apart).
KA-lineage note: the 3D port DROPPED ka_flowhead's prefix-adapted origin (Gaussian-weighted centroid of
placed neighbors around the scaffold cell) — v5 restores and strengthens that mechanism.

## v5a design (all proven components)

1. **Prefix-derived origin** (KA mechanism restored): o_j = wrap(t_j + Σᵢ wᵢ·wrap_pm(xᵢ−t_j, L) / Σᵢ wᵢ)
   over causal prefix particles with Gaussian weights wᵢ = exp(−|wrap_pm(xᵢ−t_j)|²/(2σ_o²)), σ_o = cell
   width 1.30σ; j=0 → o_0 = t_0. The scaffold selects REGION AND ORDER ONLY; it no longer defines the
   output coordinate system.
2. **Prefix-derived orientation**: `build_frames` (existing, fallback-hardened) on the two nearest
   causal prefix particles relative to o_j.
3. **Offset**: u = Rf·wrap_pm(x_j − o_j, L)/s, s = 0.65σ. |u| ≤ (L/2)/s = 4.0 exactly — the spline's
   tail_bound=4 covers the entire principal image; no wrap-truncation regime at all.
4. **Causal-kNN tokens** (k=12 nearest prefix particles of o_j), geometry-only features per token:
   frame coords (3), r (1), fixed-period Fourier(r) (8), min(1/r², 4) (1), r/1.19 (1),
   prefix-occupancy scalar n_valid/k (1), and an angular feature matched to the mW three-body kernel:
   (cos θ + 1/3) where θ is the angle at o_j between this neighbor and the nearest neighbor (1)
   [implementer may add a pooled tetrahedral summary token; features must remain geometry-only —
   no curve indices]. Per-step encoder over k+1 tokens (CLS), as the v1/v2 lineage.
5. **Head**: AR local-frame RQS spline (Spline3Head, num_bins=32, tail_bound=4.0) — demonstrated
   0.16σ resolution; multimodal per axis (the legitimate site-multimodality must survive).
6. **Exactness invariants (unchanged, co-first-class)**: ungated sample↔log_prob(preordered) on every
   draw; perturbed-weights mirror at 1e-4-scale; origin/frame are deterministic prefix functions ⇒
   per-step change of variables exact (|det Rf| = 1).
7. **Checkpoint selection includes TF structure** (the val-NLL-blindness lesson): every val cycle,
   compute TF pooled-predecessor g(r) on 64 val configs (one forward + head samples); track
   peak_err = |g(1.19) − 2.18|/2.18 and core_mass = mean g on r ∈ (0, 1.0). Save best_nll AND
   best_struct (composite = peak_err + core_mass) checkpoints; report both curves.
8. **Training**: same 72k-config bank (thin-2 primary + extension), augmentation on, 30k steps
   (warm-continue by the val/structure curves), batch 32–48.

## Go/no-go gate (pooled-predecessor TF g(r) — the metric where a perfect causal conditional MUST
match data exactly; no future-marginalization excuse)

GO for Phase-2/3 use: TF g(1.19) within 10% of 2.18 AND TF core_mass(r<1.0) < 0.02 AND exactness green.
MISS → **v5b**: radial-first exact head p(r|h)·p(n̂|r,h) (new exactness surface: r² and sphere
Jacobians + normalization tests) and/or **v5c**: v4 global body + v5 local output frame (hybrid).

## Out of scope / cautions

One-shot ESS is NOT a KPI (any base at N=64 gives ESS≈1 one-shot; the annealer is the fixer). The KA
50%-future wall still bounds FULL-ensemble one-shot structure — the gate metric is deliberately the
predecessor-pooled TF curve, which a correct conditional saturates. Energy/angular auxiliary losses
only AFTER the representation fix, if still needed.
