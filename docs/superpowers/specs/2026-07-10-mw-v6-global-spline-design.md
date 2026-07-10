# mW generator v6 — global body + circular-spline torus head (design)

**Date:** 2026-07-10 · **Status:** approved, evidence-backed · **Parents:** v5 (2026-07-10-mw-v5) ·
**Scope:** ONE retrain. Minimal, measurement-justified change on the proven-generalizing v4 body.

## Why v6 (every choice traces to a measurement, 2026-07-10)

Three isolation experiments on the N=64 mW bank settled the architecture:

1. **Free-code head-capacity overfit** (each (config,step) gets a free trainable code; fit the true
   offset): circular spline **−11.3** NLL/particle vs full-cov tanh-MDN **+1.65**. ⇒ the factorized
   circular spline is NOT expressively blocked — it is the *sharper* head; the MDN is the one that
   cannot sharpen (mode-collapse / covariance conditioning). **Refutes "revert to v4's MDN head."**
2. **Full-model overfit on 16 configs** (matched to the real trainer: lr 3e-4, clip 5.0): v5 drives
   train-NLL 4.94 → −2.70 and falling. ⇒ no bug, no capacity wall; v5's real-run plateau at val 3.46
   is a GENERALIZATION gap, not the head.
3. **v4 vs v5 generalization**: v4 (global-AR body + MDN) val **0.51**; v5 (local kNN-12 body + spline)
   val **3.46**. v5 changed body AND head; the head change should have *helped* (exp. 1), yet v5 is 7×
   worse ⇒ the **local body is the regression**. The global-frontier context (occupancy pattern, which
   cells are filled) is strongly predictive of where particle j goes — v5's aggressive localization
   discarded exactly the mid-sequence information the per-step NLL probe had flagged.

Plus the origin scan (v5 prep): median particle→origin distance — anchor **1.31σ**, every Gaussian
centroid 1.46–1.51σ, nearest-neighbor 1.57σ. ⇒ **no prefix-derived origin beats the bare anchor** (all
point into the occupied bulk, away from the vacancy the particle fills). v4 already uses the anchor.

## v6 = v4 body + circular-spline torus head (the synthesis)

- **Body: v4's `MWGlobalAR` body UNCHANGED** — global causal AR over all placed particles, `CurveRail`,
  `GeometricBias`, `CausalGeoBlock` × n_layers. This is the component that generalizes (val 0.51). Reuse
  by import; do not re-tune.
- **Head: `CircularSpline3Head` (from v5), replacing `FullCovTanhMDN`** — the one evidence-backed change.
  It is (a) sharper than the MDN (exp. 1) and (b) **toroidally exact**: uniform base on the 3-torus,
  periodic seam, no `tanh` chart distortion. This directly fixes v4's ONLY non-toroidal component (the
  MDN's `bound·tanh(z)` open-box chart, which distorts densities at cell boundaries).
- **Chart: transfer-safe.** u = wrap_pm(x_j − t_j, L)/s with **s = L/(2·bound), bound = 4.0 fixed**
  (v5 convention), anchor origin t_j = scaffold cell center. This normalizes |u| < bound for ANY N —
  unlike v4's s = (L/R)/2 which gives |u| < R and would break the fixed-bound spline at transfer
  (R = 4 at N=64 but 6 at N=216). CRITICAL for Phase 3.
- **mW three-body features (enhancement, ablatable):** v4's body has NO angular encoding, but mW's
  Stillinger-Weber three-body term is tetrahedral. Add a per-position geometric summary to the body's
  input: pooled over the anchor's k=12 nearest PLACED neighbors — mean/logsumexp of (cos θ_ij + 1/3)
  over neighbor pairs (tetrahedral), mean 1/r² (excluded volume), and occupancy count/k. Concatenated
  to `prev_proj`'s input (or a parallel `geo_feat_proj`), zero-initialized so v6 starts == v4-body.
  This is the LOWER-confidence addition; if v6's TF g(r) regresses vs a no-feature ablation, drop it.
- **TF-structural checkpoint selection (from v5):** save best_nll AND best_struct (composite =
  peak_err + core_mass on TF pooled-predecessor g(r), 64 val configs each val cycle); report both.

## Exactness (unchanged, co-first-class)

Ungated sample↔log_prob(preordered) on every draw (≤ 1e-4 per-particle); perturbed-weights mirror;
origin + chart are deterministic torus translations, |Jacobian| = 1; the frame (if used for features)
affects conditioning only. CUDA histogram must `.cpu()` before `torch.histogram` (v5 fix). Ordering
modes: canonical for training, preordered for the SMC base.

## GO/no-go gate (same as v5)

GO for Phase-2/3 use: TF pooled-predecessor g(1.19) within 10% of 2.18 AND TF core_mass(r<1.0) < 0.02
AND exactness green AND val NLL < v4's 0.51 (the body must not regress). Also record one-shot g(r).
MISS → ablate the three-body features first; then v5b radial-first head as the deeper fallback.

## Out of scope

One-shot ESS is not a KPI. Do not re-tune the v4 body. No prefix-derived origin (settled). Energy
auxiliary losses only after the representation fix, if still needed.
