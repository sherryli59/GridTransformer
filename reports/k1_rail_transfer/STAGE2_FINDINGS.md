# Stage 2 (zero-shot) findings — K=1 rail size transfer

Strategy used: **pow2** (from Stage 1; exact was disqualified by raw-cell leakage).
Checkpoint: `lj_ckpts_lj27_pbc_fixedrail_k1_ablation/lj27_pbc_continput_fixedrail_k1_fullcov/best.ckpt`.
Sampler: `sample_k1_transfer.py` (forces R per box; checkpoint has cell_size=None).
256 samples/size on CPU, temperature 0.9. Benchmark: `benchmark_lj27.py` vs size-matched MCMC.

| size | R | gr_L1 | OTgap | Uc/N | clash% | g(r) peak r | uniform OTgap | band |
|------|-----|-------|-------|------|--------|-------------|---------------|------|
| N=27 (in-dist ref) | 64 | 0.19 | 0.51 | 12.2 | ~99 | ~1.0 | 1.0 | (baseline) |
| N=64  | 128 | 0.3129 | 1.442 | 44.5 | 100.0 | — | 1.0 | RED |
| N=125 | 128 | 0.2516 | 1.629 | 45.9 | 100.0 | — | 1.0 | RED |

Reference: in-dist N=27 K=1 OTgap≈0.51; uniform-noise OTgap=1.0. Bands: green ≲0.6,
yellow 0.6–0.8, red ≳0.8.

## Verdict: NOT FEASIBLE (zero-shot)

Both transfer sizes score **OTgap > 1.0 — worse than uniform random noise**. The trained K=1
rail model does not transfer to N=64 or N=125 without retraining. gr_L1 collapses to roughly
the uniform-noise level (0.31 vs uniform 0.30 at N=64; 0.25 vs 0.25 at N=125) and clamped
energy is above the uniform base — the autoregressive generator produces structured-but-wrong
configurations, not merely degraded ones.

## Why it failed (and why the audit didn't catch it)

- **N=125 failed despite a *better* rail-input KS than N=64** (0.163 vs 0.300). Both are RED.
  So the Stage-1 audit's marginal/borderline read under-stated the problem at N=125.
- **Distribution match was necessary but not sufficient.** Stage 1 confirmed the rail input is
  approximately scale-invariant and the cartesian-delta target is near-identical (KS < 0.024).
  Neither guaranteed the *conditional* p(delta | rail, sequence-context) transfers:
  - `reference="absolute"` rail encodes an absolute box position; at a larger box the
    min-imaged absolute waypoint lands outside the coordinate range the model saw at L=3, so
    the conditioning input is an extrapolation.
  - Autoregressive error compounds over 2–5× longer particle sequences (64/125 vs 27).
  - The pow2 cell shrinks (0.031/0.039 vs training 0.047), shifting the rail-vector magnitude
    the model was tuned on.

## Recommended next step

Zero-shot is ruled out. To pursue size transfer for the rail family, choose one of:

1. **Multi-size training** — train on a mix of N=27/64/125 so the model sees the larger boxes
   and longer sequences directly. Highest-confidence path; most expensive.
2. **Translation-invariant rail + scale-invariant target** — switch `reference` from
   "absolute" to "prev_step" (the expected curve *step*, translation-invariant) and/or adopt
   the arc-length representation (Δs, fine), removing the absolute-position extrapolation. Then
   re-run this same two-stage feasibility study.
3. **Fine-tune at the target size** — cheaper than full multi-size training; tests whether a
   short adaptation closes the gap. Worth a small run before committing to (1).

Do NOT invest in (1) until a quick (2)/(3) check shows the conditional can be made to transfer
— the audit shows the *features* already align, so the failure is in the learned conditional,
which fine-tuning or a translation-invariant reference targets directly.
