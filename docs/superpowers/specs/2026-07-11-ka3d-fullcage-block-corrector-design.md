# Contractive full-cage block corrector — design

**Date:** 2026-07-11
**Status:** design (approved; plan to follow)
**Context branch:** liquid-coupling-flow

## Problem

The AR block generator (`KA3DScaffoldEBM`, `ka3d_ebm_batched.KA3DScaffoldEBMBatched`) produces **clashy
large-block proposals**, which blocks basin-crossing for the point-to-set (PTS) measurement: large
collective moves / MTM-jumps propose high-energy configs (+48/particle above equilibrium at R=2.0), so the
exact MTM can only sample garbage (see reports/logs-2026-07-11/pts_results.md).

**Root cause — verified, not assumed (`reports/logs-2026-07-11/test_halfcage.py`, K=12 block, 12 cavities,
per-particle clash rate <0.8):**
| generation | clash | isolates |
|-----------|-------|----------|
| FR (sampled prefix) | 42.8% | half-cage + drift |
| clean-prefix (data prefix) | 57.4% | half-cage, NO drift |
| full-cage (all others at data) | 29.4% | NO half-cage |

- **DRIFT effect (FR − clean) = −14.6%** -> exposure bias is REFUTED (FR is no worse than a perfect data
  prefix). This is why the past exposure-bias fixes failed: energy fine-tune COLLAPSED the model (overlap
  0.21->0.81, U/N 67->599 in `eft_gumbel_N100.out`); scheduled sampling targets a cause that isn't present.
- **HALF-CAGE effect (clean − full) = +28.0%** -> the wall is STRUCTURAL: an AR block particle conditions
  only on its causal prefix, so it cannot see the FUTURE block particles it must avoid. Matches
  `gap-is-residual-not-drift` (TF->FR ~0.002) and `full-cage-lever-needs-energy` (full-cage >> half-cage).
- **Residual:** even full-cage is 29% clashy -> ONE full-cage pass is not enough; the conditional is
  imperfect, so the fix must ITERATE (contract) the full-cage map, and iterating raw collapses (pure Gibbs,
  `full-cage-lever-needs-energy`) -> it must be CONTRACTIVE and trained by likelihood, not energy.

## Goal

A generator whose large-block proposals are CLEAN (low-clash, ~equilibrium energy) while preserving EXACT
tractable `log q` (the property the MTM/IS rely on), so large collective moves accept and cross glassy basins.

## Approach: contractive, invertible, iterative full-cage corrector on the FROZEN AR base

Do NOT retrain the AR generator and do NOT use an energy/reverse-KL objective (that collapsed before). Add a
shallow invertible flow that sees the FULL block and pushes clashes apart, composed with the base for exact
likelihood (the campaign's `tf2boltz` AR-base+flow pattern), trained by MAXIMUM LIKELIHOOD on data.

### Components
1. **Base (frozen):** `KA3DScaffoldEBM` AR block generator. `sample_block` -> block positions `x0`
   (local u-space, anchor-relative) + exact `log q_AR(x0)`.
2. **Corrector flow `F` (new, trained):** L conditional coupling layers over the K block positions in
   u-space. Each layer partitions the block particles into two sets; transforms set A with a per-particle
   affine map (scale, shift) computed by an attention/message-passing net over {ALL block particles in set
   B + the cage (retained interior + boundary), with species + relative positions}; alternate the partition
   across layers so every particle is transformed while seeing the others. Diagonal-affine coupling =>
   `log|det|` is the sum of log-scales (tractable). Conditioning on the OTHER block particles is what
   supplies the FULL CAGE that the AR base lacked; stacking L layers is the CONTRACTION that drives the 29%
   residual down.
3. **Composition (exact):**
   - Sample: `x0, lq_AR = base.sample_block(...)`; `x = F(x0 | cage)`;
     `log q(x) = lq_AR - log|det dF/dx0|`.
   - Score (MTM reverse): `x0 = F^{-1}(x | cage)`; `log q(x) = log q_AR(x0) - log|det dF/dx0|`.
   - Both directions batched-over-M (reuse `ka3d_ebm_batched`).

### Training
- MLE on carved cavity DATA blocks: maximize `log q(x_data) = log q_AR(F^{-1}(x_data|cage)) - log|det|`,
  loss `-log q / K`. Base FROZEN; train only `F`. Data manifold is clean, so `F` learns to map the base's
  clashy region onto clean configs WITHOUT an energy objective (no collapse).
- Multi-radius cavities (as the base), knn cage cap for size-transfer.

## Exactness
Invertible coupling => sample `log q` == score `log q` on the same config to fp precision (the existing
8e-3 base cat-head leak is inherited, not amplified). Gate before training with `F` = identity init.

## Validation gates (in order)
1. **Exactness:** composed sample-vs-score round-trip (identity-init F reproduces the base; trained F stays
   exact to the inherited 8e-3).
2. **Clean gate (the point):** corrected K=12 block clash < the 29% full-cage floor, and corrected block
   ENERGY ~ equilibrium (the +48/particle failure must be gone). This is the make-or-break gate.
3. **No-collapse:** corrected samples do NOT collapse (energy stays at equilibrium; overlap does not blow up
   like the energy-finetune A).
4. **MTM acceptance:** large-K (K>=8) block-MTM acceptance with the corrected proposal >> the raw AR.
5. **Basin-crossing (end goal):** MTM-jump with the corrected proposal crosses basins with CORRECT energy
   (the energy check that refuted the raw MTM-jump must now pass); qc_pair moves to a value consistent with
   an independent reference.

## Files
- `liquid_coupling_flow/ka3d_block_corrector.py` — corrector flow (coupling layers), `F`/`F^{-1}` with
  log-det, composed `sample_block_corrected` / `block_log_prob_corrected` (batched).
- `reports/logs-2026-07-11/train_corrector.py` — MLE fine-tune of `F` (base frozen).
- `reports/logs-2026-07-11/test_corrector.py` — exactness gate + clean gate + energy/no-collapse + K-sweep.

## Risks / open questions
- **Flow expressiveness:** this campaign's flows underperformed (uniform-base + coupling arms failed the
  in-dist gate). MITIGATION: the corrector rides a GOOD AR base (does most of the work) and is SHALLOW; it
  only needs to fix half-cage clashes, not model from scratch. If a shallow post-hoc corrector plateaus,
  escalate to fine-tuning the base jointly (still MLE) or Approach B (refinement head).
- **Frame/equivariance:** `F` operates in the base's local u-space; the conditioning net must be
  translation/rotation-consistent with the base (frameless + the cage relative positions). Reuse the base's
  `_periodic`/`nbr_proj`/`sp_emb` conditioning style.
- **Species:** the block move is count-preserving in species; `F` transforms POSITIONS only (species from
  the base), so no count issue.
- **Cost:** L coupling layers over K+cage per block; batched-over-M keeps it cheap (the batching win).
