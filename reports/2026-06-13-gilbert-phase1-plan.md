# Gilbert Phase-1 plan: extrapolating size transfer with constant-cell substrate + rail

**Date**: 2026-06-13
**Branch**: `hilbert-arc-repr`
**Prereqs proven**: rail conditioning solves held-out *interpolation* (L4 OTgap 0.892→0.688,
[[rail-fixes-size-transfer]]); P1 bridge-normalization collapses the absolute anchor
(α≈0.15, universal C≈0.42); P5 gilbert even-R motif statistics match pow2 Hilbert.
**Inputs**: `reports/2026-06-12-size-transfer-action-plan.md` (Phase-1 design),
`reports/2026-06-12-gilbert-curve-plan.md` (gilbert impl), `gilbert-curve-option` memory.

## What's new since the Phase-1 design

The size-transfer action plan posited two arms (C′ normalized-anchor target, B+
rail-conditioned Δs) as parallel bets. Tonight resolved the bet: **B+ (rail) is validated**
on the pow2 substrate at the held-out *interior* size. So Phase-1 is no longer "which arm
works" — it is **"does the validated rail extrapolate BEYOND the trained ladder, and does the
constant-cell substrate widen that reach?"** C′ becomes the complementary parameterization
test, not the primary hope.

## The open question Phase-1 answers

Tonight proved **interpolation** (L4 between trained L3/L5). The unsolved gate is
**extrapolation**: train a contiguous ladder, hold out sizes *beyond* it (L7, L10), and ask
whether the rail-conditioned model still generates refinable configs. The findings doc's
L10 result showed a matched cell alone does not rescue extrapolation; the rail is the new
ingredient that might.

## Substrate (Phase 0.5, user-led — prerequisite)

- **Gilbert constant-cell caches**, even-R only (P5: odd-R multicell artifact; even-R
  matches pow2 Hilbert at all scales incl. production R=86/106). Cell c = 3/64 ≈ 0.0469;
  per-box R = nearest-even(L/c). X = 1/(ρc³) constant across sizes.
- **Ladder**: train {L3 (R64,N27), L5 (R106,N125), L6 (R128,N216)}; hold out **L4
  (R86,N64) interior** and **L7 (R150,N343) + L10 (R214,N1000) exterior** for the
  extrapolation gate. (Confirm exact even-R per the nearest-even rule at build time.)
- Rebuild verification: rerun the §2 marginal A/B on the gilbert cache to confirm Δs/fine
  units collapse exactly across sizes (gilbert s-space byte-exact for even grids).

## Two arms (both on the gilbert substrate)

### Arm B+ — rail-conditioned Δs (primary; port the validated pow2 result)
- Target = the proven tight `(Δs, fine)`; rail input via `fixed_template` `rail_attn`
  (K=8, `absolute`, zero-init out_proj — all validated tonight). No noise (P6 showed it
  does nothing).
- Only change from tonight: gilbert curve + wider/contiguous ladder. The rail waypoints
  are synthesized from (N, R, box) — already gilbert-compatible via `get_curve3d`.
- **Hypothesis**: constant cell removes the per-size cell confound, so the one local
  conditional the rail steers is identical at every size ⇒ the rail's pacing fix should
  extend to exterior sizes, not just interior.

### Arm C′ — normalized absolute-anchor target (complementary parameterization)
- Target = `(pos_j − decode(j·X)) / scale(j,N)`, scale = bridge profile
  `(j(N−j)/N)^α`, α≈0.15 (P1-fitted) — non-drifting by construction, size-invariant by
  the validated normalization. `curve_rail_residual_target` + `fixed_template` machinery
  exists; the delta is the scale factor at cache build + its inverse in the sampler.
- **Why still run it**: the rail conditions a *drifting* target (Δs); C′ removes drift at
  the parameterization level. If the rail's extrapolation degrades at L10 (drift over a
  1000-step rollout), C′'s non-accumulating target is the fallback that should not.
- Watch: monotonicity is no longer structural — track the noncanonical/backward-step
  counters (tonight's rail had 3.6% at k=48; baseline arc had ~1.4% at trained sizes).

## Protocol

Per arm: gilbert ladder, dim 512 / depth 4 / heads 4, full-cov head, clean-val
checkpointing (`val/loss` + `save_last`). Warm-start from the existing pow2 baseline is
NOT valid (different substrate) — train from scratch, ~100 epochs, OR warm-start from a
short gilbert-baseline run; decide at launch based on convergence.

**Staged budget**: run both arms to **35 epochs**, gate on (i) held-out L4 Δs W1 + gen
mean, (ii) reduced-sample (128) L4 + L7 OTgap, (iii) trained-size val NLL health. Extend
the winner (or both if within noise) to 100. Tonight showed the rail's effect is visible by
epoch 2, so 35 epochs is ample to rank.

## Eval battery (per arm, per size: trained, L4, L7, L10)

1. **P3 drift** (`analyze_p3_drift_onset.py`, rail-aware) — Δs mean/W1 vs teacher-forced k.
2. **P4 closure** (`analyze_p4_closure.py`, rail-aware) — curve traversal fraction.
3. **OTgap** (`gen_arc_samples.py` → `benchmark_lj27.py`) — refinability, the headline.
4. **MCMC relaxation** (`mcmc_relax.py`, NEW) — 200-sweep Metropolis recovery of Boltzmann
   energy + g(r) vs target, with uniform cold-start control. The most physical test.
5. Monotonicity counters (noncanonical / backward-step / clamp).

## Decision gates

| observation | action |
|---|---|
| B+ green (OTgap ≲0.6) at L7 AND L10 | **extrapolation solved** — rail + constant cell is the recipe; ship, write up |
| B+ green at L7 but degrades at L10 | drift accumulates over the long rollout → promote C′ (non-drifting target) for large N; or stack Phase-2 budget guidance |
| B+ fails to extrapolate at all | constant cell insufficient; the conditional itself doesn't generalize — revisit representation (k-step anchor, Option A) |
| C′ beats B+ at L10 | normalized anchor is the large-N parameterization; consider C′+rail-input hybrid |
| relaxation: rail recovers Boltzmann, uniform does not | confirms refinability physically — the strongest claim |

## Explicitly deferred / out of scope here

- Phase-2 sampler-side budget guidance (stackable; test only if a long-rollout drift
  appears at L10).
- k-step bounded anchor (Option A) — only if both arms fail to extrapolate.
- Down-stream EGNN flow refinement integration (separate project).
