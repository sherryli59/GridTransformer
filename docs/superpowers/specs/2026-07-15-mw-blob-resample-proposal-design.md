# Cage-anchored blob-resample proposal for the mW transition kernel

**Date:** 2026-07-15 · **Branch:** liquid-coupling-flow · **Status:** design approved in brainstorm (C with expert corrections), pending spec review

## 0. Why this exists

The mW kernel campaign measured that every block proposal on disk places particles into core
overlaps: `MWPeriodicBlockAR` (250-step, non-local global-centroid anchoring) gives median β·ΔU
= +121 at K=1 — its noise-off **mode** is a core overlap (min-NN 0.68σ vs data 1.11σ); the
locally-anchored `mw_scaffold` (KA cavity port) transfers N512→N64 correctly but still gives
+55/+86 β·ΔU at K≈4/7 (full-cavity) and +14 at K=1 (partial). Root causes, both structural:
(a) anchor/cell parametrization fights the conditional (memory `deltas-exact-ordering-negative`:
centroid-anchoring sharpens, cell/grid-anchoring fights); (b) inherited PTS pinned-boundary
machinery we do not need — a blob in bulk has no frozen shell, only a cage.

This model is purpose-built for the transition kernel: resample a blob of K particles given its
surrounding cage, centroid-anchored, exact for MH, β-conditioned, size-transferable. It replaces
the repurposed scaffold model as the `two_blob`/bridge proposal and supplies the single-site
(K=1) full-cage heat-bath (KA-proven ~33% acceptance, memory `ka-heatbath-depth-lever`).

## 1. Object and interface

New module `liquid_coupling_flow/mw/mw_blob.py`, class `MWBlobProposal`. API mirrors what
`mw_kernels.two_blob_move` / the Task-6 bridge consume, plus `beta`:

- `sample_block(x, block_idx, L, beta, gen) -> (x_new, logq)`
- `block_log_prob(x, block_idx, L, beta) -> logq`

`x` is `[B,N,3]` (or `[1,N,3]` per-walker as `two_blob_move` calls it); `block_idx` the K blob
particle indices; `beta` the target inverse temperature for the proposal. **Bridge coupling:** at
rung π_λ ∝ q0^{1−λ}·e^{−λβU}, pass **β_eff = λ·β** (the Boltzmann part's effective inverse temp),
so one β-conditioned model serves every rung and the cold target. (One-line change if we later
want the proposal to always target true β and leave λ only in the acceptance ratio.)

`two_blob_move`/the bridge need `beta` threaded through their `block_model.*` calls — a small,
localized edit to `mw_kernels.py`/`mw_smc_portfolio.py`, not a redesign.

## 2. The exact conditional

Cage = the M≈24 nearest surrounding (non-blob) particles to the blob centroid (min-image),
**frozen for the entire move**. All cage particles are context for every slot; no interior/boundary
split.

### 2.1 Anchor (sharp, exactly invertible) — from `ka_localframe`

For blob slot j in generation order:
```
a_j = centroid( full cage  ∪  x_{<j} )      # PBC-safe soft/weighted centroid
u_j = x_j − a_j
```
`a_j` depends only on the frozen cage and the already-placed prefix, so the coordinate transform is
triangular with **unit-diagonal Jacobian** (log|det| = 0). β enters the head only; it never touches
this map. **Sites (below) set order only and MUST NOT replace this anchor** — anchoring to sites is
the scaffold flaw this design removes.

### 2.2 Site order (cage-only → order is exact both directions)

```
c0     = min-image centroid of the cage
Frame  = deterministic PCA frame of cage displacements about c0
sites  = c0 + Frame · pattern[K]            # pattern[K] = fixed Morton/Fibonacci points, scaled to local spacing
order  = pattern index (Morton)             # slots run in this order
```
Because sites are a function of the frozen cage alone, the slot order is identical in the forward
and reverse directions.

**Deterministic PCA orientation (stability, not balance — cage is frozen mid-move):** explicit
tie-breaking for (i) eigenvector signs (fix sign by, e.g., largest-|component| positive), (ii)
eigenvalue near-ties (canonical ordering + a documented deterministic rule), (iii) right-handedness
(force det(Frame)=+1). Tetrahedral mW cages will produce near-degenerate frames often; gate 5
tracks the PCA gap vs acceptance/NLL.

### 2.3 Particle↔site assignment — Hungarian ALWAYS (the residual noncanonical bookkeeping)

The site *order* is exact/cage-only, but assigning the K blob particles to the K slots is a
**position-dependent canonicalization** — the surviving, minimized form of the sample-order /
canonical-order issue. Handle it with a **total deterministic bijection**:

- **Hungarian matching always** (blob labels ↔ sites by min-image distance cost). NOT
  "nearest-site, reject collisions" — rejection would create state-dependent null regions.
- Operate on **labeled particle state** throughout.

### 2.4 Exact MH bookkeeping (labeled forward/reverse)

- **Forward** `q_C(y | x, cage)`: Hungarian-match blob labels **at source state x** → bijection
  σ_x (slot → label). Generate in slot order σ_x with the §2.1 anchor. `logq_fwd` = accumulated
  sample log-density under σ_x (this is the true density of the generation that produced y).
- **Reverse** `q_C(x | y, cage)`: recompute Hungarian **at proposed state y** → σ_y (may differ
  from σ_x). Score the **old labeled** coordinates x in slot order σ_y.
- Acceptance:
  ```
  log α = −β·ΔU + log q_C(x | y, cage) − log q_C(y | x, cage)
  ```
  each density using the assignment induced at **its own source endpoint**. Exact for ANY
  permutation-mismatch rate; the mismatch rate (gate 3) only measures how much noncanonical work
  the scheme is doing. Do **not** substitute a generic "canonical score" for the accumulated sample
  log-density when generated particles match different sites than their generation slots.

### 2.5 Head

Factorized 3-axis categorical (Cat3), 64 bins over ±2.5σ (≈0.078σ resolution — avoids the
`num_bins=8` spline floor from the base-quality campaign). Conditioned on `[cage kNN-transformer
context, placed-member features, site s_j features, β embedding]`. Exact normalized categorical ⇒
exact `logq`; `sample()==block_log_prob()` by shared per-slot feature construction.

### 2.6 K=1 degenerate case

One slot, no ordering, no Hungarian: the single-site full-cage heat-bath. `a = centroid(cage)`,
`u = x_i − a`. This is the primary early milestone (§5, G-K1).

## 3. Blob selection

Reuse `mw_kernels` machinery unchanged: block = K-nearest to a **state-independently drawn**
center, with the existing reverse-check (same center re-selects the same set under the proposal) so
selection probabilities cancel. No new selection exactness surface.

## 4. Training

- **Data**: carve varK blobs (K∈{1..16}, K-nearest to random centers) + their cages from banks at
  several temperatures (ambient + D0 rungs down to T\*_work); tuples `(blob, cage, β)`. Local by
  construction ⇒ train on N=512 banks, transfer to N=64 (safe big→small degree direction, memory
  `ka-block-transfer`).
- **Objective**: `L = NLL(data blobs, §2 factorization) + λ_rl · mean(adv.detach() · logq_freerun)`,
  `adv = standardize(−clamp(U_insert(free-run blob), 0, cap))` (handoff energy lever;
  `blockcond-ft-energy-lever` precedent ~50× proposal-energy cut). Per-config backward; λ_rl
  warmup after an MLE-only burn-in so the energy term cannot collapse the density.
- **β-conditioning**: sample β per batch across the training range `[1/0.09632 … 1/T*_work]`;
  the free-run reward uses that β.
- **Augmentation**: SO(3) rotation of each (blob+cage) sample (global-orientation frame ⇒
  rotational invariance, as `ka_localframe`).

## 5. Validation gates

1. **G-exact** (hard): `sample()==block_log_prob()` ≤ 1e-4/particle across K∈{1,4,8,16}, β range.
2. **G-hungarian-determinism**: repeated Hungarian evaluation on the same state returns the
   identical bijection (deterministic cost + deterministic tie-break).
3. **G-involution**: forward/reverse transition-score involution — scoring is self-consistent
   under the labeled σ_x / σ_y construction.
4. **G-mismatch-rate**: measure generated-slot vs post-generation Hungarian permutation-mismatch
   rate (quantifies the "minimal noncanonical" claim; report distribution, not just mean).
5. **G-K1**: single-site acceptance at T\*_work — median β·ΔU ≤ ~1–2, acceptance ≥ ~20%
   (beats the measured +14 baseline; target KA's ~33%).
6. **G-transfer**: train N=512, test K-blob acceptance at N=64 — median β·ΔU flat across N.
7. **G-Kgt1**: joint K-blob median β·ΔU vs measured baselines (scaffold +55 @K≈4, +86 @K≈7) —
   large, viable improvement; viable-fraction up.
8. **G-energy**: RL cuts proposal energy vs MLE-only (target ~10–50×).
9. **G-stationarity**: from equilibrium configs, the labeled-C move leaves the target invariant
   (U-distribution unchanged vs a single-site reference over the same particles) — the
   mutation-verified-gate discipline from `mw_kernels`' two-blob rig.
10. **G-pca-gap** (diagnostic): PCA eigenvalue gap vs acceptance/NLL — flags whether near-degenerate
    cage frames degrade the proposal (decides if B-control is needed).

## 6. Fallback / control

**B (canonical set-order) is retained as the documented control.** If cage-PCA/site assignment
proves unstable (G-pca-gap shows degeneracy hurts, or G-mismatch-rate is high enough to negate C's
simplicity), B's canonical-lift ordering is the fallback — same anchor (§2.1), order by canonical
lift of blob positions with explicit noncanonical accounting. C is primary; B is the escape hatch.

## 7. How it slots into the existing campaign

Replaces the scaffold/toy as the `two_blob`/bridge proposal and adds the K=1 single-site kernel.
**Reuses unchanged**: `mw_kernels` (suffix/two-blob/conveyor), the Task-6 bridge harness
(`mw_smc_portfolio.py`), the eval ledger, the reverse-check selection. Threading edits: add `beta`
to the `block_model` calls in `two_blob_move` and the bridge. Campaign consequences: Task 7's
"RL fine-tune" becomes "train `MWBlobProposal`"; D1 is re-run against it as the honest acceptance
number; Task 8 drives the bridge with it. The suffix kernel keeps using v10 (unaffected).

## 8. Non-goals

- Not a permutation-equivariant flow (that's the EGNN/coupling direction, memory
  `egnn-flow-corrector-poc`; the user chose AR with explicit ordering).
- Not a joint-K move that is plain-MH-viable at the cold target (compounding is physics; K>1 plain
  viability comes only via the K=1 sweep — the joint move's home is the SMC bridge at low λ).
- No pinned-boundary / PTS machinery.
- Not a free-energy/logZ tool (the bridge harness already owns that path).

## 9. Risks / open questions

- Near-degenerate PCA frames in tetrahedral cages (G-pca-gap; deterministic tie-break required;
  B-control fallback).
- Permutation-mismatch rate unknown until measured (G-mismatch-rate); design is exact regardless,
  but high rate erodes C's simplicity advantage.
- RL energy term collapsing the density despite the MLE anchor (λ_rl warmup + monitor NLL).
- Whether β_eff = λ·β or true-β is the better rung coupling — spec'd as λ·β, cheap to flip; decide
  empirically if bridge depth underperforms.
