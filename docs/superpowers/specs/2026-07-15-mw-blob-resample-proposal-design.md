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

New module `liquid_coupling_flow/mw/mw_blob.py`, class `MWBlobProposal`. The MH move is a
transition between two states, so the density API is **two-state** (this is load-bearing — see
§2.4): the reverse density q_C(x | y) must compute its assignment σ_y from `y` and then score
coordinates taken from `x`. A single-state `block_log_prob(x, ...)` cannot express that.

```python
selection = select(x, centers)   # -> {"centers", "block_idx", "cage_idx"}   (§1.1)

sample_block(source_x, selection, L, beta, gen) -> (target_y, logq_fwd)
transition_log_prob(target_x, source_x, selection, L, beta) -> logq
    # computes sigma from `source_x`, scores coordinates from `target_x` in that order
```

`logq_fwd` returned by `sample_block` MUST equal `transition_log_prob(target=y, source=x, selection)`
by construction (same source, same assignment). It does NOT equal a self-score
`transition_log_prob(y, y, selection)` — see G-exact (§5.1).

### 1.1 The `selection` object (makes the cage reversible — §2.3.0)

`selection` is computed once from the **state-independent center(s)** and the **fixed non-blob
coordinates**, and is passed unchanged into forward sampling and reverse scoring:

```python
selection = {
  "centers":   [c]  or  [cA, cB],   # state-independently drawn (one blob) / two-blob
  "block_idx": K (or 2K) blob particle indices,   # K-nearest to center(s), reverse-checked
  "cage_idx":  M non-blob particle indices,        # the cage — see §2.3.0
}
```
Sites, frame, Hungarian costs, and BOTH transition densities derive from this one object, so the
cage/sites/order are identical in forward and reverse by construction — never re-derived from a
blob centroid that moves under the resample.

**Bridge coupling:** at rung π_λ ∝ q0^{1−λ}·e^{−λβU}, pass **β_eff = λ·β** to the proposal head
(the Boltzmann part's effective inverse temp), so one β-conditioned model serves every rung. This
sets the proposal's *internal* target temperature; it does **not** remove the q0 term from the MH
ratio — the bridge still uses the full geometric-bridge ratio (§2.4). (β_eff = λ·β is a one-line
change if we later want the proposal to always target true β.)

**Kernel edit is more than threading β:** `two_blob_move` currently computes the reverse density as
a *self-score of the old state*. The new move needs BOTH endpoints and the `selection` object:
forward `sample_block(x, selection)`, reverse `transition_log_prob(x, y, selection)`. This is a
contained but real interface change to `mw_kernels.two_blob_move` (and the K=1 single-site kernel),
not an additive β argument.

## 2. The exact conditional

### 2.3.0 Cage definition (reversible — from the center, not the blob)

Cage = the M≈24 nearest **non-blob** particles to the **state-independent selection center(s)**
(NOT to the blob centroid — the blob centroid moves under the resample, so a cage keyed to it would
differ between x and y and break reversibility). Because the center and the non-blob coordinates are
identical at x and y, `cage_idx` is identical forward and reverse. It is stored in `selection` and
never recomputed mid-move. All cage particles are context for every slot; no interior/boundary
split.

Cage reversibility is a hard gate (§5, G-cage-reversible): the same `selection` must reproduce the
identical `cage_idx` and `block_idx` when re-derived at the proposed state y — this is the existing
`two_blob_move` reverse-check, now also covering the cage, not just the block.

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
c0     = min-image centroid of the CAGE (cage_idx from selection)
Frame  = deterministic orthonormal frame from selected cage vectors (below)
spacing= cage-only local spacing (median cage-particle nearest-neighbour distance)  # NOT blob positions
sites  = c0 + Frame · (spacing * pattern[K])   # pattern[K] = fixed Morton/Fibonacci unit points
order  = pattern index (Morton)                 # slots run in this order
```
Sites are a function of the frozen cage alone, so the slot order is identical forward and reverse.
**`spacing` must be cage-only** (e.g. median NN distance among cage particles) — never derived from
blob positions, which differ between x and y.

**Frame construction — cage-vector Gram–Schmidt, not raw PCA (reviewer point).** Eigenvectors
inside a degenerate PCA eigenspace are not fixed by eigenvalue ordering, and tetrahedral mW cages
are frequently near-degenerate. Build the frame deterministically from *selected cage vectors*
instead: pick the cage particle nearest c0 as the primary axis seed (`e1 = normalize(cage_1 − c0)`),
the next cage particle giving the largest rejection as the second (`e2` via Gram–Schmidt), `e3 =
e1 × e2` (forces right-handedness). Explicit tie-breaks when two cage particles are equidistant
from c0 or give equal rejection: order by `cage_idx` (a fixed particle label), documented. This is a
deterministic function of the frozen cage, robust where PCA is ill-posed. (Raw PCA-with-tie-breaks
is the documented alternative; G-pca-gap decides if it is ever needed.)

### 2.3 Particle↔site assignment — Hungarian ALWAYS (the residual noncanonical bookkeeping)

The site *order* is exact/cage-only, but assigning the K blob particles to the K slots is a
**position-dependent canonicalization** — the surviving, minimized form of the sample-order /
canonical-order issue. Handle it with a **total deterministic bijection**:

- **Hungarian matching always** (blob labels ↔ sites). NOT "nearest-site, reject collisions" —
  rejection would create state-dependent null regions.
- **Cost** = min-image squared distance between blob-particle coordinate and site, in float64.
  **Fixed solver** (`scipy.optimize.linear_sum_assignment`), pinned. **Deterministic tie-breaking**
  at exact or numerical cost ties: perturb-free rule = lexicographic by `(slot index, cage-derived
  particle label)`, applied inside a documented ε-band so exact ties and float-noise ties resolve
  identically on repeated evaluation (G-hungarian-determinism).
- Operate on **labeled particle state** throughout.

### 2.4 Exact MH bookkeeping (labeled forward/reverse)

- **Forward** `q_C(y | x)` = `sample_block(x, selection)`: Hungarian-match blob labels **at source
  state x** → bijection σ_x (slot → label). Generate in slot order σ_x with the §2.1 anchor.
  `logq_fwd` = accumulated sample log-density under σ_x — the true density of the generation that
  produced y.
- **Reverse** `q_C(x | y)` = `transition_log_prob(target=x, source=y, selection)`: recompute
  Hungarian **at proposed state y** → σ_y (may differ from σ_x). Score the **old labeled**
  coordinates x in slot order σ_y.
- Each density uses the assignment induced at **its own source endpoint**. Exact for ANY
  permutation-mismatch rate; the rate (G-mismatch-rate) only measures how much noncanonical work
  the scheme does. Never substitute a generic "canonical score" (a self-score) for the accumulated
  sample log-density.

**The displayed cold formula is the pure-Boltzmann-target case only:**
```
# λ = 1 (cold Boltzmann target) ONLY:
log α = −β·ΔU + q_C(x|y) − q_C(y|x)
```
**At a geometric-bridge rung the q0 term does NOT drop** (passing β_eff=λβ to the head changes the
proposal's internal temperature, not the target ratio). The move MUST use the full bridge ratio via
`ka3d_smc_bridge.geometric_bridge_log_accept`:
```
log α = (1−λ)[log q0(y) − log q0(x)]   − λ·β·ΔU   + q_C(x|y) − q_C(y|x)
```
with `log_r_reverse = q_C(x|y)`, `log_r_forward = q_C(y|x)`, and q0 = the bridge base (v10/uniform),
distinct from the proposal density q_C. `two_blob_move` already calls `geometric_bridge_log_accept`;
the change is only that `log_r_reverse` becomes the two-state `transition_log_prob(x, y, selection)`
instead of a self-score.

**Transition pseudocode (canonical):**
```python
selection = select_from_state_independent_centers(x)      # centers, block_idx, cage_idx (§1.1)
sigma_x   = hungarian(x, selection)                        # §2.3
y, logq_fwd = sample_using_source_assignment(x, sigma_x, selection, beta)   # = q_C(y|x)

if not reverse_selection_matches(y, selection):           # G-cage-reversible: cage_idx & block_idx reproduce at y
    reject()

logq_rev = transition_log_prob(target=x, source=y, selection=selection, beta=beta)  # computes sigma_y, = q_C(x|y)
# then geometric_bridge_log_accept(log_q0_current, log_q0_proposed, U, U', logq_rev, logq_fwd, lam, beta)
```

### 2.5 Head

Factorized 3-axis categorical (Cat3), 64 bins over ±2.5σ (≈0.078σ resolution — avoids the
`num_bins=8` spline floor from the base-quality campaign). Conditioned on `[cage kNN-transformer
context, placed-member features, slot ORDINAL/pattern features, β embedding]`. Exact normalized
categorical ⇒ exact `logq`; `sample()==transition_log_prob(·, source)` by shared per-slot feature
construction.

**The head receives only the slot's ORDINAL/pattern rank, never the site's absolute coordinates.**
Position enters solely through the §2.1 centroid anchor. If absolute site coordinates entered the
positional head, sites would influence *placement*, not merely *order*, softly reintroducing the
scaffold's fixed-anchor bias that this design exists to remove. (Cage geometry still reaches the
head via the kNN-transformer context — that is the physical cage, not the ordering scaffold.)

### 2.6 K=1 degenerate case

One slot, no ordering, no Hungarian: the single-site full-cage heat-bath. `a = centroid(cage)`,
`u = x_i − a`. This is the primary early milestone (§5, G-K1).

### 2.7 Two-blob mode (count redistribution)

A single compact blob has one centroid and one `pattern[K]`; the two-blob *redistribution* kernel
does not — its two lobes have a centroid between them and must allow **particle count to move
between lobes**. Resolution: **one union assignment over 2K sites**.

- `selection.centers = [cA, cB]` (both state-independent); `block_idx` = union of K-nearest to cA
  and K-nearest to cB (2K particles); `cage_idx` = M non-blob particles near *either* center.
- Sites = K around cA ∪ K around cB (each lobe's `pattern[K]` from its own center + cage frame),
  giving **2K ordered sites** in a single Morton order over the union.
- A **single Hungarian** over all 2K blob particles ↔ 2K sites. Because the assignment ranges over
  both lobes' sites, a particle originally near cA can be assigned to a cB-lobe site and generated
  there — **count-per-lobe is emergent, not fixed at K each**. This is exactly the redistribution
  the two-blob kernel needs; fixing K particles per lobe would forbid it.
- Anchor unchanged: `a_j = centroid(union cage ∪ placed)`. Head still receives ordinal-only slot
  features. MH bookkeeping (§2.4) is identical with 2K slots.
- The min-separation constraint between cA, cB (from `mw_kernels.draw_centers`) keeps the two site
  clusters resolvable for the union Hungarian.

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

### 5.1 Exactness gates (hard)

1. **G-exact**: the accumulated sample log-density from `sample_block(x, selection)` equals
   `transition_log_prob(target=y, source=x, selection)` ≤ 1e-4/particle across K∈{1,4,8,16}, β
   range. **Compare to the two-state transition density at (y ← x), NOT a self-score
   `transition_log_prob(y, y, ·)`** — a self-score recomputes σ_y and conflicts with the permitted
   σ_x≠σ_y mismatch in §2.4.
2. **G-cage-reversible**: re-deriving `selection` at the proposed state y reproduces the identical
   `cage_idx` AND `block_idx` (cage keyed to the state-independent center, §2.3.0). Moves failing
   this are rejected in-kernel; the gate asserts the rejection is rare on real states.
3. **G-hungarian-determinism**: repeated Hungarian evaluation on the same state returns the bit-
   identical bijection, including at exact and float-noise cost ties (§2.3 tie-break).
4. **G-involution**: forward/reverse transition-score involution under the labeled σ_x / σ_y
   construction.
5. **G-stationarity**: from equilibrium configs, the labeled-C move leaves the target invariant
   (U-distribution unchanged vs a single-site reference over the same particles), and the gate is
   **mutation-verified** (a deliberately swapped fwd/rev density must FAIL it) — the discipline from
   `mw_kernels`' two-blob rig.

### 5.2 Quality / performance gates

6. **G-mismatch-rate** (diagnostic): generated-slot vs post-generation Hungarian permutation-
   mismatch rate — quantifies the "minimal noncanonical" claim; report the full distribution. High
   rate → C's simplicity advantage erodes (does not break exactness) → weigh B-control.
7. **G-K1**: single-site acceptance at T\*_work — median β·ΔU ≤ ~1–2, acceptance ≥ ~20% (beats the
   measured +14 baseline; target KA's ~33%).
8. **G-transfer**: train N=512, test K-blob acceptance at N=64 — median β·ΔU flat across N.
9. **G-Kgt1**: joint K-blob median β·ΔU vs measured baselines (scaffold +55 @K≈4, +86 @K≈7) —
   large, viable improvement; viable-fraction up.
10. **G-energy**: RL cuts proposal energy vs MLE-only (target ~10–50×).
11. **G-pca-gap** (diagnostic): cage-frame eigenvalue/rejection gap vs acceptance/NLL — flags
    whether near-degenerate cages degrade the proposal (decides if raw-PCA or B-control is needed).

## 6. Fallback / control

**B (canonical set-order) is retained as the documented control.** If cage-PCA/site assignment
proves unstable (G-pca-gap shows degeneracy hurts, or G-mismatch-rate is high enough to negate C's
simplicity), B's canonical-lift ordering is the fallback — same anchor (§2.1), order by canonical
lift of blob positions with explicit noncanonical accounting. C is primary; B is the escape hatch.

## 7. How it slots into the existing campaign

Replaces the scaffold/toy as the `two_blob`/bridge proposal and adds the K=1 single-site kernel.
**Reuses unchanged**: suffix + conveyor kernels, the Task-6 bridge harness (`mw_smc_portfolio.py`)
outer loop, the eval ledger, `geometric_bridge_log_accept`, `draw_centers`/min-sep.

**Kernel change (real, not just β):** `mw_kernels.two_blob_move` (and a new K=1 single-site kernel)
must switch from a self-scored reverse (`block_log_prob(x_old, idx)`) to the two-state
`transition_log_prob(target=x_old, source=y_new, selection)`, and must build/pass the `selection`
object (centers, block_idx, cage_idx) so the cage/sites/order are shared and reversible. This is a
contained interface change, but it is more than threading a β argument, and it needs its own
re-review against the §5.1 exactness gates.

Campaign consequences: Task 7's "RL fine-tune" becomes "train `MWBlobProposal`"; D1 is re-run
against it as the honest acceptance number; Task 8 drives the bridge with it. The suffix kernel
keeps using v10 (unaffected).

## 8. Non-goals

- Not a permutation-equivariant flow (that's the EGNN/coupling direction, memory
  `egnn-flow-corrector-poc`; the user chose AR with explicit ordering).
- Not a joint-K move that is plain-MH-viable at the cold target (compounding is physics; K>1 plain
  viability comes only via the K=1 sweep — the joint move's home is the SMC bridge at low λ).
- No pinned-boundary / PTS machinery.
- Not a free-energy/logZ tool (the bridge harness already owns that path).

## 9. Risks / open questions

- Near-degenerate cage frames in tetrahedral cages (G-pca-gap; cage-vector Gram–Schmidt with
  documented tie-breaks is the primary defense; raw-PCA and B-control are fallbacks).
- Permutation-mismatch rate unknown until measured (G-mismatch-rate); design is exact regardless,
  but high rate erodes C's simplicity advantage → B-control.
- Two-blob union Hungarian at 2K sites: whether count redistribution actually flows (a particle
  reassigned across lobes) or the min-sep keeps lobes effectively independent — measure the
  cross-lobe reassignment rate; it is the whole point of the two-blob kernel.
- RL energy term collapsing the density despite the MLE anchor (λ_rl warmup + monitor NLL).
- Whether β_eff = λ·β or true-β is the better rung coupling — spec'd as λ·β, cheap to flip; decide
  empirically if bridge depth underperforms.
- `select` reproducibility: the reverse cage/block re-derivation (G-cage-reversible) must be a pure
  function of (centers, non-blob coords); any dependence on blob coords is a balance bug, not just
  waste.
