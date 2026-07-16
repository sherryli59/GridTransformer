# mW hierarchical 8-color global q0 implementation plan

**Spec:** `docs/superpowers/specs/2026-07-15-mw-hierarchical-checkerboard-q0-design.md`
**Goal:** implement and gate an exact global augmented density using an octree count model, randomized
8-color cube infilling, and a local mutation portfolio led by exact final-stage redraws and an
occupancy-preserving full-cage corrector.

## Global constraints

- Phase A is geometry and density exactness only. No long training until joint sample/score and full
  support pass.
- The production density is joint in `(x, priority, grid_shift, orientation, color_order)`. Never
  substitute a sampled path score for a marginalized `q(x)`.
- Eight parity-vector colors are required. A scalar red/black coloring is not the implementation.
- Sample a uniform permutation of the eight physical colors for each augmented state. Sampling and
  scoring use the identical strict earlier-stage mask. No other same-color or later-stage cell may
  enter the anchor, KNN context, learned tilt, or deterministic mW tilt in either path.
- Count support is `0..remaining`, never hard-capped at 16.
- Every cell-position map has exact cube support and stays inside its cell.
- Preserve the user's existing dirty worktree and do not overwrite July-14 checkpoints.

## Task 0: freeze the protocol and geometry probe

**Create:**

- `liquid_coupling_flow/mw/mw_cell_geom.py`
- `liquid_coupling_flow/tests/test_mw_cell_geom.py`
- `reports/logs-2026-07-15/mw_cell_geometry_probe.py`

Implement periodic shifted/oriented cell assignment, cell bounds, physical 8-color labels, sampled
causal color permutations, fixed reporting/Morton order, and priority grouping. Persist occupancy
histograms for N=64 and N=512, including random grid shifts.
Pin `h=L/G`, `G=2` for N=64 and `G=4` for N=512 in v1.

Tests:

- every point maps to exactly one half-open cell;
- translation of `(x,grid_shift)` leaves assignments covariant;
- same-color cube gap is `h` and exceeds `A_CUT`;
- sorting by priority is invariant to joint row permutation;
- every physical color appears at every causal stage under an enumeration of the 8! orders, and the
  earlier-stage mask is independent of Morton reporting order;
- exact boundary/tie behavior is deterministic.

Gate: reproduce the measured mean K=8 and verify shifted-grid tails before choosing cache widths.

## Task 1: exact hierarchical count distribution

**Create:**

- `liquid_coupling_flow/mw/mw_cell_count.py`
- `liquid_coupling_flow/tests/test_mw_cell_count.py`

Implement a shared octree node encoder and an 8-child integer split. Use seven sequential
full-support integer conditionals and give the remainder to child eight. Expose:

```python
sample_counts(N, G, shift, orientation, gen) -> counts, logq
count_log_prob(counts, N, G, shift, orientation) -> logq
```

Tests enumerate every composition for parent counts 0..6 and verify summed probability 1, then run
perturbed-weight sample/score tests for G=2/4. Include adversarial all-N-in-one-leaf counts.

Gate: exactness before fitting; then train count-only and report NLL per tree level and leaf K.

## Task 2: anchored cube chart and position head

**Create:**

- `liquid_coupling_flow/mw/mw_cell_chart.py`
- `liquid_coupling_flow/tests/test_mw_cell_chart.py`

Implement the anchor-centered piecewise-linear bijection from Cat3 `u in (-1,1)^3` to the full leaf
cube, with analytic forward/inverse Jacobians. Pin the density sign as
`log q_x = log q_u - log|dx/du|`. Score chart math in float64 if needed.

Tests:

- forward/inverse round trip near u=0, cube faces, and anchor clip points;
- analytic logdet matches autograd Jacobian;
- numerical integral of perturbed Cat3 plus chart equals 1;
- sample score matches at every branch.

Gate: no finite cube point may receive `-inf` from the chart/head.

## Task 3: 8-color cell AR base

**Create:**

- `liquid_coupling_flow/mw/mw_cell_q0.py`
- `liquid_coupling_flow/mw/mw_cell_data.py`
- `liquid_coupling_flow/tests/test_mw_cell_q0.py`

Build the cell generator from the July-14 KNN transformer/tilt components. Remove species, replace
ball coordinates with the cube chart, and provide global count tokens plus local K/slot/remaining
features. Sample `color_order` uniformly and batch all cells at one causal stage. APIs:

```python
sample(B, N, L, gen) -> AugmentedState, joint_logq
log_prob(AugmentedState, L) -> joint_logq
```

`AugmentedState` contains labeled `x`, iid priorities, shift, orientation, the physical-color
permutation, and enough metadata for debugging; `log_prob` must recompute counts/groups and causal
stages from state rather than trust cached metadata.

Include `sum lgamma(K_c+1) - lgamma(N+1) - lgamma(9)` in the labeled joint density. Random row
permutation is part of sampling.

Tests:

- perturbed-weight joint sample/score at N=8 and N=64;
- row-permutation invariance for `(x,priority)`;
- uniform color-order mass and sample/score agreement under multiple nontrivial permutations;
- finite density on empty cells, K>16, all-N-in-one-cell, and cell-face points;
- same-stage batch result equals a serial reference that applies the **same strict earlier-stage mask**;
- a mutation test makes a naive Morton-serial scorer expose earlier same-color cells and verifies that
  it disagrees, proving the real exactness test is sensitive to this bug class;
- per-slot sum equals block/cell/global logq.

Gate: maximum sample/score error <=1e-4 per particle and all support tests pass.

## Task 4: cached multiscale training

**Create:**

- `liquid_coupling_flow/mw/mw_cell_train.py`
- `liquid_coupling_flow/mw/launch_mw_cell_q0.sh`

Cache raw wrapped configurations and immutable observables only. Resample the grid shift and sample
one orientation uniformly from all 48 O_h signed coordinate permutations per configuration per epoch,
then recompute the cheap cell assignment on the fly; do not bake one shift into the cache. Reflections are exact symmetries of the achiral
point-particle mW target. Draw fresh priorities per epoch as well.
Draw a fresh uniform causal color permutation per configuration per epoch; do not train one physical
color permanently at one causal stage.
Warm-start compatible trunk/tilt keys from the selected July-15 slot-local-RL scaffold checkpoint
(`mw_scaffold_N512_v2_selected.pt`, step 400). Keep the July-14 step-12750 MLE checkpoint as the
density-calibration A/B; never overwrite either source artifact. Initialize cell/count adapters
separately. Treat sampling temperature and deterministic mW tilt strengths as explicit scored-density
settings, not hidden checkpoint behavior. Log:

- count NLL per level;
- position NLL/K for every K=0..16 and overflow;
- N=64 and N=512 separately;
- clash and incremental mW insertion cost by placement rank;
- throughput in configs/cells/particles per second.

Use balanced-K accumulation inside each optimizer step, but weight the reported physical validation
mean by the true data K distribution. Save last and best-by-rollout checkpoints without touching any
July-14 artifact. The controller launches the training as a harness-tracked background job, records the
exact PID, and redirects both stdout and stderr to the dated log. Checkpoint and flush validation state
incrementally so a crash is visible and resumable.

## Task 5: one-shot and initial-importance diagnostics

**Create:**

- `liquid_coupling_flow/mw/mw_cell_diagnostics.py`
- outputs under `reports/logs-2026-07-15/mw_cell_q0_*`

Compare against held-out data, v10, eRSI, and the selected July-14 local scaffold:

- U/N histograms with axes/ranges derived from data;
- full and generated-particle-centered g(r);
- tetrahedral/angular distribution and core-clash fraction;
- energy versus joint logq;
- hot-target log-weight/ESS and unique projected parents over a registered `beta_hot` sweep;
- `Var(log w)/N`, ESS/B, and effective unique-parent expectation versus `beta_hot`, with N=64 and
  N=512 shown separately;
- distributions by cell K, physical color, causal stage, and within-cell rank.

Add a named **G4 color/stage-blindness gate** on held-out N=512 teacher-forced states before any long
N=512 SMC. Reweight physical colors and causal stages to a common K distribution; require
max-minus-min position NLL <=0.20 nat/particle and every color/stage's one-slot rollout clash rate
<=1.5x the corresponding median. Save NLL, insertion cost, and clash by physical color, causal stage,
and within-cell rank. N=64 cannot test physical-color effects because G=2 has one cell per color.

Gate: no likelihood-weighted long SMC if N=64 hot-start ESS <10%, the G4 gate fails, or generated
structure shows a crystalline split peak. Retain unweighted seed evaluation even when ESS fails.

## Task 6: full-cage corrector and local mutation exactness

**Create:**

- `liquid_coupling_flow/mw/mw_cell_corrector.py`
- `liquid_coupling_flow/mw/mw_cell_mutation.py`
- `liquid_coupling_flow/tests/test_mw_cell_corrector.py`
- `liquid_coupling_flow/tests/test_mw_cell_mutation.py`

Start only after the base density gates pass, but evaluate this as the primary full-cage position
engine rather than waiting for a failed global SMC run. Adapt the full-cage coupling precedent to one
active cell at a time. Transform in unconstrained cube coordinates; keep particles inside their source
cell; condition on all frozen cells and passive priority slots.

Tests cover forward/inverse, logdet signs, cell-membership preservation, color-order reversal, and
composition sample/score. The inverse recomputes active cells and passive priority-slot parity from
`(x, priority, grid_shift, orientation, color_order)` alone; it must not trust cached forward metadata.
Train by MLE with the AR base frozen first.

Implement three explicit mutation APIs:

```python
redraw_final_stage_cell(state, cell, gen) -> proposed_state, logq_terms
apply_corrector_direction(state, cell, direction) -> proposed_state, logabsdet
local_boundary_move(state, particle, gen) -> proposed_state, proposal_terms
```

The final-stage redraw is available only for cells at `color_order[-1]` and, for the uncorrected AR
base, must numerically satisfy the simplified bridge ratio
`-lambda * beta * delta_U`. The corrector samples a uniform `direction in {-1,+1}`, flips it in the
proposed augmented state, and uses the exact directional Jacobian. Add a negative test proving that a
forward-only non-involutive deterministic map fails detailed balance.

Use sequential per-cell acceptance and refresh the cage/energy delta after each decision. Batch across
walkers, not across interacting cells within one walker. Construct an mW test where particles from two
same-color cells share a frozen three-body center and verify that independent simultaneous cell
acceptances give the wrong transition. A whole-color joint acceptance remains an exact diagnostic,
not the production default.

If the corrector has poor acceptance, add a separately checkpointed full-cage stochastic transition
warm-started from q0, with exact two-state `sample_cell_transition` and
`cell_transition_log_prob(target, source, cell)` APIs. Protect the q0 checkpoint. Add optional
two-cell count redistribution only after position kernels work; include fixed-N split, selection,
labeled assignment, active factorial, priority, and forward/reverse position terms. Ordinary local
boundary-crossing displacement remains the required ergodicity baseline.

## Task 7: SMC integration

**Modify:**

- `liquid_coupling_flow/mw/mw_thermal_smc.py`
- optionally `liquid_coupling_flow/mw/mw_smc_portfolio.py`

Add an augmented-q0 adapter. First run one-q0-correction thermal SMC, retaining auxiliaries through
the initial weight and discarding them only after the hot-target resample. Add the geometric joint
bridge as the second arm, with the Task-6 local portfolio: final-stage q0 redraw and directional
full-cage corrector early/intermediate in lambda, then increasing local boundary-crossing moves toward
lambda=1. Do not add a global whole-configuration independence-MH mutation.

Tests:

- toy target where augmented weights are analytically checkable;
- auxiliary constants and factorial terms have the correct sign;
- final-stage q0 redraw cancels to the energy-only bridge ratio, while count-changing moves retain the
  active factorial and complete forward/reverse terms;
- corrector direction/Jacobian satisfies detailed balance on a toy target;
- shared mW three-body factors force sequential or joint, not independent per-cell, acceptance;
- discarding auxiliaries after exact resampling preserves the x marginal;
- energy-evaluation ledger counts only actual mW evaluations;
- checkpoint/resume reproduces the RNG and ancestry state.

## Task 8: controlled evaluation and stop conditions

Run N=64 before N=512, except for the cheap held-out N=512 G4 teacher-forced gate. For each arm, use
matched initial RNG seeds and report full distributions:

1. existing eRSI unweighted seed-only thermal relaxation (incumbent);
2. cell q0 unweighted seed-only thermal relaxation;
3. eRSI + one-shot likelihood correction + thermal SMC;
4. v10 + one-shot likelihood correction + thermal SMC;
5. cell q0 + one-shot likelihood correction + thermal SMC;
6. cell q0 + corrector + one-shot likelihood correction, if built;
7. cell q0 geometric bridge + local MC only;
8. item 7 + exact final-stage q0 cell redraw;
9. item 8 + directional full-cage corrector;
10. item 9 + full-cage stochastic transition or two-cell count redistribution only if their isolated
    acceptance/cost gates justify them.

For arms 7--10, report accepted moves and counted mW evaluations by lambda, K, causal stage, and move
type. Verify the learned early-bridge schedule actually hands off to ergodic local/count-crossing moves
near lambda=1. Never collapse these into a single average acceptance number.

Primary score is counted mW energy units per effective independent equilibrium endpoint. Require
matched U/N, g(r), angular structure, and no crystal signature. GO to N=512 only after exactness is
green and N=64 saves at least 2x energy units or materially increases unique endpoint ancestry at the
same cost. A production GO must beat the eRSI seed-only incumbent; winning only among likelihood-
weighted arms is insufficient.

Save endpoint configurations, weights/ancestry, the energy ledger, and per-walker observable time
series after every rung/measurement unit under `liquid_coupling_flow/mw/artifacts/`. Reports and plots
are derived products, not substitutes for the raw endpoint and trajectory state.

If the cell base fails, stop by diagnosis:

- count wrong -> improve count tree/global count context;
- early causal stages wrong, later stages good -> build corrector;
- all colors clash despite correct counts -> cube positional model/anchor is wrong;
- one-shot good but ESS poor -> logq-energy ranking/calibration is wrong;
- final-stage redraw good but corrector poor -> keep q0-Gibbs and retrain/drop the transport map;
- learned position moves good but counts freeze -> increase boundary crossing or add exact two-cell
  count redistribution, not a whole-color count redraw;
- SMC matches structure but duplicates walkers -> mutation/annealing schedule, not q0 structure.
