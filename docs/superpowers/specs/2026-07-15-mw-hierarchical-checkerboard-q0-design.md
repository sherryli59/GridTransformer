# Hierarchical 8-color cell generator for an exact global mW base

**Date:** 2026-07-15
**Status:** design / implementation gate
**Goal:** turn the successful local cavity-transformer ingredients into a normalized global base
`q0` for SMC, while preserving full support and minimizing mW energy evaluations.

## 1. Decision

Build a **count-first, 8-color cell-autoregressive base followed by optional invertible
checkerboard correctors**.

For geometric SMC, use the cell construction a second time as a **local mutation portfolio**:
exact q0-conditional redraws of final-stage cells, a direction-augmented full-cage corrector, and
ordinary boundary-crossing local MC for ergodicity. Do not use whole-configuration independence MH,
and do not treat cutoff-separated same-color cells as independently acceptable under the mW
three-body target.

This is not a two-color red/black lattice.  A cubic cell receives the 3-bit color

```text
color(i,j,k) = (i mod 2, j mod 2, k mod 2) in {0,...,7}.
```

Cells of one color are generated in parallel.  At the production density, choose a physical cell
width `h ~= 2.598`, so same-color cells have a face-to-face gap `h`, larger than the mW cutoff
`A_CUT = 1.8`.  No particle in one same-color cell can be a cutoff-neighbor of a particle in another,
so there is no direct pair edge or center-neighbor three-body edge between them. Two such particles
can still be co-neighbors of a third particle in an intervening cell, so this is not a claim of exact
physical conditional independence; the optional full-cage corrector addresses that residual coupling.

The global density is defined on an augmented state
`(x, priority, grid_shift, orientation, color_order)`. Priorities supply a random, recoverable
within-cell order and eliminate canonical-order bookkeeping. `color_order` is an independent uniform
permutation of the eight physical colors. It prevents one physical color from being permanently
assigned the weak first-stage or full-cage last-stage conditional. The target augments the Boltzmann
density with independent uniform auxiliaries; all auxiliaries are discarded at the endpoint.

Do **not** attempt either of these shortcuts:

- Do not tile the box by the July-14 model's hard balls. Balls do not partition a 3-torus, so the
  result would have overlaps or holes and would not be a normalized global density.
- Do not use a state-dependent spatial checkerboard as an ordinary Real-NVP mask. Particles can
  cross cell boundaries, changing mask membership and breaking the inverse. Spatial masks are safe
  only when each transform maps its owned particles back into the same cell, or when used as a
  causal factorization rather than an invertible mask.

## 2. Why this construction

### 2.1 Measured cell geometry

At `rho = 0.4564`, both boxes admit exactly the same physical leaf-cell width:

| N | L | grid | cells | h | mean K | observed K range |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 5.19531 | 2 x 2 x 2 | 8 | 2.59766 | 8.0 | 3--13 (2,048 frames) |
| 512 | 10.39062 | 4 x 4 x 4 | 64 | 2.59766 | 8.0 | 3--14 (800 frames) |

For N=512, only 0.0312% of measured cells had `K > 12`, none had `K > 14`, and every measured
configuration had `max K <= 16`.  The N=64 and N=512 occupancy histograms are nearly identical.
This is exactly the size-transfer regime wanted by the cavity model: local `K` stays fixed while the
number of cells grows.

The observed tail does **not** justify a hard `K <= 16` model.  An exact SMC base needs positive
density on every target configuration, including rare cells with arbitrary occupancy up to N.  The
count head therefore has full integer support; `K <= 16` is only the fast-path training regime.

### 2.2 What transfers from the July-14 cavity model

Reuse or warm-start:

- the KNN transformer/context encoder (`_fc_batched` pattern);
- monatomic handling and variable-K / remaining-demand features;
- centroid-of-context-and-prefix anchoring;
- the sharp three-axis categorical position head;
- learned pair tilt and the deterministic candidate-wise mW pair/three-body tilts;
- batched sample/score APIs and per-slot log-probability accounting.

Do not reuse unchanged:

- `ball_squash` and ball scaffold: leaf support must be a cube that participates in an exact
  partition of the torus;
- fixed progressive sites as physical anchors;
- a position-derived Morton/Hungarian relabeling at score time;
- the species head.

The selected July-15 slot-local-RL checkpoint (step 400, itself descended from the July-14 model) is
the primary **trunk/tilt warm start**. The July-14 step-12750 MLE checkpoint remains a controlled
density-calibration baseline. Neither is a verbatim global model: verbatim reuse would either leave
holes in global support or repeat the sample-order mismatch. Sampling temperature and deterministic
mW tilt coefficients remain explicit normalized-density settings and are not silently inherited.

### 2.3 Why not the July-10 global AR alone

The v4/v10 global body proved that global occupancy/frontier context matters, but it is serial in N,
hard-wired to scaffold slots, and its canonical lift has already shown assignment sensitivity.  It
remains the baseline.  The proposed model retains global occupancy information through an exact
count tree while keeping continuous generation local and parallel over separated cells.

### 2.4 Relationship to the same-day blob proposal

This global-q0 line **parallels rather than replaces**
`2026-07-15-mw-blob-resample-proposal-design.md`. The cell model supplies independent global seeds and
an exact initial/bridge density. The blob model remains a two-state local transition kernel and can be
used to mutate the cell-q0 SMC population. Promotion of either line does not gate implementation of the
other; their eventual comparison is base quality versus mutation efficiency.

## 3. Exact augmented density

### 3.1 Auxiliary variables

For each particle, attach an iid priority `r_i ~ Uniform(0,1)`.  Also sample:

- a grid shift `s ~ Uniform([0,h)^3)`;
- a cubic orientation `O` uniformly from all 48 signed coordinate permutations (the full cubic group,
  including reflections). The monatomic mW point-particle potential is achiral and reflection invariant,
  so these are exact target symmetries rather than a heuristic augmentation;
- a color order `sigma` uniformly from the `8!` permutations of the physical parity-vector colors.

For a given `(s,O)`, the torus is partitioned into `C=G^3` half-open cubes. Cell boundaries have
measure zero and use a pinned tie rule.  The count vector `K=(K_1,...,K_C)` is a deterministic
function of `(x,s,O)` and satisfies `sum K_c=N`.

Within each cell, sort the particle-position/priority pairs by priority. Priorities are continuous,
so ties have probability zero; a label-index tie rule handles constructed tests.

This random order is the key exactness device. It is independent of positions, identical in sample
and score, and avoids summing over `K!` AR orders. During generation, draw the K priorities as sorted
iid uniforms; their density on the ordered simplex is `K!`. Finally uniformly permute the N output
rows. For a labeled array, the exact joint log density contains

```text
sum_c log(K_c!) - log(N!).
```

At the target, priorities are iid uniform and `(s,O,sigma)` have the same uniform laws. Thus
marginalizing the auxiliaries recovers the ordinary labeled Boltzmann target, and discarding them at
the endpoint is exact. The `-log(8!)` color-order term cancels in mutations that hold `sigma` fixed
but remains in the explicit joint density and its sample/score tests.

### 3.2 Factorization

Let `c_1,...,c_C` use a fixed reporting/storage order, but let generation stages follow the sampled
color permutation `sigma`. Let `stage_sigma(c)` be the stage of cell c's physical color. Let
`x_{c,<t}` be the priority-ordered prefix in cell c and let `P_{<stage_sigma(c)}` contain particles
from **strictly earlier causal stages only**. Other cells of the same physical color are never
included, regardless of their Morton rank. Then

```text
q0(x,r,s,O,sigma | N)
  = q_shift(s) q_orient(O) / 8!
    q_count(K_1,...,K_C | N,s,O)
    product_c product_{t=1}^{K_c}
      q_pos(x_{c,t} | K, P_{<stage_sigma(c)}, x_{c,<t}, r_{c,<=t}, s, O, sigma)
    product_c K_c! / N!.
```

Every factor is normalized on its own support. The cube partition covers the torus, the count model
supports every composition of N, and every positional factor has full support inside its cube.
Therefore `q0` is positive almost everywhere on the target support.

The priorities need not be fed numerically to the neural net in v1; they only fix the order. A
priority-gap feature is a later ablation, not part of exactness. Randomizing `sigma` is both an exact
auxiliary construction and a training intervention: every physical color is sometimes first and
sometimes last, so the model learns the full seven-color cage as an in-distribution final-stage
conditional rather than through a separate OOD call.

### 3.3 Count tree

Use a shared octree split model, not a flat 513-way head per cell:

1. Root count is fixed to N.
2. Each occupied node splits its integer count among eight children.
3. Represent the 8-way composition as seven sequential beta-binomial/binomial-mixture factors; the
   eighth child receives the exact remainder.
4. Each factor supports every integer in `[0, remaining]`.
5. The node network sees parent count/density, level, periodic node position, neighboring already
   decoded counts, and coarser ancestor counts.

Weights are shared across nodes and levels. N=64 exercises depth 1; N=512 exercises depth 2. Train
both sizes because the data already exist; treat N=64 -> N=512 as a measured transfer ablation, not
the production assumption.

### 3.4 Full-support cube head with a sharp centroid anchor

For active cell c and slot t, compute an anchor `a in (0,1)^3` from the centroid of:

- particles from strictly earlier causal stages within the physical context radius;
- the cell's already generated prefix;
- a small fixed cell-center pseudo-weight for the empty-prefix fallback.

The anchor depends only on the causal prefix and known context. The Cat3 head emits
`u in (-1,1)^3`, spanning the entire cube. Map each axis by the anchor-centered piecewise-linear
bijection

```text
p(u;a) = a (u+1),             u < 0
         a + (1-a) u,         u >= 0,
x = cell_low + h p.
```

The forward chart Jacobian is `log|dx/du| = log h + log a` on the negative branch and
`log h + log(1-a)` on the positive branch, per axis. Therefore the scored density is explicitly
`log q_x = log q_u - log|dx/du|`; the Jacobian terms are **subtracted**, not added. This preserves the sharp
centroid anchor while giving every point in the cell positive support. It also lets the existing
piecewise-uniform Cat3 head remain sharp; no ball scaffold or fixed-site positional anchor remains.

Clip the deterministic anchor only to `[eps,1-eps]` for numerical stability, with the identical
operation in sample and score. The candidate-wise mW tilt is evaluated on the physical x candidates
for all three axes.

### 3.5 Context and parallel schedule

All cell counts are known before any position is generated. Each positional conditional sees:

- the local 3x3x3 leaf-count stencil and octree ancestors (future-capacity information);
- particle KNN tokens from strictly earlier causal stages plus the current cell's priority prefix;
- cell index/level, K, absolute slot t, and remaining `K-t`;
- centroid anchor and the fixed cube bounds;
- optional previously generated cells summarized by a global count-field transformer.

Cells of one 3-bit color are evaluated as one neural batch at their sampled causal stage. They are
conditionally independent **by q0 model definition**. At the chosen h they are outside one another's
direct mW neighbor cutoff, making this a sensible approximation rather than merely a computational
trick. They may still couple in the physical target as two neighbors of a central particle in a
frozen color through the mW three-body term. Neural evaluation can therefore be parallel, but exact
MH decisions for simultaneous same-color mutations do not generally factorize.

This mask is load-bearing and identical in sampling and scoring. At score time, the centroid anchor,
KNN tokens, learned pair tilt, and candidate-wise mW pair/three-body tilt must all exclude particles in
other cells of the current color and all later stages under `sigma`. A Morton-serial scorer that
exposes earlier same-color cells defines a different density and is forbidden. The serial exactness
reference must deliberately apply the same color mask as the batched path.

A scalar chessboard color `(i+j+k) mod 2` is explicitly rejected: diagonally adjacent equal-color
cells touch and can interact. The schedule has eight colors.

## 4. Optional full-cage checkerboard corrector

After the count-first AR base has generated all positions, optionally compose L invertible correction
sweeps. For each of the eight colors:

- active particles are those currently owned by cells of that color;
- transforms map each active particle from its cube back into the **same cube**, so cell membership
  and counts cannot change;
- conditioner sees all inactive particles plus the passive half of the active cell;
- alternate fixed priority-slot parity within each cell;
- use bounded affine/spline coupling in unconstrained cube coordinates and accumulate exact logdet;
- inverse applies colors and layers in reverse order.

This is the legitimate checkerboard-flow use. It follows the repo's
`ka3d_block_corrector.py` precedent and Real-NVP coupling logic, but uses 8 spatial colors and
occupancy-preserving cube maps. It supplies full-cage information that a one-pass causal generator
cannot have.

Do not implement this until the base passes joint sample/score and support gates. A corrector cannot
repair a non-normalized base.

### 4.1 Corrector as an exact directional MCMC proposal

The full-cage corrector is also the primary learned **position-only** mutation candidate. A
deterministic forward map by itself is not a valid reversible MH proposal merely because it is
invertible. Augment the proposal with a uniform direction `d in {-1,+1}` and propose

```text
(y,-d) = (T_color^d(x), -d).
log alpha = min(0,
  log pi_lambda(y,a) - log pi_lambda(x,a)
  + log |det J_{T_color^d}(x)|).
```

The direction probabilities cancel. Counts and priorities are held fixed, so their factorial and
auxiliary terms cancel as well. The inverse must reconstruct active cells and passive priority-slot
parity from the current state, not cached forward metadata. A forward-only deterministic proposal is
valid only if the map is explicitly an involution; that is not assumed here.

This proposal is full-cage and exact, but it is not automatically invariant under `q0`, so its
acceptance need not approach one as `lambda -> 0`. That stronger property belongs to the exact
final-stage `q0` redraw below. The corrector remains occupancy preserving and is therefore nonergodic
without an interleaved boundary-crossing or count-changing move.

## 5. SMC use

Compare two exact likelihood-aware deployments plus the required seed-only incumbent protocols.

### A. One-q0-correction thermal SMC (first production target)

1. Draw independent `(x,r,s,O,sigma) ~ q0`.
2. Evaluate mW energy once per walker.
3. Weight to a hot Boltzmann distribution with
   `log w = -beta_hot U(x) - log q0(x,r,s,O,sigma) + log p_aux(r,s,O,sigma)`.
4. Resample; discard auxiliaries.
5. Continue the existing energy-only thermal ladder.

This is the simplest exact route and evaluates q0 only once. It directly optimizes the user's metric:
independent equilibrium endpoints per mW energy evaluation.

Historical mW results make this a high bar, not the incumbent. Likelihood-weighted deployments were
about 3.75--4.4x more expensive than seed-only and the measured one-shot likelihood mismatch was about
1 nat/particle, while unweighted flow seeds produced roughly 3x/2.4x seed-amortization gains. For weakly
correlated extensive errors, `Var(log w)` grows approximately linearly in N. At N=512, a non-collapsed
one-shot correction therefore plausibly requires residual hot-target mismatch of only a few `1e-3`
nats/particle, roughly two orders of magnitude below the historical result. Treat this as an empirical
overlap target, not a theorem; measure the ESS-versus-`beta_hot` curve. Lowering `beta_hot` improves the
correction but continuously approaches ordinary unweighted seeding.

### B. Geometric q0-to-target bridge (diagnostic / possible win)

Retain the auxiliaries and bridge on the joint state:

```text
pi_lambda(x,a) proportional
  q0(x,a)^(1-lambda) [exp(-beta U(x)) p_aux(a)]^lambda.
```

This may need fewer early energy relaxations if q0 is strong, but every mutation must maintain the
joint q0 score. Compare it only after A works. Endpoint x is exact after dropping a.

Primary outcome metrics are total counted mW energy units to matched endpoint distributions,
unique-parent count, pairwise overlap between walkers, U/N histogram, and g(r). One-shot NLL is only
a diagnostic.

### C. Unweighted-seed protocols (incumbent and required baseline)

Run both the new cell-q0 samples and the existing eRSI samples as unweighted initial states through the
same thermal relaxation ladder, with no likelihood correction. This finite-relaxation protocol is not
an importance-corrected exact deployment, but it is the campaign's current production champion and must
be beaten on matched endpoint distributions. A likelihood-weighted arm cannot be promoted merely for
beating another likelihood-weighted arm while losing to eRSI seed-only.

### D. Local mutation portfolio for the geometric bridge

Do not use a whole-configuration independence-MH proposal. Use local cell moves with a scheduled
handoff from learned moves early in the bridge to ordinary local moves near the target.

#### D1. Exact final-stage q0 cell redraw

For the uncorrected causal base, fix `(K,s,O,sigma)` and select a cell whose physical color is the
last causal stage `sigma(7)`. Resample that cell's priority-ordered positions and priorities from its
ordinary q0 conditional given the seven frozen colors. Because there is no later positional factor
and q0 excludes other same-color cells, this is an exact q0 Gibbs proposal for that cell. Its bridge
acceptance simplifies to

```text
log alpha = min(0, -lambda * beta * [U(y)-U(x)]).
```

This is the one learned move guaranteed to approach unit acceptance as `lambda -> 0`. Random
`sigma` means every physical color is the final stage for a subset of walkers, so this is not tied to
one spatial sublattice. Process active cells **sequentially**, refreshing their full physical cage and
energy delta after each decision; batch the same cell operation across walkers. Independent
simultaneous per-cell accept/reject decisions are forbidden unless an interaction-factor audit proves
that the old and proposed cells share no mW factor. Cutoff separation removes direct pair edges but
does not remove a three-body factor centered on a frozen particle with neighbors in two active cells.

If an invertible corrector is composed into the reported q0, the simple redraw is no longer its exact
conditional. Either run D1 against the uncorrected causal-base density or include the complete
corrected-q0 forward/reverse ratio. Never silently reuse the simplified cancellation.

#### D2. Directional full-cage corrector

Apply the direction-augmented proposal in Section 4.1 to one cell or one color sweep. Prefer
sequential per-cell acceptance to keep the energy change O(K); a whole-color joint accept is exact but
its extensive O(N/8) log ratio is expected to lower acceptance. The conditioner sees all frozen
colors, avoiding the half-cage OOD call of a fixed-order causal proposal. This is the primary
full-cage position engine to evaluate, not a claim that its learned map is q0-Gibbs.

#### D3. Full-cage stochastic block conditional, if needed

If D2 has low acceptance, warm-start a separate block-conditional model from the q0 transformer while
protecting the base checkpoint. Train it on arbitrary active cells with all other colors visible and
expose a genuine two-state API:

```text
sample_cell_transition(source_state, cell) -> target_cell, log q(target | source)
cell_transition_log_prob(target_cell, source_state, cell) -> log q(target | source)
```

Use the complete forward/reverse proposal ratio in MH. This transition may be exact without being a
factor of the global q0; exactness does not excuse OOD conditioning or missing reverse bookkeeping.

#### D4. Occupancy and ergodicity moves

D1 and D2 preserve the cell-count vector. Interleave ordinary local displacement MC so particles can
cross cell faces. An optional learned count move redistributes a fixed combined count between a
state-independently chosen pair of cells, samples the normalized split over `k=0..M`, and generates
positions/priorities with full-cage conditioning. Its reverse score must include pair selection,
split probability, labeled assignment, active-cell factorial changes, and both position densities.
Do not independently resample every active-color count under fixed total N, and do not make a
whole-color count proposal whose acceptance scales with N/8.

Schedule D1/D2/D3 most heavily at low and intermediate `lambda`; increase local displacement and
count/boundary-crossing moves toward `lambda=1`. Measure acceptance and energy cost separately by
`lambda`, K, causal stage, and move type rather than reporting one aggregate kernel number.

## 6. Exactness and quality gates

Run gates in this order; do not launch long training before gates 1--5 pass.

1. **Partition:** cell assignment is a total, deterministic periodic partition for random shifts,
   including boundary-near adversarial points.
2. **Count normalization:** enumerate all child compositions for small parent counts and verify total
   probability 1; sampled child counts always sum to the parent.
3. **Cube normalization:** numerical integration of perturbed Cat3 conditionals plus analytic
   anchor-map Jacobian equals 1; sample/score agrees at bin and anchor branch boundaries.
4. **Priority/order invariance:** jointly permuting rows of `(x,r)` leaves log q unchanged; changing
   positions without changing r never changes the order except through cell crossing, which is handled
   by regrouping then sorting r. Enumerate all color orders in a small model and verify the uniform
   `-log(8!)` term and strict earlier-stage mask.
5. **Joint sample/score:** independently accumulated sample logq equals
   `log_prob(x,r,s,O,sigma)` to `1e-4` per particle with perturbed weights, across N=8/64, multiple
   color permutations, and extreme count vectors.
6. **Full support:** finite logq for all-N-in-one-cell, empty cells, points arbitrarily near cube faces,
   and N=64/N=512 configurations. No `K<=16` rejection exists.
7. **Corrector and mutation reversibility:** forward/inverse coordinate and logdet round trip; active
   particles remain in their original cells for every color. Test the direction-augmented corrector
   proposal on a toy target and include a negative test showing that a forward-only non-involutive map
   violates detailed balance. Verify D1's q0 cancellation numerically and verify that independent
   simultaneous same-color MH decisions fail on a constructed shared three-body factor.
8. **MLE:** report count NLL and position NLL by K=0..16 plus overflow, rollout clashes by rank, and
   N=64/N=512 separately.
9. **N=512 color/stage-blindness gate:** before long N=512 SMC, run held-out teacher-forced diagnostics
   with G=4. Reweight every physical color and causal stage to a common K distribution; require
   max-minus-min position NLL <=0.20 nat/particle and no color/stage one-slot rollout clash rate >1.5x
   the corresponding median. Report insertion cost and clash by physical color, causal stage, and
   within-cell rank. Failure means the randomized causal schedule or its context mask must be fixed
   before an expensive bridge.
10. **One-shot quality:** generated U/N and centered/full g(r), core-clash fraction, and energy-vs-logq
   against held-out data. Scale plots from the data distribution. At N=64 and N=512, sweep `beta_hot`
   and save ESS/B, log-weight variance/N, unique-parent expectation, and the inferred mismatch scale.
11. **SMC efficiency:** matched endpoint U and g(r), weighted and resampled diagnostics, unique ancestry,
    cross-walker overlap, and total energy-evaluation ledger versus eRSI, v10, and local-MC baselines.

Initial go/no-go thresholds:

- exactness gates 1--7: mandatory;
- no generated core-clash fraction regression versus the selected July-14 scaffold on matched K;
- hot-start importance ESS at least 10% for N=64 before likelihood-weighted N=512, but this does not
  replace the N=512 G=4 color-blindness gate above because G=2 has only one cell per color;
- production GO only if endpoint structure matches the reference and energy evaluations fall by at
  least 2x at matched effective number of independent walkers **and beats the eRSI seed-only incumbent**.

## 7. Baselines and ablations

Use the same data split, optimizer budget, and diagnostics for:

1. eRSI seed-only (production incumbent) and eRSI likelihood-corrected;
2. v10 global AR (existing exact global baseline);
3. count tree + uniform iid positions (isolates count value);
4. count tree + 8-color position AR;
5. item 4 + global count-field tokens;
6. item 5 + checkerboard corrector;
7. cell-q0 seed-only versus likelihood-corrected;
8. 8-color versus deliberately scalar red/black (diagnostic only; expected worse);
9. priority random order versus position-derived Morton order (the latter is not promoted unless its
   full permutation accounting is implemented);
10. fixed versus randomized causal color order;
11. D1 final-stage q0 redraw, D2 directional corrector, D3 full-cage stochastic conditional, local MC,
    and their scheduled mixtures, all with acceptance/cost versus lambda and K.

## 8. Primary references and local precedents

- Dinh et al., [Density Estimation using Real NVP](https://arxiv.org/abs/1605.08803): exact coupling
  layers and checkerboard/multiscale masks.
- Papamakarios et al., [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/abs/1912.02762):
  exact change-of-variables and autoregressive/coupling tradeoffs.
- Arbel et al., [Annealed Flow Transport Monte Carlo](https://arxiv.org/abs/2102.07501): learned maps
  inside annealed SMC.
- Sun et al., [PointGrow](https://arxiv.org/abs/1810.05591): autoregressive continuous point-cloud
  generation.
- Huang et al., [OctSqueeze](https://arxiv.org/abs/2005.07178) and Kaya & Tabus,
  [neural octree coding](https://arxiv.org/abs/2106.06482): normalized hierarchical occupancy models
  with causal spatial context.
- Köhler et al., [Equivariant Flows](https://arxiv.org/abs/2006.02425): exact-likelihood flows that
  preserve physical symmetries.

Local code precedents:

- `liquid_coupling_flow/ka3d_ebm_batched.py` and `mw/mw_scaffold.py`: selected local transformer,
  Cat3 head, tilts, batched sample/score.
- `liquid_coupling_flow/ka3d_block_corrector.py`: conditional invertible full-cage corrector.
- `liquid_coupling_flow/coupling.py`: triangular coupling exactness.
- `liquid_coupling_flow/mw/mw_generator_v4.py` and `mw_generator_v10.py`: exact global AR baseline and
  evidence for global occupancy context.
- `liquid_coupling_flow/mw/mw_thermal_smc.py`: one-q0-correction thermal SMC.
- `liquid_coupling_flow/mw/mw_smc_portfolio.py`: geometric q0 bridge and energy ledger.

## 9. Main risks

- **Count model is good but geometry is not:** count NLL can improve without reducing energy. Quality
  gates are rollout energy/g(r), not count likelihood alone.
- **First colors lack a full particle cage:** count-field tokens expose future capacity; the optional
  invertible corrector supplies full coordinates after the base exists.
- **Cat3 discontinuities:** exact sample/score is mandatory near bin and anchor-map boundaries. If
  float32 branch flips exceed tolerance, score the chart/Jacobian in float64 rather than weakening
  the gate.
- **Auxiliary-variable mistakes:** never report the path log-density as a marginal physical q(x).
  SMC runs on the joint augmented state until the one-q0 correction/resampling or the bridge endpoint.
- **False same-color factorization:** cutoff-separated cells can share an mW three-body factor through
  a frozen center. Parallel neural proposal evaluation is allowed; independent parallel MH decisions
  require a proved disjoint factor graph. Sequential per-cell acceptance is the default.
- **Deterministic-MH mistakes:** invertibility alone does not make a one-way map reversible. Carry the
  direction auxiliary and exact Jacobian, or use a separately proved involution.
- **Crystal bias:** match full g(r), tetrahedral/angular observables, and energy histograms; a low-energy
  crystalline mode is not a successful liquid generator.
