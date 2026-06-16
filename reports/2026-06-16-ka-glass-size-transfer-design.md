# Design spec: size-transferable local Boltzmann sampling of a 2D Kob–Andersen glass

Status: DRAFT for review. No implementation until approved.
Context: follows `reports/2026-06-16-liquid-coupling-flow-consolidation.md`. The monatomic-LJ
line closed negative (LJ is annealable → no niche for a learned proposal). This spec defines the
follow-on: a *frustrated* system, where annealing genuinely fails, with transfer grounded in a real
invariance (locality → size), not temperature (see memory `transfer-needs-an-invariance`).

## 1. Goal & value proposition

**Goal.** Learn a genuinely-local, exact-likelihood generative model from a *small*, equilibrated
Kob–Andersen (KA) glass sample, and reproduce the *large*-system Boltzmann distribution — where
direct equilibration is infeasible because relaxation times diverge.

**Value proposition (chosen): amortize across system size.** Pay expensive equilibration (swap
Monte Carlo) *once* at a small size; the local model then transfers to larger sizes for free. This
is principled because a homogeneous supercooled liquid is a tiling of *local* environments, so a
correct local conditional is size-invariant by construction. Temperature transfer is explicitly
**out of scope** — it has no such invariance and the distribution overlap collapses on cooling.

## 2. Success criterion (first milestone)

**(b) Correctness of size-transfer.** A model trained at `N_train` reproduces the *large*-system
**local observables** within statistical error at the largest size where a reference is affordable:
partial radial distributions `g_AA, g_AB, g_BB(r)`, static structure factor `S(q)` (incl. partials),
per-particle potential energy distribution, and a local order metric (bond-orientational `ψ6`/`ψ4`
and/or Voronoi-cell statistics). Speedup/amortization (milestone a) and the transfer↔ξ map
(milestone c) are explicitly deferred until (b) holds — you cannot claim a speedup over a wrong answer.

**Honesty bound.** Transfer is principled only while the static correlation length `ξ ≤ L_train`.
`ξ` grows on cooling; the spec's claim is "transfer holds for `ξ ≤ L_train`, and breaks predictably
beyond" — locating that breakdown is itself a clean result, not a failure.

## 3. Model system

2D **Kob–Andersen** binary LJ. Standard interaction matrix (ε_AA=1, ε_AB=1.5, ε_BB=0.5;
σ_AA=1.0, σ_AB=0.8, σ_BB=0.88; cutoff 2.5σ_αβ, shifted). Composition: **65:35 A:B** (a documented
non-crystallizing 2D ratio; 80:20 crystallizes in 2D). **Open item O1: verify non-crystallization**
at the target T over long swap-MC before committing the ratio.

- **Temperature.** Train in the supercooled regime where swap MC still equilibrates feasibly
  (target `T* ≈ 0.5`, to be set by O2 below). Above the experimentally-relevant slowdown, but
  structured enough that *unaided* annealing/MD cannot equilibrate the large system.
- **Sizes.** `N_train` ≈ 256 (large enough to contain local glass structure and `ξ`); validate
  transfer at `N ∈ {256, 512, 1024, 2048}` with swap-MC references at each; "use" regime (no
  reference) is larger. **Open item O2:** confirm swap MC equilibrates 256–2048 at the chosen T in
  a tractable budget; set T accordingly.

## 4. Architecture — two contenders, benchmarked head-to-head

Shared requirement: exact likelihood, **cheap one-pass `log q`** (the SMC corrector evaluates it on
every proposed move), built from **bounded-neighbourhood / constant-neighbour-count** conditioning so
it is size-extensive. The only thing that varies between the two arms is the generative model.

- **Arm 1 — transformer-AR.** Curve-ordered autoregressive placement; conditioner is a transformer
  with **distance-cutoff geometric attention** (each token attends only to placed particles within
  `r_c`); continuous position head (circular spline / mixture) → exact triangular `log q` in one
  masked forward pass. Carries species labels. Heritage of the repo's GridTransformer.
- **Arm 2 — spatial coupling flow.** Checkerboard/interleaved spatial partition of particles;
  alternately transform set-A positions conditioned on set-B's **real** local neighbourhood
  (cutoff GNN), then swap; several layers. Exactly invertible, cheap triangular `log q`, genuinely
  local. (The earlier particle-coupling stall used a *noise* auxiliary; conditioning on real
  neighbours is the fix.)

**Central hypothesis the benchmark tests:** the coupling flow is **drift-free** (parallel transform),
while the transformer-AR accumulates **exposure-bias / AR drift** over the long generation sequence —
the documented cause of the original size-transfer failure (`arc-cell-discontinuity`). Prediction:
the AR's *raw-proposal* quality degrades faster with N than the coupling flow's. If the AR wins
anyway, the drift story was incomplete — also worth knowing.

Equivariant continuous flow (flow-matching / EGNN CNF): **deferred to future work** — its `log q` is
an ODE solve, too costly for this SMC-heavy comparison.

## 5. Corrector — swap-augmented SMC

The local flow proposes **structure**; the corrector relaxes **identities**. For a KA glass the SMC
kernel must include **swap moves** (propose A↔B identity swaps) alongside single-particle
displacement moves — swap MC is the reason KA is equilibrable at all. Bridge as before
`(1−β) log q + β (−U/kT)`; reuse `smc.py` (kernel extended with swaps). The flow's exact `log q`
makes the importance weights exact.

## 6. Validation strategy (resolves the circularity)

We never need a reference at the size we ultimately *use* the model. Validate over the
reference-affordable range and report the **cost curve**:
1. Generate swap-MC equilibrated references at `N ∈ {256…2048}` (also the `N_train` training set).
2. Train each arm at `N_train`; generate + SMC-correct at each larger `N`.
3. Compare milestone-(b) observables to the reference at each `N` (correctness), and record cost
   (flow+SMC sweeps/wall to converge) vs the swap-MC baseline cost (trend → amortization).
Extrapolation to un-referenceable `N` is then *earned* by the validated trend, not assumed.

## 7. Plan & gates (incremental, each gated)

- **P0 — system setup.** KA energy (species σ/ε matrix, partials), swap MC, observables
  (partials g(r), S(q), ψ6, per-particle U). GATE: O1 (no crystallization) + O2 (swap MC
  equilibrates 256–2048 at chosen T). If fail → adjust ratio/T or stop.
- **P1 — hardness confirmation (reuse the gate methodology).** Show uniform+SMC / plain MD CANNOT
  equilibrate the large system in a sane budget (the analogue of `hard_state_scan.py`, now expected
  GO). Establishes the baseline the flow must beat. If uniform+SMC anneals it → wrong T/size, back to P0.
- **P2 — build + unit-test both arms.** Exact-`log q` checks (invertibility / log-det vs autograd),
  genuine-locality checks (conditioner magnitude flat vs neighbour count — the test that caught the
  `_ctx_c` extensivity), species handling.
- **P3 — train both at N_train; in-distribution validation.** Match milestone-(b) observables at
  `N_train`. GATE: both reproduce the training-size glass.
- **P4 — the benchmark (milestone b).** Size-transfer 256→2048: raw-proposal quality vs N (overlap
  / energy vs uniform baseline — NOT ESS, which saturates), then flow+SMC correctness of observables,
  then cost. Head-to-head transformer-AR vs coupling flow. Plus the **flow-vs-uniform control** at
  large N (does the flow beat uniform+SMC *here*, where the easy-LJ control said it didn't).

## 8. Metrics & known pitfalls (carried from the LJ work)

- **Do not trust ESS alone.** It saturates (an `r⁻¹²` tail makes good and bad proposals both read ~0)
  and reads ~100% after a terminal resample on a wrong ensemble. Always pair it with a structural
  observable (overlaps, g(r), energy).
- **Judge transfer by raw overlaps / observables vs the uniform baseline,** not by plain-IS ESS.
- **Run the flow-vs-uniform control under identical SMC** — the only honest test of whether the
  learned model earns its keep.

## 9. Risks / open questions

- O1 crystallization; O2 equilibration budget (above).
- AR drift may sink Arm 1 at large N (that's the hypothesis); coupling-flow trainability is Arm 2's risk.
- `ξ` may already exceed feasible `L_train` at interesting T → transfer window narrow. Mitigation:
  characterize `ξ(T)` first, pick T with `ξ` comfortably inside `L_train`.
- Swap-MC reference cost at N=2048 may be heavy; cap the validated range where it's affordable.

## 10. Out of scope (this milestone)

Temperature transfer (unprincipled); 3D; equivariant CNF; the speedup (a) and ξ-map (c) milestones.
