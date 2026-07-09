# Liquid-mW annealed SMC — design

**Date:** 2026-07-09 · **Branch:** liquid-coupling-flow · **Status:** approved design, pre-plan
**Goal:** build and validate an annealed-SMC equilibrium sampler for liquid mW, then quantify how much the
locally-trained, size-transferable AR generator (3D port of the flowhead lineage) reduces its cost as the
SMC base. Value regime = expensive-energy targets (MLIP/ab-initio); mW is the cheap validation proxy.

Decisions taken with user: state point = ambient 300 K; sizes = train N=64 → zero-shot N=216 (512
appendable later); approach = geometric λ-path with q₀ in the path (Approach 1); primary cost metric =
energy evaluations, not wall-clock.

---

## 1. System, units, state point

mW = Stillinger–Weber, Molinero–Moore (2009) monatomic-water reparameterization:

- ε = 6.189 kcal/mol, σ = 2.3925 Å, cutoff a = 1.8 (in σ), A = 7.049556277, B = 0.6022245584,
  p = 4, q = 0, γ = 1.2, cosθ₀ = −1/3, λ₃ = 23.15 (tetrahedral strength; named λ₃ to avoid clashing
  with the annealing parameter λ).
- Energy: U = Σ_{i<j} φ₂(r_ij) + Σ_i Σ_{j<k ∈ nbr(i)} φ₃(r_ij, r_ik, θ_jik) with
  φ₂(r) = A ε [B (σ/r)⁴ − 1] exp(σ/(r − aσ)) for r < aσ, else 0;
  φ₃ = λ₃ ε (cosθ_jik − cosθ₀)² exp(γσ/(r_ij − aσ)) exp(γσ/(r_ik − aσ)) for both r < aσ.
  Both terms vanish smoothly at the cutoff (no shift needed).
- Internal units: reduced (σ = ε = 1). Reports carry both reduced and real.
- State point: T = 300 K → **T\* = k_B·300/ε = 0.0963** (β\* = 10.38); ρ = 0.997 g/cm³ →
  n = 0.03333 Å⁻³ → **ρ\* = n·σ³ = 0.4564**. Ensemble: NVT, cubic box, PBC, min-image.
- Boxes: N=27 → L=3.90σ, N=64 → L=5.20σ, N=125 → L=6.49σ, N=216 → L=7.79σ. All satisfy
  L/2 > aσ = 1.8σ (N=27 marginally: 1.95σ — acceptable, flagged).

## 2. The sampler (Approach 1: geometric λ-path)

Path: π̃_λ(x) ∝ q₀(x)^{1−λ} · (e^{−βU(x)})^λ, λ: 0 → 1, so log π̃_λ = (1−λ)·log q₀(x) − λβU(x).

- **Init:** B particles i.i.d. from q₀, log w = 0.
- **Incremental weights** (λ → λ′): Δlog w_i = (λ′−λ)·(−βU(x_i) − log q₀(x_i)).
- **Adaptive schedule:** λ′ = largest value in (λ, 1] with ESS ≥ ess_target·B, default ess_target = 0.6
  (the `ka_local_smc` value; bisection; direct port of
  `ka_local_smc.next_beta` with U replaced by βU + log q₀). Progress floor 1e-4; the `ka_local_smc`
  stall guard (resample when ESS lands exactly on target) carries over.
- **Resampling:** multinomial on ESS-trigger, weights reset to 0.
- **Mutation (π_λ-invariant):** single-site displacement MH sweeps; acceptance
  α = min(1, exp[(1−λ)(log q₀(x′) − log q₀(x)) − λβ(U(x′) − U(x))]) with symmetric Gaussian proposals
  (step adapted at λ=1 target during Phase-1 tuning, then frozen). ΔU via a pair/triplet-local row update
  where possible; correctness anchor = full-recompute agreement test.
- **Uniform base degenerate case = Phase 1:** log q₀ = −N·log V = const ⇒ the log q₀ terms drop from
  weights-differences and MH ratios; the path reduces to β-annealing with β_eff = λβ. The identical
  code path is therefore certified exact before any NN exists (fault isolation for Phase 2: only the two
  log q₀ terms are new).
- **log Z estimator:** Σ over rungs of log-mean incremental weight (needed for the c measurement, §5).

Guards inherited from ordmh (reports/logs-2026-07-09/ordmh_results_note.md): cached log q₀ carried on the
canonical ordering, asserted equal to a fresh re-evaluation each rung; per-step conditional normalization
grid-checked; full-support base (spline tails ⇒ density > 0 everywhere in the box).

## 3. Module layout — new subpackage `liquid_coupling_flow/mw/` (ordmh-style isolation)

| Module | Contents | Depends on |
|---|---|---|
| `mw_energy.py` | batched torch SW (two-body + triplets within cutoff), min-image; O(N·k²) triplet enumeration; chunked eval helper for big config stacks (the 10.4 GiB lesson) | torch only |
| `mw_bedrock.py` | grid-quadrature ⟨U⟩ at N=2 (two-body only) and N=3 (the only analytic exercise of the three-body term), halving-error control | mw_energy |
| `mw_reference.py` | displacement-MC reference (adaptive step), budget-doubling flatness + collection-drift proofs, per-unit saves; doubles as Phase-2 training-data generator | mw_energy |
| `mw_smc.py` | λ-path SMC per §2; base passed as an object exposing `sample(B)` and `log_q(x)`; per-rung saves | mw_energy, base |
| `mw_base.py` | `UniformBase` (log q₀ = −N log V) and the generator adapter | — |
| `mw_generator.py` | 3D local-frame AR generator (§4) | gilbert3d, torch |
| `tests/test_mw_*.py` | §6; the independent numpy SW implementation lives inside test_mw_energy.py, never imported by main code | — |

Artifacts → `liquid_coupling_flow/mw/artifacts/`; logs → `reports/logs-<date>/`; every run > 30 min has
per-unit/per-rung saves (CLAUDE.md checkpoint rule); full configs + per-config observables saved, never
summaries only.

## 4. The generator (Phase 2/3): 3D port of the local-frame AR flowhead

Architecture class fixed by prior evidence: AR transformer + exact spline head = the only class with a
one-forward exact log q (required in the weight path — EGNN-CNF integrator-error densities are
disqualified there), with geometry-invariant local-frame conditioning = the only conditioning that
transferred across N without a cliff.

- **Ordering:** `gilbert3d_path` on a 4×4×4 grid (N=64) / 6×6×6 (N=216) — defines the factorization
  order AND the per-step anchor t_j = scaffold cell center, regenerated at each N (the rail/localframe
  mechanism measured to transfer). Hard invariant: **the curve enters conditioning ONLY through t_j**;
  no feature may encode curve indices or arc-length of neighbors (index-style curve conditioning is the
  measured transfer-killer).
- **Local frame (the one genuinely new component):** per step j, an orthonormal triad built by
  Gram–Schmidt from the displacement vectors to the two nearest already-placed neighbors (min-image).
  Fallback hierarchy: j=0 → box frame at a fixed anchor; j=1 → axis from the single neighbor completed
  to a triad by a fixed rule; near-collinear pair (|cos| > 0.99) → third-nearest neighbor substitutes.
  Deterministic, config-dependent only through placed particles ⇒ exactness preserved.
- **Conditioning:** k-NN placed neighbors (k = 12, 3D cage scale) with positions expressed in the local
  frame + fixed-period physical radial encodings (size-invariant). Monatomic ⇒ no species machinery.
- **Head:** per-coordinate factorized rational-quadratic-spline conditional for the residual w.r.t. the
  frame anchor (same head family as `ka_flowhead`, widened to 3 coordinates).
- **Density bookkeeping:** q₀(x) evaluated via the deterministic canonical (curve) ordering; exactness
  tests = sample↔log_prob agreement (per-step, 1e-5 scale) + noncanonical-fraction counter. Note: the AR
  density is not local — a single-site move changes downstream conditionals — so each mutation MH
  evaluation costs one teacher-forced forward (B-parallel). Acceptable at N≤216; in the value regime the
  NN forward is the cheap component by assumption.
- **Training:** MLE on ~2–4k decorrelated reference configs at N=64 (from `mw_reference`), val split,
  val-loss checkpointing (the exposure-bias-noise-run lesson: never monitor train loss).

## 5. Phases and gates

**Phase 1 — skeleton, trivial base, no NN.** Gates, in order; each blocks the next:

- **G0 energy exactness:** torch SW vs independent numpy SW vs hand-analytic values (dimer at chosen r;
  trimer at chosen angles so φ₃ is hand-checked) — machine precision (≤1e-10 relative).
- **G1 bedrock:** quadrature ⟨U⟩ vs displacement-MC reference at N=2 AND N=3 (three-body exercised);
  tolerance = quadrature halving error + 3·MC standard error.
- **G2 SMC exactness:** uniform-base λ-SMC at N=8 and N=64 vs long displacement-MC reference on ⟨U⟩,
  P(U) (total-variation on a shared binning), g(r); plus mutation-kernel detailed-balance unit test and
  the §2 weight-path guards. Pass = agreement within combined error bars.
- **G3 rung scaling:** T_trivial(N) at N = 27, 64, 125, 216 at fixed ess_target and fixed per-rung
  mutation budget → the ∝N baseline line (slope = c_trivial). This is the yardstick Phase 2 is measured
  against.

**Phase 2 — generator as base (N=64).** Entry gate: ordering-spread measurement on mW reference configs
(log q̃ under K=32 random orderings on M=64 configs; pre-registered re-check of ordmh's benign-tax claim
in 3D — recorded whatever it shows, it simply enters c). Then:

- Swap base; measure **c** per base: c = (1/N)·(E_{q₀}[log q₀ + βU] + log Z), log Z from the SMC
  estimator; the **c-difference** between bases is Z-free (log Z cancels), so the headline number
  Δc = c_trivial − c_gen carries no estimator systematic.
- **Primary result:** T_generator vs T_trivial at matched ess_target and matched mutation budget
  (identical number of MH sweeps per rung, one sweep = N site attempts);
  pre-registered success = rung ratio consistent with the c-ratio (the T ∝ cN prediction), and
  **energy evaluations per effective sample** reduced. NN forwards counted and reported separately.
- Exactness re-gate: generator-base SMC at N=64 must reproduce the same reference observables as G2.

**Phase 3 — zero-shot transfer (N=216).** The N=64-trained generator deployed unmodified (curve
regenerated at 6×6×6; nothing in conditioning references it). Repeat the Phase-2 measurement at N=216:
success = rung reduction survives with c roughly N-independent; exactness re-gate at N=216 vs its own
displacement-MC reference.

## 6. Testing

- `test_mw_energy.py`: G0 triple-check; translation/rotation/permutation invariance; cutoff smoothness
  (U continuous through r = aσ); chunked-vs-full agreement.
- `test_mw_smc.py`: next_λ bisection properties (monotone, respects floor); resample statistics;
  mutation DB on a 2-particle system (empirical transition-ratio vs Boltzmann ratio); uniform-base
  reduction (log q₀ terms provably absent from ratios); weight-guard asserts fire on deliberate
  corruption.
- `test_mw_generator.py`: frame orthonormality + covariance (rotating the neighbor displacement vectors
  rotates the frame so frame-coordinates are invariant; box-lattice translations of config ⇒ identical
  density — global rotation invariance is NOT claimed, the box-fixed scaffold breaks it); fallback-path
  coverage (j=0, j=1, collinear); sample↔log_prob consistency; normalization spot-check by 1D quadrature
  on a conditional; prefix-permutation invariance (permuting storage order of placed particles ⇒
  identical conditioning tensors — features are geometry-only, no curve indices).

## 7. Risks (pre-registered)

1. **Ambient mW is easy** → c_trivial may be small, leaving little headroom. The c-ratio criterion is
   the pre-registered readout precisely so "generator adds little because the problem is easy" is a
   measurable outcome, not a post-hoc excuse. (KA/LJ precedent: easy-state head-starts were ~nil.)
2. **AR clash tail at λ≈0** → −βU spans wildly on raw base samples; the adaptive Δλ absorbs it as tiny
   first rungs (measured KA precedent). Monitor rung-1 ESS and Δλ₁.
3. **Ordering spread in 3D/tetrahedral** may exceed the 2D-WCA 0.027 nats/particle → measured at
   Phase-2 entry; enters c honestly.
4. **3D frame degeneracies** → explicit fallbacks + equivariance tests (§6).
5. **N=27 box marginal** (L/2 = 1.95σ vs cutoff 1.8σ) → G3 only; if artifacts appear, drop to
   N=32-in-noncubic or start the scaling line at 64.

## 8. Out of scope

One-shot generation fidelity; architecture changes beyond the 3D port; beating classical samplers on
cheap-energy mW (value case is expensive-energy targets); supercooled state points (follow-up arm);
N=512 (appendable — transfer is zero-shot); Phase-2 retraining at any N other than 64.
