# AMENDMENT: joint continuous (σ,x) flow as the transition kernel

**Date:** 2026-07-16 (evening pivot)
**Amends:** `docs/superpowers/plans/2026-07-16-joint-sigma-x-block-gate.md` (Tasks 3–5)
**Status:** USER DIRECTIVE — supersedes the discrete-permutation design.

## The directive (verbatim intent)

> "We don't want the flow to propose the relax step after a swap. I want the flow to
> continuously change both position and species, and this acts as a transition kernel."

## Why the old design failed (measured 2026-07-16)

Discrete σ-permutation → position-only relax transport is LONG and MULTIMODAL:
- identity relax (pure thermal, 400 sw, T=0.085): RMS move 0.347
- similar-σ swap + relax: RMS 0.348
- random 2-transposition + relax (the old training target): **RMS 1.02, 90pct max move 6.35 (≈ box)**

FM regresses the mean of a one-to-many map → loss floor 1.01 vs v=0 floor 1.38; pilot gate acc ~0.
The permutation *creates* the hard overlap that positions must then undo. Co-evolving σ removes it.

## New design

### State & ensemble
- Per block particle: z = (x, u) ∈ R⁴, where u = Φ⁻¹(F(σ)) with
  F(σ) = (a − σ⁻²)/(a − b), a = SIG_MIN⁻², b = SIG_MAX⁻² (SIG_MIN=0.725, SIG_MAX=1.61);
  inverse σ(u) = (a − Φ(u)(a−b))^(−1/2); Φ = standard normal CDF (numba: 0.5*(1+erf(u/√2))).
- Target (semi-grand in u-space): **π(x, u) ∝ exp(−βU(x, σ(u))) · Π_i φ(u_i)**, φ = std normal pdf.
  Marginal composition = P(σ) ∝ σ⁻³ by construction (matches the canonical system's composition).
  Classical swap MC (permutations) is ALSO valid in this ensemble → τ comparisons remain fair.
- Energy: existing `poly/model.py` kernels with σ(u) — NO energy code changes.

### Kernel (exact MH)
- Flow proposes z_new from base z_old + Gaussian noise (x-width BASE_W=0.35; u-width BASE_W_U
  measured from training pairs, logged), forward RK4 t:0→1 on the JOINT field.
- logq: reverse fixed-grid RK4 (24 steps) accumulating div over ALL block dims:
  div = div_x(vel) [existing analytical] + Σ_i ∂u̇_i/∂u_i [new, autograd/vmap-jvp exact].
- **A = min(1, exp(−βΔU) · Π_i[φ(u′_i)/φ(u_i)] · q_rev/q_fwd)** — no permutation factor.

### Training pairs (deployment-matched, SHORT transport)
- `make_joint_block_pair`: block = k-NN; env frozen; short block-local JOINT MC
  (displacement sweeps + single-site u-moves u′=u+δη with accept min(1, [φ(u′)/φ(u)]e^{−βΔU}));
  record (x_old,u_old) → (x_new,u_new). Both channels move continuously → unimodal transport.

### Gate metrics (Task 5 revision)
- acceptance vs k at bracket temps, PLUS **σ-mobility**: E[Σ_i|Δσ_i|] per ACCEPTED move and
  per unit wall-clock — acceptance alone is gameable by tiny moves.
- Baselines: (a) classical pair-swap anchor; (b) single-site u random-walk at matched
  per-move σ-displacement (the naive semi-grand kernel — THE bar to beat); (c) x-only flow.

### Crux tests (carry over + new)
1. propose/logq self-consistency on a PERTURBED net (<1e-4).
2. Dummy-padding inertness on BOTH channels (k_max=k vs k+4 agree <1e-6; finite logq).
3. **u-divergence exactness**: reverse-RK4 path-sum vs forward-accumulated −∫div on a perturbed
   net (<1e-4), and Σ∂u̇/∂u vs brute-force autograd Jacobian trace on one step (<1e-6).
4. Semi-grand u-kernel exactness: at β=0 the single-site u-walk must reproduce φ(u)
   (KS or moment check), i.e. σ-marginal = P(σ).

### File plan (disjoint sets)
- `liquid_coupling_flow/poly/semigrand.py` — numba: u↔σ maps, single-site u-sweep kernel,
  joint block relax; + `liquid_coupling_flow/tests/test_poly_semigrand.py`.
- `liquid_coupling_flow/poly/joint_flow.py` — `JointBlockFlow` (batched propose/logq_of on
  (x,u)); + `liquid_coupling_flow/tests/test_poly_jointflow.py`.
- `reports/logs-2026-07-17/poly_joint_pairs.py`, `poly_train_jointflow.py`, gate revision — after
  the two modules land.
- Old position-only `block_flow.py` stays committed as the refuted variant (do not delete).

### Unchanged
- Banks (4×N=300 + held-out N=600) remain the data/eval substrate; N-scaling and m_env
  controls carry over verbatim (gate is N-agnostic; m_env read from ckpt).

## DECISION 2026-07-17 (user): swap-endpoint now, semi-grand later

Measured (clean frames, frozen x unless noted, T=0.085):
- Unconstrained semi-grand (prior only): mean sigma 0.998->0.82, U/N +0.32->+0.02 (deflates; wrong state point).
- Volume-conserving pair exchange (sum sigma^3 exact): shape still drifts (median 0.917->0.846, p95 1.495->1.602), U/N ->+0.20. Also wrong state point.
- Classical direct swap acc vs |dsig|: 0.44 (<0.1), 0.026 (0.1-0.2), 0.0000 beyond 0.2 (0/524).
- Single-pair swap + 400sw block relax transport: rms|dx| 0.12->0.39 smoothly in |dsig|, p90max <=1.05 -- SHORT and near-unimodal (vs random 8-perm RMS 1.02 multimodal). Clean-frame thermal RMS ~0.12 (old 0.35 was bank-bug-inflated).

PHASE 1 (NOW) -- swap-endpoint kernel, canonical fixed-composition ensemble:
- Kernel: pick pair (i,j) in the k-NN block; sigma(t) = (1-t)sigma_old + t sigma_swapped as
  TIME-DEPENDENT CONDITIONING of a position-only flow (labels sigma_to_bin(sigma(t)) recomputed
  per RK4 step); endpoint = exact permutation + accommodated x. logq = position divergence only
  (block_flow.py's validated machinery). A = min(1, exp(-beta dU) q_rev/q_fwd); q_rev conditioned
  on the reverse path sigma_new -> sigma_old. No mu, no ensemble change; swap-MC baseline at the
  IDENTICAL state point.
- Gate axis: acceptance vs |dsig| (bins above) x T; flow must beat classical acc(|dsig|),
  esp. >=1% in the 0.2+ bins where classical = 0.
PHASE 2 (LATER, gated on phase-1 success): true semi-grand with calibrated mu(sigma); J2's
joint (x,u) flow module is the phase-2 asset (commit it; do not delete).
