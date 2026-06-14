# liquid_coupling_flow — overnight progress (2026-06-14)

## UPDATE 13:50 — particle flow RESOLVED to autoregressive placement (was: stuck at identity)

The "stuck at identity" particle-training bug is understood and largely fixed:
- **Diagnosis:** a coupling/augmented flow cannot create *excluded volume* because each
  particle's update is conditioned on a variable that is NOT its real neighbours at
  generation time (uniform or noisy auxiliary). Noisy-aux *fit* the density (−logq 103→50)
  but generation broke (train/test mismatch, ESS 0.01%). Confirmed via `diagnose_particles.py`.
- **Fix = autoregressive placement** (`particles_ar.py`): place particles one at a time, each
  drawn from a circular-spline conditional shaped by the REAL already-placed particles. Exact
  triangular log q(x), clean reweighting (no auxiliary). Conditioning variables exist at
  generation time ⇒ no bootstrap.
- **Monotone trajectory** (N=16 2D LJ, forward-KL on MCMC data, x-space ESS = reweight to LJ):
  overlaps **0.71 → 0.34 → 0.12 → 0.061**, ESS **0% → 0.01% → 0.13% → 0.60%** as the
  conditional sharpens (bins/capacity). Clearly works; needs more sharpness to reach usable ESS.
- **Next levers (in flight / ordered):** more bins/capacity/steps (a bins=24 9000-step run is
  running → `/tmp/lcf_ar3.out`, artifact `phaseE_ar.png`); then **apply the SMC corrector**
  (`smc.py`) to rescue residual ESS; then the **size-transfer test** (train small N, eval larger
  N — the conditioner is local, so it should transfer). A sharper 2D conditional (couple the two
  coords more strongly, or kNN-local context) is the main modelling lever.
- Demos: `demo_particles_ar.py` (AR, the live one), `demo_particles_mle.py` (augmented, for the
  record), `diagnose_particles.py` (the gradient/excluded-volume diagnostic).

---


**Branch:** `liquid-coupling-flow`
**Context:** the pivot from the stuck AR-over-Hilbert-curve transformer to a *local,
exact-likelihood coupling normalizing flow* for size-transferable Boltzmann sampling of
**disordered liquids**. Rationale + literature + scoop check (you are NOT scooped — Scalable-BG
2509.25486 does crystals only; liquids are its stated future work) are in
`reports/2026-06-14-getting-unstuck-new-directions.md`.

**Goal:** match all statistics (or give reweightable likelihoods) + size-transfer. Success route:
exact `log p` (one pass, triangular Jacobian) → importance reweighting; SMC/annealing to fix ESS at
scale; locality (cutoff) for size transfer because a homogeneous liquid is locally stationary.

## What's built and VERIFIED (all tests machine-precision in float64)

| Module | Purpose | Test / result |
|---|---|---|
| `base.py` | DiagGaussian, UniformTorus base distributions | — |
| `transforms.py` | AffineElementwise (RealNVP) | core test |
| `transforms_spline.py` | RQ spline (R) + **circular RQ spline (torus)** | `test_spline` roundtrip ~1e-13, log-det vs autograd ~1e-15 |
| `coupling.py` | masked coupling layer (triangular Jacobian) | `test_flow_core` |
| `conditioner.py` | MLP + **PeriodicMLP** (sin/cos, seam-continuous) | — |
| `flow.py` | Flow: `sample()`, `log_prob()`, `forward_kld()` | `test_flow_core`: logp(sample)==logp(eval) 1e-6, log-det==autograd 1e-7 |
| `smc.py` | **annealed SMC/AIS corrector** + periodic RW-Metropolis | `test_smc`: ESS 6.5%→**99.3%**, mean 1.993 (true 2.0) |
| `energy.py` | periodic LJ energy (the Boltzmann target) | `test_energy`: two-body analytic, transl-inv 4e-16, min −1 |
| `particles.py` | **augmented local particle coupling flow** (cutoff GNN) | `test_particles`: joint invertibility 6e-14, joint log-det==autograd **1e-15** |

### Demo results (proof-of-concept)
- **Phase A** (`demo_phaseA.py`): affine flow learns `eight_gaussians`, NLL 3.85→1.29 (affine is the
  weak transform — establishes the training loop).
- **Phase C/D** (`demo_phaseC.py`): **circular-spline flow on a 5-mode torus mixture → NLL −1.30,
  importance-sampling ESS 97.6%.** ← keystone: exact-logp flow reweights to ~perfect ESS on a
  PERIODIC domain.
- **SMC** (`test_smc.py`): a deliberately-mismatched proposal (ESS 6.5%) rescued to **99.3%** with the
  correct reweighted mean. ← the exactness-at-scale mechanism.

Together these prove both exactness routes the project needs: **direct reweight** when the flow is
good, **SMC** when it isn't.

## How to run
```bash
PY=/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python
cd /mnt/ssd/GridTransformer
# tests
for t in test_flow_core test_spline test_smc test_energy; do $PY -m liquid_coupling_flow.tests.$t; done
# demos (write PNGs to liquid_coupling_flow/artifacts/, ~3 min each on CPU)
$PY -m liquid_coupling_flow.demo_phaseA
$PY -m liquid_coupling_flow.demo_phaseC
```

## NEXT STEP — the particle flow (the real liquid), design notes

The toys prove the machinery. The integration piece is a flow over PARTICLE configurations
`x ∈ [0,L)^{N×d}`. The non-obvious crux is **how to make the coupling local AND invertible**:

- A coupling that updates particle *i* cannot use *i*'s own position to find its neighbours
  (params must not depend on the variable being transformed) — that breaks invertibility.
- **Fix = augmented coupling** (Scalable-BG): carry an auxiliary copy `a` (same shape as `x`).
  Alternate: update `x | a` (GNN/neighbour-graph built from the *fixed* `a`), then `a | x`. The
  conditioning set supplies BOTH features and the neighbour graph, so locality + invertibility
  coexist. Reweight in the JOINT `(x,a)` space (density exact there); the x-marginal is just the
  reweighted x's. Target joint `π(x,a) = e^{−U(x)/kT} · r(a)`, `r` = uniform/Gaussian.
- Conditioner = **local message passing**: per particle, aggregate over neighbours within a cutoff of
  `MLP(rbf(min-image r_ij), unit vec)` → per-particle embedding → circular-spline params. Cutoff +
  homogeneity ⇒ size transfer (the whole point).

Build order:
1. ✅ DONE `particles.py`: ParticleGNNConditioner + AugmentedCouplingLayer + ParticleFlow.
2. ✅ DONE `test_particles.py`: joint invertibility (6e-14) + joint log-det vs autograd (1e-15). PASS.
3. ⚠️ OPEN ISSUE — particle-flow TRAINING does not yet converge (machinery is proven; this is
   localized to the GNN conditioner / augmented optimisation, NOT the flow framework):
   - `demo_particles.py` (reverse-KL, N=16 2D, soft core): hot/slow anneal got `<U>/N` 93→11 but it
     stalls clumped at the soft-core floor, ESS 0%. Reverse-KL is mode-seeking (known-hard).
   - `demo_particles_mle.py` (forward-KL on 32k MCMC configs, data `<U>/N = −1.56`, correct fluid):
     **`−log q` plateaus at 103.0 from step 400 = exactly the uniform-base constant `2Nd·log L`** →
     **the flow never leaves the identity map** (samples uniform, 72% overlap, ESS 0%).
   - **Diagnosis:** the SAME Flow/spline/training code fits the torus toy to 97.6% ESS, so the bug is
     in `particles.py` — the GNN conditioner produces ~no gradient/variance (flow stays volume-
     preserving). DEBUG NEXT, in order: (a) check `ParticleGNNConditioner` output variance + grad norm
     on data; (b) non-zero / smaller-scale init of `node_mlp` last layer (zero-init may be a flat
     start *for the GNN specifically*); (c) try **split-particle coupling** (condition x-updates on a
     structured subset of x) instead of a uniform-noise auxiliary `a` — conditioning x on *noise* a
     can't inject x–x correlations, the likely root cause; (d) larger lr / more layers.
4. **Size-transfer test (the headline claim)** — only after (3): train small N, evaluate joint ESS /
   g(r) at larger N with the SAME weights (local conditioner ⇒ should transfer). Pair with `smc.py`.
5. Then 3D; add rotation equivariance (v1).

**Honest status:** the exact-likelihood periodic coupling-flow + SMC + energy stack is verified and the
toys prove the thesis (torus ESS 97.6%, SMC 6.5%→99.3%). The particle *integration* is scaffolded and
its likelihood is proven exact, but training is stuck at identity — a concrete, isolated bug for the
next session, with the split-particle-vs-noise-auxiliary hypothesis as the prime suspect.

## Open decisions for you
- **Auxiliary `a`**: torus-uniform (both sets spatial, clean neighbour graphs) vs. Gaussian (Scalable-BG
  style). I lean torus-uniform for symmetric, local neighbour graphs in both update directions.
- **Equivariance**: v0 uses relative min-image features (→ translation invariance). Full rotation
  equivariance (e.g. swap-CoM, vector messages) is a v1 refinement; not needed for first transfer signal.
- **Float precision**: float32 spline inverse is ~3e-4; for tight reweighting at scale, run `log_prob`
  in float64 (the math is exact there). Cheap given one-pass likelihood.
