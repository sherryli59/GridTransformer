# liquid_coupling_flow — overnight progress (2026-06-14)

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
1. `particles.py`: ParticleGNNConditioner + AugmentedCouplingLayer + ParticleFlow (joint sample/log_prob).
2. Tests (CRITICAL, same as toys): joint invertibility + joint log-det vs autograd on a tiny N=4 2D system.
3. Train on a small 2D LJ liquid (N≈16–32), reweight against `energy.py`, report ESS + g(r).
4. **Size-transfer test**: train at small N, evaluate joint-space ESS / g(r) at larger N with the SAME
   weights (the headline claim). Pair with `smc.py` to rescue ESS at the larger sizes.
5. Then 3D, then energy-based training (reverse-KL / energy-weighted FM) to drop the data requirement.

## Open decisions for you
- **Auxiliary `a`**: torus-uniform (both sets spatial, clean neighbour graphs) vs. Gaussian (Scalable-BG
  style). I lean torus-uniform for symmetric, local neighbour graphs in both update directions.
- **Equivariance**: v0 uses relative min-image features (→ translation invariance). Full rotation
  equivariance (e.g. swap-CoM, vector messages) is a v1 refinement; not needed for first transfer signal.
- **Float precision**: float32 spline inverse is ~3e-4; for tight reweighting at scale, run `log_prob`
  in float64 (the math is exact there). Cheap given one-pass likelihood.
