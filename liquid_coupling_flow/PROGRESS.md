# liquid_coupling_flow — overnight progress (2026-06-14)

## UPDATE (cont.) — 3D works; size-transfer is SMC-carried (raw flow does not transfer)

Two parallel runs. Code: `mcmc.py` (d-general MCMC), `particles_ar_nd.py` (d-general AR
flow, reduces to 2D ARParticleFlow, invertibility ~1e-6 at d=2 AND d=3), `size_transfer.py`,
`train_3d.py`. Figures: `artifacts/size_transfer_gr.png`, `artifacts/lj3d_smc_reweighted.png`.

**3D LJ (N=32, rho*=0.65, T*=1.0) — SUCCESS.** d-general flow trains in 3D (-logq 126.8->78.8,
well below base 124.7 -> genuinely learning), raw overlaps 0.088 (learns 3D excluded volume),
flow+single-particle SMC reproduces energy AND g(r): reweighted <U>/N -2.759+/-0.004 (MCMC
-2.770), SMC ESS 53.1%. Architecture generalizes to 3D. (Box small: cutoff 1.8 ~ L/2, g(r) to
first shell only; bigger N / more steps would tighten ESS + the mild ~0.4% energy under-relax.)

**Size transfer (2D, const rho=0.64, N=16->36->64) — PARTIAL / diagnostic.** Trained at N=16
only, weights load at any N (no N-dependent shapes). Results:
  N   raw_ov  IS_flow  IS_unif  SMC_ESS  <U>/N_rw (MCMC)
  16  0.055   0.31%    0.04%    82.9%    -1.553 (-1.559)
  36  0.219   0.04%    0.04%   100.0%*   -1.555 (-1.571)
  64  0.433   0.04%    0.04%   100.0%*   -1.538 (-1.577)
- g(r) transfers well at ALL sizes (reweighted tracks MCMC through all shells, even N=64=4x).
- BUT the RAW FLOW does NOT transfer: plain-IS ESS collapses to the uniform-noise level (0.04%
  = uniform) by N=36; flow advantage (8x over uniform at N=16) vanishes. Overlaps balloon to 0.43.
- Energy is biased high and the bias GROWS with N (Dlt 0.006->0.016->0.039) = under-relaxation:
  SMC does all the work (proposal ~ uniform) and a fixed 24-step bridge under-builds the first
  peak more as N grows. *The "SMC ESS 100%" is a MISLEADING post-terminal-resample artifact;
  the honest metric is the reweighted observable, which shows the bias. Always check an
  observable alongside ESS.*
- ROOT CAUSE (both runs triangulate it): the coord-0 context `_ctx0` is a GLOBAL DeepSets SUM
  over placed particles -> extensive, ~Nx magnitude at larger N -> pushes head0 off-manifold ->
  proposal collapses to uniform. 3D works because it's evaluated at the train size (in-distribution).
- FIX TRIED: made `_ctx0` intensive (mean not sum), retrained N=16, re-ran transfer. RESULT =
  LATERAL, not the hoped win. mean helped large-N (overlaps N64 0.43->0.37, energy bias
  0.039->0.007) but HURT train size (-logq 31.5->32.8, overlaps 0.055->0.092, SMC ESS 83->63%).
  `_ctx0` was NOT the sole culprit.
- METRIC INSIGHT: plain-IS ESS is SATURATED by the r^-12 tail (any >few% overlap -> one bad pair
  dominates weights), so flow (0.37 overlaps) and uniform (0.72) BOTH read 0.04% -- it cannot see
  transfer. The honest transfer metric is RAW OVERLAPS vs the uniform baseline (~0.72 at all N):
  by that measure the flow DOES partially transfer (N64 0.37 vs 0.72 uniform, ~2x better than
  noise) at BOTH sum and mean. Always check a structural observable, not ESS alone.
- SMC budget firm-up (3D, `firm_3d.py`): n_bridge 30->50 lifted 3D ESS 53%->85% and fixed the
  energy bias (-2.759->-2.776 vs MCMC -2.770) -- confirms residual energy bias = under-relaxation.
- REMAINING BOTTLENECK (next levers): (1) `_ctx_c` excluded-volume context is still L-EXTENSIVE
  (Gaussian-weights over a coord-0 strip spanning the full box height -> message sum grows with L);
  only `_ctx0` was fixed. (2) AR placement is inherently OOD at larger N (late particles see far
  more placed neighbours than at train size). Fixing (1) = normalize/cap `_ctx_c` to a true local
  2D neighbourhood. Net: exactness-at-scale SOLID (flow+SMC matches Boltzmann 2D N16->64 AND 3D);
  locality->STRONG flow transfer only PARTIAL (flow beats noise but still a weak large-N proposal,
  SMC essential).

## UPDATE (cont.) — SMC corrector VALIDATED on particles: AR flow -> Boltzmann, ESS 96.7%

The end-to-end claim now holds on the 2D LJ liquid (N=16, L=5, kT=1): the AR flow is a
size-local proposal with exact log q, and annealed SMC corrects it to the Boltzmann target.

- **Diagnostic plots** (`plot_ar_diagnostics.py` -> `artifacts/ar_energy_gr.png`): the raw AR
  flow reproduces medium-range structure (g(r) tracks MCMC beyond the first shell) but has a
  SOFT hard core -- spurious g(r) below 0.9 sigma, under-built first peak, 41% of configs above
  the energy cap. ~5% per-particle overlaps -> r^-12 wrecks energy -> x-ESS ~0.2%. (g(r) 2D
  shell normalization validated on a uniform ideal gas: flat g=1.000.)
- **SMC fixes it** (`apply_smc.py`): bridge (1-beta) log q + beta (-U/kT), resample + MCMC.
  | kernel | ESS | acc | unique configs | energy median U/N |
  | flow alone (plain IS) | 0.17% | - | - | +0.92 |
  | global RW moves       | 84.6% | 0.01 | 1402/4000 (35%) | -1.60 |
  | **single-particle**   | **96.7%** | **0.47** | **3688/4000 (92%)** | **-1.572** (MCMC -1.559) |
  Energy + g(r) of the single-particle SMC ensemble are indistinguishable from MCMC
  (`artifacts/ar_smc_single.png`).
- **Two bugs found + fixed:** (1) `smc.py` RWMetropolis accept-mask assumed flat [M,D] toy
  inputs -> broke on [M,N,d] particle configs (latent: toys never exercised the rank). Fixed to
  broadcast over trailing dims. (2) global whole-config moves give acc ~ (single-acc)^N ~ 0.01
  in a dense liquid -> SMC degenerated to resampling-only (diversity 35%, high ESS was
  misleading). Added `SingleParticleMetropolis` (one particle at a time, exact bridge via full
  log q recompute) -> acc 0.47, genuine rejuvenation, 92% unique.
- **Why this matters for the thesis:** single-particle SMC *relaxes* overlaps instead of
  discarding them, so it does NOT depend on the flow having a clean tail to resample -> it should
  keep working at larger N where the flow alone degrades. NEXT: the size-transfer test (train
  small N, SMC-correct at larger N) and then 3D.

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
