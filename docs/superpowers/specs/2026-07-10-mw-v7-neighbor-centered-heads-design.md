# mW generator v7 — neighbor-centered output charts (radial-first + frame arms)

**Date:** 2026-07-10 · **Status:** approved · **Parent:** v6 spec · **Scope:** one module, two head
variants, controlled A/B on the same body/bank/trainer.

## Why (v6 verdict, measured)

v6 (global body + per-axis circular spline on the ANCHOR-offset chart) plateaued at val ~1.0 with TF
peak_err FLAT at ~0.55 from step 0 through 48k while core_mass improved 0.93→0.30 ⇒ expressiveness
ceiling, not slow learning: the first shell is a thin SPHERE |x−neighbor|=1.19, and a factorized
p(u_x)p(u_y)p(u_z) cannot concentrate mass on a sphere (axis coupling required). The free-code overfit
that ranked the spline over the MDN used point targets (factorizable); shell-shaped conditionals are
not. v4's full-cov MDN could tilt (crude coupling) — hence its better val (0.51) despite softer modes.
Fix = make the RADIAL structure around a PHYSICAL particle a coordinate of the chart.

## Shared exact construction (both arms)

For step j: locator n1_j = nearest already-placed particle to anchor t_j (deterministic causal; the
ka_localframe origin). Offset v = wrap_pm(x_j − n1_j, L). Density on the torus:

  p(x_j | h_j) = w(h)·p_local(v | h)·1[v ∈ D] + (1 − w(h))·(1/L³)

- p_local is a normalized density on the FIXED PHYSICAL domain D (below); w = sigmoid head.
- Integrates to exactly w + (1−w) = 1 over the box for any D that fits the fundamental domain ✓.
- Full support (uniform component) ✓ — required for the SMC base.
- SIZE-TRANSFER-NATIVE: D and p_local are physical/local (N-independent); the uniform term adapts via
  L³. The learned component is manifestly size-invariant — aligned with the project's transfer thesis.
- Steps with no placed neighbor (j=0): force w=0 (pure uniform; exact).
- Log-density via logsumexp of the two branches (−∞ shell branch outside D). Sampling: Bernoulli(w),
  then the local inverse transforms or a uniform box draw. Sample and log_prob share all parameter
  heads ⇒ mirror-exact by construction; verified by the standard ungated + perturbed tests.

## Arm A — v7-radial (the complete fix)

p_local(v) = p_r(r|h) · p_dir(n̂ | r, h) / r², r = |v|, n̂ = v/r; D = ball(r_cap = 2.5σ)
(2.5 < L/2 = 2.598 at N=64; fits at every larger N; covers first AND second shells: 1.19, 1.85).
- p_r: RQS spline on [0, r_cap], uniform base (scale/shift of the existing bounded spline; exact
  log-Jacobians). Multimodal in r ⇒ both shells representable SHARPLY in 1D.
- p_dir: a = cosθ ∈ [−1,1] via bounded RQS (tail_bound=1 ⇒ domain exactly [−1,1]; uniform base 1/2)
  then φ via circular RQS (period 2π, uniform base 1/(2π)); measure dΩ = da dφ ⇒ normalized on the
  sphere with NO sinθ pitfalls (cosθ as the variable absorbs the Jacobian; pole φ-degeneracy is
  measure-zero). AR conditioning within the step: r-embed → a-head; (r,a)-embeds → φ-head (the
  Spline3Head ctx-growing pattern; angles embed as sin/cos).
- The 1/r² Cartesian Jacobian is exact; clamp r ≥ 1e-6 in logs (excluded core has ~zero density there
  anyway; the uniform branch keeps support).

## Arm B — v7-frame (the cheap mechanism test)

p_local(v) = factorized bounded RQS over u = R_frame·v/s_c on [−1,1]³ (uniform base (1/2)³;
|det R|=1; s_c = 1.5σ so the rotated cube ALWAYS fits the fundamental domain: √3·1.5 = 2.598 ≤ L/2
at N=64, safer at larger N). Frame: e1 = unit wrap_pm(t_j − n1, L) (points at the vacancy region ⇒
u₁ ≈ radial), e2 via Gram–Schmidt with the 2nd-nearest placed neighbor (existing fallback hierarchy),
e3 = e1×e2. NOTE the torus caveat that killed naive rotation (v5 spec correction): the rotation is
exact ONLY because the support is capped to the inscribed cube — the circular/global chart cannot be
rotated. Covers the first shell only (r ≤ 1.5–2.6 in corners); the uniform absorbs the rest.

## Body, trainer, gates (shared; from v6)

v6 body verbatim (global AR + rail + geo-bias + zero-init three-body features); trainer with
multi-bank (thin-2 + extension), augmentation, AdamW 3e-4 / clip 5.0, best_nll + best_struct + last,
TF pooled-predecessor g(r) composite every val cycle (CUDA histogram .cpu() fix). NEW exactness test
beyond the standard two: 3D NORMALIZATION QUADRATURE — for a few fixed h (perturbed weights),
numerically integrate p(x) over the box (48³ midpoint grid) and assert = 1 within quadrature error
(the mixture + Jacobian surfaces are new; this is their bedrock test).

GO gate (either arm): TF g(1.19) within 10% of 2.18 AND core_mass < 0.02 AND exactness green AND
val < 0.51. Expectation: radial GO, frame partial (first shell only) — if BOTH miss on the peak,
the residual is context, not chart, and the next lever is the body/features, not heads.
