# Curve-guided spline-flow placement head for the local-frame AR generator — Design Spec

**Date:** 2026-06-26
**Status:** DRAFT (awaiting user review)
**Context:** KA 2D glass (65:35 binary LJ, ρ=1.2, T\*=0.5), N=100. Pivot from training-side exposure-bias
mitigation (negative) to an **architectural** fix.

## 1. Motivation & goal

The exposure-bias campaign established, by direct measurement, that the AR generator's structural defect is
**the per-step conditional itself**, not rollout drift. The g(r) evidence (placed-vs-true, full cage):
contact peaks come out at the **right positions** but ~½ the **height**, with the excluded-volume core
**filled** (`g(r<σ)` ≈ 0.27–0.55 vs data ≈ 0.02), and **TF ≈ FR** (the data→TF gap dominates the TF→FR
drift). Two structural causes:

1. **A binned categorical placement head cannot represent a sharp contact peak + hard wall** — it is
   intrinsically a smoothing (192 bins, uniform jitter). This is the amplitude/core wall.
2. **The AR model conflates two jobs** — walking the Hilbert curve *and* reading the already-placed
   neighbors (whose meaning shifts with curve position). This is the known "AR over an unordered set"
   failure; capacity spent on the traversal starves the local conditional.

**Goal (in-distribution N=100 first):** make free-run g(r) recover **sharp contact peaks AND an empty
excluded-volume core**, by (B) replacing the categorical head with an **exact continuous** head sharp enough
to represent the hard core, and (A) **giving the model the curve** so its capacity goes to the local
conditional. Staged: **B first** (isolates the measured dominant lever), then **A**.

Inspiration (precise mechanisms pulled at implementation time): **Quetzal** (causal transformer → small
dedicated continuous position head; we keep the *structure*, not its diffusion likelihood), **QueryAR**
(learnable position queries for "where the next thing goes"), **InertialAR** (local inertial frames).

## 2. Core invariant

**Exact likelihood is preserved end-to-end** (the SMC corrector / importance weighting depend on it — this
is why the head is an exact spline flow, NOT a diffusion head). `ka_localframe.py` is **not edited**; the
new model is a subclass in a new file. The model's **exactness gate** — `sample(return_logq=True)` returns
`log_prob(pos, sp)` to float precision — must hold for the flow head exactly as it does for the categorical
head. Every variant defaults **off**; the validated local-frame model remains the untouched baseline.

## 3. Success criteria

**Primary (in-dist N=100, free-run):** partial g(r) recovers data structure — contact peaks rise toward
data (g_AB → ~7.1, g_AA → ~3.7, g_BB main shell → ~2.4) **and** the excluded-volume core empties
(`g(r<σ)` → ~0.02), measured with the TF/FR + peak-position tooling already built
(`ka_exposure_lf` / the structure-figure script). Peak **positions and count** are co-primary with height.

**Mandatory gate — training converges** (§6): the spline-flow head must train stably to a lower position-NLL
than the categorical baseline, with zero NaN/inf and the exactness gate intact. A sharper head that will not
train is a fail.

**Out of scope for the verdict:** size-transfer to held-out N (follow-on); SMC retuning.

## 4. Feature B — exact RQS spline-flow placement head (Phase 1, primary lever)

### 4.1 Mechanism
Replace the binned `(a,b)` categorical ([ka_localframe.py:150-152]) with a **conditioned autoregressive
rational-quadratic spline (RQS) flow** over the normalized offset `(a,b) ∈ [−arc_range, arc_range]²`,
factorized `p(a|h)·p(b|a,h)` (mirrors the current head; reuse `transforms_spline.py` + the `KAARFlow`
pattern, [ka_flow_ar.py:24]). Context `h_j` from `_local` conditions the spline parameters (Quetzal
structure: transformer context → tiny continuous head). The **species head stays categorical** (discrete;
exact; unchanged).

### 4.2 Exactness (drop-in for the categorical density)
`ab = wrap(xo − origin)/arc_scale`. Per-particle position log-density (physical):
`log p(x_j) = log p_flow(ab_j | h_j) − d·log(arc_scale)` — the continuous flow density replaces the
categorical `P(bin)/bin_area`; the `arc_scale` Jacobian is the same `jac` term as today, the `bin_w` `vol`
term disappears. Total `log_prob = Σ_j[ log p_flow(ab_j|h_j) + lp_s_j ] − N·d·log(arc_scale)`. Sampling:
`ab = flow.sample(h_j); x_j = remainder(origin + ab·arc_scale, L)`. The exactness gate holds because RQS is
exactly invertible and `flow.sample`/`flow.log_prob` share the transform.

### 4.3 Frame option (InertialAR-inspired) — a toggle, not the default
`frame_mode ∈ {scaffold, inertial}`. `scaffold` (default) = current global-orientation offset.
`inertial` = rotate `(a,b)` into a **local frame built from the placed neighbors** (orientation from the
neighbor geometry — e.g. the n1→n2 axis the original docstring described but the model never used, or PCA of
the KNN relative positions). The rotation is a function of *placed* particles ⇒ `|det|=1` ⇒ exactness
preserved; it makes the conditional rotation-aligned and potentially sharper. Tested as a variant against
`scaffold`.

### 4.4 Convergence safeguards (first-class, per the "make it converge" requirement)
- **Identity init** — the conditioned spline initializes to ≈ identity, so the head starts as its base
  distribution (≈ current behavior) and cannot blow up early.
- **Bounded domain + linear tails** — RQS on the box with linear tails ⇒ finite density everywhere, no NaN
  on out-of-range points.
- **Monotonicity guards** — softplus derivatives + minimum bin width/height (standard RQS), keeping the
  transform invertible and well-conditioned.
- **grad-clip 5.0** (codebase default), short **lr warmup**, AdamW.

## 5. Feature A — QueryAR curve conditioning (Phase 2, capacity-freeing)

Give the model the curve so its capacity goes to the local conditional. **Soft conditioning only** (the flow
domain is unchanged ⇒ exactness untouched):
- **QueryAR position query** — a learnable position-query token for the next particle's scaffold coordinate
  `s_j`, interleaved into the transformer input, so the model attends to an explicit "place-here" signal and
  the flow predicts the offset *given* the query.
- **Curve-rail x-attention + arc-length coords** — reuse the validated D2 rail + D3 arc-length features
  ([[curve-rail-validated]], [[turn-token-design]]).
- **Δs range** as a soft region-prior feature (the bounded "where can it land" signal). The **hard**
  Δs-bounded support (flow domain = curve region) is **Approach 3, explicitly deferred** (the per-particle
  data-dependent support complicates the exact Jacobian).

Measured as an increment over B-alone.

## 6. Convergence gate (explicit, blocking)

Before any flow-head variant is promoted past Phase 1:
1. **Overfit-tiny** — train on a handful of configs (~8) for a few hundred steps; flow NLL drops near its
   floor (proves the head *can* fit; no optimization pathology).
2. **NLL beats categorical** — full-run position-NLL/N decreases (smoothed) and is **lower** than the
   categorical baseline's *physical* position log-density (apples-to-apples: both are densities over x).
3. **No NaN/inf** in the spline log-det across the entire run (asserted).
4. **Exactness gate** — `sample` logq == `log_prob` (§2) still passes.
Any failure blocks promotion and is reported.

## 7. Testing & validation

1. **Spline round-trip unit test** — forward∘inverse identity + analytic log-det vs autograd (reuse
   `tests/test_spline.py` patterns). CPU, artifact-free.
2. **Exactness gate test** — `sample(return_logq)` logq == `log_prob` on a tiny random-weight model.
3. **Trivial-init parity** — at identity init the flow density is its base; assert a sane finite log_prob and
   a passing exactness gate (guards the wiring).
4. **Convergence gate** (§6) — run as part of training, reported.
5. **Structural metric** — in-dist N=100 free-run g(r) (peaks position+count+height, core fill) vs the
   categorical baseline and data, via the existing structure tooling.

## 8. Phases & deliverables

- **Phase 1 (B):** `liquid_coupling_flow/ka_flowhead.py` — `KAFlowHeadModel(KALocalFrameModel)` with the
  AR-RQS spline placement head (exact), `frame_mode` toggle (scaffold|inertial), convergence safeguards;
  unit tests; training + the §6 convergence gate; in-dist N=100 g(r) vs categorical baseline.
- **Phase 2 (A):** add QueryAR position query + curve-rail/arc-length + Δs-range conditioning; measure the
  increment on the same g(r) metric.
- **Figure/report:** free-run g(r) (AA/AB/BB, peaks marked) for {categorical baseline, +B(scaffold),
  +B(inertial), +A+B} vs data; convergence-gate evidence (overfit-tiny curve, NLL-vs-categorical, NaN-free).

## 9. Scope

**In:** the exact RQS spline-flow head (B) with the inertial-frame option; QueryAR soft curve conditioning
(A); the convergence gate; the in-dist N=100 g(r) verdict; all §7 checks.
**Out (explicit follow-ons):** size-transfer; the hard Δs-bounded support (Approach 3); SMC retuning;
diffusion-head variants (incompatible with the exactness invariant).

## 10. References (consult for precise mechanisms at implementation)

Quetzal — transformer + continuous position head (structure only; we use exact flow, not its diffusion
likelihood). QueryAR — interleaved learnable position queries. InertialAR — local inertial frames. GraphGPS —
positional/structural encodings. VAR — coarse-to-fine (the deferred Approach C). Existing repo building
blocks: `transforms_spline.py`, `ka_flow_ar.py` (`KAARFlow`), `ka_localframe.py` (`KALocalFrameModel`).
