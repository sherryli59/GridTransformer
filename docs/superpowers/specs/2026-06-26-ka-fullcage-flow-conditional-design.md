# Full-cage spline-flow conditional (TF g(r) milestone) — Design Spec

**Date:** 2026-06-26
**Status:** DRAFT (awaiting user review)
**Context:** KA 2D glass (65:35 binary LJ, ρ=1.2, T\*=0.5), N=100.

## 1. Motivation & goal

The A+B campaign measured, in-distribution at N=100:
- **B (exact spline-flow head):** no improvement over the 192-bin categorical — position log-density 2.78→2.70,
  g(r) unchanged. **The head's expressiveness is not the bottleneck.**
- **A (curve/GPS conditioning):** a small, real positive — position log-density 2.70→2.90, g(r) cores −10% —
  but only a ~10% dent on a ~4× structural gap.

The conclusion: the conditional `p(offset | context)` is **broad because the context is missing
config-specific information** (the causal half-cage). The one lever prior work showed actually moves the peak
is the **full cage** — the non-causal conditional that sees *all* neighbours: NonCausalLF lifts teacher-forced
g_BB 1.25→**2.00** (toward data 2.40) ([[full-cage-lever-needs-energy]]).

**Goal (this milestone):** combine the **full-cage context** (NonCausalLF) with the **exact spline-flow head +
curve conditioning** already built, and measure whether the resulting full-cage *flow* conditional pushes
**teacher-forced g(r)** past NonCausalLF's categorical (g_BB ~2.0) toward data (2.4). This is the cheap, direct
test of "does a sharp head help once the cage is full" — before wiring it into the corrector.

**What this is NOT (scope guard):** not a generative model (a full-cage conditional is non-generative — every
particle depends on all others) and not a corrector. We never iterate the bare conditional (pure Gibbs
collapses, [[full-cage-lever-needs-energy]]); the energy-guarded MH kernel handles iteration in the Phase-2
follow-on.

## 2. Architecture & core invariant

`NonCausalCurveFlow(KACurveFlowModel)` (new file `liquid_coupling_flow/ka_noncausal_flow.py`) overrides **only**
`_local` with NonCausalLF's non-causal version (each particle's context = all neighbours except self,
referenced to the full-cage soft-centroid origin). Everything else — the exact spline-flow head
(`SplineFlowHead`), curve conditioning (`_curve_feat`), and the flow `log_prob` — is inherited from
`KACurveFlowModel` unchanged. So the **only** difference vs Phase-1 A is *what the conditional sees* (full cage
vs causal half-cage).

Rejected alternatives: multiple inheritance (`NonCausalLF` + `KACurveFlowModel`) — MRO fragility; a
`noncausal=` flag on `KACurveFlowModel` — edits the verified Phase-1 file. The single `_local` override is the
low-risk path.

**Core invariant — exact likelihood.** The flow density is exact; the full-cage `log_prob` is an exact
pseudolikelihood. Exactness is **required** for the Phase-2 MH-kernel proposal (the kernel's Hastings ratio
evaluates `q`). `ka_localframe.py`, `ka_flowhead.py`, `ka_curveflow.py`, `ka_noncausal.py`, and
`ka_mh_kernel.py` are **not edited**.

## 3. Training

Full-cage pseudolikelihood, from scratch (~20k steps; no warm-start — the full-cage context differs from
Phase-1's causal context):
`loss = -(m.log_prob(augment(data[idx], L), s, canonical=False) / N).mean()`
`canonical=False` (non-causal ⇒ no causal species-budget mask, exactly as NonCausalLF trains). Convergence
safeguards inherited from the flow head: identity-init spline, grad-clip 5.0, lr warmup, NaN guard.

**Convergence gate (blocking):** overfit-tiny NLL drops near its floor with no NaN; full-run NLL is finite and
decreasing. (No "beats categorical" gate here — the comparison is structural, §4.)

## 4. Milestone measurement — teacher-forced full-cage g(r)

For each particle, build its **full-cage context from the TRUE neighbours**, sample the offset from the flow,
place it: `ctx, origin = m._local(xo, so, sc, L, N)` (non-causal, all true neighbours); `ctx += _curve_feat`;
`ab = flow.sample(ctx)`; `x̂ = origin + ab·arc_scale`. Compute **cross g(r)** (placed-j vs the TRUE neighbour
positions — single noise source, no double-resampling) for AA/AB/BB: peak height, peak positions, and core
fill `g(r<σ)`.

**Reference rows:** data, NonCausalLF (categorical full-cage, g_BB ~2.0), and the full-cage flow. (NonCausalLF
must be trained/available; if its checkpoint exists reuse it, else train it the same way for the comparison.)

## 5. Success gate

**Full-cage flow TF g_BB peak materially exceeds NonCausalLF's ~2.0, toward data 2.4**, AND cores drop toward
data (~0.02), AND peak positions match data. If met → **Phase-2 follow-on:** wire `NonCausalCurveFlow` into
`ka_mh_kernel.py` as the proposal (replacing the categorical NonCausalLF), measuring corrector
acceptance/mixing/sweeps-to-data-g(r) — gated, separate plan; the kernel already exists. If not met → a sharp
head adds nothing beyond NonCausalLF once the cage is full, and the lever is the corrector/energy, not the
conditional.

## 6. Testing & validation

1. **Non-causal `_local` correctness** — the context for particle j includes all neighbours `k≠j` and excludes
   self (assert: perturbing `x_j` leaves `context_j` unchanged only via the self-mask; the gathered neighbour
   set has size `knn` from the non-self pool). Reuse NonCausalLF's masking.
2. **Exactness gate** — `log_prob` is finite and the flow head's `sample`↔`log_prob` round-trip holds
   (inherited `SplineFlowHead` property); a `_curve_feat`-zero parity vs the same model without curve.
3. **Convergence gate** (§3) — overfit-tiny + no-NaN, reported.
4. **Milestone structural metric** (§4) — TF cross g(r) AA/AB/BB peak/core/positions vs data + NonCausalLF.

## 7. Scope

**In:** `NonCausalCurveFlow` (the `_local` override), full-cage pseudolikelihood training + convergence gate,
the teacher-forced full-cage g(r) verdict, all §6 checks.
**Out (explicit follow-ons):** wiring into `ka_mh_kernel.py` (Phase 2, gated on §5); any corrector / SMC /
pure-Gibbs iteration (the conditional is non-generative — never iterated here); size-transfer; the
inertial-frame variant.

## 8. Deliverables

- `liquid_coupling_flow/ka_noncausal_flow.py`: `NonCausalCurveFlow` + training + the convergence gate.
- A TF full-cage g(r) eval (cross g(r) for {data, NonCausalLF, full-cage flow}) + figure.
- Tests (`tests/test_noncausal_flow.py`): non-causal `_local`, exactness, curve-zero parity.
- A results addendum (the §5 verdict: did full cage + sharp head beat 2.0 toward 2.4?).
