# Mitigating exposure bias in the local-frame AR generator — Design Spec

**Date:** 2026-06-25
**Status:** DRAFT (awaiting user review)
**Context:** KA 2D glass (65:35 binary LJ, ρ=1.2, T\*=0.5), N=100. Target model: the **causal**
geometry-invariant `KALocalFrameModel` (`liquid_coupling_flow/ka_localframe.py`), from-scratch AR
generation, **in-distribution N=100 first** (size transfer is a documented follow-on, not this gate).

## 1. Motivation & goal

From-scratch AR generation places particle `j` conditioned on its own previously-placed prefix
(soft-centroid origin + KNN, [ka_localframe.py:108-132]). At training the prefix is the true data
(teacher-forced); at inference it is the model's own samples. The current result shows **significant
exposure bias** — the free-run (FR) placement clash/overlap rises above the teacher-forced (TF) level,
worst at curve closure (`j>80`). The **standard noise-scheduling fix already failed**:
`KALocalFrameSS` (`ka_localframe_ss.py`) perturbs the prefix with isotropic Gaussian noise and did not
close the gap.

This campaign tries two literature methods, **adapted to this model**, with a deliberate ordering set by
the user's research taste:

- **Primary — geometric soft labels (RQ-Transformer, target-side).** Soften the one-hot `(a,b)`-bin
  target into a distance-weighted categorical, plus stochastic target sampling. Bet: the head currently
  fits a *spiky* per-step conditional on a fine continuous-coordinate grid; geometric smoothing yields a
  *calibrated* conditional, improving free-run quality directly and adding drift-robustness.
- **Secondary — scheduled sampling (GapDiff, input-side).** Feed the model its own one-step placements
  (not Gaussian noise) into the conditioning prefix, on a GapDiff arc-annealing schedule. Targets the
  drift term specifically.

They attack **different terms** of the TF/FR decomposition (conditional quality vs accumulated drift) and
should compose; we validate independently before combining.

### 1.1 Why soft labels are a stronger bet here than in RQ's setting

The `(a,b)` head bins a **continuous physical coordinate** into `n_bins=192` fine bins
(`bin_w = 2·arc_range/n_bins ≈ 0.031` normalized units, [ka_localframe.py:75]). The true conditional
`p(a|cage)` is smooth, but a one-hot target on a *single* sample teaches a delta and treats neighbor bins
(physically almost-correct) as equally wrong as far bins. On a fine grid with one-sample targets that is a
high-variance, spiky histogram estimate. A geometric soft label is **kernel-density smoothing of the
target** — it shares statistical strength across neighbor bins for a calibrated density per unit data.
RQ's own ablation (soft-label-*alone* hurt, FID 14.06→14.87) came from a **discrete VQ codebook** where
neighbor-code distance is not literal geometry; that caveat is much weaker here, where neighbor-bin
distance *is* physical distance. **Failure mode kept honest:** over-smoothing (τ too large) broadens the
proposal → *more* overlap/clash. So the central experiment for this arm is a **τ-sweep** with
over-smoothing tracked.

## 2. The core invariant (causal analog of the MH-kernel invariant)

**Neither feature may touch the inference path.** Feature B changes only the **training target**; Feature
A changes only the **training input** (the prefix). `sample()` ([ka_localframe.py:214-245]) and
`log_prob()` ([ka_localframe.py:134-160]) stay byte-for-byte unchanged. Consequently the model's
**exactness gate** — `sample(return_logq=True)` returns `log_prob(pos,sp)` exactly
([ka_localframe.py:240-244]) — must still pass identically after each feature. That gate replaces the
MH-kernel's detailed-balance check as the guardrail. Both features default **off**; at their trivial
config each must be a numerical **no-op** equal to the current one-hot NLL / teacher-forced path.

## 3. Success criteria

**Co-primary (both required), all on free-run N=100 samples:**
- **Absolute FR quality:** `g_BB` peak rising toward data (~2.40) and spurious `g(r<0.88)` falling;
  FR placement-clash *level* (mean over `j`, and the closure mean `j>80`) falling. (0.88 = σ_BB, the B–B
  LJ contact diameter — a physical cutoff, not a knob.)
- **FR→TF drift gap:** `FR_clash(j) − TF_clash(j)`, especially at closure.

These are deliberately **co-primary**: soft labels are expected to lower **both** TF and FR (a better
conditional), so the drift *gap* may not shrink even when absolute FR quality clearly improves — judging
soft labels by the gap alone would read a real win as a null. Scheduled sampling is the opposite (closes
the gap with TF ~fixed). Reporting both keeps attribution honest.

**Diagnostic (not a gate):** per-index TF/FR overlap and median min-NN-to-predecessors; early/mid/late
split.

## 4. Feature B — geometric soft labels + stochastic target sampling (PRIMARY)

### 4.1 Mechanism
Acts on the factorized head ([ka_localframe.py:150-152]): `la = log_softmax(head_a(ctx))`,
`lb = log_softmax(head_b(ctx + bin_a_emb(ba)))`. Let `a* = ab[...,0]`, `b* = ab[...,1]` be the continuous
normalized targets (`ab = wrap(xo − origin)/arc_scale`). Replace the one-hot gather with a **soft target**
over bins, per axis:

```
P_a(i) ∝ exp(−(c_i − a*)² / τ),   c_i = _bin_center(i),   loss_a = −Σ_i P_a(i)·la(i)
```

and `P_b` analogously, conditioned on the a-bin used for `head_b`. Equivalent to KL up to the (constant)
target entropy.

- **τ parameterized in bins.** Expose a config `soft_sigma_bins` (the soft width in *bins*); set
  `τ = (soft_sigma_bins · bin_w)²`. Interpretable for the sweep; default `0` ⇒ one-hot.
- **Boundary handling.** Renormalize `P_a, P_b` over the in-range bins `[0, n_bins)` (consistent with
  `_bin`'s clamp). Data rarely approaches the edge (arc_range over-covers), so this is near-inert, but it
  is implemented and asserted.

### 4.2 Stochastic target sampling (the RQ component that moved the needle)
Instead of always binning to the containing bin, **sample** the target a-bin `ã ~ Q_τ(·|a*)`; use
`bin_a_emb(ã)` for `head_b` conditioning **and** as the a-target; then sample `b̃ ~ Q_τ(·|b*, ã)`. With
stochastic sampling off (soft-only), `head_b` conditions on the deterministic containing bin `_bin(a*)`,
as today.

### 4.3 RQ 4-way ablation (mandatory — soft-alone can hurt)
Run `{neither, soft-only, stochastic-only, both}` and **measure** rather than assume both-is-best, plus the
`soft_sigma_bins` sweep (e.g. `{0.5, 1, 2, 4}` bins) on the best ablation cell. **Early gate:** if an arm
*raises* FR clash (the over-smoothing / RQ soft-only risk), flag and do not promote.

### 4.4 Implementation shape
A training-only subclass / loss (mirroring how `KALocalFrameSS` adds `log_prob_ss` beside the base
`log_prob`). The base `log_prob` / `sample` are untouched, so inference and the exactness gate are
unchanged. Warm-start from `ka_localframe_N100_20k.pt` and/or train from scratch (decide per result).

## 5. Feature A — self-conditioned scheduled sampling (SECONDARY, drift-specific)

### 5.1 Mechanism (GapDiff, own-placements not noise)
The genuinely new content vs the failed `KALocalFrameSS`: feed the model's **own placement errors**, not
isotropic noise. **A1 — two-pass detached (primary):**
1. **Pass 1 (`no_grad`, teacher-forced):** `ctx,origin = _local(xo,...)`; sample each particle's own
   one-step placement `x̂_j` from the heads (origin + (bin_center+jitter)·arc).
2. **Mixed prefix:** per-particle `Bernoulli(1−p_T)` mask `m_j`; `xmix_j = m_j ? x̂_j.detach() : xo_j`.
3. **Pass 2 (grad):** `ctx,origin = _local(xmix,...)`; target `ab = wrap(xo − origin)/arc` (TRUE position
   vs the drifted origin — the `KALocalFrameSS` pattern); NLL (soft-NLL if combined with B).

No backprop through sampling / through the chain (GapDiff's explicit requirement; bounds cost to ~2×).
Causality is preserved (particle `j`'s context uses only `k<j`; we swap whole-particle positions).

**Schedule.** Arc/quarter-circle anneal `p_T(ρ) = max(√(r²−(rρ)²)/r, floor)` with progress
`ρ = step/total`, `r=2`, `floor=0.5`, `p_T(0)=1` (floor engages in the last ~13% of training). `r`,`floor`
exposed; floor enforced (never `p_T→0`) per GapDiff's bias-reinforcement guard.

### 5.2 Staged escalation (gated, not v1)
**A2 — full rollout** (real `sample` loop, mixing own/true placements as the chain proceeds, loss on true
targets) reproduces *accumulated* multi-step drift. **A1 injects only one-step errors** (each `x̂_j` is
placed from the *true* prefix), so it may under-represent the **closure** input distribution where drift
is worst. **Decision rule:** if A1 closes early/mid but the closure gap survives, escalate to A2 — that is
the AR↔diffusion mapping subtlety made measurable, not a failure. A3 (k-step hybrid) is a further option.

## 6. Phases & deliverables

- **Phase 0 — diagnostic + baselines (prerequisite).** Build a **local-frame** per-index TF-vs-FR
  clash/overlap diagnostic (the existing `ka_exposure_tf_vs_fr.py` / `ka_exposure_curve.py` target the
  *arcnorm gridformer*; the local-frame `_local`/`sample` paths differ — extend `tf_clash`,
  [ka_localframe.py:269-293]). Measure the **baseline** (`ka_localframe_N100_20k.pt`) and **re-measure the
  failed noise null** (`ka_localframe_ss_N100.pt`) on the *same* diagnostic as the documented control.
- **Phase 1 — Feature B (PRIMARY):** soft labels + stochastic sampling; 4-way ablation + τ-sweep.
- **Phase 2 — Feature A (SECONDARY):** A1 two-pass scheduled sampling; A2 staged per §5.2.
- **Phase 3 — combine** best-B with A; validate independently first; report nulls as legitimate.
- **Figure:** TF/FR clash-by-index for `{baseline, noise-null, +B(best), +A, +B+A}` with a `g_BB`
  (peak + spurious) panel and the data reference line.

## 7. Correctness & validation (each guards a distinct failure)

1. **Exactness gate (PRIMARY):** `sample(return_logq=True)` logq == `log_prob`, unchanged after every
   feature. *Guards: a feature leaking into the inference path.*
2. **Trivial-config parity:** soft loss with `soft_sigma_bins=0` + stochastic-off equals the current
   one-hot NLL to float noise; scheduled sampling with `p_T≡1` equals teacher-forced NLL. *Guards: silent
   non-no-op / silent always-on.*
3. **Soft-target normalization:** `P_a, P_b` sum to 1 over in-range bins (boundary renormalization
   asserted). *Guards: the geometric kernel + edge handling.*
4. **Inference source-diff:** `sample()` / `log_prob()` bodies are byte-identical to pre-campaign
   (mechanical check). *Guards: the §2 invariant directly.*

## 8. Scope

**In:** the local-frame TF-vs-FR diagnostic; geometric soft labels + stochastic sampling (4-way ablation +
τ-sweep); two-pass scheduled sampling (A1); the in-dist N=100 co-primary benchmark; all §7 checks.
**Out (explicit follow-ons):** size-transfer to held-out N (the endgame, gated separately); A2/A3
escalations (gated on the closure gap, §5.2); the residual/half-cage floor — the part neither A nor B can
fix, which needs the energy/full-cage corrector ([[full-cage-lever-needs-energy]]); we will only *label*
where the remaining gap is drift (addressable here) vs floor (not).

## 9. Deliverables

- A per-index TF-vs-FR clash/overlap diagnostic for `KALocalFrameModel` (new script or extended
  `tf_clash`).
- Soft-label training (subclass/loss beside base `log_prob`, inference untouched), config `soft_sigma_bins`
  + `stochastic_target` (default off).
- Scheduled-sampling training (own-placement two-pass, extends the `KALocalFrameSS` pattern), config
  `sched_r`, `sched_floor` (default off / `p_T≡1`).
- One figure (§6) + a short results report with the §3 co-primary numbers and an explicit drift-vs-floor
  attribution.
