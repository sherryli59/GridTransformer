# Scoping: is cavity-PTS-with-learned-samplers worth continuing, and how to tailor the samplers to basin search?

**Date:** 2026-07-16. **Status:** decision document. Every claim below is traceable to a measured number from
this campaign (file refs inline). Question posed by user after the flow-seed tie (flow_vs_random_relax.out);
user hint: "when doing suffix SMC we do see some basin crossing at small lambda."

---

## 1. The premise, stated falsifiably

Original premise: *a learned cavity-conditioned sampler can replace or dramatically accelerate the exact-MCMC
measurement of G_PTS(R) in the KA glass.*

That premise, in its direct form ("the model samples, exactness fixes it up"), is now **refuted five
independent ways** (§2). The scoped question is whether a *narrower* premise survives:

> **Learned components deployed inside an exact (T,λ) ladder, targeted at the two measured bottlenecks —
> top-of-ladder decorrelation and basin discovery — with exchange (never importance weights) as the
> transport mechanism.**

## 2. Evidence review — every arm, by sub-problem

The task decomposes into three sub-problems. Each failure in the campaign is a failure of exactly one.

| Sub-problem | What it needs | Verdict |
|---|---|---|
| **(i) Validity** (clash-free configs) | beat r⁻¹² | learned models fail directly; cheap classically |
| **(ii) Basin search** (find/cross to the boundary-selected state) | rare-event proposal + recognition | THE bottleneck; learned crossing real **only at small λ** |
| **(iii) Exact transport/weighting** | carry (ii) to λ=1 without bias | IS = extensive wall (×4); exchange works but diffuses |

### Arms tried and their measured verdicts

| Arm | Sub-problem attacked | Measured result | Failure mechanism |
|---|---|---|---|
| AR direct sampling @λ=1 | i+ii | raw U/n +36..+1e11 (relax_race) | half-cage AR → clashes |
| AR as relax seed | ii | ties random, all budgets | basin search unaffected by seed |
| eRSI flow (uniform base) @λ=1 | i+ii | coarse shells ✓, hole filled; +5.6/p; ties random; survivor q_c 0.50 vs 0.66 | FM-L2 can't build r⁻¹² wall; samples *generic* packing, not *the* basin |
| Island/stratified SMC | iii | ESS→1, family bleed; 0.345 vs truth 0.77/0.54 | extensive weight variance (4th appearance) |
| Interior blob MTM @λ=1 | ii | acc 0.000@λ=0, 0.28–0.34@λ>0 | q₀-tax (AR context shift) + thin-target |
| **Suffix/Morton-tail kernel** | ii | **acc 0.948@λ=0**, 0.58–0.59 all λ; data-seed→0.584 (mixes); alien head frozen | clean q₀-Gibbs; crossing capability REAL at small λ; head never refreshed |
| Full-regen independence @high λ | ii | **never tested as a mutation**; p(ref basin)≈1e-3/draw; logq₀(ref) z=−0.5 (no underweight); tempered λ=1 weight favors ref by **+330..670 nats** once equilibrated | draw frequency + recognition timescale |
| FL/tether weights | iii (bypass) | validated vs PT at R=2.0 but w_ref=1 regime only; alien anchors +100–200 nats (strawmen) | user distrust upheld: completeness + tether F unauditable |
| Shrinkage-PT (ours) | ii+iii | correct (Eq.2, λ̃, exchange all verified); TRIPS=0 at our budgets; flow f(r) = severed sigmoid while acceptance lies at uniform 0.32 | **round-trip diffusion starvation**; BCY budget = 5e7 sweeps/cavity ≈ 10 round trips |
| Random-restart relax | ii | p=0.04@R=2.0, **0.00@R=2.3**; τ≈60–114k sweeps | basin search cost, pure |
| Plain MC below ξ | — | q_c=0.643 = BCY ✓ (3 routes agree) | no basin search needed below ξ |
| Learned heat-bath (A2), blockcond-FT, discovery+quench | kernels/discovery | genuine positives (depth gain @1.11×; dE/p 742→14; **basin discovery ≈16 quenches/cavity**) | the components that DID earn their keep |

### The λ-asymmetry (the user's remembered signal, confirmed and sharpened)

The record (ka3d-pts-smc memory, diag_tail_kernel/diag_basin_mass/diag_quench_recognition):

- **Small λ is where learned crossing works.** Suffix resample = exact conditional of q₀ → MH ratio telescopes
  → acc 0.948 at λ=0. Interior blob is rescued from 0.000 to ~0.3 only at low λ. "Redistribution move's home
  is LOW-LAMBDA tempered rungs" was already the measured design consequence.
- **q₀ genuinely finds basins.** True attraction-basin mass p≈1e-3 per full energy-free draw (quench-verified,
  not look-alikes); logq₀(ref) sits *inside* the draw distribution (z=−0.5) — no coverage wall.
- **The wall is RECOGNITION, not proposal**: the +330–670-nat acceptance advantage of the ref basin belongs to
  the *equilibrated* config; a raw cross-basin draw is many nats uphill and gets rejected. Basin identity
  becomes energy-visible only after within-basin equilibration (the recognition-timescale wall) — except via
  the **T=0 capped quench**, which recognizes in ~400 steps (U_IS separates 3–4σ).
- Today's addition: at λ=1 the *relaxation* bottleneck is the same basin search (~20k sweeps), which is why
  seeding (AR or flow) buys nothing — the seed changes (i), not (ii).

**Unifying diagnosis: every failure is the same failure.** Learned crossing exists at small λ; each arm then
tried to move it to λ=1 by a mechanism that breaks: importance weights (extensive-variance wall), ladder
diffusion (round-trip starvation), or one-shot generation at λ=1 (recognition wall / generic packing). The
premise doesn't need a better generator — it needs a better **transport** of the small-λ capability, and the
only exact transport that survives the evidence is **Metropolis exchange**.

## 3. Tailored designs (ranked)

### T1 — Generator as the ladder's top-rung regenerator ("J-walking-style PT") ← primary
Plain PT's cost is round-trip diffusion: N_indep ~ T·D/NR² (measured TRIPS=0 at 4e4 sweeps). Replace the top
rung's *dynamics* with **independence-MH from the AR model against the soft deformed target π_{λ_top}**:
- Exact: AR has 1-pass exact logq (its one unambiguous asset); acceptance = standard independence-MH.
- Evidence it will fire: suffix acc 0.948@λ=0 and 0.58 at all λ; blob rescued at low λ; soft cores forgive
  the model's 0.05σ-scale placement errors (the r⁻¹² objection vanishes off λ=1).
- Effect: the top rung decorrelates in O(1/acc) sweeps instead of never; the ladder becomes a one-way
  pipeline (fresh configs flow down), N_indep ~ T·D/NR — an **~NR× (≈20×) structural win** over round trips,
  on top of numba throughput. PT's exactness untouched (per-rung invariance is all that's needed).
- Risk: independence-MH acceptance may decay exponentially with n at fixed λ (extensive KL). This is THE gate
  (G1) — if acc collapses for n≳100 at every λ connected to 1 by a short ladder, T1 dies for the R≳ξ regime.

### T2 — Discovery-seeded multi-init PT ← combine with T1
Basin *finding* by dynamics is what PT pays 5e7 sweeps for; the classical pipeline already solves it: pool →
box-q̃ prescreen → ~16 capped quenches → recognition (measured, ~16 quenches/cavity). Seed replica stacks in
**every discovered basin**; exchange equilibrates populations; the certificate is stack agreement (dual-init
generalized to k-init). No FL weights anywhere. This was already written into memory as "REMAINING:
exchange/bookkeeping seeded in every discovered basin, not search" — it was never built.

### T3 — Generator-seeded tempered-transition move (single-chain, exact)
Neal-style composite move: up-anneal current config in λ, draw fresh from the generator at the top (exact
density ratio), down-anneal; accept the whole path with the telescoping product. Pays the recognition time
*inside* the move, so acceptance compares *equilibrated* configs — exactly what the +330–670-nat measurement
says is decisive. Risk: path-weight variance (a per-move, short-path cousin of the IS wall). Heavier; build
only if T1/T2 close.

### T4 — λ-conditioned rung-to-rung flows (CRAFT/AFT on the shrinkage ladder)
The FM failure was a 0.9σ transport; **rung-to-rung λ-transport is ~0.01–0.05σ** — squarely in FM's sweet
spot, and soft targets remove the r⁻¹² mismatch. Flows transport walkers between adjacent π_λ with small
incremental weights (AFT/CRAFT — already flagged in memory as "the only ARM-1 path"). Endgame lever for
shortening the ladder at R≳ξ where BCY themselves need ~ξ^{3/2} replicas. Build last.

### Explicitly NOT worth continuing
- Any generator sampling π_{λ=1} directly (5 refutations: AR raw, AR seed, flow raw, flow seed, one-shot IS).
- Any IS/SMC transport of model density across rungs without exchange (4 refutations of extensive weights).
- Repulsive-prior retrain *for its own sake*: fixes g(r) cosmetics, not basin search (today's tie proves
  clash-freeness isn't the binding constraint).

## 4. The competitive bar and the unique-value question

- The bar is **numba plain-PT** (5,117 sweeps/s single-replica ≈ 30 CPU-h per BCY-budget cavity, parallel
  across cores), not our dead GPU-python strawman. Any learned addition must beat *that* at matched compute.
- BCY already own the measurement at 2M CPU-h. Reproduction is calibration, not contribution. The learned
  stack's structural advantages are exactly two:
  1. **Amortization across disorder** — one trained model serves every cavity × R × (eventually) T; BCY pay
     full price per cavity. (This is where T1/T2's per-cavity marginal cost ≈ 0 matters.)
  2. **The R≳ξ frontier** — BCY's own fine-tuning bottleneck (~ξ^{3/2} replicas, 98%→ threshold failures,
     T=0.45 budgets exploding). If T1+T2 cut certificate-closure time ≥5× there, that's a real paper.
- Prerequisite work item either way: an equilibrated **ρ=1.2, T≈0.5, N=4096** box (today's R=2.6 crash:
  2R+r_c=7.70 > L=7.53 for N512). The mosaic T=0.55 bank is not a substitute (v2 PT diverged on it).

## 5. Staged gates, kill criteria, verdict

| Gate | Experiment | Cost | Pass | Kill |
|---|---|---|---|---|
| **G1** | acc(λ) of full-regen + suffix independence-MH (AR exact logq; flow secondary) on deformed targets, R=2.0 & 2.6, n up to ~100 | hours | some λ* with acc ≥ 5–10% AND short ladder λ*→1 | acc collapses with n at every usable λ |
| **G2** | numba PT vs numba PT + T1 regenerator: round-trips/N_indep + dual-init closure time, R=2.0 | ~1 day | ≥5× closure speedup at matched compute | <2× |
| **G3** | + T2 multi-init at R=2.6–3.0 (multi-basin regime, new bank): k-init agreement | ~1–2 days | certified G_PTS points where plain PT (matched compute) fails to close | no advantage over plain PT |
| **G4** | CRAFT rung-flows to shorten the R≳ξ ladder | ~week | rung count ↓ ≥2× at fixed exchange acc | FM per-rung fails |

**Verdict: conditionally worth continuing.** The generator-as-sampler premise is dead and should stay dead.
The regenerator/discovery premise (T1+T2) is alive: every ingredient is individually measured (suffix 0.948,
p≈1e-3, quench recognition, exchange exactness, numba throughput), the mechanism it attacks is the measured
bottleneck (round-trip starvation + basin finding), and the transport is exact by construction. It is also
cheap to falsify: G1 is hours, G2 a day. If G1 or G2 kill, the honest endpoint is a negative-results
methodology synthesis (extensive-weight wall ×4, recognition-timescale wall, acceptance-lies-flow-tells,
generation-can't-shortcut-basin-search ×5) plus the validated classical pipeline — which is a real, if
unglamorous, contribution.
