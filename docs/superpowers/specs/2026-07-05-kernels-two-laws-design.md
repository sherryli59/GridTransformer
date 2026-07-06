# Kernels that obey the two laws — making learned generators earn their keep in MCMC/IS/SMC — design spec

**Date:** 2026-07-05 (user-designed umbrella, controller-refined). **Branch:** liquid-coupling-flow.
**Target:** 2D Kob-Andersen glass (65:35 A:B, ρ=1.2, T*=0.5, N=100/256, deterministic k=7 clusters).

## The two laws (distilled from the campaign's negatives)

Every learned deployment that FAILED asked the model to *sample precisely from scratch* and either used the
sample as an IS weight or forced it into the population; the precision wall (~0.1 placement in a dense glass)
then killed it. Every deployment that WON used the model for the *slow mode* behind an *exact corrector*:
swap-and-breathe (species channel, 12.4× via I-MTM, MH-guarded) and flow seeds (warm-start head-start).

1. **Never force a learned move.** Learned transport belongs behind accept/reject (MH) or a per-rung-exact
   weight (cSMC / trained AFT), never as always-applied IS transport. (P2 verdict [[local-smc-pilot]]:
   forced moves pay ×p ancestral diversity per move; MH merely pays acceptance rate.)
2. **Aim at the slow mode.** A low-acceptance move is worth it *iff* it decorrelates a mode MC can't
   (species, activated basin hops). 1% acceptance × structural decorrelation = win; 1% × local displacement
   = useless (MC does it free). [[swap-breathe-kernel]]

Corollary the campaign proved repeatedly: **these models are good DENSITIES (data-fit) and bad SAMPLERS
(generation).** ARM-FULL is "diffuse-but-calibrated" (Hastings +4.26: it knows the true cluster is likelier
than its own draws) [[cluster-move-gate-nogo]]; the non-causal full-cage conditional reaches g_BB 2.00 (data
2.40) yet pure Gibbs collapses without an energy guard [[full-cage-lever-needs-energy]]. Deploy them as
scorers / conditionals under a corrector, not as generators.

## Success criteria (umbrella)

- **Primary (depth):** SMC end ⟨U⟩/N −3.137 → −3.26 at N=100 at matched wall-clock (close the mutation-limited
  gap that ARM-0 left open).
- **Transfer:** N=256 zero-shot gap < 0.12/N vs the *rebuilt sharp* reference (all learned parts N=100-trained).
- **Amortization:** one kernel set serves all β-rungs with no per-rung retraining; acceptance profile roughly
  flat across rungs (the anti-init-dependence property SB lacked).

## Stage 0 — PT-ladder dataset (the owed N=256 reference, instrumented)

`ka_reference.py` currently simulates all M rungs but persists only cold `x[0]` (:72,:112). Change:
- **Save all rungs:** snapshot `x[:]` per recorded sweep → `pt_ladder_{N100,N256}.pt` = {β_rung → ~thousands of
  decorrelated configs}. One run buys three dividends: (i) trustworthy N=256 reference [[ka-reference-underconverged]]
  (prerequisite for every sharp transfer claim), (ii) A2 training data (per-rung conditionals), (iii) Stage-B
  event-mining raw material.
- **Longer equilibration** per the under-convergence diagnosis (N=256 true eq ≈ −3.26, current −3.21).
- **Convergence GATES (honest, all must pass before the dataset is trusted):** seed-pair agreement (|Δ⟨U⟩/N|
  < 2e-3), drift-free tail (2nd-half vs 1st-half slope ~0), finite-size direction sanity (N=100 vs N=256
  intensive, correct sign), exchange acceptance in-band across the ladder.
- Output also records per-rung β, so downstream FiLM conditioning reads β directly.

## Stage A1 — cSMC cluster move around existing ARM-FULL (NO retraining)

**Refactor** `ClusterProposal`: hoist the shared per-step body of `sample`/`log_q` into
`_step(i, placed_u, placed_sp, ctx_tok, q_scaf, ctx_u, ctx_sp, sp) → (ctx, sample_step, logprob_step)`.
`sample`/`log_q` call it in a loop (old outputs BYTE-IDENTICAL, regression-tested); the cSMC kernel calls it
per filter step.

**Kernel** (`ka_cluster_csmc.py`), a k-step particle filter placing one cluster given frozen environment:
- M candidates per step; **proposal** = ARM-FULL's per-step conditional (exact per-step log_q from `_step`);
  **incremental potential** g_i = exp(−β·ΔU_inc), ΔU_inc = interaction of candidate i with all already-placed
  cluster particles + all environment particles. LJ is pairwise-additive so Σ_i ΔU_inc = U_cluster(incl. env
  coupling) exactly. **Compute ΔU_inc IN-FRAME** (rigid transform preserves distances; reuse `ctx_u`/`placed_u`,
  no lab round-trip).
- **Resample between steps** (systematic/multinomial); **the current cluster is retained as the reference path**
  (particle Gibbs / Andrieu–Doucet–Holenstein) ⇒ the kernel is EXACTLY π(x_C | x_env)-invariant, **no outer MH**.
  Clash candidates die at their own step; the 0.72⁷ per-particle compounding that poisoned single-shot cluster
  moves is gone by construction.
- **Ancestor sampling (PGAS, Lindsten 2014) built in from the start** (flag, default on): resample the reference
  path's ancestor at each step. Near-free at k=7; immunizes the mixing gate against reference-path *sticking*
  (which would otherwise read as a false "move doesn't help").
- Placement order = existing fixed slot order (seed + scaffold distance); cSMC is valid for any fixed order.

**Validity gates (before any benchmark):** M=1 reduces to the plain single-shot move (exact); species/count
tripwires; **stationarity-from-reference** (start at a reference config, run the kernel only, marginals invariant).
**GA1 (utility):** rearrangement-per-cost vs MTM at β=2; then depth-at-matched-wall-clock inside the SMC stack
(drop cSMC into the [[local-smc-pilot]] mutation slot beside displacement).

## Stage A2 — β-conditioned full-cage single-site heat-bath (the ONE new model)

Purpose-built per-site conditional q(x_i | cage, species, β) (`ka_heatbath.py`):
- **Local-frame encoder**, degree-capped k-NN (the block-transfer size lesson [[ka-block-transfer]]); **spline head**
  for exact per-site density (must score an arbitrary x_i, not just sample); **β via FiLM** embedding; trained on
  ALL Stage-0 rungs (per-rung NLL), cold-rung loss up-weighting if needed.
- **Kernel** = single-site MH, k=1 ⇒ zero compounding:
  α = min(1, e^{−βΔU} · q(x_i | cage, β) / q(x_i′ | cage, β)).
  The non-causal experiment already proved this conditional class is sharp once it sees the full cage (g_BB 2.00);
  the MH guard supplies exactly the energy check whose absence made pure Gibbs collapse [[full-cage-lever-needs-energy]].
- **EXACTNESS INVARIANT — frozen cage (named, tested):** the k-NN cage AND its induced local frame are computed
  from the OTHER particles only, IDENTICAL for the forward proposal of x_i′ and the reverse density of x_i. Never
  from the moving particle's own position (that shifts the neighbor set/frame under the move and breaks detailed
  balance silently — the arcnorm-anchor / EGNN-frame-mix failure class [[arcnorm-anchor-bug]] [[traceable-egnn]]).
  Enforced by a detailed-balance symmetry test (forward×reverse ratio consistency) in the gate suite.

**GA2 gates:** per-rung held-out NLL + single-site clash rate; acceptance × mean-jump vs displacement sweeps at
matched cost; in-SMC depth (partial ≤ −3.20, full −3.26); zero-shot N=256 gap vs the rebuilt reference;
amortization (acceptance profile ~flat across rungs).

## Stage B — learn the move (CONTINGENT, pre-committed, not built unless triggered)

**Trigger:** A closes < ~half the gap, OR acceptance is high but depth plateaus (⇒ the missing class is collective,
not single-site). **Then:** mine rearrangement events from Stage-0 trajectories (displacement bursts within a
neighborhood between decorrelated snapshots); train a two-way block conditional q(x′_block | x_block, env) with
exact forward/reverse log_q; MH-guarded (law 1). **Fallback** if PT exchanges yield too few clean local events:
harvest events from swap-MC runs just above T*.

## Risks (honest)

- cSMC reference-path + ancestor-sampling bookkeeping is subtle → M=1 reduction + stationarity-from-reference
  gates BEFORE any utility benchmark; PGAS on by default.
- Stage-0 rebuild is days-scale GPU, but already owed and triple-purpose.
- A2 may win acceptance yet not depth if the tail is collective — that is NOT a failure, it is B's trigger firing
  with its dataset already in hand.
- Spline training across rungs may need cold-rung loss weighting.

## Build order

Stage 0 launched FIRST (long run equilibrates while A1 is built) → A1 refactor + cSMC kernel + gates (buildable
now, reuses ARM-FULL + I-MTM + SMC harness) → A2 (needs Stage-0 data) → B (contingent). Every stage has a
pre-committed gate and a durable-results obligation ([[record-simulation-data]]).
