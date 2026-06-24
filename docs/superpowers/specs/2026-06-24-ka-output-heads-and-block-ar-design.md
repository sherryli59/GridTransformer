# Design: KA local-frame output-head comparison + block-AR (joint co-determination)

**Date:** 2026-06-24
**Branch:** liquid-coupling-flow
**Status:** design (approved in brainstorming; pending spec review)

## Motivation

The geometry-invariant local-frame model (`liquid_coupling_flow/ka_localframe.py`) **transfers the
conditional** across system size (teacher-forced clash << random at held-out N=36/256, no cliff) but its
**free-running generation fails** — overlaps 0.27–0.53 and a broken g_BB, *at the trained N too* — because
of rollout drift (exposure bias). A one-step teacher-forced diagnostic showed the current species-blind
head already reproduces species-resolved structure given a clean prefix, so drift is the dominant lever.

Two open questions remain that 1-step TF cannot answer:
1. **Output-head coupling.** The current head is fully factorized with **species ⊥ position**. Does
   coupling species and the relative position help once errors are allowed to *compound*?
2. **Block-AR.** Can **joint co-determination** of several atoms at once (so they see each other) attack
   the 50%-future-neighbor blindness that drives the drift?

This spec defines an experiment to answer both, on a yardstick stronger than 1-step TF, while preserving
the two hard requirements that define the project.

## Hard constraints (carry-through, every variant)

- **Exact likelihood.** Heads: per-particle coordinate change `(a,b)->x` has |det|=1 (unchanged). Block
  flow: exact log-density via the flow log-det. Required for downstream SMC importance weighting.
- **Geometry-invariant conditioning.** No curve positional encoding; condition only on frame-relative
  neighbour positions + species. This is what makes the conditional size-transferable; must not regress.
- **Evaluated at trained N=100 AND held-out N=36/256.** Transfer stays a co-goal, not an afterthought.

## Evaluation (shared harness, primary deliverable)

**Primary — k-step partial rollout.** Generalizes the 1-step TF check into a controllable drift dial:
- From real configs (curve-ordered), seed the model with the **true prefix** up to a start index `p`.
- **Free-run** the model for `k` steps, generating atoms `p..p+k-1` from their own (drifting) prefix.
- Measure on the freshly-generated atoms vs data: **clash fraction**, **g_BB(r<0.88)** spurious B-B
  contacts, and **g_AA/g_AB first-peak** fidelity.
- Sweep `k ∈ {1, 2, 4, 8, 16, 32}`, averaged over many start indices `p`.
- Output: metric-vs-`k` curves per variant. **Winner = slowest degradation with k.** k=1 reproduces the
  1-step diagnostic; large k approaches full free-run.

**Secondary — exact held-out NLL/N** (free from `log_prob`): a clean density-fit scalar per variant.

Both reported at N=36/100/256. The same harness scores Phase A heads and Phase B blocks identically.

## Phase A — single-particle output heads (do first)

Minimal deltas on the existing local-frame binned-AR. `c` = local context from `_local`. Selected by a
`head_mode` flag on `KALocalFrameModel`; the `(a,b)->x` coordinate change (and thus exactness) is identical
across all three.

| ID | Factorization | Tests |
|----|---------------|-------|
| **H0** baseline (current) | `P(s\|c)·P(a\|c)·P(b\|a,c)` | species ⊥ position |
| **H1** species→position | `P(s\|c)·P(a\|s,c)·P(b\|s,a,c)` | coupling via species embedding fed into position heads |
| **H2** joint (s,a) | `P(s,a\|c)·P(b\|s,a,c)` | explicit 384-way joint softmax over (species × a-bin), then b given (s,a) |

Notes:
- **H1 and H2 are the same model family** (any joint over (species, a-bin)); H1 is the AR / additive-
  embedding parameterization, H2 the explicit joint-softmax. So **H0 vs {H1,H2}** tests whether coupling
  matters; **H1 vs H2** tests whether parameterization matters (the smaller effect).
- **Fully-joint (s,a,b)** is rejected: a 2·192·192 ≈ 74k-way head ≈ 14M params, larger than the model.
- Each head keeps b autoregressive on its predecessors (`b|...,a`), which is exact and loses no
  expressiveness (P(a)P(b|a) is a universal 2D density).

**Deliverable A:** k-step rollout curves + NLL for {H0, H1, H2} at N={36,100,256}, and the verdict on
whether species↔position coupling slows drift onset.

## Phase B — block-AR joint co-determination (after Phase A)

Gets its own detailed spec/plan once Phase A picks the within-block head. High-level approach:
- Curve-order atoms; group into **blocks of K consecutive** atoms. Autoregressive **over blocks**; within
  a block, a **conditional normalizing flow** models the **joint** over the K atoms' (species, local
  positions) given the **pre-block prefix** context only (so the K atoms are co-determined and see each
  other symmetrically — the drift attack).
- Origins of the K block atoms depend only on the pre-block prefix → geometry-invariant, exact
  (per-block flow log-det). Species: joint categorical over the block (small K) or within-block AR.
- **Sweep K ∈ {2, 4}.**
- **Risks to manage in the Phase-B spec:** (1) flow stability/exactness; (2) size-transfer of a flow —
  global flow-matching did *not* transfer, so the block flow must condition only on geometry-invariant
  local features and stay local/small; (3) the within-block head reuses Phase A's winner.

**Deliverable B:** same k-step rollout + NLL harness applied to the K∈{2,4} block flows vs the Phase-A
single-atom winner.

## Components / files

- `ka_localframe.py` — add `head_mode ∈ {factorized, species_pos, joint_sa}` to `KALocalFrameModel`;
  branch `log_prob`/`sample`/`_step` accordingly. Persist `head_mode` in checkpoints.
- `ka_localframe_kstep.py` — **new** k-step partial-rollout eval harness + NLL; emits comparison plots.
- `ka_localframe_block.py` — **new (Phase B)** block-flow model; interface stubbed now, built after Phase A.
- A small comparison driver to train {H0,H1,H2} and run the eval.

## Efficiency

- **KNN=8 + bf16 autocast** (optionally `torch.compile`) → ~30 min/train (vs ~85 min at KNN=16 fp32).
- Reuse the hardened `train()` (periodic checkpoint every eval + OOM-safe N=256 eval).
- Phase A = 3 trainings; Phase B = 2 (K=2,4).

## Testing / validation

- **Exactness per variant:** geometry round-trip + a `sample()`↔`log_prob()` self-consistency check
  (the open item we flagged) — sampled configs' empirical log-density must match `log_prob`.
- **Regression:** H0 must reproduce the current N=100 numbers (TF-clash ~0.09, FR overlap ~0.27).
- **Harness determinism:** fixed seeds; k-step curves reproducible.

## Out of scope (YAGNI)

- SMC integration (the separate, subsequent go/no-go experiment).
- Fully-joint (s,a,b) single-particle head (parameter-prohibitive).
- Wider single-atom context window (only if a head result motivates it).

## Decisions locked in brainstorming

- Primary metric = **k-step partial rollout** (+ exact NLL secondary).
- **Phase A (heads) before Phase B (block flow).**
- Phase A heads = **H0, H1, H2**; **H2 = joint (s,a) then b|s,a**.
- Block size **K ∈ {2, 4}**; block = **joint co-determination** (local flow).
- **Exact likelihood + size-transfer** mandatory in every variant.
