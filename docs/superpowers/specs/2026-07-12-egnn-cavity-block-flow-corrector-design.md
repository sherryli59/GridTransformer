# Boundary-conditioned EGNN cavity block-flow corrector (Option B) — Design Spec

**Date:** 2026-07-12
**Status:** design (pre-plan)
**Branch:** liquid-coupling-flow

## Goal

An **exact-likelihood, full-cage, equivariant flow** that corrects the wrong *relative structure* of an
AR-generated cavity block, deployed as a **block MOVE** inside the PT/SMC cavity ladder. It replaces the
sharpened-AR block move (the `MSEED_MOVE` arm that won the three-arm frontier benchmark, commit ac1f9bd,
4/12 vs alien 1/12) with a learned equivariant transport that fixes the half-cage structure error the AR
cannot.

## Why this succeeds where 5 prior correctors failed

Each prior corrector died on ONE axis; B is the first to clear all four:

| corrector | died because | B's answer |
|-----------|--------------|------------|
| coupling-flow MLE (Tasks 1-4) | composed-MLE needs the frozen AR's non-smooth log-prob gradient | trained by TRANSPORT (flow-matching), never differentiates AR log-prob |
| deterministic relaxation | descent removes overlaps but cannot RESTRUCTURE; folds on r^-12 | a LEARNED velocity field restructures; exact divergence, no folds |
| per-particle / half-cage | each particle blind to half its cage | full-cage EGNN message passing (the g_BB 1.25->2.00 ingredient) |
| uniform-base coupling flow | can't build hard-core exclusion from structureless base | base = AR sample (species+coarse structure already right) => SHORT transport |
| anchor-lookahead / OT-deterministic | mean-collapse / loose anchors | conditional flow (per-cage) does not collapse |

## Architecture

### Base
An AR block sample `x0` (the trained `KA3DScaffoldEBMBatched`, ckpt `ka3d_cavity_ebm3ax_rho115_best.pt`),
with its exact `logq_AR(x0)` from `block_log_prob_b`/`sample_block_b`. Species come from the AR and are FIXED
through the flow (species proven fine: three-arm + corrector-diagnosis row C = +0.9/particle). `pos_temp`
sharpening on the AR base is allowed (exact for the tempered density).

### Flow
Cage-conditioned EGNN velocity field `v_theta(x, t | cage)` integrated `x0 -> x1` over `t in [0,1]`:
- **Cage context (frozen, not in log-det):** boundary shell particles + retained interior particles. They
  send messages, never move, never enter the divergence. Exactly `ConditionalEGNN`'s existing contract.
- **Full-cage among block particles:** block particles message-pass with EACH OTHER simultaneously (the
  non-causal property the AR lacks) plus the frozen cage.
- **Conditioning features:** species (node feature), R (global feature, like the AR's R_embed), t.
- **Exact log-det:** analytic k-NN traceable divergence (`egnn_traceable`, k-NN trim design
  2026-06-30-knn-local-traceable-egnn-design.md, exact to ~1e-6, linear-in-P => size transfer). ISOLATED
  (non-periodic) energy/geometry branch => dodges the periodic-branch bugs.
- **Integrator:** dopri5 on the augmented `[x_block, logdet]` ODE (`EGNNClusterFlow` already has this).

### Composed exact log-q (carry-the-latent)
The Markov state carries the AR pre-image `x0`. The corrected block is `x1 = Flow(x0)`.
`logq_composed(x1 ; x0) = logq_AR(x0) + integral_0^1 -div v dt`.
No inversion needed (carry x0). Used directly in the MTM/SMC weight `u = -beta*U(x1) + logq_composed`.

## Training objective (THE crux)

**Conditional flow-matching**, AR-block base -> data-block target, minibatch-OT coupling. Rationale:
composed-MLE is DEAD (AR-gradient wall). Flow-matching learns `v_theta` to match the transport vector field
from AR samples to data blocks; it NEVER touches the AR log-prob gradient. Exact log-q at inference comes
from the divergence integral (evaluated, not differentiated).

- **Coupling:** within a minibatch of (cage, AR-block, data-block) triples sharing the same cage, use
  minibatch-OT (Sinkhorn or exact small-batch) between AR samples and data blocks to define target pairings;
  avoids the mean-collapse that killed deterministic OT.
- **Loss:** rectified-flow / conditional-FM MSE on the velocity along the straight-line (or OT-map)
  interpolant `x_t = (1-t) x0 + t x1`, cage frozen.
- **Data:** carved cavities from the 112-chain training set `ka3d_train_N4096_T0.5_rho1.15.pt`; held = the 16
  reference chains (SAME chain-level zero-leakage split as the AR retrain). Multi-radius {1.6, 2.0, 2.5, 3.0}.
  For each cavity: sample AR blocks (the base) + the data block (the target).

## Deployment

Drop-in replacement for the block move in `frontier_threearm.py` / the PT+SMC ladder:
`AR-propose block (pos_temp) -> EGNN-flow-correct -> MTM/SMC accept with exact composed log-q`, at hot rungs
(beta-softened) so K>=6 blocks pass (confirmed usable, ac1f9bd). Also usable as a whole-interior seed
(block = all).

## Gates (in order; each blocks the next)

1. **Exactness:** analytic divergence vs finite-difference/brute-force autograd trace (n_species=2, double
   precision) to ~1e-6; composed-logq carry-the-latent round-trip.
2. **Structure fix:** flow-corrected AR block reaches the good basin the AR block cannot — beat the +6.9/particle
   inherent-structure floor (corrector-diagnosis) toward the +0.9 data floor; g_BB peak toward 2.0.
3. **Move quality:** flow-corrected block-MTM acceptance at the cold rung (beta=2) beats the sharpened-AR
   K>=6 acceptance (currently ~0); hot-rung acceptance up.
4. **Frontier:** convergence at R=2.5/3.0 beats the `MSEED_MOVE` baseline (>4/12) at matched budget.

## Reuse / file plan

- REUSE: `ka_cluster_egnn.py` (`ConditionalEGNN` frozen-cage context + cluster-restricted divergence,
  `EGNNClusterFlow` dopri5 augmented ODE, rep_prior), `egnn_traceable.py` k-NN analytic divergence,
  `KA3DScaffoldEBMBatched` (frozen AR base + sample/score).
- NEW: `liquid_coupling_flow/ka3d_cavity_egnn_flow.py` — cavity-block flow (base=AR, cage=boundary+retained,
  FM training + composed-logq inference). `reports/logs-2026-07-*/train_cavity_egnn_flow.py` — FM training
  (chain-level split). Deployment hook in the PT/SMC ladder.
- Ckpt: `liquid_coupling_flow/artifacts/ka3d_cavity_egnn_flow.pt`.

## Risks / open

- **Expressiveness on hard-core** (13.3% free-cluster history): mitigated by short AR->data transport + full
  cage context; Gate 2 is the decisive test.
- **OT coupling stability** at variable block/cavity size: fall back to independent coupling if Sinkhorn is
  finicky; measure.
- **Cost:** dopri5 + per-stage analytic divergence per move; k-NN trim keeps it linear-in-P. Budget vs the
  cheap sharpened-AR move must be justified by the convergence gain (Gate 4).
- **Species-fixed assumption:** re-audit that flowing positions at fixed AR species reaches low energy
  (row C says yes for DATA positions; confirm for flow-corrected).
