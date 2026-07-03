# ClusterProposal continuous-head upgrade — design spec

**Date:** 2026-07-03 (autonomous, per user directive: "spec the continuous-head upgrade... explore and compare
multiple options... mixture/flow head + scale + pair features; I'll be gone so just be autonomous")

## Goal

Revive proposal-A (`ClusterProposal`, the AR-in-frame cluster generator) as a competitive MH proposal by
replacing its binned placement head with an exact continuous head, adding pair-distance features, and scaling —
testing whether an AR transformer can approach the EGNN CNF's clash rates while keeping its decisive engineering
advantage: **exact log_q in one forward pass** (no ODE, no divergence integration, none of the integrator
machinery the EGNN's log_q required).

## Numbers that frame the target (all measured this campaign)

| generator | overall per-particle clash | cage | intra | log_q cost |
|---|---|---|---|---|
| A sharpened (1.9M, 64 bins) | 44% | — | — | 1 forward (exact) |
| EGNN small 0.13M | 18.1% | 12.8% | 7.0% | ODE + divergence |
| EGNN big 1.64M | **13.3%** | 7.8% | 6.8% | ODE + divergence |
| data | 0% | 0% | 0% | — |

MH acceptance (EGNN big): 1.02%; clash-free fraction 50%; the binding constraint beyond clash is fine energy
precision. An AR at EGNN-level clash would be preferred for the MH kernel on cost/exactness grounds.

## Why A's 44% is NOT evidence of an AR ceiling (three campaign findings)

1. **Capacity lesson (2026-07-03)**: "overfit==generalization ⇒ structural" was falsified for the EGNN
   (17.5→7.1% at 12× params). A's sharpening ablation (1.9M, 40k) predates this; its "diminishing returns"
   conclusion is equally suspect.
2. **Head mechanism**: A's sampled positions are `bin_center + U(-bin_w/2, bin_w/2)` — the density is FLAT
   inside every 0.094-wide bin. A hard-core boundary crossing a bin is unrepresentable: a built-in clash floor
   at exactly the precision scale the glass demands (~0.1).
3. **AR is structurally fine for clash avoidance**: each step conditions on the cage + ALL placed particles —
   everything it must avoid is visible. The half-cage/50%-future problem degrades joint ENERGY quality
   (acceptance|no-clash), not exclusion.

## Prior art in-repo (the user's remembered spline flow) — and the critical caveat

**`liquid_coupling_flow/ka_flowhead.py`** already contains `SplineFlowHead(d_model, num_bins, tail_bound)`:
an exact AR rational-quadratic-spline flow head (Durkan RQS via `transforms_spline.RQSplineElementwise`,
Gaussian base, linear identity tails, factorized `p(a|h)·p(b|a,h)`, identity-initialized). Reviewed, tested
(`tests/test_spline.py` passes today), trained once end-to-end.

**CAVEAT — the 2026-06-26 null result** (reports/2026-06-26-ka-spline-flow-head-results.md): on the FULL-CONFIG
local-frame AR generator, swapping the 192-bin categorical for this spline head changed NOTHING (position
log-density 2.70 vs 2.78; identical g(r)). Verdict then: the head was not the bottleneck — that model's
conditional is intrinsically broad (soft-centroid context, per-step half-cage). **Why it need not transfer
here**: the cluster task conditions on the FULL 48-particle cage — the EGNN proved this conditional is sharp
(cage-clash 7.8%, mean min-r 0.85 vs data 0.88). But the null result mandates two design decisions:
(a) the upgrade must BUNDLE head + context/pair features + scale (a head swap alone risks a repeat null if A's
context representation is the binding constraint); (b) the experiment must include a head-only ablation arm so
the outcome is attributable either way.

## Options compared

### Placement head (per AR step, 2D in-frame offset u ∈ ~[-3,3]²)

| option | density class | sharp boundaries | exact log_q | status | verdict |
|---|---|---|---|---|---|
| **O1: RQS spline head (reuse `SplineFlowHead`)** | monotone piecewise-rational CDF per dim, `p(a|h)p(b|a,h)` | YES — knots can place derivative kinks at contact edges | YES (closed-form) | EXISTS, tested, identity-init | **PRIMARY** |
| O2: mixture of K 2D Gaussians | smooth mixture | needs many components to fake an edge (Gaussian tails leak into the core) | YES | ~40 lines new | fallback if O1 trains unstably |
| O3: deeper per-step flow (2-3 stacked RQS layers + rotations) | as O1, more expressive | YES | YES | ~60 lines on top of O1 | escalation if O1 underfits (check via overfit probe) |
| O4: finer bins + learned sub-bin density | piecewise-linear | partially | YES | new code | dominated by O1 (RQS ⊃ piecewise-linear); rejected |

Decision: **O1**, `num_bins=16` (the 2026-06-26 run used 8 and flagged it conservative), `tail_bound=3.5`
(covers the box=3 in-frame range; linear tails + Gaussian base replace the old hard `-69` out-of-box penalty —
strictly better behaved for log_q of arbitrary MH reverse points). Escalate to O3 only if the Task-overfit
probe floors.

### Pair/distance features (the "context" half of the bundle)

| option | what | cost | verdict |
|---|---|---|---|
| **P1: per-step radial features on every token** | append `[|Δu|, 1/|Δu|² (clamped), σ_pair-scaled |Δu|]` of each token (cage + placed) TO THE CURRENT query slot's scaffold position, recomputed each AR step | tokens rebuilt per step (k=7 small) | **CHOSEN** — direct excluded-volume inductive bias; the transformer no longer infers distances from raw coords |
| P2: attention pair-bias (gridformer edge_bias style) | distance-dependent additive attention bias between all token pairs | invasive (custom encoder layer) | deferred — P1 first |
| P3: LJ pair-energy feature | per-token shifted-LJ energy vs query σ_pair | trivial add-on to P1 | include (1 extra channel, uses `ka_energy.SIGMA`) |

### Scale + training

- d_model 128→**256**, n_head 4→**8**, n_layer 3→**6**, n_ctx 16→**32** (~4-6M params; transformer steps are
  cheap relative to EGNN).
- 40k steps, batch 128, AdamW lr 3e-4 (500-step warmup + cosine, as in the existing `train()`), grad-clip 5,
  `augment` on. NLL objective unchanged: `-log_q(true cluster | true surroundings)`.
- fp32 (GeForce fp64 pitfall is on record).

## Experiment arms (attribution, per the 2026-06-26 mandate)

- **ARM-H (head-only ablation)**: spline head at the OLD sharpened scale (d_model 192, n_layer 4, n_ctx 32,
  no new pair features), 15k steps. If ≈44% → head-not-bottleneck confirmed for the cluster task too; if it
  drops materially, the within-bin-uniform mechanism was real.
- **ARM-FULL**: spline head + P1/P3 pair features + full scale, 40k steps. The production candidate.
- Optional (only if ARM-FULL beats 20%): overfit-4-configs probe at ARM-FULL scale for the capacity read.

## Interfaces and exactness invariants

- `ClusterProposal(head="bins"|"spline", pair_feats=False, ...)` — default `"bins"` keeps every existing
  checkpoint loadable and behavior byte-identical (`_load` passes stored arch keys only).
- `sample()` and `log_q()` must stay exact mirrors: the spline head returns the continuous log-density directly;
  DELETE the `-2·log(bin_w)` and dequantization-noise lines in the spline path; keep the frame Jacobian (=1)
  argument untouched. Out-of-box queries: spline linear tails + Gaussian base give a finite, correct density
  (no `-69` clamp in the spline path).
- Round-trip test: `sample()` then `log_q()` on the sampled cluster agree to <1e-4 (it's the same forward
  factorization — no integrator, so this must be near-exact; tolerance covers dtype noise only).
- The g(r)/split gate protocol is unchanged (`gate_measure`, `ka_cluster_egnn_diag.py split` semantics) so all
  numbers stay comparable; MH acceptance via the existing `mh_accept.py` protocol (AR variant: logq from the
  model's own sample/log_q — no integrator args).

## Gates / success criteria

1. **Exactness gate** (before training): round-trip logq consistency; spline identity-init sanity (untrained
   spline ≈ Gaussian base density); suite `tests/test_ka_cluster.py` + new head tests green.
2. **ARM-H readout**: attribution only, no go/no-go.
3. **ARM-FULL split gate**: overall per-particle clash — success tiers: <30% (mechanism confirmed, keep
   pushing), <18% (matches small EGNN: AR is competitive), ≤13% (matches big EGNN: AR preferred for MH on
   engineering grounds — proceed to acceptance estimate).
4. **MH acceptance** (if tier ≥2): same 640-move protocol; compare 1.02%.

## Risks

- Repeat of the 2026-06-26 null on ARM-FULL (context representation still the wall even with pair features) →
  the outcome is still decisive: it would localize A's deficit to the transformer-context pathway vs the EGNN's
  message-passing geometry, closing the AR-revival question with attribution.
- Spline training instability (the old full-config run was stable; identity init + warmup + clip carried over).
- Per-step token rebuild cost (~7× context encoding) — acceptable; the transformer is small.
