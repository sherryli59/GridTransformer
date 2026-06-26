# Exact spline-flow placement head (Phase 1 / Feature B) — results

**Date:** 2026-06-26  **Verdict: NEGATIVE** (the head is not the bottleneck).

## What was built (all reviewed, all correct)
`KAFlowHeadModel` (`liquid_coupling_flow/ka_flowhead.py`): the binned `(a,b)` categorical placement head of
`KALocalFrameModel` replaced by an exact autoregressive RQS spline-flow head (Gaussian base, identity-init,
AR `a`→`b|a`), with an InertialAR local-inertial-frame toggle. Exact likelihood preserved (round-trip + the
model exactness gate pass). A Critical inertial-frame R-consistency bug (log_prob vs sample) was caught in
review and fixed with a genuine fail-before/pass-after regression test.

## Convergence (gate a): PASS
Overfit-tiny nll/N 4.79 → −1.23 (no NaN); full 20k training stable, no NaN, final nll/N −0.49.

## Likelihood (gate b): FAIL
Apples-to-apples **normalized position log-density** (per particle, in the (a,b) coordinate, before the
shared `arc_scale` Jacobian):
- **flow: 2.704**  vs  **categorical baseline: 2.783** → flow is marginally WORSE, essentially equal.

*Correction on the record:* the full-training `nll/N −0.49` is NOT comparable to the categorical's
"`nll/N ~4.7`" quoted earlier — the latter (from the soft-label ablation log) omitted the `jac` term the
flow's loss includes. The only valid comparison is the normalized position log-density above; it shows no
improvement.

## Structure (the verdict metric): NO CHANGE
Free-run and teacher-forced partial g(r), peak height / core fill, vs data (`ka_flowhead_gr_N100.png`):

| pair | data peak | cat-TF | flow-TF | cat-FR | flow-FR | data core | flow-TF core | flow-FR core |
|---|---|---|---|---|---|---|---|---|
| AA | 3.83 | ~1.9 | 1.82 | 1.85 | 1.75 | 0.023 | 0.345 | 0.319 |
| AB | 7.28 | ~3.2 | 3.14 | 2.61 | 2.44 | 0.024 | 0.496 | 0.318 |
| BB | 2.34 | ~1.5 | 1.48 | 1.37 | 1.29 | 0.019 | 0.260 | 0.263 |

The flow's g(r) — TF and FR — is identical to the categorical's. Contact peaks still undershoot ~2× and the
excluded-volume core is still filled (~0.3–0.5 vs data ~0.02).

## Interpretation
Two very different head classes (8-bin exact flow vs 192-bin categorical) converge to the **same** position
log-density AND the **same** g(r). The head's expressiveness is therefore **not** the bottleneck. The
conditional `p(offset | soft-centroid context)` is **intrinsically broad**: the soft-centroid origin + KNN
context does not determine the placement sharply, so no output head can sharpen it. The wall is in the
**context/representation the conditional sees** (the soft-centroid washing out precise neighbour geometry;
the half-cage), consistent with [[full-cage-lever-needs-energy]] and [[gbb-persistently-wrong]].

**Hypothesis B (a sharper exact head closes the g(r) wall) is falsified.** Feature A (curve conditioning,
Phase 2) was premised on freeing capacity for a sharper head — that premise is undercut; A should be
reconsidered or redirected at the context/representation, not the head.

## Caveats (bug-hypothesis kept open)
- `num_bins=8` (the implementer flagged it as conservative). The g(r) being *identical* across an 8-bin flow
  and a 192-bin categorical is strong evidence bin count is irrelevant, but a `num_bins=32` retrain would
  make the negative airtight.
- The result is in-dist N=100, scaffold frame. The inertial-frame variant's *quality* (vs scaffold) was not
  evaluated (only its exactness was fixed).

## Deliverables
`ka_flowhead.py` (head + model + training + convergence gate + inertial toggle), `ka_flowhead_eval.py`
(g(r) verdict), tests (`test_flowhead*.py`), checkpoint `ka_flowhead_N100_k8_scratch.pt`, figure
`ka_flowhead_gr_N100.png`.

---

## Feature A (GPS/QueryAR curve conditioning) — increment over B

`KACurveFlowModel` (`ka_curveflow.py`): adds a per-particle, deterministic curve feature (periodic encoding
of the scaffold position `s_j` + arc-length `(j+0.5)/N`) to the flow's conditioning. Zero-init ⇒ starts == B;
warm-started from B; 20k steps. **Result: a small but consistent POSITIVE (and the controller's "deterministic
curve adds nothing" prediction was wrong).**

- **Position log-density:** B 2.704 → **A 2.901** (+0.20 nats/particle; now above categorical 2.783).
- **Free-run g(r)** (peak / core, data core ~0.02):

  | arm | AA | AB | BB |
  |---|---|---|---|
  | data | 3.88 / 0.039 | 7.38 / 0.024 | 2.47 / 0.019 |
  | categorical | 1.85 / 0.346 | 2.63 / 0.288 | 1.39 / 0.288 |
  | +B flow | 1.76 / 0.346 | 2.43 / 0.293 | 1.31 / 0.272 |
  | +A curve | 1.86 / 0.321 | 2.72 / 0.262 | 1.37 / 0.253 |

  Cores drop ~10% across all three pairs, peaks nudge up — consistent direction + corroborating likelihood ⇒
  real signal.

**Interpretation:** giving the explicit curve coordinate lets the model spend the neighbour context on
*structure* instead of *position-inference* (the proposed decoupling). So the conditional was partly
position-inference-limited, which the curve relieves — a genuine, cheap lever. **But the gain is modest**
(~10% core reduction; AB peak 2.72 vs data 7.38): the dominant wall (broad conditional from missing
config-specific information / the half-cage) remains. Next lever: the full cage (real added config info,
g_BB 1.25->2.00 in prior work), optionally combined with curve conditioning.

**Note:** `ka_flowhead_eval.py` hardcodes `KAFlowHeadModel` and cannot load the `curve_proj` checkpoint; the
Feature-A g(r) above was run with the `KACurveFlowModel` loader directly (eval-script generalization is a
follow-up nicety, not a result-affecting issue).
