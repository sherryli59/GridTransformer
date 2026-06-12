# Scoping: "predict deviation from the Hilbert center" instead of cumulative Δs

Goal: replace arc_repr's drifting along-curve step (Δs = c_j − c_{j−1}, relative to previous
particle) with a non-cumulative anchor, to kill the exposure-bias drift that collapses
held-out Δs generation (see [[arc-cell-discontinuity]]; held-out L4/L10 gen Δs mean ~0.88 vs
data 0.99).

## The mechanism is already fully built (no new representation code)

`curve_rail_residual_target` predicts `residual_j = pos_j − decode(j·X)` (the per-index Hilbert
center, `fixed_template_anchors`, lj_transferable.py:290) instead of `pos_j − pos_{j−1}`.
Wired end to end:
- target baked at cache build (`_build_item_from_index` L1297; batch build L1432) — **needs a
  fresh cache**, cannot reuse the arc cache whose deltas are pos_j−pos_{j−1}.
- preprocess flags: `--use_curve_rail --curve_rail_mode fixed_template
  --curve_rail_residual_target --curve_rail_k --curve_rail_reference absolute --cell_size`
  (multi-size via multiple `--data_h5`). **Requires `--no_augment_shift`** (absolute anchor
  needs fixed phase).
- train flags: matching `--lj_transfer_curve_rail_*` + `--lj_transfer_cell_size`.
- model: standard arch `rail_attn` cross-attention consumes waypoints in both `forward` and
  KV-cache `forward_step`; sampler adds the anchor back (`sample_lj.py:952`,
  auto-detected from checkpoint). `/mnt/ssd` has 186 GB free (cache ≈15 GB).

## BUT: measurement says the global template anchor is loose AND size-dependent

Residual `|pos_j − decode(j·X)|` (min-imaged) on real MCMC data:

| size | role | \|resid\| mean | per-axis std | mid-curve | uniform |
|------|------|---------------:|-------------:|----------:|--------:|
| L3 N=27   | train    | 1.03 | 0.64 | 1.16 | 1.17 |
| L5 N=125  | train    | 1.31 | 0.83 | 1.51 | 1.95 |
| L10 N=1000| held out | 1.76 | 1.12 | 1.83 | 3.90 |

The target is barely tighter than uniform (L3), **grows with box size** (1.03→1.31→1.76), and
bulges mid-curve — the order-statistic random walk (liquids lack long-range order, so the
absolute index template only loosely predicts pos_j). This reintroduces size-dependence and a
high-entropy target — almost certainly why the single-size K=1 rail failed despite a passing
marginal audit (`lj_ckpts_lj27_pbc_fixedrail_k1_*`).

## The tension (load-bearing)

**Tight + size-invariant + non-drifting cannot all hold.** Tight & size-invariant ⇒ a *local*
reference (prev particle, sharp via short-range order). Non-drifting ⇒ an *absolute* reference
(index template), which liquids make loose. arc Δs = tight+invariant but drifts; template
anchor = non-drift but loose+size-scaling.

## Options (a fork — pick before spending compute)

- **A. Bounded-local k-step anchor** (`pos_j − pos_{j−k}`): keeps a tight, size-invariant
  target; bounds drift accumulation to ~N/k hops instead of N steps. Tunable k. Partial fix,
  not elimination. Uses the existing offset/lookahead rail (prev_step reference).
- **B. Exposure-bias mitigation on the existing arc rep** (scheduled sampling /
  `continuous_input_noise`): keep the proven tight size-invariant arc target; make generation
  robust to its own drift in training. No new cache; retrain arc + noise. Attacks the diagnosed
  cause (drift) directly without giving up tightness.
- **C. Run the global-template residual multi-size anyway** as a data point (it's built);
  interpolated L4 might still benefit even if extrapolated L10 won't. Costs a cache rebuild +
  100-epoch train on something the measurement says is loose.

Recommendation: **B primary** (keeps what works, targets the actual failure), **A** as the
interesting middle if we want a representation change. C only if we want to close the loop on
the literal idea.
