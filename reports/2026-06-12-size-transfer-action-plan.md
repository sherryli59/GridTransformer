# Size-transfer action plan: probes, gilbert substrate, two training arms

**Date**: 2026-06-12
**Branch context**: `hilbert-arc-repr`, multi-size arc runs
**Inputs**: `reports/multisize_arc/SIZE_TRANSFER_FINDINGS.md` (consolidated findings),
`reports/2026-06-12-exposure-bias-mitigation-plan.md` (Plan A, first run in flight),
`reports/2026-06-12-gilbert-curve-plan.md` (gilbert implementation, in progress).

## Goal and success criterion

**Extrapolation, not just interpolation**: a model trained on a small-size ladder must
generate refinable configurations at sizes beyond the ladder. Primary gate: **held-out
L10 (N=1000) OTgap ≤ ~0.6** from training that excludes it; secondary: held-out interior
size beats its zero-shot baseline (L4: 0.795). Marginal W1 audits are necessary but not
sufficient (K=1 rail lesson: passing distribution audit, failed generation).

Consequence: any **output parameterization must be exactly size-invariant** — there is
always a size beyond the ladder. Inputs and sampler corrections carry no such constraint.

## Structural principle (division of labor)

Three independent, stackable levers, each owning one invariance:

| lever | what it makes size-invariant | mechanism |
|---|---|---|
| gilbert + constant cell | the **local conditional** | same cell size, same X=R³/N, same LJ-to-curve length ratio at every size |
| bridge normalization | the **global parameterization** | divide absolute-anchor residual by its deterministic N-scaling |
| corruption + rail conditioning | the **rollout** | model learns to steer back to an absolute reference from off-manifold context |

The findings doc's "load-bearing tension" (tight + size-invariant + non-drifting cannot
coexist) dissolves only if each invariance is handled by the lever that owns it.

### Key new fact: the absolute anchor's size-scaling is deterministic

Measured per-index absolute-anchor residual stds (findings §5: 0.636 / 0.826 / 1.127 at
N=27/125/1000) match **N^(1/6)** to 1–3% (predicted ratios 1.291 / 1.826 vs measured
1.299 / 1.772). Interpretation: order-statistic bridge fluctuation in s-space (~√N cells)
pushed through Hilbert locality |Δpos| ~ |Δs|^(1/3). If the per-index *shape* also
collapses (probe P1), the absolute anchor becomes non-drifting AND size-invariant after
normalization. Constant cell does NOT remove this effect — it is particle counting, not
cell geometry — so Arm C′ needs the normalization even on the gilbert substrate.

## Phase 0 — probes on existing artifacts (no training; run now, pow2 substrate)

- **P1 (gates Arm C′): bridge-normalization collapse.** Divide measured per-index
  absolute-anchor stds by `(j(N−j)/N)^(1/6)` (and global `N^(1/6)`). Pass = L3/L5/L10
  curves overlap within ~10%. Fail ⇒ replace Arm C′ with plain Arm B.
  **RESULT 2026-06-12: PASS** (`analyze_p1_bridge_collapse.py`,
  `figs/p1_bridge_collapse.png`). Cross-size spread 0.528 raw → **0.106** normalized
  (prev-particle control: 0.074). Best-fit exponent α = 0.158/0.150/0.146 per size
  (theory 1/6 ≈ 0.167) with a **universal prefactor C ≈ 0.41–0.42 (±1.3%)** across
  N=27–1000. Note the global and per-index normalizations give identical cross-size
  spread (they differ by a common function of j/N); prefer the per-index profile for
  the Arm C′ target since it also flattens the within-sequence variance the head sees.
  Optionally use the fitted α ≈ 0.15 instead of 1/6. **Arm C′ is GO.**
- **P2: T=1.0 ablation.** Regenerate L4 (+L5 control) from `multisize_arc_fullcov`
  without tempering. Tempering sharpens a right-skewed Δs toward its sub-mean mode, and
  bites harder on the flatter held-out conditional; if Δs gen mean recovers toward 0.99,
  fix the temperature policy before judging any arm.
  **RESULT 2026-06-12: tempering exonerated.** T=1.0 reproduces the T=0.9 drift within
  noise (suffix means 0.870/0.847/0.787/0.727 across k=0/16/32/48 vs
  0.867/0.843/0.781/0.716). Only effect: backward-step (noncanonical) rate rises at
  T=1.0 (32% vs 22% at k=0) — tempering mildly helps monotonicity. No temperature-policy
  change needed.
- **P3: drift-onset.** Teacher-force a data prefix of k ∈ {0, N/4, N/2, 3N/4} steps at
  L4, free-run the rest, measure Δs stats of the suffix. "Compounds from step 1" vs
  "late attractor" chooses corruption schedule for Arm B+ (iid vs scheduled).
  **RESULT 2026-06-12 (`analyze_p3_drift_onset.py`, `figs/p3_drift_onset_T{0.9,1}.png`):
  drift is immediate AND the bias is position-dependent, not context-dependent.** First-8
  free-step bias: −0.06 (k=0, early curve) vs −0.25/−0.28 (k=32/48, late curve) — a
  perfectly clean 48-step data prefix makes the next-step bias 4× WORSE than free-running
  at early positions. Textbook exposure bias predicts the opposite. The conditional's
  *pacing* is miscalibrated at late curve positions for the unseen size (hypothesis: the
  model keys on absolute s-progression; at L4 position 48, cumulative s numerically
  matches a late-L5 position). Also: 22% of free-run samples take a backward step
  (data: 0%). **Consequences:** (i) iid-noise training (P6 in flight) attacks only the
  small context-sensitive component — prediction: improves W1 modestly, leaves L4 Δs
  mean well below 0.93; (ii) scheduled sampling would not fix it either (not a context
  problem) — the B1→B2 escalation clause is expected to be moot; (iii) the template
  anchor (C′ target / B+ rail input) directly supplies the missing pacing reference and
  is the mechanism-matched fix; (iv) P4's mean-drift share is partially answered:
  gen mean ≈ 0.87 at k=0 ⇒ ~13% curve underfill.
- **P4: closure audit.** Distribution of total generated arc length vs N per size.
  Gen Δs mean 0.889 ⇒ rollouts end ~11% short ⇒ underfilled box; quantifies how much of
  L4's OTgap is mean-drift alone.
- **P5: gilbert motif stationarity.** Measure, separately, (a) the jump fraction
  (Δs above the jump threshold) and (b) the turn/motif histogram
  (`analyze_hilbert_lj_jumps.py` machinery) across gilbert grids at every candidate
  R in [8, 20], plus the resolution-matched pow2 Hilbert baselines (R=8, 16) for
  reference. **Gate: if the jump fraction varies by more than 2× across R ∈ [8, 20],
  restrict the training/eval ladder to R values whose jump fraction is within 30% of
  the nearest pow2 baseline, and document the restriction in the gilbert plan.**
  On failure, localize it: jump statistics vs turn statistics, and whether it hits
  all non-pow2 R or a specific class (e.g., odd/prime side lengths) — the restricted-R
  set follows from that localization, not from abandoning the substrate.
  **RESULT 2026-06-12 (`analyze_p5_gilbert_motifs.py`, `figs/p5_gilbert_motifs.png`):
  gate fails numerically but the cause is benign and the remedy is free.** Jump-fraction
  max/min across R∈[8,20] is 3.1 (L4) / 5.7 (L5) — over the 2× threshold — but the spread
  is entirely the **odd-R multi-cell-step artifact**: odd grids have Euclidean steps up to
  ~2.8 cells (multicell fraction 0.8–3%), even grids are perfectly face-continuous
  (multicell 0%, max step 1.0). At EVEN R the gilbert jump fraction matches the pow2
  Hilbert baseline at every scale, including production R=86 (0.0009 vs hilbert 0.0009,
  Δs mean 0.995). **Restriction: even R only** — which the production nearest-even-R rule
  already enforces, so it costs nothing. Curve straight-step fraction is lower for gilbert
  (~0.05) than Hilbert (~0.10) but stationary across R, so motif statistics are
  size-stable on the even-R ladder.
- **P6: exposure-bias noise readout — RESULT 2026-06-12 (`arc_noise_ds006`, 20 ep,
  evaluated via `P3_CKPT=...noise.../best.ckpt` on the P3 harness).** Prediction confirmed,
  in fact more strongly: the noise run did **not** move the held-out drift at all. L4
  suffix Δs mean / W1 at k=0/16/32/48 are 0.866/0.840/0.772/0.723 and 0.129/0.162/0.230/
  0.275 — **statistically identical to the baseline** (0.867/0.843/0.781/0.716,
  0.129/0.160/0.221/0.281). Δs gen mean 0.866 ≪ 0.93 escalation threshold; W1 did not even
  improve. Clean val NLL stayed flat through the full noise ramp (40.2→40.6, no regression),
  so the conditional was undamaged — the corruption simply has no purchase on a
  position-dependent (not context-dependent) failure. **Consequences:** (i) iid noise is
  dropped from Arm B+; (ii) the B1→B2 scheduled-feedback escalation is moot (confirmed: not
  a context problem); (iii) **Arm B+ collapses to rail-conditioning alone** — see P7.
- **P7 (in flight): rail-conditioning probe.** `arc_rail_k8` (warm-start, fixed_template
  curve rail, K=8 waypoints, `absolute` reference, NO noise, 20 ep) — the mechanism-matched
  test of whether feeding the pacing reference as a cross-attention INPUT fixes the
  position-dependent drift. Reuses the existing non-rail L3L5 cache via the on-the-fly
  fixed-template synthesis unlock (`LJTransferableCachedDataset(use_curve_rail=...)`);
  inputs need not be size-invariant, so the N^(1/6) target issue does not apply.
  `CurveRailAttention.out_proj` is now **zero-initialized** so the warm-start is an exact
  no-op at step 0 — any movement is causally the rail's. **Discriminating eval (CPU):** the
  P3 teacher-prefix harness (`P3_CKPT=...rail.../best.ckpt`). Success signature: the k=48
  first-8-step bias collapses from −0.28 toward the −0.06 early-position level, L4 Δs gen
  mean → ≥0.93, W1 < 0.08. Outcome decides whether Arm B+ ships as rail-conditioning and
  whether the rail input stacks onto Arm C′.

## Phase 0.5 — gilbert constant-cell substrate (in progress, user-led)

Rebuild caches at constant cell size c for the full ladder (R = L/c any integer; X =
1/(ρc³) exactly constant across sizes). Then rerun the cheap §2 marginal A/B on the
rebuilt cache to confirm Δs/fine units collapse exactly. Widen the ladder: train
{L3, L5, L6}, hold out **L4 (interior)** and **L7 + L10 (exterior)** for the
extrapolation gate. Expectation-setter: the L10 result already showed matched cell alone
does not rescue free-running — gilbert removes a training-side confound (mixed cells in
the train set) and shrinks per-step interpolation error, but is substrate, not fix.

## Phase 1 — two training arms in parallel, both on the gilbert substrate

Protocol per arm: train the gilbert ladder, dim 512 / depth 4 / heads 4, full-cov head,
clean-val checkpointing (`val/loss` monitor + `save_last`, in train.py since 2026-06-12).

**Staged budget, not a flat 100 epochs.** Prior multi-size runs showed the verdict
forming early (EP18_STATUS: trained-size OTgap trends visible by epoch 18; arc-probe
variants separated by epoch 10). Stage A: run both arms to **35 epochs**, then gate on
(i) held-out L4 Δs W1 and gen mean, (ii) a reduced-sample (128) L4 OTgap read,
(iii) trained-size val NLL health. If one arm clearly loses (worse on both (i) and (ii)
outside run-to-run noise), extend only the winner to 100 epochs; if they're within
noise, extend both. This halves wall-clock to a decision when the comparison is lopsided.

- **Arm C′ — normalized absolute anchor** (launch iff P1 passes). Target =
  `(pos_j − decode(j·X)) / scale(j,N)` with the P1-validated bridge scale; fine component
  unchanged. Non-drifting by construction (no telescoping decode). Builds on
  `curve_rail_residual_target` + `fixed_template`; delta = one scale factor at cache
  build + inverse in sampler. Watch: monotonicity is no longer structural — track
  noncanonical/backward-step counters from the likelihood audit.
- **Arm B+ — rail-conditioned Δs** (revised post-P6: noise dropped). Keep the tight
  size-invariant `(Δs, fine)` target; add the absolute reference as *input* via the
  `rail_attn` cross-attention (zero-init, fixed_template, K=8). P6 showed context
  corruption has no purchase on the position-dependent failure, so Arm B+ is now rail
  conditioning alone — the form validated by the P7 probe. Targets must be size-invariant
  to extrapolate; inputs (the rail) need not be.

**Shared eval battery**: teacher-forced NLL (all sizes), Δs/fine marginal W1 at held-out
sizes, OTgap at trained + held-out sizes (reduced sample count at N=1000 if OT cost
bites), closure audit, monotonicity counters.

## Phase 2 — sampler-side bridge/budget guidance (stackable)

Budget-aware Δs decoding: bias/reweight the mixture so the running arc sum tracks the
remaining `(N−j)·X` budget (hard closure constraint: the rollout must reach the curve
end). Test first on the existing checkpoint — doubles as the mean-drift vs mode-collapse
ablation — then stack on the winning arm.

**Success criterion (set by P4):** P4's closure audit quantifies how much of the held-out
OTgap is attributable to mean-drift underfill alone; that number is Phase 2's expected
ceiling. **Ship** budget guidance if it recovers at least half of the P4-predicted
OTgap share AND the held-out OTgap beats its zero-shot baseline. **Drop** it if it
corrects the mean (gen Δs mean ≥ 0.97) but improves held-out OTgap by < 0.05 — that
outcome is itself the answer (mode collapse dominates, mean drift is secondary) and
gets recorded as the ablation result.

## Decision gates

| observation | action |
|---|---|
| P1 fails to collapse | Arm C′ → plain Arm B (noise-only control) |
| P2 recovers Δs mean at T=1.0 | ~~fix temperature policy~~ RESOLVED: no recovery; keep T=0.9 |
| P5: jump fraction varies >2× across R∈[8,20] | restrict ladder to R within 30% of nearest pow2 baseline; document |
| P6 hits Stage-1 gates | B+ keeps iid noise; if mean stuck <0.93, scheduled feedback |
| epoch-35 stage gate: one arm loses on both L4 Δs W1 and OTgap-128 | extend only the winner to 100 epochs |
| Phase 2 fixes mean (≥0.97) but OTgap gain <0.05 | drop budget guidance; record "mode collapse dominates" as the ablation result |
| C′ green at L10 | new canonical rep; fold in Phase-2 + B+'s corruption |
| both arms fail at L10 | next round: k-step bounded anchor (deferred: only slows accumulation) |

## Deliberately deferred

- **Option A (k-step anchor `pos_j − pos_{j−k}`)**: strictly dominated for now — trades
  away the absolute reference without eliminating accumulation.
- **Exact incremental scheduled sampling (B2)**: only if the two-pass approximation
  clearly helps but saturates.
- **Trained-size-coverage fallback**: if extrapolation stalls entirely, ship multi-size
  at trained sizes and study transfer on the side.
