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
- **P2: T=1.0 ablation.** Regenerate L4 (+L5 control) from `multisize_arc_fullcov`
  without tempering. Tempering sharpens a right-skewed Δs toward its sub-mean mode, and
  bites harder on the flatter held-out conditional; if Δs gen mean recovers toward 0.99,
  fix the temperature policy before judging any arm.
- **P3: drift-onset.** Teacher-force a data prefix of k ∈ {0, N/4, N/2, 3N/4} steps at
  L4, free-run the rest, measure Δs stats of the suffix. "Compounds from step 1" vs
  "late attractor" chooses corruption schedule for Arm B+ (iid vs scheduled).
- **P4: closure audit.** Distribution of total generated arc length vs N per size.
  Gen Δs mean 0.889 ⇒ rollouts end ~11% short ⇒ underfilled box; quantifies how much of
  L4's OTgap is mean-drift alone.
- **P5: gilbert motif stationarity.** Compare turn/jump statistics
  (`analyze_hilbert_lj_jumps.py` machinery) across gilbert grids at several non-pow2 R.
  Gilbert locality is Hilbert-*like* but not exactly self-similar; verify local curve
  statistics are size-stationary before betting the substrate on it.
- **P6 (in flight): exposure-bias noise readout.** `arc_noise_ds006` (Plan A iid Δs
  noise 0.06, warm-start, 20 ep) lands ~tonight. Stage-1 gates: L4 Δs gen mean → 0.99,
  L4 Δs W1 < ~0.08, clean val NLL regression ≤ +0.03, L3/L5 OTgap stay ≤ 0.6/0.65.
  Outcome scopes Arm B+: clear win ⇒ B+ keeps iid noise; NLL fine but mean stuck < 0.93
  ⇒ escalate to scheduled feedback inside B+.

## Phase 0.5 — gilbert constant-cell substrate (in progress, user-led)

Rebuild caches at constant cell size c for the full ladder (R = L/c any integer; X =
1/(ρc³) exactly constant across sizes). Then rerun the cheap §2 marginal A/B on the
rebuilt cache to confirm Δs/fine units collapse exactly. Widen the ladder: train
{L3, L5, L6}, hold out **L4 (interior)** and **L7 + L10 (exterior)** for the
extrapolation gate. Expectation-setter: the L10 result already showed matched cell alone
does not rescue free-running — gilbert removes a training-side confound (mixed cells in
the train set) and shrinks per-step interpolation error, but is substrate, not fix.

## Phase 1 — two training arms in parallel, both on the gilbert substrate

Protocol per arm: train the gilbert ladder, 100 epochs, dim 512 / depth 4 / heads 4,
full-cov head, clean-val checkpointing (`val/loss` monitor + `save_last`, in train.py
since 2026-06-12).

- **Arm C′ — normalized absolute anchor** (launch iff P1 passes). Target =
  `(pos_j − decode(j·X)) / scale(j,N)` with the P1-validated bridge scale; fine component
  unchanged. Non-drifting by construction (no telescoping decode). Builds on
  `curve_rail_residual_target` + `fixed_template`; delta = one scale factor at cache
  build + inverse in sampler. Watch: monotonicity is no longer structural — track
  noncanonical/backward-step counters from the likelihood audit.
- **Arm B+ — drift-robust Δs.** Keep the tight size-invariant `(Δs, fine)` target; add
  the absolute reference as *input* via the built D2 `rail_attn` cross-attention; train
  with context corruption (form chosen by P3/P6). Targets must be size-invariant to
  extrapolate; inputs need not be.

**Shared eval battery**: teacher-forced NLL (all sizes), Δs/fine marginal W1 at held-out
sizes, OTgap at trained + held-out sizes (reduced sample count at N=1000 if OT cost
bites), closure audit, monotonicity counters.

## Phase 2 — sampler-side bridge/budget guidance (stackable)

Budget-aware Δs decoding: bias/reweight the mixture so the running arc sum tracks the
remaining `(N−j)·X` budget (hard closure constraint: the rollout must reach the curve
end). Test first on the existing checkpoint — doubles as the mean-drift vs mode-collapse
ablation — then stack on the winning arm. Band-aid, not fix: corrects the mean, not the
collapse.

## Decision gates

| observation | action |
|---|---|
| P1 fails to collapse | Arm C′ → plain Arm B (noise-only control) |
| P2 recovers Δs mean at T=1.0 | fix temperature policy before judging arms |
| P5 shows non-stationary gilbert motifs | reconsider substrate or restrict R choices |
| P6 hits Stage-1 gates | B+ keeps iid noise; if mean stuck <0.93, scheduled feedback |
| C′ green at L10 | new canonical rep; fold in Phase-2 + B+'s corruption |
| both arms fail at L10 | next round: k-step bounded anchor (deferred: only slows accumulation) |

## Deliberately deferred

- **Option A (k-step anchor `pos_j − pos_{j−k}`)**: strictly dominated for now — trades
  away the absolute reference without eliminating accumulation.
- **Exact incremental scheduled sampling (B2)**: only if the two-pass approximation
  clearly helps but saturates.
- **Trained-size-coverage fallback**: if extrapolation stalls entirely, ship multi-size
  at trained sizes and study transfer on the side.
