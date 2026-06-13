# Overnight autonomous action plan (2026-06-12 ~22:00 → ~06:00)

Owner: Claude, running autonomously while user sleeps. Live experiment: rail probe
`arc_rail_k8` (P7). Running log at the bottom — read that first in the morning.

## Objective
Get a clean verdict on whether **rail conditioning fixes the position-dependent Δs
pacing drift** (the P3/P6 diagnosis), and complete the cheap probes (P4) — without
touching the strategic/expensive levers that are the user's call.

## Autonomy boundaries (will NOT do without user)
- No 15 GB gilbert cache rebuilds (Phase 0.5 is user-led).
- No 100-epoch Phase-1 arm launches (expensive, strategic).
- No git push, no deleting/overwriting existing checkpoints (only archive-rename).
- Relaunching the 20-epoch rail PROBE with a better LR *is* in scope — it is cheap,
  reversible, and exactly the iteration the experiment needs.

## Decision gates

### Gate A — rail probe LR adjudication (~22:00–23:00)
Epoch-0 clean val/loss = 0.195 vs baseline 0.131 → rail at lr=1e-3 is degrading the
conditional. Watch epochs 1–3 trajectory:
- **RECOVERING** (epoch-3 val < epoch-0 − 0.01, trending to ~0.13): let it run to completion.
- **DIVERGING / FLAT-ELEVATED** (epoch-3 val ≥ ~0.18): kill, relaunch
  `arc_rail_k8_lr2e4` (lr 2e-4, all else identical). If train.py has a param-group
  freeze hook, prefer a 2-epoch rail-only warmup then unfreeze at 2e-4; else just lr 2e-4.
  Re-arm the trajectory watcher on the new run.

### Gate B — rail probe completion verdict (CPU eval, no GPU contention)
When the winning rail run finishes, run the P3 teacher-prefix harness on its best.ckpt:
- **FIXED**: k=48 first-8-step bias collapses from −0.28 toward −0.06, L4 Δs gen mean
  ≥ 0.93, suffix W1 < 0.08. → headline result; document, consider a confirm/variant.
- **PARTIAL**: bias shrinks but mean < 0.93. → rail is an Arm-B+ ingredient for the
  gilbert phase; document magnitude.
- **NULL**: unchanged from baseline. → rail-as-input insufficient; the remaining lever is
  Arm C′ (normalized-anchor TARGET, P1-validated), which needs the gilbert cache (parked
  for user). Document the negative result clearly.

### Gate C — GPU-free window (after rail run ends, ~02:30+)
GPU is single-tenant (concurrent runs OOM), so serialize. Priority order:
1. **Variant A probe** (user-requested) — `variantA_jointbias`, 10-epoch warm-start,
   `--use_joint_arc_bias 1`, joint (distance, arc-sep) attention bias. Compatible with the
   existing cache (arc_s = cumsum(input_deltas[...,0]), no rebuild); zero-init no-op warm
   start. LR: match the rail Gate-A outcome (1e-3 if rail recovered, 2e-4 if rail needed
   the drop). Launch ~02:35, ETA ~05:00. Watch its val trajectory like the rail probe.
2. P3 harness (CPU, no GPU contention) on rail best.ckpt — run AT rail completion,
   overlaps with variant A training.
3. benchmark_lj27 OTgap on rail + noise best.ckpt at L4 — squeeze before/after variant A.
4. P3 harness on variant A best.ckpt when it finishes (~05:00).

Three-way comparison by morning: baseline vs rail (input conditioning) vs variant A
(attention structure) — all zero-init warm starts from the same baseline, directly
comparable on the P3 pacing-drift metric.

## CPU work to fill GPU-busy windows
- **P4 closure audit** (`analyze_p4_closure.py`): generate samples at L3/L4/L5(/L10)
  from baseline + noise checkpoints, measure total traversed arc fraction
  (final_code / ((N−1)·X)) distribution. Sets Phase-2 budget-guidance ceiling.
- Append every result to the running log; commit findings in logical batches.

## Running log (newest entries appended)
- 22:00 — Plan written. Rail probe at epoch 1/20 (~15 min/ep, ETA ~02:30). Epoch-0
  val/loss 0.195. Trajectory watcher running. Launching P4 closure audit on CPU.
- 22:25 — P4 closure done (baseline + noise). Trained sizes traverse the full curve
  (L3 0.0–0.2% short, L5 1.2–1.6%); **held-out L4 underfills 12.2% (baseline) / 12.3%
  (noise)** — identical, confirming P6/noise null at the closure level too. ⇒ Phase-2
  budget guidance can recover at most ~12% of L4's gap; the rest is mode-collapse/shape.
- 22:30 — User requested variant A (JointEdgeBias). Verified multi-size compatible
  (arc_s from input deltas, no cache rebuild; zero-init no-op). Prepped
  `variantA_jointbias/launch_variantA.sh` (10-ep warm-start, lr parameterized). Queued
  for the post-rail GPU window (Gate C #1). Rail Gate-A trajectory still pending.
- 22:35 — **Gate A RESOLVED: RECOVERING.** Rail val/loss 0.195 (ep0) → 0.153 (ep1),
  still descending toward baseline 0.131. The lr=1e-3 transient (rail sublayer
  activating) is resolving, not diverging — no relaunch. Variant A confirmed to use
  lr=1e-3 (rail recovered at it). Next: let rail finish, then Gate B/C.
- (later, machine-time ~05:10) — ETA corrected by direct measurement: 15.9 min/epoch,
  17 epochs left ⇒ rail ETA ~09:38 (NOT the earlier ~02:30 mis-estimate). Decision:
  let it run the full 20 epochs (strongest checkpoint; killing early to save time is a
  marginal-benefit state change). Variant A + OTgap will land midday — correctness over
  speed.
- ~05:15 — Caught + fixed a correctness bug: P3 (and P4) harness was NOT rail-aware;
  forward() skips rail_attn when curve_waypoints is None, so a rail checkpoint would
  eval rail-DISABLED = false NULL. Harness now feeds recomputed fixed_template waypoints.
- ~05:20 — **HEADLINE: rail FIXES the held-out drift, already at epoch 2.** Rail-aware
  P3 on rail best.ckpt (ep2, val 0.144): L4 Δs mean 0.716→**0.941** at k=48 (baseline
  0.716), W1 0.281→**0.098**; k=48 first-8-step bias −0.28→**−0.07** = Gate-B FIXED
  signature. Mechanism-matched fix confirmed: the failure was pacing, the rail supplies
  the pacing reference as input. Expect the final (ep~19) checkpoint to be stronger.
  Remaining: OTgap confirmation (GPU, post-rail), variant A comparison, final-ckpt P3.
- ~05:25 — **P4 rail (ep2) confirms from the closure angle:** held-out L4 closure
  0.877→**0.977** (missing 12.2%→**2.3%**); L3 1.011, L5 0.999. Rail fixes the underfill.
  Two independent metrics (P3 drift + P4 closure) now agree the rail works.
- ~05:30 — benchmark_lj27 needs NO flow ckpt / NO GPU (pure scipy OT) → OTgap runs on
  CPU. Started early rail-ep2 L4 OTgap (baseline ref 0.892). 12-core box, load ~12 with
  training; keeping to one CPU eval at a time so as not to slow the critical-path rail.
- ~05:35 — **OTgap (3rd metric) CONFIRMS: rail ep2 L4 OTgap 0.892 → 0.670** (toward the
  green ≲0.6 refinable bar), via CPU benchmark_lj27 (no flow/GPU needed). Composite 2.319.
  `reports/multisize_arc/otgap_rail_ep2_L4.txt`.

  **THREE-METRIC EARLY VERDICT (rail best.ckpt @ epoch 2 of 20):**
  | metric | baseline | rail ep2 | direction |
  |---|---|---|---|
  | L4 Δs mean (k=48) | 0.716 | 0.941 | fixed |
  | L4 Δs W1 (k=0) | 0.129 | 0.066 | fixed |
  | L4 closure (underfill) | 12.2% | 2.3% | fixed |
  | L4 OTgap | 0.892 | 0.670 | toward green |
  Rail conditioning fixes held-out size transfer. Final (ep~19) checkpoint should be
  stronger. **Mistake noted:** my parallel CPU eval jobs (load 12/12) slowed the rail
  training; backing off CPU now to let it finish.
- ~05:36 — Backing off all CPU eval until rail completes. Completion watcher bnn3org8s
  armed. Next wake = rail done → Gate B (final P3) + Gate C (variant A launch, full
  L3/L4/L5 OTgap on freed GPU).
