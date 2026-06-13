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
Run on GPU in priority order:
1. benchmark_lj27 OTgap on rail best.ckpt at L3/L4/L5 (the refinability verdict).
2. benchmark_lj27 OTgap on noise best.ckpt at L4 (completes P6 at the OTgap level).
3. If rail FIXED and time remains: one confirmation variant (e.g., K=16 or from-scratch
   rail) — 20 epochs only.

## CPU work to fill GPU-busy windows
- **P4 closure audit** (`analyze_p4_closure.py`): generate samples at L3/L4/L5(/L10)
  from baseline + noise checkpoints, measure total traversed arc fraction
  (final_code / ((N−1)·X)) distribution. Sets Phase-2 budget-guidance ceiling.
- Append every result to the running log; commit findings in logical batches.

## Running log (newest entries appended)
- 22:00 — Plan written. Rail probe at epoch 1/20 (~15 min/ep, ETA ~02:30). Epoch-0
  val/loss 0.195. Trajectory watcher running. Launching P4 closure audit on CPU.
