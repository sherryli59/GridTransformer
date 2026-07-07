# Stage B — learned collective move, event-mined from the PT ladder — design spec

**Date:** 2026-07-06. **Trigger (measured, per the kernels-two-laws spec):** A2's n_heatbath sweep PLATEAUED
(~-3.14, no trend n=1..8, n=8@2x wall no deeper) => the residual ~0.13/particle to the true -3.276 is
COLLECTIVE — multi-particle rearrangements that no single-site move can express, however sharp. Both
pre-committed trigger conditions fired (closes <half the gap; acceptance high but depth plateaus).

## What Stage B learns (the "move model" — different object than A1/A2)

A1/A2 learn STATE conditionals q(x_block | env) — where a block should sit given surroundings. Stage B learns a
TRANSITION conditional q(x'_block | x_block, env): given the CURRENT positions of a small mobile set (+ shell),
propose where they hop TOGETHER. Trained on real relaxation events mined from the PT ladder trajectories, it
targets exactly the barrier-crossing move class the plateau says is missing. MH-guarded (law 1) with the exact
two-way density: alpha = min(1, e^{-beta dU} q(x|x',env)/q(x'|x,env)) — the SAME network scores both directions
(a conditional density, not a flow; reverse scoring is one forward pass).

## Staging — GB0 is the only committed build; the model is gated on measured event morphology

### GB0 — event miner + data gate (build now, `ka_events.py`)
Ladder layout (per rung, per seed block): index = t*B_rep + b, t = 0..nc-1, B_rep = 8 replicas; adjacent-time
same-replica pairs (t*B+b, (t+1)*B+b) are 8 MC-sweeps apart. Two seed blocks; never pair across the boundary.
- **Exchange rejection (data subtlety):** PT swaps whole configs between rungs every 10 sweeps (cold exch acc
  0.49) => a large fraction of adjacent pairs differ by a REPLACED config, not dynamics. Filter: per-particle
  min-image displacement d_i; if frac(d_i > d_mobile) > 25% => exchange/contaminated, REJECT. (Double-exchange
  within 8 sweeps looks clean — rare, acceptable contamination.)
- **Event definition:** clean pair with max_i d_i >= d_event (cage-break scale) => mobile set {i: d_i > d_mobile}.
  Defaults d_mobile = 0.35 (>> vibration ~0.1-0.15, < hop ~ nn-dist 0.95), d_event = 0.6. Recorded per event:
  mobile-set size k, spatial extent (max pairwise min-image distance in the set), connectivity (single cluster
  at radius 1.4?), species composition, rung/beta.
- **GATE GB0:** the two COLD rungs (beta 2.0, 1.81) together yield >= ~200 clean localized events (k <= ~10,
  single-cluster) => GO to the move model. ELSE the pre-committed fallback: harvest events from long swap-MC
  runs just above T* (0.55-0.6) where hops are frequent but structure is still glassy.
- Deliverable either way: event-statistics report (counts/rung, k-distribution, extent, species mix) — this
  MEASURES the collective move class and fixes the model's block size.

### GB1 — move model + kernel (design finalized only after GB0's morphology numbers)
Sketch: reuse the cluster machinery [[cluster-move-gate-nogo]] — env frame from `frame_ctx_slots` (frozen-cage
exactness class, config-independent); tokens = env + CURRENT block positions (role-embedded); per-step spline
heads place the block's NEW positions AR within the event (exact joint log_q, one forward per direction);
beta-FiLM (ladder events span all rungs). Kernel: pick a seed site, take the k-NN block matching the measured
event size, propose, MH-accept with the two-way ratio. Gates mirror A2: DB/frozen-inputs invariant test,
stationarity-from-reference (mixed), acceptance, in-SMC depth vs the A2 stack — the bar is pushing past ~-3.14
toward -3.276 at near-matched wall.

## Data note (incident + mitigation)
`pt_ladder_N100.pt` was LOST from artifacts/ (existed 11:44, gone 16:40, deleter unknown — possibly a parallel
session cleanup). The protocol is seeded => regeneration reproduces it (relaunched 16:46, ~2.6 h). Mitigation:
after regen, keep `pt_ladder_N100.pt.bak` alongside (guards the observed failure mode: accidental deletion).
Miner development proceeds against `pt_ladder_N256.pt` (identical layout) meanwhile.

---
## GB0 RESULT (2026-07-06): GO — measured design inputs for GB1
- PT-ladder mining REFUTED structurally (exchange contamination 40-65% + 8-sweep window < event duration ~tens
  of sweeps => canonical N=100 cold rungs: 0 events). Fallback harvester (displacement-only, no swaps/PT) is
  strictly better and nearly free (~1.7 ms/sweep); **dt=100 = the calibrated window** (dt=8 fragments, dt=400
  merges: k 6-23 multi-cluster).
- **Move class (stable across runs): 1-8 particle, single-cluster, compact (extent <~ 2 sigma) string hops.**
- Bank: ka_event_bank_N100.pt — 652 localized events (147 @ beta 2.0, 199 @ 1.81, 306 @ 1.63), full (xa,xb)
  config pairs. GATE PASSED (346 >= 200 at beta>=1.81).
- GB1 fixed parameters: k_block = 8 (mobile set + nearest fill-up to fixed size); beta-FiLM over the harvest
  betas; two-way exact conditional; ONE-SHOT MH first, A1-cSMC composition (per-step filtering with retained
  reference) as the pre-committed fallback if one-shot acceptance floors (the A1 disease).
