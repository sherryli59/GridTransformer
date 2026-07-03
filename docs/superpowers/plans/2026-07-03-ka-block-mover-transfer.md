# KA Block-Mover Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure train-small→sample-big: N=100-trained two-time joint flow + block-relabel mover accelerating equilibration to the PT reference at unseen N=256/36 (sweeps-saved).

**Architecture:** Pure port of gated IPL44 machinery (spec §3): observable hook in `run_chain`, KA data path in the trainer, thin KA adapter/benchmark module. No new algorithms.

**Tech Stack:** existing `ipl_swap_smc` kernels (system-agnostic), `joint_flow` two-time, `ka_energy`/`ka_observables`, PT reference artifacts.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-03-ka-block-mover-transfer-design.md` (gates T0/T1/T2 + swap-safety check).
- lr ≤1e-3 warmup+EMA; best+last ckpts only; IPL test suite (13) stays green; commit per task.
- KA: T*=0.5 (β=2), ρ=1.2; L=√(N/1.2); thresholds ⟨U⟩/N −3.0/−3.1; fixed 65:35 labeling from artifacts.
- GPU: PID-verified liveness only (no pgrep-pattern checks); nohup + PID-keyed watchers.

---

### Task 1: observable hook + KA smoke tests
- [ ] `run_chain(..., obs_fn=None)`: default keeps `ipl_gr_partials`-based `gbb_peak`; if given, `obs_fn(x, s) -> float` fills the `gbb_peak` slot. IPL tests unchanged and green.
- [ ] Test (append to test_swap_smc.py): KA wiring smoke — `ka_energy` in `position_sweep` on 16 reference configs at β=2 preserves ⟨U⟩/N within 3%; pair + block4 moves at 65:35 preserve counts.
- [ ] Commit.

### Task 2: KA data path in the trainer
- [ ] `train_joint_flow.py`: `--system ka` (argv[8] or tag prefix `ka`): load `artifacts/ka_reference_N100.pt` (x 14400, broadcast fixed s), L=ref["L"], nB=int(s.sum()), val=last 1000. Everything else identical (global OT bank, two-time).
- [ ] Smoke 200 steps, then launch `ka100tt` 20000 64 4 (≈30 min, background + PID watcher).
- [ ] Commit.

### Task 3: E0 — geometry-table transfer check (Gate T0)
- [ ] Script/inline: `denoiser_eval` (t_pos=1, t_spec=0 and 0.9) with the ka100tt model at N=100 val; then rebuild model at (N=36, L=5.4772) and (N=256, L=14.6059), load same weights, eval on held-out references.
- [ ] Gate T0: unseen-N acc_mismatch within 10 pt of train-N. Kill → stop, report.

### Task 4: `ka_block_smc.py` — E1/E2 benchmark driver
- [ ] Movers: random-pair, block4, block8 (frozen geometry table at (t_pos=1,t_spec=0), 8 attempts/sweep, table cached per sweep). Seeds: uniform, flow-samples (`jf.sample`). Position step tuned for β=2 (start 0.12, report acceptance).
- [ ] Metrics: sweeps + wall-clock to ⟨U⟩/N ≤ −3.0 and ≤ −3.1 (batch-median), g_BB via `partial_gr` as sanity; B=128.
- [ ] E1 at N=100 (Gate T1: block ≥2× vs random on sweeps, wall-clock ahead).
- [ ] E2 at N=256 with the N=100 weights (Gate T2 = the claim); N=36 bonus row.
- [ ] Swap-safety: 2000-sweep block4 run from N=100 reference seeds — no drift below PT ⟨U⟩/N (else investigate before any claim).
- [ ] Commit after each run bank; plots to `artifacts/`.

### Task 5: report + memory
- [ ] `reports/<date>-ka-block-transfer-results.md`: T0/T1/T2 table, sweeps-saved vs the historical local-frame head-start, swap-safety result. Update `ka-glass-size-transfer` + `ipl44-lever-push` memories + MEMORY.md.
