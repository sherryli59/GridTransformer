# D0 regime-probe verdict (mW N=64)

Ladder run 2026-07-14/15 (`d0_regime_probe.py`, task b5b67a1tv). 4 of 5 rungs completed before the
run was SIGTERM'd (exit 143) mid-final-rung — almost certainly GPU contention (a 17 GB process
appeared; 54 GB RAM free rules out OOM), NOT a fault in the run. Per-rung saves preserved all four
completed rungs; rung 5 (T*=0.055) has a partial checkpoint (`d0_probe_T0.0550_N64_ckpt.pt`, 13:55).

## Results

| T*    | U/N     | coll_drift | max_window_drop | g(r) 1st peak | g(r) 2nd shell | verdict |
|-------|---------|------------|-----------------|---------------|----------------|---------|
| 0.0963| −1.6271 | 0.0034     | +0.0275         | 2.48@1.16-ish | broad          | plateau, liquid |
| 0.0850| −1.6567 | 0.0022     | +0.0314         | —             | —              | plateau, liquid |
| 0.0750| −1.6946 | 0.0024     | +0.0429         | 2.48@1.16     | 1.38@1.87      | **T*_work** clean plateau, liquid |
| 0.0650| −1.7490 | 0.0051     | +0.0509         | 2.90@1.14     | 1.61@1.85      | **T*_hard** borderline plateau, liquid (NOT crystal) |
| 0.0550| (incomplete — SIGTERM) | | | | | deferred |

Plot: /mnt/ssd/GridTransformer/reports/logs-2026-07-14/d0_regime_probe.png

## Decisions

- **T*_work = 0.075** (β = 13.33): coldest rung with coll_drift < 0.005 and a confirmed liquid g(r)
  (single broad 2nd shell at 1.87, no split). Plain MC still plateaus here → trustworthy reference,
  and the supercooled structure is developed enough that collective kernels have room to help.
- **T*_hard = 0.065** (β = 15.38): drift just over threshold (0.0051), 2nd shell sharpening to
  1.61@1.85 but still a single peak — a genuine supercooled liquid, not crystallized. This is the
  stress point for the kernel campaign.
- **Crystallization ruled out** at both 0.075 and 0.065 by the single-peaked 2nd shell (mW crystal
  would split it). The `max_window_drop` climbing with cooling is equilibration relaxation from the
  random start, not a post-plateau step-drop.

## Artifacts

- `mw_bank_supercooled_N64.pt` — 1600 configs at T*_work=0.075 (built from the 0.075 rung; no new
  GPU run needed). Feeds the kernel campaign and the MWBlobProposal training β-range.
- β-range training data = the 4 completed rungs (β ≈ 10.4 … 15.4): `d0_probe_T{0.0963,0.0850,0.0750,0.0650}_N64.pt`.

## Deferred

- Rung 5 (T*=0.055) can be finished later by warm-starting `mc_run(..., init_cfgs=)` from the 13:55
  checkpoint when the GPU frees — an even harder point for T*_hard, but not blocking (the 4 rungs
  already give T*_work, T*_hard, and a full β-range).
