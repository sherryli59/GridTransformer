# Learned-augmented PT + transfer/scale crossover — design spec

**Date:** 2026-07-07 (user-directed queue after the GB1 selection-symmetry wall). Both pivots reuse VALIDATED
kernels (A2 heat-bath: exact-mixed, 33% accept, amortized flat over all beta-rungs) and have no discovered
walls. They share infrastructure and run in sequence.

## Pivot 1 — learned-augmented PT ("PT+A2")

**Claim to test:** inserting the beta-conditioned full-cage heat-bath sweep into every PT rung multiplicatively
accelerates PT equilibration. A2's flat 0.32-0.37 acceptance across the ENTIRE ladder (GA2 amortization) is
exactly the property this needs. Exactness is free: each added sweep is pi_beta_rung-invariant MH, always mixed
with displacement (the quasi-non-ergodicity caveat never binds); PT's invariant distribution is unchanged —
pure acceleration. **First deliverable = the missing trustworthy N=256 reference**
([[ka-reference-underconverged]]: current one is 0.067/particle shallow at 16k sweeps).

**Construction** (`ka_reference.parallel_tempering`): optional `hb_ckpt`/`hb_every` — every hb_every PT sweeps,
run one `single_site_mh_sweep` over ALL rungs at per-chain beta (the sweep already takes tensor beta). Default
off => existing paths untouched. Overhead budget: at hb_every=10 the heat-bath adds ~20-50% wall; tune so the
matched-wall comparison is fair.

**Gates:**
- **PT0 (transfer pre-gate, N=256):** A2 was trained at N=100; its conditioning is scaffold-local (n_ctx
  nearest slots, in-frame, fixed-period features) so zero-shot N=256 use is plausible but UNPROVEN. Gate before
  trusting it inside a reference generator: (i) frozen-cage DB invariance test at N=256 (numerically checkable
  at any N, no reference needed); (ii) acceptance on N=256 cold-ladder configs (healthy = same order as N=100's
  33%); (iii) short mixed stationarity from the current N=256 cold rung (band check).
- **PT1 (control, N=100):** PT+A2 vs plain PT at MATCHED WALL from the same seeds. Bar: PT+A2's cold rung
  reaches <= -3.270 (plain PT reached -3.264 and stalled ~0.012 shallow of the true -3.276) OR reaches -3.264
  in clearly fewer sweeps. Both arms judged by the honest gates (seed agreement, cold-drift, finite-size).
- **PT2 (the deliverable, N=256):** PT+A2 with n_equil scaled (>= 32k) + gates. Success = finite-size direction
  fixed (N=256 cold ~= -3.27x, consistent with N=100), seed-pair + cold-drift PASS. Output: trustworthy
  `pt_ladder_N256.pt` + reference (unblocks every sharp transfer claim in the campaign).

## Pivot 2 — transfer/scale crossover

**Claim to test:** PT degrades with N faster than the N=100-trained learned stack (per-sweep costs scale the
same O(N^2) dense-energy here, so the comparison isolates MIXING; PT's known pathology is equilibration time,
measured at N=256). Find the crossover N* where the stack is deeper at matched wall.

**Design** (`ka_scale_crossover.py`): for N in {100, 256, 576}: (a) plain PT for wall W, record cold <U>/N(t);
(b) the A2-stack SMC (flow seeds + disp + SB + heat-bath, adaptive ladder) at the same W, record final +
trajectory. Metrics: U/N at matched wall vs N; wall-to-threshold. At N>=256 there is no trusted reference —
the arms are compared AGAINST EACH OTHER (that is the point: at large N "which sampler is deeper per compute"
is the operative question). N=576 feasibility pre-checks: _scaffold(576) exists; flowhead seeds via the
knn=32 degree cap ([[ka-block-transfer]] validated at 256); SB/heat-bath zero-shot (PT0-style DB gates first).

**Order:** PT0 -> PT1 -> PT2 (overnight) -> crossover (reuses PT2's plain-PT arm at N=256; adds N=576).
Oracle test (in flight) decides Stage-B's fate independently; these two proceed regardless of its verdict.

**Risks:** A2 N-transfer could fail PT0 (then PT+A2 runs N=100-only and the N=256 reference falls back to
longer plain PT — still a deliverable, just slower); N=576 scaffold/seed machinery may need small fixes
(budgeted, pre-checked before any long run); matched-wall fairness requires identical hardware conditions
(sequential runs on the same GPU, no concurrent jobs).
