# KA block-mover transfer — results (train-small→sample-big)

**Date:** 2026-07-03 · **Spec:** `docs/superpowers/specs/2026-07-03-ka-block-mover-transfer-design.md`
· **Model:** `jf_ka100tt_best.pt` (two-time joint flow, 64|4, trained N=100 on the PT reference, val acc 0.9993)
· **Benchmarks:** `liquid_coupling_flow/artifacts/ka_block_bench_N{100,256}_ka100tt.{png,pt}`, `ka_bench.log`

## 0. TL;DR

1. **Gate T0 (conditional transfer): PASS — with the campaign's key discovery.** The N=100-trained geometry
   table transfers to N=36 out of the box (mm 0.937) but **collapsed to chance at N=256 (0.504)**. Cause:
   full-graph **sum aggregation** — node input mass scales with N. Fix: **k=32 degree cap at eval, no
   retrain** → **0.9905 at N=256** (cost at train size: ~1pt). The learned message function was already local;
   only the graph construction was size-dependent. This is the "genuinely-local architecture" crux resolved.
2. **Swap-safety: PASS.** 2000 sweeps of block4 from PT-reference seeds stay at the PT level (min −3.300 vs
   ref −3.274, drift 0.026 < 0.03). KA's non-additive σ blocks the IPL44 demixing failure, as designed.
3. **Gate T1/T2 (sweeps-saved): the mover alone under-delivers on KA; the seed+mover stack nearly makes the
   2× bar at unseen N.** N=100: flow+block8 = 200/460 sweeps (to −3.0/−3.1) vs uniform+random 590/1200 —
   **2.6–3× combined**, but block-vs-random alone is only 1.1–1.7×. **N=256 (unseen, the headline): 1335 →
   730 = 1.83× combined** (wall 812→512 s) — a real, cliff-free transfer at 2.56× the training size,
   just under the 2× letter of the gate.
4. **Why the mover is marginal on KA (and the deeper lesson):** at T*=0.5 the KA species labeling is nearly
   *determined* by geometry (denoiser 0.999) — species aren't the frustrated channel; position relaxation is.
   Random-swap acceptance 0.5% confirms KA is swap-unfriendly. So: **where species moves are safe (KA,
   non-additive) they are not rate-limiting; where they are rate-limiting (IPL44-like additive) they are not
   safe (demixing).** The block mover's natural home is continuous polydispersity (Berthier-style swap
   systems) — identity exchange both safe and rate-limiting.

## 1. E0 — geometry-table transfer (Gate T0)

N=100-trained two-time model, geometry query (t_pos=1, t_spec=0), acc/acc_mismatch:

| N | full graph | k=32 cap |
|---|---|---|
| 36 (unseen ↓) | 0.954/0.937 | — |
| 100 (train) | 0.9997/0.9996 | 0.993/0.988 |
| 256 (unseen ↑) | 0.647/**0.504** | **0.994/0.9905** |

Mechanism: sum-agg over all N−1 neighbors ⇒ activations scale with degree; up-transfer explodes them
(down-transfer shrinks, tanh-tolerable). `JointSpeciesFlow.knn` (torus k-NN edges) added + full-graph
equivalence test; the `acc_mismatch` metric was essential (overall acc reads 0.975 while mismatch sits at
chance — label-copying masks the collapse).

## 2. E1 (N=100) and E2 (N=256, model trained at N=100, knn=32)

Sweeps (t-to-threshold) to batch-median ⟨U⟩/N ≤ −3.0 / ≤ −3.1; PT-ref −3.274 (N=100), −3.200 (N=256).

| size | seed | mover | →−3.0 | →−3.1 | swap acc |
|---|---|---|---|---|---|
| 100 | uniform | random | 590 | 1200 | 0.005 |
| 100 | uniform | block8 | 520 | 1035 | 0.75 |
| 100 | flow | random | 265 | 770 | 0.001 |
| 100 | **flow** | **block8** | **200** | **460** | 0.87 |
| 256 | uniform | random | 1335 | — | 0.006 |
| 256 | uniform | block8 | 1015 | — | 0.70 |
| 256 | flow | random | 1065 | — | 0.002 |
| 256 | **flow** | **block8** | **730** | — | 0.74 |

- **Seed effect transfers:** flow seeds save 2.2× at train N, 1.25× at 2.56× the size — consistent with (and
  replicating) the historical local-frame head-start (~300 sweeps saved), now with a size-transferred generator.
- **Mover effect is real but small on KA** (1.1–1.7×): with species geometry-determined, most block redraws
  reproduce the current labels (acceptance 0.7–0.94 is largely null moves); the rare genuine relabels help
  most in the −3.0→−3.1 stretch at train N (770→460 with flow seeds).
- **Wall-clock:** frozen-table block moves cost ≈ nothing extra (position sweeps dominate) — sweep gains ≈
  wall gains, unlike the IPL44 pair-proposer's 4× wall penalty.
- **No transfer cliff anywhere**: every learned component (seeds, table, mover) retains its sign and rough
  magnitude at N=256. The claim that fails is only the *magnitude* bar (1.83× vs 2×), not the transfer.

## 3. Safety (KA swap-safety under strong species moves)

Block4 from PT-reference seeds, 2000 sweeps: U/N trace flat at −3.27…−3.30 (drift-below 0.026, threshold
0.03). No demixing-type runaway — the IPL44 §4 failure mode is absent on the non-additive KA, as predicted.

## 4. Gate table

| gate | bar | result | call |
|---|---|---|---|
| T0 conditional transfer | mm within 10pt of train-N | 0.9905 @256 (k-cap), 0.937 @36 | **PASS** (via knn fix) |
| T1 mover at train N | block ≥2× vs random | 1.1–1.7× (mover), 2.6–3× (stack) | mover FAIL / stack PASS |
| safety | no drift below PT | 0.026 | **PASS** |
| T2 transfer claim | ≥2× vs uniform+random @256 | **1.83×** (flow+block8) | NEAR-MISS |

## 5. What this buys the program

- **The size-transfer thesis is now demonstrated at the component level**: a geometry-reading network trained
  at N=100 works at N=256 (0.99) once the architecture respects locality (degree cap). That was the crux.
- The end-to-end 1.83× at unseen N is dominated by the **flow seed**; the species mover adds little on KA
  because species there are geometry-slaved. The mover's payoff regime (frustrated species + safe swaps) points
  to **polydisperse systems** as the right next substrate if the mover is to be a headline; otherwise the seed
  line (flow → SMC) is the one to push (e.g., longer flow training, bigger N targets, N=576/1024 references).
- All infrastructure is now system-agnostic and tested (15/15): kernels, two-time trainer, knn-capped
  transfer eval, benchmark driver (`ka_block_smc.py` e1/safety modes).

## 6. Honest caveats

- −3.1 threshold unreached by all N=256 cells within 1500 sweeps (chains at −3.01…−3.08) — T2 measured at
  −3.0 only; longer runs would sharpen the comparison.
- nB from the artifacts is 39 (N=100) / 95 (N=256), i.e., 61:39 / 63:37 — not exactly 65:35 as the spec
  assumed; read from data, no impact on conclusions.
- The N=256 flow seeds used the knn-capped sampler end-to-end; its position-channel quality at unseen N was
  validated only through the downstream benchmark (it helps: 1335→1065), not by a dedicated structural audit.
- Block acceptance counts include null redraws; the mover's true "work rate" is lower than 0.7–0.9 suggests.

---

## Addendum (2026-07-04): the entire pipeline, end-to-end at unseen N=256

User reframing: the deliverable is the FULL pipeline — flow generates imperfect (x,s) → species-flow-corrected
relaxation → equilibrium — with size transfer of the whole thing. Executed with full trajectory retention
(`ka_e2e_generate.py`, `ka_e2e_correct.py` (MH), `ka_e2e_mala.py` (tamed MALA); trajectories
`artifacts/ka_e2e_traj_*.pt`, raw gen `ka_e2e_gen_N256.pt`).

**Raw generation (N=100-trained model @ N=256, B=1024 in 101 s):** g(r) peak positions correct, amplitudes ~2×
smeared, spurious B–B contact peak; 3.1%/particle core overlaps → U/N median +105; species 28.9% wrong
(`ka_e2e_gen_N256.png`).

**Position kernel upgrade — tamed MALA:** naive MALA froze at the raw configs (acceptance ~0: LJ-core forces
propose absurd jumps); per-particle force cap in the proposal (exact) fixes it. All-particle batched moves:
13 ms/step vs 2.3 s/sweep for sequential MH (whose bottleneck is measured to be launch latency, not energy
math — local-ΔU O(N)/move gave only 1.3×; checkerboard cells pay only at N≥576).

**Final plateaus at N=256 (all learned parts trained at N=100, knn=32):** `ka_e2e_final_energy_dist.png`

| arm | plateau U/N | note |
|---|---|---|
| PT reference | −3.198 | |
| flow → MALA+block8 | **−3.189** | energy distribution ≈ ref in shape+width; ~40 min from raw gen |
| uniform → MALA+block8 | −3.188 | same depth, ~2× slower → **flow seeds = speed, not depth** |
| flow → MALA+random | −3.162 | plateaued shallower → **mover = depth: quenched species-disorder penalty ~0.03/particle that random swaps (0.8% effective) cannot relax** |
| flow → MH+block8 | −3.115 @ 3000 sweeps | position kernel is the largest lever |

**Structure endpoint** (`ka_e2e_final_gr.png`): gAA, gAB indistinguishable from PT ref; gBB main shells match,
residual B–B contact shoulder (1.4 vs 0.95) — the known slowest mode. Species-oracle disagreement 28.9% → 1.7%.
τ_U ≈ 27–35 iterations, kernel-independent at stationarity; block-move effective rate →0 at equilibrium
(corrector, not mixer — its work happens during relaxation, 4.4% effective early).

**Caveats:** 1–2 of 128 chains stuck at nan (raw configs with exactly coincident particles: r²=0 →
tamed force nan; full-recompute MH also cannot recover inf configs — inf−inf=nan). Fix = r² clamp in
forces/energy rows; excluded from stats. MH+random arm capped at 1250 sweeps (crossed −3.0; purpose served).

**Bottom line:** the entire pipeline transfers: N=100-trained generator + geometry table drive a 256-particle
glass from scratch generation to PT-reference-level equilibrium (energy distribution and g(r)) in ~40 min,
with clean component attribution: MALA = biggest speed lever, flow seeds = 2× time-to-plateau, learned block
mover = final depth (removes quenched species disorder).

## Addendum 2 (2026-07-04): corrected convergence + decorrelation analysis

Re-analysis at the full 6000 MALA iterations (each = 10 MALA steps + species block), MH arms dropped.
Plots: `ka_e2e_pt_convergence.png`, `ka_e2e_decorrelation.png`, `ka_e2e_gr_overlay.png`; arrays
`reports/logs-2026-07-04/e2e_analysis_data.pt`. Analysis script `scratchpad/e2e_analysis.py` (copied to
`reports/logs-2026-07-04/`).

**The PT N=256 reference is NOT converged (supersedes "PT −3.198" and "energy dist ≈ ref" above).**
Recomputed with `ka_energy`: PT mean U/N = **−3.2112** (std 0.038, skew 0.15, kurt ≈0 — near-Gaussian
*marginal*). But block-means in save-order drift **monotonically −3.202 → −3.219** (−0.016/particle) and are
still trending down in the last blocks. The stored `plateau_drift=0.0011` measured the wrong window and hid
this. Consequences: (i) the true equilibrium is *below* −3.219; (ii) the 14400 configs are autocorrelated PT
snapshots, so effective-N ≪ 14400 (thin data, jagged when compared against). The corrector's residual gap
(MALA plateau ≈ −3.186 vs PT −3.211) is therefore partly a *moving-target* artifact: PT keeps deepening AND the
MALA arms are themselves still creeping (block8 −3.177→−3.185 over iters 5000–6000). **Neither ensemble is
fully converged; both approach a common deeper equilibrium.** A converged PT N=256 reference (longer run /
decorrelated saving) is the prerequisite for any sharp gap claim.

**Decorrelation (detrended per chain, window it>4000):**

| arm | τ_U (energy) | τ_s (species) |
|---|---|---|
| flow → MALA+block8 | 21 | 166 |
| uniform → MALA+block8 | 19 | 187 |
| flow → MALA+random | 20 | 103 |

- **τ_U ≈ 20 iters, arm-INDEPENDENT** (the three energy-ACF curves lie exactly on top of each other) → energy
  and structural decorrelation are governed entirely by the MALA *position* dynamics; the species mover and the
  seed do not change the decorrelation rate. (Detrending removes the residual downward drift that had inflated
  the earlier "27–35" figure — the detrended ~20 is the cleaner fluctuation autocorrelation time.)
- **Species is the slow mode** (τ_s 5–9× longer) and IS mover-dependent: random shows a fast initial drop
  (the ~0.8% accepted swaps flip labels ~uncorrelated) then a frozen tail; block8 decays gradually. The 2000-it
  window only marginally resolves this slow glassy mode, so τ_s are rough / lower-bound.

**g(r) overlay** (`ka_e2e_gr_overlay.png`; full pipeline, uniform→block8, raw egnn flow, **PT dashed**): g_AA and
g_AB from both correctors sit exactly on the PT dashed line; raw egnn flow is badly smeared (broad first peak,
spurious B–B contact). g_BB main shells match; the pipeline slightly under-fills PT's intermediate r≈1.4 feature
— but PT is itself under-converged there, so that residual is partly PT noise. Full-pipeline and uniform→block8
are **structurally indistinguishable**, reinforcing flow-seed = speed-not-depth.
