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
