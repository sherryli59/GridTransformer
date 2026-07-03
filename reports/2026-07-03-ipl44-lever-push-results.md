# IPL44 lever push — results

**Date:** 2026-07-03 · **Branch:** liquid-coupling-flow · **Spec:** `docs/superpowers/specs/2026-07-02-ipl44-lever-push-design.md`
· **Plan:** `docs/superpowers/plans/2026-07-02-ipl44-lever-push.md` · autonomous execution (user delegated)

## 0. TL;DR

1. **Training procedure >> architecture** for the joint flow's denoiser (G1): retraining the *same 23.7k model*
   with EMA+warmup lifted species accuracy 0.83→0.98; a 25× bigger net added only +2.6pt on the honest
   misassigned-site metric (0.757→0.783). Per-species OT is a clear negative for the joint flow.
2. **The two-time curriculum unlocked a geometry oracle.** The coupled-t model reads species from pure geometry
   at chance (0.507 — t=0 also noises positions, so the query is off-manifold). Decoupling (t_pos, t_spec) and
   retraining turns the same architecture into a **0.978-accurate, s-independent misplaced-particle detector**
   at (t_pos=1, t_spec=0).
3. **k-site block relabeling from the frozen geometry table is the wall-clock winner** (G2-block): exact by
   construction (enumeration-verified incl. a sharp-table regression), one denoiser call per sweep. block4
   reaches the reference band in **50 sweeps / ~5 s** from uniform (random: 170 / 11 s; learned pairs: 115 /
   43 s) — **3.4× on sweeps and 2× on wall-clock vs random; 8× wall-clock vs learned pairs.**
4. **Professor forcing: KILLED** (G-PF): 4 configs, discard 0.99→0.975 at best (gate <0.9); insensitive to β
   over 10× and to segment/discriminator size; the strong config destabilized the density (NLL 0.116→0.47).
   Silver lining: PF fine-tuning at small β *improved* the exact likelihood 40% (`pf_b003_best.pt`, nll 0.070
   vs 0.116) with unchanged generation — density and rollout quality are nearly decoupled, again.
5. **Discovery: the joint (x,s) ensemble ≠ the quenched reference ensemble — the mixture DEMIXES.** Strong
   species movers legitimately (kernel exactness proven) drive the system to a phase-separated basin: U/N
   0.260 vs 0.332, unlike-NN 0.416 vs 0.511, g_BB peak 6.23 vs 4.20. Two-sided check: uniform-seeded chains
   plateau at U≈11.2 (2000 sweeps stable); reference-seeded chains stay trapped at 14.6. The additive
   σ_AB=(σ_AA+σ_BB)/2 makes this system **not swap-safe** — the textbook reason swap-MC uses non-additive
   mixtures, rediscovered here by a neural proposer that is *too good* at finding the pathway.

**Program consequence:** the lever works — but its home field is the **non-additive KA 2D glass** (65:35,
swap-safe, PT references at N=36/100/256 exist), which is precisely the system for the next-phase
train-small→sample-big claim. On IPL44, species movers are off-target for any comparison against the
(species-fixed, quenched) Grenioux benchmark.

## 1. G1 — scaling grid (denoiser quality)

Val = 1000 held-out configs; `acc_mismatch` = accuracy on the still-misassigned sites (the honest metric;
overall acc saturates at ~0.98 by copying the input labels).

| tag | params | acc | acc_mismatch | ece | note |
|---|---|---|---|---|---|
| v1r | 23.7k | 0.983 | 0.757 | 0.0006 | same size as historical v1 (0.83 overall) |
| mid | 121.7k | 0.984 | 0.772 | 0.0006 | |
| big | 596.9k | 0.984 | **0.783** | 0.0004 | G2-pair model |
| midsp | 121.7k | 0.979 | 0.683 | 0.0012 | per-species OT — **negative** |
| **midtt** | 121.7k | 0.984 | **0.930** @ t_spec=0.9 / **0.978** @ t_spec=0 | 0.001 | two-time; the geometry table |

Verdict: G1 PASS (monotone scaling), but the *procedure* effects dwarf the *size* effects — EMA+warmup
(0.83→0.98 overall) and the two-time curriculum (0.507→0.978 geometry query) are each worth far more than 25×
parameters. Supports the user's architecture+training hypothesis with the emphasis on training.

## 2. G2-pair — learned pair swaps (big model, 12 cells, 1500 sweeps, B=128)

| seed | random | learned@0.99 | sweeps gain | wall-to-band |
|---|---|---|---|---|
| uniform | 170 | 115 | 1.48× | 11 s → 43 s (**loses 4×**) |
| tfbank | 95 | 60 | 1.58× | 6 s → 24 s (loses) |
| flow | **5** | 5 | — | — |

- Mechanism confirmed: swap acceptance 0.4–0.7% (random) → 24–31% (learned) — the denoiser finds misplaced pairs.
- **Formally below the ≥2× gate**; above the <20% kill bar. Wall-clock inverted at 597k-param prices (16
  denoiser calls/sweep).
- **The quiet headline: flow seeds.** The retrained joint flow's own samples reach the band in 5 sweeps —
  a 34× head-start over uniform. Proper training turned the v1-era flow into a near-equilibrium generator.
- Artifacts: `data/bench_swap_big.png`, `data/bench_swap_curves_big.pt`, log `data/bench_full.log`.
  (v1r ablation deferred by user in favor of the block-relabel arm.)

## 3. G2-block — block relabeling from the frozen geometry table (midtt, 600 sweeps)

Movers: `random`/`pair-tt` (live-s pairs at the on-manifold (t_pos=1, t_spec=0.9) query) vs `block-k`
(frozen (t_pos=1, t_spec=0) table, **one denoiser call per sweep**, conditional-Bernoulli k-site redraw,
Gumbel-top-k x-only site selection).

| seed | mover | sweeps→band | t→band | acc |
|---|---|---|---|---|
| uniform | random | 170 | 11 s | 0.011 |
| uniform | pair-tt | 115 | 43 s | 0.279 |
| uniform | **block4** | **50** | **5 s** | 0.946 |
| uniform | block8 | 40 | 5 s | 0.824 |
| tfbank | random | 95 | 6 s | 0.007 |
| tfbank | pair-tt | 55 | 24 s | 0.307 |
| tfbank | **block4** | 55 | **7 s** | 0.967 |

- **block4: 3.4× sweeps + ~2× wall-clock vs random; ties pair-tt on sweeps at 3–4× less wall-clock.** The
  one-call-per-sweep amortization fixes the wall-clock inversion, as designed.
- Caveat that led to §4: from uniform seeds the block chains pass *through* the band and keep descending —
  the band metric assumes the reference ensemble is the chain's target, which turned out false.
- k≥6 is a known **mixing pathology** (independence-proposal freeze at sharp tables: state-change rate <1%),
  documented in the regression test — not a bias. Use k ≲ N/8 blocks.
- Artifacts: `data/bench_block_midtt.png`, `data/bench_block_midtt.pt`, `data/bench_block.log`.

## 4. The ensemble discovery — demixing under species moves

Two-sided convergence (block4, 2000 sweeps, `annealed_check.py`): uniform-seeded chains plateau at
**U≈11.2**; reference-seeded chains hold **14.6** — broken ergodicity, two basins, e^{βΔU}=e^{34} energy
advantage to the low basin. Structure of the low basin (300-sweep anneal from uniform):

| | U/N | unlike-NN | g_AA | g_AB | g_BB |
|---|---|---|---|---|---|
| quenched reference | 0.332 | 0.511 | 4.35 | 3.99 | 4.20 |
| annealed basin | 0.260 | 0.416 | 3.25 | 3.64 | **6.23** |

**Interpretation: size segregation / demixing.** Like-clustering (unlike-NN ↓, g_BB ↑↑) lets each region pack
its own size optimally, lowering the purely-repulsive energy. Possible because σ_AB is exactly additive here.
The quenched glass reference is *metastable against phase separation*; species moves unlock the pathway; the
learned block mover finds it in ~50 sweeps. Kernel exactness was verified before believing this
(enumeration stationarity incl. sharp-table structured-target regression, 13/13 tests): the chains sample the
joint ensemble *correctly* — the joint ensemble simply isn't the glass.

Implications:
- Any species-moving sampler on IPL44 is **off-target for the Grenioux comparison** (their target is
  species-fixed). Pair swaps at low acceptance merely stay quenched-like transiently (why §2 looked sane).
- The lever's proper arena is a **swap-safe (non-additive) system — the KA 2D glass**, where swap-MC is our
  validated workhorse and PT references exist. That is also the designated transfer-phase system.
- Standalone capability worth noting: a learned, MH-exact block proposer that finds thermodynamic phases
  hidden behind kinetic barriers in ~50 sweeps.

## 5. G-PF — professor forcing (killed)

Pre-gate PASSED (drift share 0.44: FR clash 0.056 vs TF 0.031 — real drift to attack, unlike KA's
all-residual gap; `data/pf_pregate.png`). But 4 configs produced no usable gain:

| config | discard 0.99→ | note |
|---|---|---|
| β=0.03 | 0.975 | nll *improved* 0.116→0.070 (`pf_b003_best.pt`) |
| β=0.10 | 0.979 | |
| β=0.30 | 0.975 | |
| β=0.10, freelen16, GRU128 | 0.986 (=baseline) | destabilized: val nll 0.116→0.47; NLL-guard refused every ckpt |

Verdict: KILL (gate <0.9 unreachable; insensitive to both knob families). Third failed training-procedure fix
for the AR rollout gap (after soft labels, scheduled sampling) — the FR–TF drift on these AR models resists
training-time fixes; correctors (SMC/energy-guarded) remain the lever.

## 6. Assets & code (all committed)

- `liquid_coupling_flow/ipl44/ipl_swap_smc.py` — exact swap-MH pair kernel (paired re-evaluation), position
  Metropolis, chain runner, **conditional-Bernoulli DP + block-relabel kernel** (randomized x-only Gumbel-top-k).
- `liquid_coupling_flow/tests/test_swap_smc.py` — 13 tests: enumeration stationarity (pair, block, sharp-table
  regression), CB-DP vs brute force, kawasaki vectorization exact-equality, equilibrium preservation.
- `joint_flow.py` — **two-time (t_pos, t_spec) conditioning**, per-species OT option, `denoiser_eval`
  (acc_mismatch, t_pos/t_spec), 6× faster training (vectorized Kawasaki + OT bank).
- `train_joint_flow.py`, `bench_swap.py`, `bench_block.py`, `pf_pregate.py`, `ipl_pf.py`, `annealed_check.py`.
- Checkpoints (`ipl44/data/`, gitignored): `jf_{v1r,mid,big,midsp,midtt}_best/last.pt`,
  `pf_{b003,b010,b030}_best.pt` (+ lasts). Plots: `bench_swap_big.png`, `bench_block_midtt.png`, `pf_pregate.png`.

## 7. Recommendation for the next spec

Port the two winning pieces to the **KA 2D glass** and run the deferred headline experiment:
1. Retrain the joint flow two-time on KA data (65:35 — Kawasaki machinery is count-preserving, ports directly).
2. Use the geometry table + block-k mover inside the existing swap-SMC corrector; measure **sweeps-saved to
   the PT reference** (the validated metric) at train-N, then at unseen N — the train-small→sample-big claim.
   KA's non-additive σ makes it swap-safe, so the demixing failure mode of §4 does not apply.
3. Keep flow seeds in the design: the 34× seed head-start (§2) plus per-sweep block acceleration stack.
