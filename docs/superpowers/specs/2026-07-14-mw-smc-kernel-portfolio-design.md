# mW SMC transition-kernel portfolio (suffix + two-blob), judged by energy-evals saved

**Date:** 2026-07-14 · **Branch:** liquid-coupling-flow · **Status:** spec approved in brainstorm, pending user review

## 1. Goal

Port the glass-line collective transition kernels — **ordering-suffix resample** and **two-blob
redistribution** — to the mW (Stillinger–Weber monatomic water) liquid, deployed inside a tempered
SMC bridge, and determine whether they reduce the number of **expensive SW energy evaluations**
needed to equilibrate at supercooled conditions versus plain single-site MC.

**Primary metric:** total SW energy evaluations (full `mw_energy` calls normalized to `du_move`
locality; count both mutation and bridge-reweighting evals) to reach matched equilibration depth
(⟨U⟩/N plateau + structural observables). Baseline: `mw_reference.mc_run` from identical seeds.
Per user directive: judge by full distributions (U histograms, g(r), per-walker spread), never a
single scalar.

## 2. Why the bridge is primary (not plain block-MH)

At mW ambient T\* = 0.09632, β ≈ 10.4; supercooled is colder. A K-particle block proposal needs
total energy excess ≲ a few × T\* for non-vanishing plain-MH acceptance. The glass line measured
this wall directly: block-conditional fine-tuning cut proposal excess ~50× and acceptance still sat
at ~1% ("fine energy precision" limit). Plain-MH collective moves at target β are presumed dead
(acc ≈ 0) — but D1 below *measures* the forecast instead of assuming it, since small K may survive
(glass A2 heat-bath: K=1 full-cage, 33% accept).

The tempered bridge π_λ ∝ q0^{1−λ} e^{−λβU} is what converts "acceptance 0 at λ=1" into useful
work: collective kernels only need to mix the population at low-to-mid λ, before the weights
sharpen; filtering handles the rest. All glass suffix/two-blob positives were bridge results.

## 3. Kernel portfolio and λ-scheduling

Each kernel is MH'd with the **exact** geometric-bridge ratio
(`ka3d_smc_bridge.geometric_bridge_log_accept`): full log-q0 ratio required for any proposal that
is not an exact conditional of q0 (a true ordering-suffix is; anything else is not).

| Kernel | What it does | Where it pays | Energy cost / attempt |
|---|---|---|---|
| **Suffix resample** (global AR v10) | Redraw the tail (random length m ∈ [n/2, 3n/4]) of the canonical ordering from q0's conditional | λ ≈ 0 (near-Gibbs, glass acc 0.95); basin mixing | ~full recompute — use sparingly, low λ only |
| **Two-blob redist** (block-conditional model) | Jointly regenerate the union of two well-separated K_r-blobs; particle **count** redistributes between them under the union budget (monatomic — no species, cleaner than glass) | low-to-mid λ; the density-fluctuation mover, can touch the ordering-prefix head | local (two cavities + shells) |
| **Single-site `du_move` sweeps** | Standard displacement MC | all λ, the λ→1 workhorse | 1 local eval |
| **(Conditional) K≤2 learned heat-bath** | Full-cage single/pair-site regeneration | all λ, only if D1 shows acceptance clears ~10% | local |

**Ergodicity of the composition** (the "fixed subset" concern): suffix length is random per move,
blob centers are drawn fresh per move, and single-site sweeps touch everything — no particle or
region is pinned across a run, and no per-region particle count is conserved (two-blob breaks the
canonical single-blob constraint). Validity: a cycle/mixture of π_λ-invariant kernels is
π_λ-invariant; irreducibility is a property of the composition, not any single member.

**Re-rooted suffix (option, gated by D2):** redraw the ordering origin (torus shift) per move and
resample the suffix of the shifted ordering — the suffix subset then varies spatially per move.
This is exact only with the full q0 ratio (proposal is the conditional of q0_shifted, not q0);
viable iff v10's log-q0 is sufficiently shift-invariant. v10 *was* trained with exact torus-shift
+ O_h augmentation, so D2 is a genuine measurement, not a formality.

**Delayed-acceptance lever (new for mW):** two-stage MH — stage 1 filters with a cheap surrogate
ΔU (2-body only, or the model's learned tilt), full 3-body SW eval only for stage-1 survivors.
Exact, and attacks eval cost through cheaper *rejections*, complementary to better proposals.
Nothing in the repo does this yet.

## 4. Order of work

- **D0 — pick the regime.** Plain-MC struggle probe: run `mc_run` at a T\* ladder below ambient
  (e.g. 0.09 → 0.06) and pick the highest T\* where plain MC visibly fails to plateau in an
  affordable eval budget. At easy states nothing saves evals (LJ-line lesson); the metric needs
  room to show a win. Also generate the supercooled training/seed banks here.
- **D1 — acceptance forecast, no chains.** Sample K-blocks from the existing
  `mw_block_N64_mle.pt` (and the mW-ported scaffold model if a checkpoint exists), histogram
  β·ΔU vs K at the D0 temperature. Acceptance is bounded by exp(−β·ΔU): read off K\* where
  ~10% is reachable, decide whether the heat-bath row and any plain-MH deployment exist at all.
- **D2 — shift-spread on v10.** Score a bank of configs under log q0 at random torus origins;
  plot the spread distribution. Decides the re-rooted-suffix option (spread ≪ typical MH slack
  → viable).
- **Fine-tune if D1 demands it:** free-run capped-energy REINFORCE on the block model per the
  2026-07-14 handoff (reuse `train_capacity_rl.py` pattern, mW energy swapped in, no species).
- **Main experiment — bridge ablations at D0's T\*:** SMC with mutation portfolios
  {single-site only} vs {+suffix} vs {+two-blob} vs {+both}, alien/data-seed bracket style,
  scored in total SW evals vs the plain-MC baseline to matched depth. Save all configs,
  per-walker observables, and eval counters (record-simulation-data directive); checkpoint each
  rung to disk as it completes (checkpoint-incrementally directive).

## 5. Gates

- **GO:** some portfolio reaches a depth/observable target the plain-MC baseline cannot reach at
  ≥3× the eval budget, or reaches the shared target with ≥2× fewer evals — reproduced across
  ≥2 seeds, distributions inspected.
- **NO-GO:** bridge overhead eats the collective-kernel gains at every T\* tried (i.e., plain MC
  matches or beats total-eval cost at matched depth). Negative is publishable-to-memory: it would
  localize the kernels' value to the glass (species) setting.

## 6. Assets reused (do not rebuild)

`mw_energy`/`du_move`/`mw_energy_chunked`; `mw_reference.mc_run` (baseline + banks, has
`init_cfgs` seed hook); `mw_generator_v10` (global AR, exact density, shift+O_h augmented —
suffix + D2); `mw_block_ar` + `mw_block_N64_mle.pt` (block-conditional — D1, two-blob);
`mw_scaffold.py` (KA cavity model ported to mW cavities); `ka3d_smc_bridge.geometric_bridge_log_accept`
(exact ratios); `mw_smc.py` (λ-scheduler/ESS machinery); glass diagnostics as templates
(`diag_suffix_cross.py`, `diag_twoblob_redist_cross.py`).

## 7. Non-goals

- Free-energy / logZ estimation (mw-ersi line owns that; different metric).
- Beating swap-MC-style specialist kernels — mW is monatomic, no swap analogue.
- New architectures. Everything runs on existing models; only fine-tuning is in scope.
- SMC as an equilibration recommendation at *ambient* T\* (known: costs evals, no room to win).

## 8. Risks / open questions

- Suffix moves at λ>0 may decay fast in acceptance even along the bridge for m ~ N/2 (glass ran
  them at cold-ratio-valid special cases; mW's stiffer 3-body network may punish tails harder).
  Mitigation: random-length schedule biased smaller as λ grows.
- Two-blob redist in mW redistributes *count*, not species; if supercooled-mW density
  fluctuations are not the relevant slow mode, the move may be neutral. The ablation isolates it.
- N=64 is small for basin structure; if bridges trivially equilibrate at N=64, escalate box size
  before declaring anything.
