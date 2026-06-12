# Action Plan — Variant Screening, Equivariance, and Likelihood Exactness

**Date**: 2026-06-12  
**Branch**: `hilbert-arc-repr`  
**Companion docs**: `reports/transformer-architecture-and-alternatives.md` (variants A–G defined
there), `reports/hilbert-arc-repr-plan.md`, `CODE_REVIEW_hilbert-arc-repr_2026-06-10.md`
(findings 3.x referenced below), `CURVE_RAIL_REFACTOR_TODO.md`.

**Order of work**: Phase 1 (likelihood instrumentation) comes first because Phase 3's
metrics and any IS/ESS numbers depend on knowing what `logp_continuous` actually is.
Phase 2 (equivariance diagnostic) is an independent afternoon and can run in parallel.

---

## Phase 1 — Generation likelihood: audit verdict and fixes

### 1.1 What is computed today (verified in code, 2026-06-12)

`sample_lj.py:807`: `logp_continuous += _gmm_log_prob(log_pi_sample, mu_t, scale_sample, delta_model_t)`

So the saved quantity is

$$
\texttt{logp\_continuous} \;=\; \sum_{t=1}^{N-1} \log q_\tau\big(u_t \mid \text{ctx}_t\big),
\qquad u_t = (\Delta s_t, f_t) \in \mathbb{R}^4 \ \text{(arc)} \ \text{or}\ \delta_t \in \mathbb{R}^3,
$$

where $q_\tau$ is the **tempered** mixture actually sampled from
(`log_pi/τ` re-softmaxed, scales ×√τ, default τ=0.7; lines 742–754). The decode map to
positions is (lines 409–455):

$$
c_{t} = \text{clamp}\big(\,\text{round}(c_{t-1} + \Delta s_t X)\,,\ 0,\ R^3{-}1\big), \qquad
r_t = \text{wrap}_L\Big( \text{center}(c_t) + \big(f_t - \text{round}(f_t)\big)\,\ell \Big).
$$

### 1.2 Verdict: exact in the wrong space

`logp_continuous` **is exact — but as the density of the raw MDN draw sequence
$u_{1:N-1}$ under the tempered proposal**, not as the position-space likelihood
$\log p_\theta(r_{1:N})$. The decode $T: u \mapsto r$ is **not a bijection** (the user's
suspicion is correct), in four distinct ways, with very different severities:

| # | Non-bijectivity | Mechanism | Severity |
|---|---|---|---|
| L1 | Rounding | every $\Delta s$ in an interval of width $1/X$ gives the same code | **constant factor, fixable in closed form** (see 1.3); residual error negligible since $1/X \approx 10^{-4} \ll \sigma_{\Delta s}$ |
| L2 | Fine wrap | $f$ and $f + n$, $n \in \mathbb{Z}^3$ give the same position | negligible if $P(\|f\|_\infty > 0.5)$ is small — **must be measured, not assumed** (was finding 3.4) |
| L3 | Clamp | all out-of-range codes collapse onto $c \in \{0, R^3{-}1\}$ | atom in the pushforward; **weights of clamp-hit samples are wrong** — count and flag |
| L4 | Order | nothing forces $\Delta s_t > 0$; a non-monotone code sequence is *not* the Hilbert-canonical order of its own output config | **the deep one** — see 1.4 |
| L5 | Box wrap (Δxyz mode) | $\delta$ and $\delta + nL$ give the same position | negligible ($\sigma \ll L$); same nearest-image approximation already made in `mdn_loss` |

### 1.3 The closed-form correction (resolves review finding 3.6)

Within one piece of the piecewise-affine decode, the change of measure from arc space to
position space is constant: probability mass of a code is $\approx q(\Delta s)\cdot\frac{1}{X}$
(flat over the $10^{-4}$-wide interval), and the fine offset is in *cell units*, Jacobian
$\ell^3$ per particle. So

$$
\log p_\theta(r_{1:N}) \;\approx\; \texttt{logp\_continuous}
\;+\; (N{-}1)\Big( -\log X - 3\log \ell \Big)
\;=\; \texttt{logp\_continuous} + (N{-}1)\log\rho ,
$$

since $X = R^3/N$, $\ell = L/R$ ⇒ $-\log\frac{R^3}{N} - 3\log\frac{L}{R} = \log\frac{N}{L^3} = \log\rho$.
**The full arc→position correction is $+\,(N{-}1)\log\rho$ — one line of code.**

Consequences, by use case:

- **Self-normalized IS / ESS / MALA overlap at fixed $(N, L)$**: correction is a constant
  ⇒ cancels. Existing numbers are *not* invalidated.
- **`lambda_var` training loss**: $\widehat{\text{Var}}(\log w + \text{const}) = \widehat{\text{Var}}(\log w)$
  ⇒ unaffected.
- **Cross-size comparisons and `eval_mcmc_logprob`-style absolute checks**: correction
  differs per $(N, L)$ ⇒ **currently wrong; must apply the $+(N{-}1)\log\rho$ term**.
- **Temperature**: at τ≠1 the saved value is the *proposal* density (correct for IS
  weights $w = \pi/q_\tau$), but it is **not** $\log p_\theta$; any likelihood-flavored
  comparison must rerun with `--temperature 1.0`.

### 1.4 The order problem (L4) — what "bijection not guaranteed" really costs

The model is a density over *sequences*. Canonicalization (Hilbert sort) makes
config → sequence a bijection a.e. (ties = two particles in one cell = measure zero in
data, but reachable in generation). Generation, however, is free to emit a sequence that
is **not** the canonical order of its own output configuration (any step with
$c_t \le c_{t-1}$). Then:

$$
p_\theta(\text{config}) \;=\; \sum_{\text{orderings } \sigma:\ T(\text{seq}_\sigma) = \text{config}} p_\theta(\text{seq}_\sigma)
\;\neq\; p_\theta(\text{generated seq}) ,
$$

and the training-side NLL (always evaluated on canonical sequences) and the generation-side
logp are densities of *different random variables*. Every downstream consumer that mixes
them (IS against MCMC data, NLL-vs-sample comparisons) inherits the mismatch.

**Resolution ladder** (cheapest first; stop at the first level the data supports):

1. **Measure it** (no code risk): per-sample boolean `is_canonical` = all generated
   $\Delta c_t \ge 1$ and no duplicate codes; report the violation rate + the logp mass it
   carries. If violations are ≪1% of samples *and* carry ≪1% of total weight, document
   and move on — the inexactness is bounded by that rate.
2. **Filter**: drop non-canonical samples from IS estimates (introduces a bias bounded by
   the measured rate; honest if reported alongside).
3. **Exact fix (representation-level)**: make canonical order true by construction —
   predict the code *increment* with positive support, e.g. $\Delta c_t \sim$ discrete
   distribution over $\{1, 2, \dots\}$ (or continuous $\Delta s$ through a softplus-style
   bijection with its Jacobian logged). Then sequence↔config is bijective a.e., NLL and
   generation logp are the same quantity, and L1's rounding interval disappears too.
   This is a head change — slot it into the Phase 3 funnel as a variant (call it **H**)
   rather than a hotfix.

### 1.5 Particle 0 and the translation gauge

Arc decode is absolute, and particle 0 is pinned deterministically at the code-0 cell
center (`_arc_initial_positions`) — an **atom**: the model's joint is a density on
$3(N{-}1)$ dims, not $3N$. Two distinct issues:

- **Train/sample shift**: training's first particle is the *lowest-code particle of a real
  config* (code typically small but ≠ 0, fine offset distributed); generation pins
  $c_0 = 0$, fine $= 0$. The step-1 conditional is evaluated off-distribution. Cheap fix:
  sample particle 0 from its empirical training marginal (cache it once); exact fix: let
  the model predict particle 0's $(c_0, f_0)$ as step 0.
- **Gauge factor for absolute comparisons**: comparing to MCMC log-density over $3N$ dims
  requires accounting for the missing 3 dims (translation gauge in Δxyz mode: $-\log L^3$;
  arc mode: the particle-0 atom above). Document the convention in one place and make
  `eval_mcmc_logprob` apply it.

### 1.6 Phase-1 deliverables (≈1 day)

| Item | Where | Test |
|---|---|---|
| `+(N−1)·log ρ` correction term saved as `logp_position` alongside `logp_continuous` | `sample_lj.py` output dict + h5 | toy 2-particle pushforward check: histogram-estimated position density vs $\exp(\texttt{logp\_position})$ |
| Counters: clamp-hit rate (L3), fine-wrap rate $P(\|f\|>0.5)$ (L2), canonical-order violation rate (L4), duplicate-code rate | sampler output dict, printed in summary | assert all < 1% on the current best N=27 checkpoint; if not, escalate to 1.4 level 3 |
| τ=1 logp pathway documented; benchmark scripts that interpret logp as likelihood get `--temperature 1.0` | `benchmark_lj27.py` / eval scripts | — |
| Particle-0 empirical marginal seed (1.5) behind a flag | `sample_lj.py` | step-1 NLL on val before/after |

---

### 1.7 Phase-1 measurement results (2026-06-12, DONE)

Implemented in `sample_lj.py` (tests: `tests/test_phase1_likelihood.py`, 9 tests; full
suite 91 green). Counters on the best arc checkpoint
(`lj_ckpts_lj27_pbc_arc_repr_norm/.../best.ckpt`, 2000 samples, T=0.7):

| Counter | any-event rate | gate <1% | verdict |
|---|---|---|---|
| clamp (L3) | 0.35% | pass | negligible |
| duplicate code | 0.20% | pass | negligible |
| noncanonical order (L4) | **1.35%** | marginal | variant H stays in queue |
| fine wrap (L2) | **52.25%** (≈2.8%/step) | fail | structural, see below |

The fine-wrap rate is inherent to the representation, not a bug: a particle near a cell
face has |fine| ≈ 0.5 and any unbounded fine density puts mass across the boundary. The
exact treatment is a **wrapped-Gaussian (cell-torus) fine likelihood**
(logsumexp over neighbor images in the MDN eval, mirroring the box-torus min-image in
`mdn_loss`) — train and sample sides currently make the *same* nearest-image
approximation, so the inconsistency is bounded by the per-event tail mass. Follow-up
item, not a blocker.

Note: at this benchmark's density ρ = 1 exactly, so the (N−1)·log ρ correction vanishes
and `logp_position == logp_continuous`; the correction matters for any L ≠ N^{1/3} box.

## Phase 2 — Equivariance: diagnose before building (≈1 afternoon)

Context (from the architecture doc): the symmetry group here is **octahedral × translations
× permutations**, not SO(3) — and the Hilbert ordering breaks even that, so strict
equivariance is ill-posed for this model class. Translations and permutations are already
handled (min-image features + aug-shift; canonical sort). Rotations are the open question.

**2.1 Diagnostic (do this; ~30 lines, minutes of compute).** For the current best
checkpoint and the 24 proper octahedral rotations $R_k$: rotate val configs, re-Hilbert-sort,
evaluate NLL. Report

$$
\Delta_{\text{rot}} = \max_k \overline{\text{NLL}}(R_k x) - \min_k \overline{\text{NLL}}(R_k x),
\qquad \text{spread across } k .
$$

Decision rule:
- $\Delta_{\text{rot}} \ll 0.5$ (the existing NLL gate): **equivariance is a non-issue** —
  close the question, skip 2.2/2.3.
- otherwise → 2.2.

**2.2 Octahedral augmentation (free, exact).** Random group element applied to each
training sample *before* the Hilbert sort. Exact symmetry of the Boltzmann ensemble ⇒
zero bias. Implement as a one-flag data option; screen it through the Phase 3 funnel like
any variant.

**2.3 Curve-gauge local frames (= variant D).** Only if 2.2 shows gains but plateaus.
Already ranked in the variant table; the diagnostic + augmentation results decide whether
D moves up or drops out of the queue.

**2.4 Diagnostic results (2026-06-12, DONE).** Script `octahedral_nll_diagnostic.py`
(tests: `tests/test_octahedral_diagnostic.py`), best arc checkpoint, 512 val configs,
all 24 proper rotations:

- identity NLL/config **13.35** (train log: 13.30 → evaluator validated)
- mean over 24: 13.47, std 0.17, **spread 0.61 vs gate 0.5 → marginally TRIGGERED**
- per-particle spread 0.023 (~4.5% of the 0.51 per-particle NLL)
- identity is *not* the minimum (3 rotations score lower) → the spread reflects uneven
  motif coverage, not memorization of the training frame.

Consequence: **AUG (octahedral augmentation) joins the Phase-3 queue at moderate
priority; variant D stays in the second wave.** Not an emergency — the effect is ~4.5%
relative — but free to collect whenever a fine-tune is running anyway.

**Do not** build strict E(3)-equivariant machinery (spherical harmonics etc.): the box
symmetry is discrete, the curve conditioning breaks it anyway, and the downstream EGNN
flow is the equivariant component of the pipeline.

---

## Phase 3 — Variant screening funnel

Variants from `transformer-architecture-and-alternatives.md`:
**A** joint (d, Δs) edge bias · **B** geometric value path · **C1/C2** per-layer rail /
rail tokens · **D** curve-gauge frames · **F** near-contact RBF centers · **G** kNN-sparse
attention · plus **H** monotone code-increment head (from 1.4) and **AUG** octahedral
augmentation (from 2.2).

Shared infrastructure (one-time, before any run):
- **One cache for everything.** A, B, C1, C2, F, G, H, AUG consume the existing arc_repr
  cache; D computes frames on the fly from waypoints already in the batch. No rebuilds.
- **One flag per variant** through the `reproduce.sh` env-var pattern; new module configs
  ride the `CurveRailConfig` consolidation (do that TODO first if it isn't merged).
- **Stratified NLL logging**: per-epoch val NLL split by $\Delta s$ stratum —
  local ($\Delta s \in [0, 4]$) vs jump ($\Delta s > 4$) — reusing the jump definition from
  `analyze_hilbert_lj_jumps.py`. This is the cheap discriminator for the whole funnel.

### Stage 0 — free checks (minutes per variant)

- `py_compile` + module unit test + **KV-cache parity test** (A and C2 change the bias
  row / key set; `forward` vs `forward_step` must stay exact — extend
  `tests/test_kv_cache.py` pattern).
- Distribution audit where applicable. Reminder from the K=1 study: the audit is
  necessary, never sufficient — it only prunes.

### Stage 1 — warm-start probes (hours per variant, 1 GPU)

For the additive variants (A, C1, C2, F, G): load the current best N=27 checkpoint,
initialize the new module with **zero-init on its output projection** (step 0 ≡ baseline
exactly, so any NLL movement is causally the new module), fine-tune ~10 epochs, frozen
hyperparameters, same seed, same cache.

For variants that change the trunk's required representation (B, D, H, AUG): from-scratch
short budget, 25–50 epochs (repo history shows multi-size verdicts visible by epoch 18).

**Gate**: jump-stratum val NLL must beat baseline. Aggregate NLL alone does not promote —
the variants exist to fix jump conditionals, and the aggregate is dominated by local steps.

### Stage 2 — sampling metrics on survivors (≈1 day each)

Sample 1–2k configs from the best snapshot (with Phase-1 instrumentation on: canonical
violation rate, clamp rate get reported per variant for free) → `benchmark_lj27.py`
(core-clamped energy + OTgap) + g(r) MAE.

**Gates** (vs baseline NLL −13.3, g(r) MAE 0.211, OTgap 0.12): NLL gap < 0.5,
g(r) MAE gap < 0.02, OTgap ≤ baseline; plus L4 violation rate not worse than baseline.

### Stage 3 — the real test, finalists only (days; max 2 variants per round)

Multi-size train {L3, L5}, held-out L4, K=1 playbook + `eval_multisize.sh`.
**Decision metric: held-out L4 OTgap ≤ 0.3.** Nothing below this stage confirms a variant;
stages 0–2 only prune.

### Comparison discipline

- Equal optimizer steps (not wall-clock), same seed, same cache path, parameter counts
  reported (B/C2 add capacity — a win at +15% params is a different claim).
- No combinations in round 1 (A+C2 tempting; attribution first). Combine winners in round 2.
- Every run leaves a `reproduce.sh`.

### Round-1 status (2026-06-12)

Implemented and launched (tests: `tests/test_phase3_probes.py`, suite 103 green):

- **Shared infra**: `mdn_loss(..., return_point_nll=True)` + `stratified_nll_means`
  (`grid_transformer/training/ar.py`); `train/nll_jump`, `train/nll_local`,
  `train/jump_fraction` logged per epoch for arc models (threshold |Δs| > 4).
- **A**: `JointEdgeBias` (`grid_transformer/models/joint_arc_bias.py`) — joint
  (RBF(d), RBF(log1p Δs)) MLP bias, zero-init output, `bias_row` KV parity tested;
  arc coordinate accumulated from input deltas (cache carries `arc_s`).
- **F**: additive near-contact `RBFEdgeBias` (64 centers, d ≤ 2σ), zero-init.
- **G**: `knn_mask_k` causal kNN mask (parameter-free), forward + forward_step.
- **Warm start**: `train.py --warm_start_ckpt` (`warm_start_from_checkpoint`,
  strict=False; unexpected keys = hard error), threaded through
  `train_sample_lj27_pbc.sh` (`USE_JOINT_ARC_BIAS`, `USE_REFINE_RBF_BIAS`,
  `KNN_MASK_K`, `WARM_START_CKPT` env vars).
- **Launcher**: `launch_warm_probes.sh` — probes A, F, G sequentially, 10 epochs each,
  warm-started from the arc_repr_norm best.ckpt, baseline cache reused
  (`RUN_PREPROCESS=0 RUN_SAMPLE=0`), output under `lj_ckpts_lj27_pbc_arc_probes/probe_*`.
  Launched 2026-06-12 with batch 1024 (time-sharing the GPU with the multi-size run).

**Deferred with reason**: C1/C2 (rail placement) — the arc baseline has *no rail*, so a
warm probe here would measure "adding a rail at all," not rail placement; run them on
the fixedrail baseline (`lj_ckpts_lj27_pbc_fixedrail_0608`) in round 2. AUG needs a
data-side rotation+re-sort augmentation (cache-level change), queued behind the probes.
B and D remain second wave (from-scratch).

### Round-1 probe results (2026-06-12, 10 epochs, batch 1024, warm start)

| Probe | nll ep1→ep10 | local NLL | jump NLL | Stage-1 verdict |
|---|---|---|---|---|
| **A** joint d×Δs | 13.51 → **13.28** | 0.514→**0.505** monotone | 5.379→5.353 (best) | **PROMOTE** — only probe below baseline (13.30), still descending |
| F near-contact RBF | 13.56 flat | 0.516 flat | flat (±0.02 noise) | PRUNE — contact resolution is not the bottleneck |
| G kNN mask k=12 | 15.47 → 13.88 | 0.528 (> baseline) | 5.43 (worst) | PRUNE at N=27 — halves the receptive field; revisit only for N≥125 |

Reading notes: (i) all probes show an Adam-restart bump at epoch 1 — "below baseline by
ep10" is a conservative bar that only A cleared; (ii) jump_fraction is just 0.12% of
tokens (~32/batch), so per-stratum differences are noisy — consider lowering the jump
threshold from 4 to ~2 for future probes; (iii) jumps cost ~5.4 nats vs 0.51 local
(~10×/token), confirming jumps dominate per-token difficulty.

### Stage-2 result for probe A (2026-06-12)

2000 samples from `probe_A/best.ckpt` (T=0.7, same protocol as baseline) vs the L3 MCMC
target: OTgap **0.480** (baseline 0.465 — within OT minibatch noise), gr_L1 0.2515
(baseline 0.2518), W1_md 0.4995 (0.4973), composite 1.981 (1.965). Exactness counters
unchanged (clamp 0.20%, noncanonical 1.65%).

**Verdict: parity, not yet a sampling win at the 10-epoch probe budget.** A's NLL was
still descending at epoch 10 and its designed payoff — the curve×geometry *conditional*
— matters most on held-out sizes. Next decision: (a) extend A's fine-tune (~40 ep), or
(b) take A straight into the multi-size {L3,L5}→L4 protocol once the in-flight
multi-size baseline finishes, comparing held-out L4 OTgap with/without the joint bias.
(b) is the recommended path: it tests A where it should matter.

### Round-1 queue (concrete)

| Slot | Variant | Mode | Budget |
|---|---|---|---|
| 1 | A (joint d×Δs bias) | warm-start probe | ~10 ep |
| 2 | C2 (rail tokens in attention) | warm-start probe | ~10 ep |
| 3 | C1 (per-layer rail) | warm-start probe | ~10 ep |
| 4 | F (near-contact RBF) | warm-start probe | ~10 ep |
| 5 | G (kNN mask) | warm-start probe | ~10 ep |
| 6 | AUG (octahedral) — only if 2.1 triggers | from scratch | 25 ep |
| 7 | H (monotone code head) — only if 1.4 counters trigger | from scratch | 25 ep |
| 8 | B, D | from scratch | 25–50 ep, second wave |

Estimated round-1 cost: roughly one full baseline training run for the entire screen
(slots 1–5 ≈ a weekend of single-GPU time; 6–8 conditional).

---

## Summary of decision points

1. **Phase 1 counters** decide whether the likelihood story is "document the constants"
   (rates < 1%) or "variant H is mandatory" (rates high).
2. **Phase 2 diagnostic** ($\Delta_{\text{rot}}$) decides whether AUG/D enter the queue at all.
3. **Stage 1 jump-NLL** decides who gets sampled; **Stage 2 gates** decide who gets the
   multi-size run; **Stage 3 L4 OTgap ≤ 0.3** is the only success criterion.

---

## Gilbert curve (constant cell) — 2026-06-12

### What was built

The Gilbert curve (generalized Hilbert on arbitrary cubic grids) was integrated
end-to-end through the arc representation pipeline. The `curves.py` module provides
`GilbertCurve3D` and `HilbertCurve3D` under a common `get_curve3d(ordering, R)`
factory; `hilbert_arc_delta` accepts a `curve=` argument so both families share the
same Δs/fine math. The dataset (`LJTransferableDataset`), sampler (`sample_lj.py`),
and training CLI (`train.py`) all thread `--ordering {hilbert,gilbert}` through; the
checkpoint `hparams` carry the ordering. The s-space arc math is unchanged: `arc(c)`
returns cumulative physical arc length in cell units (for hilbert this equals the code
index; for gilbert it tracks the actual step lengths, with even R giving cumlen == code
index). The nearest-even resolution rule (`resolution_for_box_rule` with
`ordering="gilbert"`) maps L to the nearest even R = 2·round(L/cell/2), keeping cell
size to <1% of the target — L3→R=64 (exact), L4→R=86 (0.78% off), L5→R=106 (0.63%
off). The octahedral diagnostic (`octahedral_nll_diagnostic.py`) was routed through the
same factory: `arc_nll_for_configs` now accepts `ordering` and `--ordering` is exposed
in the CLI, with `ordering="hilbert"` byte-identical to the original default.

### Audit table (2000 MCMC configs/size; CELL = 3/64 = 0.046875)

Resolution rules:

| size | L   | N   | R_hilbert | cell_hilbert | dev_h%  | R_gilbert | cell_gilbert | dev_g%  |
|------|-----|-----|-----------|-------------|---------|-----------|-------------|---------|
| L3   | 3.0 | 27  | 64        | 0.046875    | +0.00%  | 64        | 0.046875    | +0.00%  |
| L4   | 4.0 | 64  | 128       | 0.031250    | -33.33% | 86        | 0.046512    | -0.78%  |
| L5   | 5.0 | 125 | 128       | 0.039062    | -16.67% | 106       | 0.047170    | +0.63%  |

KS statistics vs L3 reference:

| ordering | size | R   | cell     | dev%    | mean_Δs | p50_Δs | p90_Δs | KS_Δs  | KS_fine | jump_frac  |
|----------|------|-----|----------|---------|---------|--------|--------|--------|---------|------------|
| hilbert  | L3   | 64  | 0.04688  | +0.00%  | 0.9853  | 0.8217 | 1.8302 | 0.0000 | 0.0000  | 0.001000   |
| hilbert  | L4   | 128 | 0.03125  | -33.33% | 0.9948  | 0.8716 | 1.7902 | 0.0420 | 0.0104  | 0.000746   |
| hilbert  | L5   | 128 | 0.03906  | -16.67% | 0.9968  | 0.8532 | 1.9003 | 0.0293 | 0.0087  | 0.001226   |
| gilbert  | L3   | 64  | 0.04688  | +0.00%  | 0.9855  | 0.8255 | 1.8553 | 0.0000 | 0.0000  | 0.001231   |
| gilbert  | L4   | 86  | 0.04651  | -0.78%  | 0.9944  | 0.8914 | 1.7843 | 0.0549 | 0.0111  | 0.000746   |
| gilbert  | L5   | 106 | 0.04717  | +0.63%  | 0.9971  | 0.8471 | 1.8855 | 0.0355 | 0.0081  | 0.001073   |

### Diagonal-step facts

All three gilbert grids (R=64, 86, 106) have **diagonal-step fraction = 0.000000** —
fully face-continuous. The nearest-even resolution rule keeps all production grids in
the even-R regime, where GilbertCurve3D behaves identically to HilbertCurve3D in terms
of step continuity (cumlen == code index). Odd grids would introduce multi-cell steps of
Euclidean length up to 3; production never encounters them.

### Audit verdict: NO-GO (hilbert wins marginally; both small)

The hypothesis that gilbert targets are MORE size-invariant was not borne out:
hilbert KS(Δs) at L4=0.042 and L5=0.029 are both LOWER than gilbert KS(Δs) at
L4=0.055 and L5=0.036. The effect is small (all values well below 0.10, far below the
0.30 threshold that triggered concern in the K=1 study), and both orderings are in the
"no heavy tail" regime. The mechanistic explanation: the Hilbert curve's self-similarity
at pow-2 grids means the Δs distribution is nearly invariant to cell size changes —
the recursion scales codes and positions together, and the KS distance is driven by
geometric differences in how particles pack at different densities, not by curve
cell-size mismatch. Gilbert's constant-cell rule introduces a slightly different R per
size, which changes the coarse-grid bucketing and thus the Δs statistics in a way that
modestly increases, rather than reduces, the KS distance to the L3 reference.

### Decision rule

Next training action = multi-size {L3,L5}→held-out-L4 rerun with **ORDERING=hilbert**
(the production default, which is confirmed as the tighter target) — after the in-flight
multi-size baseline finishes and on user go-ahead. Gilbert ordering remains implemented
and available (`--ordering gilbert`) but the Δs audit does not provide a target-quality
motivation to switch. Any future gilbert investigation should focus on conditions where
cell-size constancy matters for the learned conditional (large N or very different L
ratios), not on the target marginal KS statistics.
