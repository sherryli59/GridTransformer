# Architecture review: continuous head vs binned-discrete vs codebook (× factorized)

**Date**: 2026-06-14. Code-grounded review of the AR output-head choices, with the DW4
ESS benchmark (`benchmark_dw4_ess.py`, commit 5bb9d0b) as empirical confirmation.
Question driving it: do these choices make sense, do they give exact likelihoods (up to a
normalization constant), which is best, and what would improve them?

## The two axes

1. **Head / output distribution**: continuous MDN (`use_continuous_head`), binned-discrete
   (`binned_discrete`: a categorical over a regular grid of displacement bins), or codebook
   (`discrete` + a learned VQ codebook of K displacement vectors).
2. **Factorized or not** (`is_factorized`): predict the spatial axes one-at-a-time
   autoregressively, or jointly.

## Exact-likelihood analysis (the load-bearing question)

The downstream use (ESS / importance reweighting / Boltzmann matching) needs a proper
**continuous density** q(x) over displacement space, exact up to a global constant.

- **Continuous MDN — YES, a proper continuous density.** `mdn_loss` evaluates a Gaussian
  (full or diagonal covariance via `scale_tril`) mixture log-density with the correct
  Gaussian normalizer + Cholesky log-det. The relative→absolute reconstruction is a unit-
  Jacobian cumsum, so logq over positions = logq over deltas + const. Exact up to a
  constant. **This is the only head that is a continuous density.**

- **Binned-discrete — an exact PMF, a density only after dequantization.** It is a
  categorical over a *uniform* grid (`cell_size = box/R`, constant volume). As a continuous
  density it is `PMF / cell_volume` — piecewise-constant — *if* you assume uniform-within-
  cell. But the sampler places each draw at the **cell center** (`delta_t =
  codebook.index_select(...)`, sample_lj.py:1051) with **no dequantization**, so the model's
  realized support is a grid (a discrete measure), not a density. For ESS this still *works*
  (the constant cell volume cancels, so `logq = log PMF` gives a valid ESS that measures how
  well the PMF matches the cell-integrated target), but it is **resolution-limited**: the
  model can never resolve structure finer than a bin, which caps achievable ESS.

- **Codebook — NOT a valid continuous density; ESS is also biased.** The support is K
  learned vectors with **unequal Voronoi volumes**. (a) A finite point set can never place a
  sample where a continuous Boltzmann target wants it → structural ceiling. (b) The implied
  density `PMF / Voronoi_volume` has a *non-constant* volume term that does **not** cancel in
  ESS, so the naive `logq = log PMF` ESS is biased, not just low. The codebook is the right
  tool for genuinely categorical/VQ data (images), but it is a **category error for
  continuous molecular coordinates**.

### Factorized or not — both exact for the continuous head
`is_factorized` sets `continuous_out_dim=1` + an `axis_emb`, i.e. predict each spatial axis
autoregressively (x, then y|x, …). An AR factorization over axes is **exact**, not an
approximation — so factorized-continuous is a proper density too, just a different (often
cheaper) parameterization than the joint full-covariance MDN. For **discrete**, factorized =
a per-axis categorical (vocab = bins, tractable); the non-factorized joint discrete would
need bins^dim categories (intractable), which is exactly why the **codebook** exists — it is
the workaround for "non-factorized discrete," bought at the cost of the density validity above.

## The permutation caveat — applies to ALL heads, and is the prime suspect for low ESS

This is the subtlety the bare ESS number hides. An AR model defines a density over an
**ordered** particle sequence. The DW4/LJ target is **permutation-invariant**. The exact
perm-invariant model density of an unordered config is
  q_set(x) = Σ over the N! orderings σ of q(σ(x)).
Using the single generated ordering's logq (what `logp_continuous` stores) **undercounts**
q_set whenever a sample is not in canonical order, inflating its weight and deflating ESS.
For a model trained on canonically-ordered data this is benign *only if* sampling stays
canonical; otherwise it is a real, representation-level ESS leak — **not** a head-choice
issue and **not** model miscalibration. For DW4, N=4 ⇒ 24 orderings, so a perm-corrected
`q_set = logsumexp_σ logq(σ(x))` is cheap to compute and is the decisive test.

## Empirical: DW4 ESS (10k samples, T=1.0, verified energy/sign/gauge)

| architecture | ESS% |
|---|---|
| continuous (MDN, full-cov) | **2.26** |
| binned-discrete, factorized | 0.90 |
| binned-256, factorized | 0.61 |
| codebook (4096) | 0.07 |

**The ordering is exactly what the likelihood theory predicts**: continuous (true density) >
binned (resolution-limited PMF) > codebook (invalid density + biased ESS). The codebook's
0.07% is structural, not bad luck. **But the absolute values are ~20× below the ≥40% an easy
system should allow** — and that gap is *not* explained by the head choice (continuous is
already best). Two testable causes, not yet separated:
- **Permutation leak** (above): its magnitude depends on the *non-canonical generation
  rate* — for samples the model emits in canonical order, the single-ordering logq already
  equals q_set, so this only bites to the extent the model emits out-of-order draws. Cheap to
  settle on DW4 (24 orderings); measure it before assuming it dominates.
- **Genuine fit quality**: the continuous model's own samples sit ~1 energy unit above the
  target mean with a heavy low-logq tail, and a tokenized-AR-GMM has a lower ESS ceiling than
  a normalizing flow regardless.
The benchmark verified energy, logq sign, coordinate gauge, and temperature, so it is not one
of the usual pipeline bugs — but "the models are just miscalibrated" is premature until the
permutation correction is measured.

## Conclusions

- **Which is best:** the continuous MDN head — the only proper continuous density, and
  empirically the best ESS. Full covariance over diagonal where correlations matter.
- **Codebook:** drop it for continuous coordinates; it cannot represent a continuous
  Boltzmann density and its ESS is biased by unequal Voronoi volumes.
- **Binned-discrete:** defensible as a simple/robust baseline, but resolution-capped; if
  used for likelihood work, **dequantize at sampling** (uniform-in-cell) so the realized
  distribution is the density the logq claims.

## Improvements (ranked)

1. **Fix the permutation handling** (biggest likely ESS win, all heads): generate in
   canonical order and reject/resort non-canonical draws, OR report the perm-summed
   `q_set` (cheap for small N), OR move to a permutation-equivariant likelihood. Test on
   DW4 first (24 orderings).
2. **Standardize on the continuous head; retire the codebook** for this data.
3. **More expressive exact density per step**: a small normalizing flow head (exact,
   invertible) instead of / on top of the GMM captures curved, multimodal conditionals the
   GMM smears — strictly better than adding mixtures past saturation.
4. **If keeping discrete, add sampling dequantization** so its likelihood is a real density.
5. **Train quality**: the continuous model's own samples sit ~1 energy unit above the target
   mean with a heavy low-logq tail — more/better training (and the items above) should lift
   the absolute ESS toward the easy-system expectation.
