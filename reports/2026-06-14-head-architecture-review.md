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

The real test is NOT "is q a continuous density" — for ESS, q only has to be correct **up
to a single GLOBAL constant** (the constant cancels in `w = p/q`). The discriminator is
therefore *constant vs non-constant* missing normalization, not density vs mass.

- **Continuous MDN — exact up to a constant.** `mdn_loss` evaluates a Gaussian (full or
  diagonal covariance via `scale_tril`) mixture log-density with the correct Gaussian
  normalizer + Cholesky log-det. The relative→absolute reconstruction is a unit-Jacobian
  cumsum, so logq over positions = logq over deltas + const. The only missing factors (the
  partition function Z, the Jacobian) are global constants → cancel. ✓ Best on resolution
  and expressiveness.

- **Binned-discrete — ALSO exact up to a constant** (earlier draft wrongly demoted this).
  It is a categorical over a *uniform* grid (`cell_size = box/R`). Its implied density is
  `PMF / cell_volume`, and because the grid is uniform, `cell_volume` is **one global
  constant** → it cancels exactly like Z. So `logq = log PMF` gives a perfectly valid ESS:
  you are importance-sampling against the *bin-discretized* Boltzmann target, exact up to a
  constant. The cell-center placement (`delta_t = codebook.index_select(...)`,
  sample_lj.py:1051, no dequantization) does not break this — within a uniform cell the
  dequantized density is flat, so evaluating it at the center is consistent with evaluating
  `U` at the center. The ONLY real cost is **resolution**: you match a coarse-grained
  target, capped by bin size. Not an exactness problem. ✓ (coarser than continuous)

- **Codebook — the genuine outlier: off by a NON-constant.** It is a categorical over K
  *learned, irregularly-spaced* vectors, so its implied density is
  `PMF / Voronoi_volume(code)` and the Voronoi volumes are **unequal across codes**. The
  missing factor is therefore **not a single global constant** — it is a per-sample quantity
  `log Voronoi_volume(code_i)` that rides along in every log-weight and does **not** cancel.
  So `logq = log PMF` gives a **biased** ESS, not merely a coarse one. The fix isn't "train
  better" — it is "use equal-volume cells," at which point it is just (non-factorized)
  uniform binning again. The codebook is the right tool for genuinely categorical/VQ data
  (images); for continuous coordinates it buys nothing over uniform bins and corrupts the
  importance weights. ✗

### Factorized or not — both exact for the continuous head
`is_factorized` sets `continuous_out_dim=1` + an `axis_emb`, i.e. predict each spatial axis
autoregressively (x, then y|x, …). An AR factorization over axes is **exact**, not an
approximation — so factorized-continuous is a proper density too, just a different (often
cheaper) parameterization than the joint full-covariance MDN. For **discrete**, factorized =
a per-axis categorical (vocab = bins, tractable); the non-factorized joint discrete would
need bins^dim categories (intractable), which is exactly why the **codebook** exists — it is
the workaround for "non-factorized discrete," bought at the cost of the non-constant
Voronoi normalization above.

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

**The ordering matches the likelihood theory**: continuous (exact-up-to-const, fine
resolution) > binned (exact-up-to-const, coarse) > codebook (biased by non-constant Voronoi
normalization). NOTE the continuous and binned numbers are directly comparable *valid* ESS
estimates; the codebook's 0.07% is a **biased** estimate (the per-code volume term is
omitted), so it should not be compared head-to-head — it is suspect, not merely low.
**But the absolute values are ~20× below the ≥40% an easy system should allow** — and that gap is *not* explained by the head choice (continuous is
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

- **Which is best:** the continuous MDN head — exact up to a constant AND finest resolution,
  and empirically the best ESS. Full covariance over diagonal where correlations matter.
- **Binned-discrete:** also exact up to a constant (uniform cell volume cancels), just
  resolution-capped — a legitimate, valid baseline; its ESS numbers are trustworthy. Finer
  bins → closer to the continuous head.
- **Codebook:** drop it for continuous coordinates. Not because of "finite support" per se,
  but because its *unequal Voronoi volumes* are a non-constant normalization that biases the
  importance weights. Equal-volume cells = uniform binning, so the codebook buys nothing here.

## Improvements (ranked)

1. **Fix the permutation handling** (biggest likely ESS win, all heads): generate in
   canonical order and reject/resort non-canonical draws, OR report the perm-summed
   `q_set` (cheap for small N), OR move to a permutation-equivariant likelihood. Test on
   DW4 first (24 orderings).
2. **Standardize on the continuous head; retire the codebook** for this data.
3. **More expressive exact density per step**: a small normalizing flow head (exact,
   invertible) instead of / on top of the GMM captures curved, multimodal conditionals the
   GMM smears — strictly better than adding mixtures past saturation.
4. **If keeping binned-discrete**, it's already ESS-valid; add sampling dequantization only
   if you need genuine continuous *samples* (uniform-in-cell), and use finer bins to lift the
   resolution cap toward the continuous head.
5. **Train quality**: the continuous model's own samples sit ~1 energy unit above the target
   mean with a heavy low-logq tail — more/better training (and the items above) should lift
   the absolute ESS toward the easy-system expectation.
