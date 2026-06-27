# AR transformer vs eRSI on N=44 IPL — benchmark results

**Date:** 2026-06-27
**Branch:** liquid-coupling-flow · `liquid_coupling_flow/ipl44/`
**Reference:** Grenioux et al. 2025, *Riemannian Stochastic Interpolants for Amorphous Particle Systems*
(code `github.com/h2o64/learndiffeq`, data Zenodo 17966995)
**Spec/plan:** `docs/superpowers/specs/2026-06-26-ipl44-benchmark-vs-ersi-design.md`,
`docs/superpowers/plans/2026-06-26-ipl44-benchmark-vs-ersi.md`

## 1. What was tested

Head-to-head on the paper's **N=44 IPL** system, *same energy + observables in/out, only the generator
differs*: our **autoregressive +A curve-flow transformer** (`KACurveFlowModel`, exact likelihood) vs the
paper's published **eRSI**. The fairness lever is that the energy (`SoftSphere.U`), the radial distribution,
and the equilibrium data are **theirs, reused unmodified** (vendored read-only in `ipl44/learndiffeq/`); we
only swap the generator. The contribution under test is **importance-sampling efficiency**: how much IS does
the AR generator need (discard fraction, ESS), and do the reweighted observables match.

**System (matched exactly):** 2D, N=44, 50:50 binary (22/22), inverse-power-law r⁻¹² (BHHP soft-sphere,
purely repulsive), σ=[[1.0,1.2],[1.2,1.4]], ε=1, cutoff 2.5σ shifted to 0; **ρ=0.5 → L=√88≈9.38**, **T=0.1
(β=10)**. Energy-match gate: reference ⟨U⟩=14.71 (std 0.64), finite, tight — passed.

**Matched budget:** 1000 epochs (= 78 125 steps at B=128 over the M=10⁴ released configs), matching eRSI's
1000 epochs. **Parameter count: 5.54M** — see the capacity caveat in §5.

## 2. ⚠️ A data bug was found and fixed first (do not skip)

The **first** full run returned discard 0.967 / ESS 1 / raw ⟨U⟩~5×10²⁵ — which would have read as a clean (if
damning) AR negative. It was a **data bug**, found by diagnostics, not the generator:

- The released Zenodo configs store **unwrapped MD coordinates** — ~50% of coordinates lie outside [0,L)
  (raw range ≈[−29, 35]).
- The energy is **wrap-invariant** (minimum-image `gram_torus`), so the energy-match gate passed *blind* to
  it (14.707 raw == wrapped).
- The AR curve-order/scaffold assume positions in [0,L); unwrapped coords mis-registered, making the
  local-frame offset std measure **1.42** instead of the true wrapped **0.66**. The *max* offset coincided
  (~2.5 both), so the **max-based** support-coverage gate also missed it, and tuned `tail_bound=4.0` — ~1.5×
  too wide for the 8-knot spline.
- Training accidentally dodged it (`augment` applies `remainder(·,L)` every step), so the **model learned the
  right offset distribution** — but with a needlessly coarse spline, and any raw-data evaluation was corrupted
  (logq(real data)/N = **−13.3**, while the model scored its own samples at −1.0).

**Fix** (commit `0c26008`): wrap reference data into [0,L) at load; re-tune `tail_bound`=`arc_range`=**2.75**
to hug the true support; retrain. After the fix, **logq(real data)/N = −0.10** (well-calibrated, and now
correctly *above* the model's own samples at −1.36). All numbers below are the **corrected, fairly-tuned**
model. (See memory note `ipl44-unwrapped-data-bug`.)

## 3. Results (corrected model, 200k samples; figure from 16k)

| Axis | Ours (AR +A) | eRSI | eFM | RSI |
|---|---|---|---|---|
| **Discard fraction** (U > 2·U_ref,max), IS-free | **0.987** | 0.03 | 0.84 | 1.00 |
| **ESS at R=2×10⁵** (self-normalized IS) | **≈1** (flat in R) | up to ~1.8×10⁶ | — | — |
| Reweighted ⟨U⟩ (ESS-degenerate, see below) | 20.9 | — | — | — |
| Reweighted c_V (ESS-degenerate) | 0.0 | — | — | — |
| Reference ⟨U⟩ (target) | 14.66 | — | — | — |
| Parameters | 5.54M | 22k–580k | — | — |

(U and c_V are **total** energies/capacities for the 44-particle system, not per-particle; no cross-paper
U/c_V comparison is drawn — the eRSI U/c_V cells are "—" — and the filled comparisons, discard and ESS, are
dimensionless.)

- **Discard 0.987** — between eFM (0.84) and RSI (1.0); i.e. our AR transformer is **as bad as the paper's
  non-eRSI baselines** on the IS-free measure. **0.0%** of 16 384 generated configs fall inside the reference
  energy band (min generated U = 22.4 > reference max U = 16.6).
- **ESS collapses to ~1 and stays there** for every R from 10² to 2×10⁵ (the ideal ESS=R line; eRSI reaches
  ~1.8×10⁶). A single sample carries essentially all the weight, so **IS does not rescue this generator** — the
  reweighted ⟨U⟩=20.9 and c_V=0.0 are degenerate single-sample artifacts, not estimates. This is *more*
  negative than the spec's hopeful framing ("IS may rescue reweighted observables at higher cost"): here the
  q/p overlap is so poor that IS provides essentially zero effective samples.

## 4. Mechanism — the failure is localized, not gross (the honest nuance)

The generator is **not** producing structurally-random configs. The **g(r) for r>0.4 tracks the target
closely** (first-shell peak ≈3 at r≈1, second feature at r≈2.3; the generator slightly under-fills the first
shell). The failure is a small fraction of **rare-but-fatal near-contacts** per config (nearest-neighbour
distance median 0.586 vs target 1.10) that, under r⁻¹²×β=10, dominate the energy (βU ~10⁴–10¹⁶). A few
overlaps per otherwise-reasonable config are enough to discard it.

This is the **causal-AR half-cage limitation**, sharply evidenced here: the model's *density* is excellent
(logq(data)/N −0.10) — it knows what good configs look like — but its *autoregressive sampling* must place
each particle without seeing its future neighbours, so the conditional is necessarily **broad**, and that
breadth includes the near-contact placements the stiff cold potential cannot tolerate. The
re-tuned spline did not help — a sharper *marginal* concentrates mass at the centroid mode without enforcing
the *joint* exclusion. (Consistent with this codebase's prior KA findings: `gap-is-residual-not-rollout-drift`,
`full-cage-lever-needs-energy` — the lever is a full-cage/energy corrector, not the conditional or the head.)
The identified data bug is fixed; this residual failure matches the independently-documented AR limit and the
paper's own RSI/eFM baselines.

## 5. Verdict and caveats

**Verdict:** On the N=44 IPL stiff-repulsive cold target, the autoregressive +A curve-flow transformer is
**strictly dominated by eRSI on every axis** — discard ~0.99 (vs 0.03), ESS ~1 (vs ~10⁶) — and, decisively,
**importance sampling cannot rescue it** (ESS collapse ⇒ no reliable reweighting). The result is what the
paper's equivariant/Riemannian structure was designed to avoid, and matches their non-eRSI baselines. The
deeper lesson is consistent with prior work in this repo: for a stiff target, the causal-AR conditional's
breadth is fatal, and the fix is a full-cage energy-guarded corrector, not a sharper head or representation —
which is **out of scope** here (spec §8) and the natural follow-on.

**Caveats (honest framing):**
1. **Capacity asymmetry — the headline caveat.** Our generator is **5.54M params ≈ 10× eRSI's largest (580k)
   and ≈250× its smallest (22k)**. The comparison tests IS efficiency, not capacity; but a loss this complete
   at 10× the parameters states plainly that the gap is structural, not a capacity deficit.
2. The reweighted ⟨U⟩/c_V are **degenerate** (ESS≈1) and must not be read as observable estimates.
3. The reused g(r) function includes self-pairs / lacks ideal-gas shell normalization → a small-r spike
   (g≈54 at r→0); identical for all curves (fair), clipped to r>0.4 in the figure.
4. The identified data bug is fixed and the density is now well-calibrated; the residual generation failure is
   evidenced as the AR half-cage limit, but per standing practice the possibility of further issues is left
   open rather than declared absent.

## 6. Artifacts

- Corrected checkpoint: `liquid_coupling_flow/ipl44/data/ipl44_curveflow.pt` (gitignored; tail_bound 2.75,
  arc_range 2.75, knn 16, n_B 22, 1000 epochs, 5.54M params).
- Benchmark figure (committed code output): `data/ipl44_benchmark.png`; **clean report figure**
  (zoomed g(r), log10(βU) energy, ESS-vs-R): `data/ipl44_benchmark_clean.png`.
- Code: `ipl44/ipl_energy.py` (energy/g(r)/data adapters), `ipl44/ipl_model.py` (model + trainer +
  support-coverage gate), `ipl44/ipl_benchmark.py` (discard/ESS/reweighted + figure). Diagnostics in the
  session scratchpad (`ipl_diag*.py`, `ipl_fig_clean.py`).
