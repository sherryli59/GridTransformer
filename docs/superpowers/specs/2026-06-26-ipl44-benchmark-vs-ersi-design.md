# AR transformer vs eRSI on N=44 IPL — benchmark design spec

**Date:** 2026-06-26
**Status:** DRAFT (awaiting user review)
**Reference:** Grenioux et al. 2025, *Riemannian Stochastic Interpolants for Amorphous Particle Systems*
(arXiv 2512.16607), code `github.com/h2o64/learndiffeq`, data Zenodo record 17966995.

## 1. Goal

Head-to-head: run our **autoregressive transformer** generator on the paper's **N=44 IPL** system and compare
against their published **eRSI** (equivariant flow) numbers, on the **same data in, same energy + same
observables out — only the generator differs.** The contribution being tested is *importance-sampling
efficiency*: **how much IS does each generator need** (ESS, discard fraction), and do the reweighted
observables match. We expect, and will report plainly, that our AR generator's *raw* quality is worse (its
half-cage core-fill defect produces high-energy configs IS discards) while IS may rescue the reweighted
observables at higher sample cost — i.e. the "is IS doing the work" question, answered with a number.

## 2. System (Option 1 — IPL, CONFIRMED) — match exactly

2D, **N=44**, **50:50 binary** (22/22), **inverse-power-law `r^-12`** (BHHP soft-sphere, purely repulsive),
`σ = [[1.0,1.2],[1.2,1.4]]`, `ε = 1`, cutoff `2.5σ` (shifted to 0 at cutoff), **density ρ = N/L² = 0.5** ⇒
`L = √(44/0.5) = √88 ≈ 9.38`, **T = 0.1** (β = 10). All length scales differ from our KA work (ρ=1.2) and
must be **re-derived for ρ=0.5**, not inherited.

## 3. Reuse-their-code invariant (the fairness lever)

**Do not reimplement** the energy or observables — call theirs behind thin adapters in
`liquid_coupling_flow/ipl44/`:
- **Energy `U_star`:** `learndiffeq...particles.distributions.soft_spheres.SoftSphere` (subclass of
  `KobAndersen`); the summed energy is `KobAndersen.U(a, x)` (species `a`, positions `x`). A mismatched
  cutoff/shift silently corrupts loss AND reweighting.
- **ESS / IS estimator:** `particles/callbacks/ess.py` — `log_w = log_p_target − log_q_proposal;
  w = softmax(log_w); ESS = 1/Σw²`. It consumes the proposal's exact `log q` directly.
- **g(r) observable:** their radial-distribution code (`particles/callbacks/marginals.py`).
- **Data:** Zenodo `ipl44_T0.1_positions.pt` + `ipl44_T0.1_species.pt` — the exact equilibrium configs.
  **Do not regenerate** our own MC data (introduces an equilibration confound).

**Task-0 validation gate:** evaluate `U_star` on a batch of *reference* configs; the mean must match the
dataset's reported equilibrium energy. If not, the energy adapter is wrong — fix before anything else.

## 4. The exact-log-q requirement (what goes into the IS weight)

Our model's `sample(return_logq=True)` returns `log_prob(pos, sp)` to float precision (the exactness gate),
including the arc-scale change-of-variables Jacobian; the **flow head has no jitter term** (continuous
density). So `log_q_proposal = the model's logq` is a fully-normalized joint density. Two non-Jacobian items:
- **Species (correctness):** `U(a,x)` takes species `a`; the model generates species jointly. Fix the
  composition (22/22, canonical) and use the **same species labels** in `logq` and in `U` so the IS weight
  `w = −βU − logq` is over one space. (Decide: condition on fixed species and drop `lp_s` from the weight, OR
  define the target as the joint Boltzmann × uniform-over-arrangements — pin in the plan.)
- **Permutation (not a blocker):** our AR `q` is over curve-ordered, *labeled* configs; the Boltzmann target
  and their eRSI are permutation-invariant. Self-normalized IS with `w = −βU − logq` is **valid for any
  normalized proposal**; our broken permutation simply shows up as **lower ESS** — which is exactly the
  quantity the comparison measures, not a bug to fix.

## 5. Model & matched budget

**Generator:** the AR local-frame transformer, **re-trained from scratch on the IPL reference data** — the
plain categorical/`neither` baseline (the exposure-bias-campaign winner; NOT the null/harmful soft-label or
scheduled-sampling variants). *(Optional upgrade: the +A curve-flow head, marginally best on KA g(r) — flag
for the user; default categorical baseline.)* IPL adaptations: `_Lof/arc_scale` use **ρ=0.5**; KNN count,
`arc_range`, any Boltzmann-inverted priors **re-derived for the looser ρ=0.5 cage**; canonical mask uses 22/22.
**Support-coverage GO/NO-GO** (the single-site-spec §5.6 out-of-range fraction ≈ 0) recomputed on the *IPL*
reference data — if KA-tuned `arc_range` truncates the IPL cage, every downstream number is on a truncated
target. Blocking.

**Matched budget (fairness):** their 1000 epochs on ~10⁵ samples — match epochs + dataset size; **report our
parameter count next to their ~22k (smallest) / ~580k (largest)** so a win states its capacity.

## 6. Comparison (the contribution) — same axes as the paper

- **Discard fraction (IS-free, co-primary):** fraction of generated configs with energy > 2× target max.
  Theirs: RSI 100%, eFM 84%, eRSI 3%. Where do we land?
- **ESS vs R (the single most informative):** does our ESS plateau, at what R̄? Theirs (eRSI N=44) up to
  ~1.8×10⁶. Fewer ⇒ better q/p overlap; more ⇒ IS working harder.
- **Reweighted observables:** `U`, `c_V`, `g(r)` after IS vs target — report **both raw (dotted) and
  reweighted**, as their Figs 3–4.
- **One figure** mirroring Figs 4/9: g(r) (target / our-raw / our-reweighted), energy histogram p(U) vs q(U),
  ESS-vs-R with eRSI's published curve overlaid.

## 7. Honest-framing requirement

Report **discard fraction AND ESS R̄ as co-primary** (the IS-free and IS-cost measures), never only the
reweighted observables (that hides a weak generator behind IS). A worse raw generator rescued by IS at higher
cost is a legitimate, useful result — stated as such.

## 8. Scope

**In:** Option-1 IPL N=44 only; reuse-their-energy/ESS/g(r); adapt + train the AR generator at matched budget;
the §6 comparison + figure; all validation gates (energy match, support coverage).
**Out (follow-ons):** Option-2 ternary KA; the full-cage/corrector threads; any change to *their* code beyond
thin read-only adapters; size-transfer.

## 9. Deliverables

`liquid_coupling_flow/ipl44/`: adapters wrapping their energy/ESS/g(r) + the IPL dataset loader; the
IPL-adapted generator + training; the §6 comparison script + figure; a results report with the discard
fraction, ESS R̄, and reweighted-vs-published table. Plus the Task-0 energy-match + support-coverage gate
evidence.
