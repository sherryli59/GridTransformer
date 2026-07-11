# mW eRSI flow base — design

**Date:** 2026-07-10 · **Branch:** liquid-coupling-flow · **Status:** approved design, pre-plan
**Goal:** train an equivariant Riemannian Stochastic Interpolant (eRSI) flow on the mW reference bank and
deploy it as a **non-causal one-shot importance-sampling proposal** for the liquid-mW sampler — the first
base that can cross the future-wall (produce medium-range structure the causal AR bases v1–v10 could not).
**Reference:** grenioux.pdf (eRSI paper; code vendored at liquid_coupling_flow/ipl44/learndiffeq).

## Why this direction (campaign context)

The v1–v10 base search proved, from both architecture and objective sides, that a **causal** AR generator
cannot install second-shell order: the true conditional marginalizes future particles, so a per-step causal
model is future-blind by construction. A **non-causal** flow (EGNN velocity sees every particle at every
integration step) is the trilemma exit: it can encode medium-range order. eRSI is that flow, demonstrated in
the paper on the 2D binary IPL glass (our own IPL44 lineage), with OT-straightened paths and a tractable
divergence-based likelihood.

## Locked decisions (brainstorm)

1. **Integration mode = one-shot IS.** Draw X ~ flow, weight `w = e^{−βU(X)}/q_flow(X)` with the exact
   composed likelihood evaluated ONCE per sample. Uses the flow's generation likelihood (the user's
   requirement), is exact, and pays the ODE cost once per sample (not per site-move). The flow-density
   geometric-SMC mode (q_flow in every mutation ratio, ~640× cost) is shelved as prohibitive. An
   **energy-only SMC tail** (resample + a few cheap local-MCMC sweeps; flow log_q only at init) is the
   graceful ESS booster if one-shot ESS is marginal — exact, reuses smc_run.
2. **Reuse = hybrid.** TRAIN with the vendored `RiemannianFlowMatching` + OT + `egnn_traceable` Lightning
   loop (don't reimplement exactness-critical FM/OT/divergence). DEPLOY via a thin mW-native wrapper.
3. **De-risk = straight to 3D mW** (no separate 2D-IPL reproduction), with cheap exactness checkpoints
   (G-a/G-b) baked in so a break localizes without the IPL milestone.
4. **Velocity = kNN-TRIMMED `EGNN_dynamics(max_neighbors=k)`**, k≈12–16 (cage scale) — linear-in-P via the
   central-force flash-div trim (essential for Phase-3 transfer to N=216+; the periodic branch's two fixed
   bugs get a fresh 3D verification in G-a).

## Components

### Training (vendored, adapted) — new mW-native pieces feeding the vendored pipeline
- **`mw_ersi_data.py`** — adapter: reference bank `[n,64,3]` → the pipeline's `(species, X)` format with
  `n_species=1` (constant species vector), positions wrapped to `[0,L)³`. Reuses the thin-2 + extension
  banks and the event-ordered (leakage-safe) val split.
- **3D monatomic config**: `dim_phys=3`, `n_species=1`, `L=(64/RHO_STAR)^{1/3}`, velocity =
  `EGNN_dynamics(max_neighbors=k)` (exact analytical divergence), `ot_particles=True`
  (single-group Hungarian, trivial for monatomic), symmetry augmentation (translations + signed
  coordinate permutations + trivial species perm) via the pipeline's `DataAugmentationCallback`.
- Trained by the **simulation-free MSE flow-matching loss** (the pipeline's `training_step`) — no ODE
  integration during training. Checkpoint = the velocity field (+ config to reconstruct).

### Deployment wrapper — `mw_ersi.py` `MWeRSIFlow` (the clean seam)
- `sample(B, gen) → X`: integrate the ODE base→target (uniform torus → flow), fixed deterministic solver,
  few steps (OT ⇒ near-straight path).
- `log_q(X) → [B]`: composed likelihood = uniform-base log-density + `−∫ div v̂ dt` reverse-integrated
  target→base via the kNN-trimmed `egnn_traceable` **exact analytical divergence** (NOT Hutchinson ⇒
  deterministic ⇒ the consistency a valid IS weight needs). Fixed solver ⇒ `log_q` is a deterministic
  function of X.
- Optional `GeneratorBase`-compatible shim so tonight's mini-SMC harness/gates apply unchanged.

### Measurement — `mw_ersi_eval.py`
One-shot IS: `logw = −βU(X) − log q_flow(X)`; report ESS/B, IS-reweighted ⟨U⟩ and g(r), and the raw-sample
g(r) shells. Head-to-head vs the ladder in the same energy-eval accounting: {uniform, excluded-volume,
v10-AR, eRSI}. Optional energy-only SMC tail via smc_run for the ESS boost.

## Exactness & gates (each blocks the next)

- **G-a — 3D exact divergence**: kNN-trimmed periodic `EGNN_dynamics` analytical divergence vs brute-force
  autodiff `div v̂` in 3D, machine precision, several k and random configs. (Re-verifies the fixed periodic
  branch in 3D BEFORE training.)
- **G-b — composed-likelihood integrity**: (i) `sample→log_q` round-trip — `log q_flow` of a drawn X equals
  the density accumulated during generation (fixed solver, both directions), ≤1e-3; (ii) normalization —
  small-N coarse quadrature of ∫q_flow ≈ 1, and solver-step-count convergence of `log q_flow` (quantifies
  the residual integration error we accept; OT should make it near-flat).
- **G-c — STRUCTURE (pre-registered headline, go/no-go on the premise)**: raw-sample one-shot g(r) shows a
  SECOND SHELL — g(1.85) approaching the reference 1.19 and first shell approaching 2.12. If the non-causal
  flow cannot make shells one-shot, the direction fails at N=64 and we stop (no amount of IS fixes a base
  with no structure). This is the capability every causal base lacked.
- **G-d — ESS / amortization**: one-shot ESS/B usably above the AR base's ≈1/B; IS-reweighted ⟨U⟩ matches
  the reference within error; energy-evals vs the baseline ladder.

## Risks (pre-registered)

1. **Regime**: even a structured flow may show modest energy-eval savings on the easy liquid; the honest
   readout is ESS + reweighted-observable quality, not a large rung win. The big payoff is a glass (future
   campaign).
2. **Integration error in log q_flow**: costs IS efficiency (ESS), not exactness (one-shot IS is exact for
   any deterministic q_flow). G-b quantifies it; OT minimizes it; exact-analytical-div removes estimator
   noise the paper tolerates.
3. **3D + three-body port**: the velocity learns three-body structure from data (implicit); G-a/G-b localize
   any port bug to the divergence before training spends GPU.
4. **Vendored-pipeline 3D support**: a `dim_phys=3` smoke is plan step 0 (before the data adapter), since the
   paper ran 2D.

## Out of scope

The flow-density geometric-SMC mode (prohibitive). 2D-IPL reproduction (skipped per de-risk decision).
Retraining the mW energy/reference/SMC/gates (all reused as-is). Phase-3 transfer to N=216 is a follow-on
once G-c/G-d pass at N=64 (the kNN trim makes the velocity size-agnostic, so it is a deployment-only step).
