# Progress report: learned kernels below the swap-arrest (NBC polydisperse glass)

**Date:** 2026-07-17. **Branch:** mw-cell-q0 (poly commits interleaved; files disjoint from mW work).
**System:** NBC continuously-polydisperse soft spheres (v(r)=(σij/r)¹² + smoothed cutoff, nonadditive
σij, P(σ)∝σ⁻³ on [0.725,1.61]), N=300, ρ=1. Swap MC's home turf; the campaign question:

> Can a LEARNED kernel extend equilibration below the temperature where classical swap MC arrests —
> the gap Ciarella 2023 left open (they refuted only global/one-shot generation, never local kernels)?

---

## 0. Substrate (validated foundation)

- **Model kernels** (`poly/model.py`, numba): energy exactness-gated vs independent torch f64 to 1e-9;
  ⟨σ⟩=1.000 and C²-smooth cutoff verified analytically (sympy).
- **Swap T-ladder banks** (4 independent runs, N=300, budget 10⁷ sweeps/rung): located the swap-arrest
  operationally. Swap speedup (τ_local/τ_swap): ~15× at T=0.10 → 1–3.7× at T=0.065-0.075. τ_swap at
  0.065 ≈ 58k sweeps and climbing steeply; deepest rungs still running. Held-out N=600 bank for
  N-scaling controls descending in parallel.
- **CRITICAL DATA BUG found & fixed:** the bank collected 16 x-frames while swap kept permuting σ
  between captures but saved only the final σ → frames 0-14 paired positions with a stale
  σ-assignment (U/N +13 vs +0.32 equilibrium; only frame 15 consistent). All early pair datasets and
  the first pilot were built on this. Fix: per-frame σ banking + full regeneration
  (`poly_bank_fixed_run{1,2,3}.pt`, clean tight energy bands at every rung to T=0.058-0.065).
  Detection rule now in memory: gate any frame bank with per-frame total_U.

## 1. PHASE 1 — one-shot swap-endpoint flow kernel: ORACLE-LIMITED (closed, negative)

**Design evolution (each step measured):**
1. *Random 8-particle σ-permutation → position-only relax as FM target* (original plan): transport
   RMS 1.02 with box-scale tails = one-to-many map FM cannot fit. Refuted; the permutation itself
   creates the catastrophe.
2. *User directive → continuous joint (σ,x) flow.* Ensemble probes killed the naive versions:
   prior-only semi-grand DEFLATES (σ̄ 0.998→0.82, U/N→+0.02 — particles shrink to relieve pressure);
   volume-conserving pair exchange (Σσ³ exact) still shape-drifts (U/N→+0.20). Both = wrong state point.
3. *Decision: swap-endpoint kernel* — σ(t) interpolation as time-dependent conditioning of a
   position-only flow; endpoint = exact pair swap + accommodated positions; canonical ensemble
   preserved; exact reverse-RK4 logq (review-verified: both MH legs byte-consistent, ΔU bookkeeping
   cross-checked to 1.6e-4, reverse-leg convention verified numerically to 0.0).

**The measured chain to the verdict (all on clean held-out frames, T=0.085, k=8):**

| proposal | dead-bin (|Δσ|>0.2) acceptance | note |
|---|---|---|
| classical direct swap | 0.0000 (0/524) | acc = 0.44 / 0.026 / 0 / 0 / 0 across |Δσ| bins |
| identity + noise w=0.05 | ~1e-13 | ΔU +2.4; ANY isotropic block noise is fatal at β=11.8 |
| trained flow, base 0.35 | 0.0000 | ΔU +110 (yet 130× below random noise — flow finds soft modes) |
| trained flow, base 0.05 | 0.0000 | ΔU +3.5 + logq asymmetry −9 |
| **ORACLE (exact relaxed endpoint)** | **~1-2% (tail events)** | ΔU med +2.5 to +4.5 |
| oracle at k=16/24/32 | flat | the cost does NOT relax away with block size |

**Verdict (committed, `poly_gate_verdict_phase1.md` + full per-attempt data):** the wall is
THERMODYNAMIC — a dissimilar-pair swap against a (even partially) frozen environment costs 25-50 kT
of local free energy that block relaxation cannot pay. A perfect proposal is capped; learning quality
is not the binding constraint. This sharpens Ciarella's negative to local kernels, recapitulates the
KA fine-energy-precision wall, and explains BCY's global-deformation choice.

**Fixed en route:** base-noise scale must match the cage scale (~0.1), never hot-liquid scales
(0.35); dummy-padding poisoned logq via unfiltered kNN neighbor gather (fixed + inertness-tested);
a top-|Δσ|-bin stratifier filter that was silently a no-op (review catch).

## 2. PHASE 2 — gradual-σ semi-grand ("pay in installments"): IN PROGRESS, calibration-blocked at cold T

**Rationale:** the phase-1 wall is a vertical (frozen-env) cost. Many small accepted σ-steps with
position relaxation interleaved pay the adiabatic cost instead. Requires the semi-grand ensemble
π ∝ exp(−β(U−Σμ(σᵢ)))Πφ(uᵢ) with μ(σ) calibrated so composition = P(σ).

**Built + certified:** u↔σ Gaussianized maps and single-site u-MH (β=0 and β=2-quadrature exact);
joint (x,u) flow with exact two-channel divergence (phase-2 accelerator, review-passed incl.
no_grad/torch.func interaction probe); μ(σ) as 32-bin lookup with Boltzmann-inversion loop;
kernel-mix gate harness (P2b) measuring τ_α, σ-decorrelation, composition guard per mix
{disp | +swap | +u(μ) | +both}.

**Calibration results:**
- **T=0.2: SOLVED.** Iterative loop to its noise floor, then the finisher (150k-sweep fixed-μ
  measurement, 1.9M σ-samples + one full-strength Newton step): U/N +0.5206 vs canonical +0.52
  (exact), mean σ 0.9935, L1 2.2%. Residual frac_sup ~6-7% is tail-mixing hysteresis (worst bin
  wanders among outermost ~1%-mass bins under repeated Newton steps).
- **P2b smoke at T=0.2:** naive gradual-σ MATCHES classical swap (τ_α 512 = 512, τ_σ 2048 = 2048;
  u-acc 0.80); disp-only control 4× slower with NO σ-decorrelation (wiring confirmed). First
  faint positive — but warm-T parity is expected; the question lives below arrest.
- **T ≤ 0.1: EQUILIBRATION-LIMITED (the current obstacle).** At every cold rung the composition pins
  ~1.5% inflated (mean σ 1.013-1.017), U/N sits 0.08-0.11 above canonical, L1 plateaus 4-9%, and
  Newton steps chase hysteresis (at 0.085 they worsened the tail 17→34%). Cause: τ_swap ≥ 30-60k
  sweeps, so 150k-sweep measurements are only ~2.5-5τ — each snapshot is non-equilibrium. The glass
  now slows the calibration itself; this is the same physics the gate wants to probe, biting earlier
  than expected.

## 3. Assets (all committed, all reviewed)

| asset | state |
|---|---|
| model/energy kernels + tests | exactness-gated |
| clean frame banks, 3 runs × T∈[0.058,0.2] + N=600 descending | done / running |
| classical swap acc-vs-|Δσ| + oracle + oracle-vs-k measurements | committed with full data |
| swap-endpoint flow (σ(t) conditioning, exact logq, batched) + gate harness | review-certified; reusable for ANY kernel drop-in |
| joint (x,u) flow (two-channel exact divergence, 21.6 prop/s @B=64) | review-certified, parked |
| semi-grand substrate + μ machinery + finisher + ladder driver | committed |
| P2b kernel-mix gate harness | committed, smoke-passed |
| μ(T): 0.2 certified; 0.1/0.085/0.075/0.065 plateaued with documented bias | needs cold-rung fix |

## 4. The open decision (recommendation attached)

Cold-rung μ needs one of:
- **(a) vex+μ hybrid (recommended):** σ-channel = volume-conserving pair exchange (Σσ³ exact — kills
  the mean-σ inflation structurally; machinery probed: acc 0.44 at T=0.085) + small μ correcting only
  the residual shape drift. Calibration becomes easy because the hard moment is pinned by construction.
  Physically the honest continuous analog of swap (exchange, not creation/annihilation).
- **(b) Brute-force equilibration-aware ladder:** ≥750k-sweep measurements per Newton round at the
  cold rungs (~6-8h overnight). No redesign risk; cost scales with the glass slowdown; may still
  plateau if σ-tail mixing is intrinsically slow.
- **(c) Fast preview at T=0.1 with documented +1.5% composition bias** — cannot answer the
  below-arrest question, only extend the warm-parity observation.

## 5. Session-level negatives & lessons now in memory

- One-shot exact-density block proposals for dissimilar exchange: oracle-limited (do not retry).
- Base-noise scale must match cage scale; even then, near-deterministic proposals inherit a
  contraction-asymmetry logq penalty.
- Frame banks must snapshot every co-evolving array per frame; gate with per-frame total_U.
- Iterative histogram calibration saturates at per-iteration sampling noise; the finisher
  (long fixed-μ measure + single Newton step) is strictly better — until equilibration itself
  becomes the binding constraint.
