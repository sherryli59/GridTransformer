# Local-SMC pilot — unified annealed sampler from validated local components — design spec

**Date:** 2026-07-04 (user-approved direction). **Goal:** a minimal, asymptotically-exact annealed-SMC Boltzmann
sampler for the 2D KA glass whose every learned component is LOCAL (⇒ size-transfers), assembled from the
campaign's validated pieces — and a clean measurement of whether learned local transport earns its keep over
mutation-only SMC.

## Lessons this design encodes (with corrections)

- **A1's ESS=1 was a one-shot whole-config flow**: extensive weight variance ⇒ population collapse. Fix =
  bridge with many small annealing steps + resampling (standard SMC), with LOCAL transport keeping each step's
  transport residual per-particle-bounded and cheap (O(P·k) exact divergence).
- **CORRECTED variance claim**: locality does NOT make per-rung weights N-free. The annealing increment
  fluctuation is Δβ·(U−⟨U⟩) ~ Δβ·√N ⇒ **rung count must scale ~√N** (or Δβ ~ 1/√N) for constant ESS under size
  transfer. Locality makes that scaling affordable, not unnecessary. The schedule is a transfer-dependent knob.
- **Integrator-in-weights caution**: continuous-flow log-densities carry integrator error (measured: real
  trained field RK4 self-consistency ~0.02 mean / 0.1 max) — acceptable in MH acceptance, BIAS when compounded
  in weights over many rungs. Weight-path learned components must have integrator-free exact densities ⇒ the
  AR cluster proposal (one-forward exact log_q) is the weight-safe learned transport; the EGNN-CNF is not
  (without reversible/tight-adaptive integration).
- **jf_ka100tt cannot be the SNF transport cheaply**: its position ODE runs on the NON-traceable trunk (no
  cheap exact ΔlogJ). It stays in the seeding role.
- **References**: N=100 PT reference is solid (−3.260); N=256 PT reference is UNDER-CONVERGED (true eq ≈
  −3.26/−3.27; parallel campaign Addendum 3). N=256 gates are directional until the reference is re-run.

## Architecture (pilot = two arms + shared machinery)

Population of B=256 (N=100) / 128 (N=256) weighted chains; annealing ladder β: 0.8 → 2.0 (T 1.25 → 0.5,
matching the PT ladder's span); adaptive Δβ (choose next Δβ so predicted ESS drop ≈ target, cap rungs ~40 at
N=100); multinomial resample when ESS < 0.5·B.

- **Seeds (both arms)**: flow-generated configs + EXACT init weights w₀ = π_{β₀}(x)/q_gen(x) — generator =
  `KAFlowHeadModel` (`ka_flowhead_N100_k8_scratch.pt`, exact log_prob via sample(return_logq)). Species from
  the generator; positions canonicalized per the swap-breathe state rules.
- **Mutation kernel at each rung (both arms; π_{β_k}-invariant, exactness insurance)**: per rung apply
  n_mut rounds of [tamed-MALA or parallel-displacement sweeps + swap-and-breathe sweep + optional MTM moves].
  All three learned/exact kernels are validated: MTM (stationarity gate), swap-and-breathe (G1 0.153%
  equilibrium exchange, G2 stationarity w/ species observables), MALA/displacement (parallel campaign / this
  campaign). β_k is passed through (kernels take beta).
- **ARM-0 (mutation-only SMC, the baseline that must be beaten)**: weights update ONLY by annealing:
  log w += −Δβ·U(x). This alone is expected to work (it is plain annealed SMC with a richer kernel stack) —
  it isolates the bridging fix for A1.
- **ARM-1 (learned stochastic local transport)**: after each reweight, additionally apply one AR-cluster
  TRANSPORT sweep whose weight contribution is exact (SNF stochastic-kernel form): for each cluster move
  x→x′ with proposal q(x′_C|S) and backward kernel L = q(x_C|S) (the same conditional — valid backward kernel
  choice for an independence block proposal):
      log w += log π_{β_k}(x′) − log π_{β_k}(x) + log q(x_C|S) − log q(x′_C|S)
  (energy part = −β_k·ΔU_clu, cluster-local ⇒ bounded increment; q terms one forward each, integrator-free).
  NOTE this is IS-reweighted transport, not MH — moves are always applied, weights absorb the mismatch. The
  proposal = ARM-FULL spline `ClusterProposal` (exact log_q; species-aware). Transport sweep = N seeds.
- Optional ARM-1s variant (flagged, only if ARM-1's weight variance is dominated by q mismatch): use
  swap-and-breathe transpositions inside the transport sweep so species also transport across rungs.

## Why ARM-1 might win (the hypothesis being tested)

Mutation kernels only equilibrate WITHIN π_{β_k}; transport moves the population TOWARD π_{β_{k+1}} before
reweighting, reducing the weight variance per rung ⇒ fewer rungs / higher ESS / better end quality at matched
wall-clock. The campaign's recurring negative (flow adds ~nothing over SMC at easy states) is the null
hypothesis; the glass at T*→0.5 with a validated local proposal is the best-case regime for rejecting it.

## Interfaces (new module `liquid_coupling_flow/ka_local_smc.py`)

- `smc_run(arm, N, B, beta0=0.8, beta1=2.0, ess_target=0.6, max_rungs=40, n_mut=2, seed=0, device)` →
  dict(traj of (beta, ESS, U/N mean/std, acc rates), final pos/s/logw, wall-clock) — FULL trajectory +
  final population saved to `artifacts/smc_pilot_{arm}_N{N}.pt` (results-durability rule).
- Reuses: `KAFlowHeadModel` (seeds+logq), `swap_breathe_sweep`, `mtm_move`, `cluster_energy`,
  parallel-displacement sweep (per-row species via canonicalization after each species-changing sweep),
  `ClusterProposal.sample/log_q`, `ka_energy`.
- Adaptive Δβ: bisect so that effective sample size of exp(−Δβ·U) hits ess_target per rung (standard).

## Gates

- **P0 (plumbing/exactness)**: weights finite through a 3-rung dry run; ESS init sane (seed weights not
  degenerate — measures generator calibration at β₀); species counts invariant; resampling preserves counts.
- **P1 (the A1 fix, N=100)**: ARM-0 completes β 0.8→2.0 with ESS never collapsing to ~1 (target: ESS ≥ 0.2·B
  at every rung after resampling policy) and end population ⟨U⟩/N within 0.02 of −3.260 with g_BB ≈ 2.30.
  This alone is a campaign headline (exact sampler reaching the PT-solid reference without tempering).
- **P2 (transport attribution, N=100)**: ARM-1 vs ARM-0 at MATCHED WALL-CLOCK: rungs needed / final ESS /
  end ⟨U⟩/N + g_BB. Verdict either way; if ARM-1 ≤ ARM-0, the learned-transport line closes with attribution
  (q-mismatch variance vs energy-variance breakdown logged per rung).
- **P3 (transfer, N=256, zero-shot: all learned parts N=100-trained)**: ARM-0 (and ARM-1 if P2 positive) at
  N=256 with rung count scaled ×~1.6 (√N rule): ESS alive end-to-end; end U/N ≤ −3.21 (beats the soft PT
  reference) — directional until the N=256 reference is re-run. Generator seeds at N=256 via the knn-capped
  sampler (parallel campaign validated its downstream utility).

## Risks / honest notes

- The flowhead generator's samples are poor (28.5% clash) — but weights are exact and the bridge starts at
  β₀=0.8 where π is forgiving; if init ESS is degenerate, raise β₀ or pre-relax seeds with a few MALA sweeps
  (kernel-invariant at β₀, weight-free).
- ARM-1's always-move IS-transport can INCREASE variance if q is poor at intermediate β (trained at T*=0.5
  equilibrium only) — this OOD-in-β risk is precisely what P2's per-rung variance breakdown diagnoses; the
  spec deliberately avoids retraining per rung in the pilot (AFT/CRAFT per-rung training = the staged upgrade
  if P2 shows promise choked by β-OOD).
- Wall-clock: SB+MTM sweeps are the slow kernels (~15 s/sweep at B=128); n_mut and MTM usage are budget knobs;
  the pilot targets ≤ ~1.5 h per arm per size.
- The N=256 reference rebuild is a listed prerequisite for sharp P3 claims (separate task, not this pilot).
