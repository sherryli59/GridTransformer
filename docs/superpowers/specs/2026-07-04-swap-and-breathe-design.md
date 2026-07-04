# "Swap-and-breathe" — species-transposing cluster move (v1: adjacent pairs) — design spec

**Date:** 2026-07-04 (approved in-session). **Goal:** open the EQUILIBRIUM species-exchange channel in the 2D
binary KA glass (N=100, 65:35, T*=0.5), which is measured DEAD today.

## The problem (measured, not assumed)

- Position-preserving A↔B swaps at equilibrium: **0 acceptances / 102,400 trials** (this session; consistent
  with binary-KA swap-unfriendliness; parallel campaign measured 0.5% during relaxation).
- The parallel campaign's block-relabel mover (jf_ka100tt geometry table) works DURING RELAXATION (removes
  ~0.03/N quenched species disorder) but its effective rate **→0 at equilibrium** (species geometry-slaved,
  redraws are null moves). See reports/2026-07-03-ka-block-transfer-results.md.
- Species is the SLOW mode at stationarity: τ_s ≈ 100–190 vs τ_U ≈ 20 (same report, Addendum 2).
- Physics: an equilibrium A↔B exchange requires the σ=1.0/σ=0.88 pockets to DEFORM in the same move —
  label-only moves cannot do this; hence "swap-and-breathe".

## The move (v1)

One exact Metropolis–Hastings move, assembled from validated components:

1. **Seed slot** ~ Uniform{0..N−1}; cluster = `cluster_slots(seed, sc, k=7, L)` — deterministic,
   position-independent (same as the validated MTM kernel). Cluster members are mutual near-neighbors ⇒ the
   swapped pair is automatically ADJACENT (v1 scoping).
2. **Transposition**: let s_cl be the cluster's species pattern, nA·nB its unlike-pair count. If nA·nB = 0
   (single-species cluster), the move ABORTS (counts as rejected; no bias — abort is state-symmetric via the
   deterministic cluster). Else pick one unlike pair uniformly (prob 1/(nA·nB)) and transpose its labels →
   s′_cl. nA·nB is invariant under transposition ⇒ forward/reverse selection probabilities cancel.
3. **Breathe**: propose x′_C ~ q(· | S, s′) — resample ALL k cluster positions from the exact-log_q generator
   (`ClusterProposal` head="spline", pair_feats=True; ckpt `ka_cluster_flow_full_N100.pt`) conditioned on the
   cage S (x_R, unchanged) and the SWAPPED pattern s′. The generator is species-aware by construction
   (σ-pair features).
4. **Accept** with min(1, exp(−βΔU) · q(x_C | S, s) / q(x′_C | S, s′)); on acceptance the state's cluster
   species become s′ and positions x′_C.

Detailed balance: seed selection symmetric; cluster deterministic; transposition-selection factors cancel
(nA·nB invariant); frame/conditioning x_C-independent (tested keystone); q densities exact (one forward each).
ΔU must use the species-dependent energy for both patterns (cluster-involving terms only; rest–rest cancels —
`cluster_energy` already takes per-row species).

## State representation (the known pitfall)

Global state = (positions[B,N,2], species[B,N]). Species now CHANGE per move ⇒ the fixed-canonical-species
assumption from the benchmarks does NOT hold downstream of this kernel. Rules:
- The kernel operates in slot-ordered coordinates per move (on-the-fly `_curve_order`, scatter back), like the
  benchmark mtm_step; it must scatter back BOTH positions and species.
- Any vectorized displacement kernel running alongside must take per-row species OR re-canonicalize (sort by
  species, permuting positions) after each swap-and-breathe move — v1 gates use per-row-species-safe metrics
  (`ka_energy._matrix` supports [B,N] per-row species — verified: ka_energy.py docstring "per-config [B,N] ->
  [B,N,N]"; `cluster_energy` likewise).
- Global counts 65:35 asserted invariant after every move (cheap tripwire).

## Interfaces

New file `liquid_coupling_flow/ka_swap_breathe.py`:
- `swap_breathe_move(P, pos[B,N,2], s[B,N], seed, sc, L, beta) -> (pos', s', accepted[B], info)` — one move at
  a given seed for all chains (per-chain transposition choice and accept).
- `swap_breathe_sweep(P, pos, s, sc, L, beta, n_moves=N)` — random seeds.
- `info`: acceptance, EXCHANGE rate (accepted AND transposition changed the pattern — the payoff events),
  mean log-ratio components (for diagnosis).
- Reuses: `cluster_slots`, `ClusterProposal.sample/log_q` (already take s[B,N]), `cluster_energy` (per-row s).

## Gates

- **G0 exactness plumbing**: species counts invariant; sample/log_q round-trip on swapped patterns < 1e-4;
  scatter-back bijectivity test (positions+species).
- **G1 (decisive, cheap)**: equilibrium acceptance + exchange rate over ≥10k moves from the N=100 reference
  (random slice). Success = ANY stable nonzero exchange rate (baseline exactly 0). Report the ratio-component
  breakdown (is rejection energy-driven or q-driven?).
- **G2 stationarity**: random-slice hold (U/N, g_BB) with swap-and-breathe interleaved into displacement
  sweeps, ~50 sweeps — same protocol/thresholds as the MTM gate. ALSO track species-composition observables
  (fraction of B with B-neighbors, g_AB/g_BB) — the new failure surface is species-distribution bias.
- **G3 payoff**: species autocorrelation τ_s with/without the kernel interleaved (baseline τ_s 100–190,
  parallel campaign protocol); and per-accepted-exchange wall-clock.

## Staged extensions (NOT v1)

- v2 MTM-ification: species part makes proposals state-dependent ⇒ needs full Liu I-MTM with M−1 reverse
  draws under s (not the reverse-free Barker trick). Only if G1 shows small-but-nonzero acceptance.
- v3 denoiser-guided seed/pair selection from `jf_ka100tt_best.pt` (bias toward geometry-ambiguous sites;
  selection probs enter the ratio). Only after G3.
- v4 bigger k / annealed breathe if k=7 pockets are too tight for exchange.

## Risks

- q(·|S, s′) is queried on swapped patterns — locally in-distribution (all local patterns occur in data) but
  possibly sharper-OOD at specific sites; G1's component breakdown diagnoses this.
- ARM-FULL's 32.7% per-particle clash caps single-try acceptance; expectation ~0.1–1% acceptance, lower for
  true exchanges. Any nonzero stable rate passes G1 (channel currently dead).
- If G1 = 0: conclusion is itself valuable — even coupled local relaxation cannot open binary-KA equilibrium
  swaps at k=7 ⇒ escalate k or close the channel with attribution.
