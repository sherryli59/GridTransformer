# Multiple-Try Metropolis (MTM) cluster kernel — design spec

**Date:** 2026-07-03. Follow-up to the ARM-FULL acceptance measurement (0.31%): the AR proposal is
**diffuse-but-calibrated** (clash-free hit rate 10.3%, but acceptance|no-clash 2.99% BEATS the EGNN's 2.03%,
Hastings term +4.26). Its bottleneck is single-shot hit rate — exactly what many-candidate kernels fix — and its
log_q is one exact forward pass, so candidates are nearly free on GPU. This spec defines the exact kernel.

## The structural fact this design rests on (verified)

`ClusterProposal`'s conditioning is **x_C-independent**: the frame (`frame_from_positions`, docstring: "IFF
ctx_pos is x_R-only — frame_ctx_slots guarantees it") and all context tokens are built from x_R (the non-cluster
particles) and the fixed scaffold only, and x_R does not move during a cluster move. Therefore the proposal is a
**pure independence sampler on the block**: q(y | state) = q(y | S) with S fixed within the move, identical for
forward and reverse. (Contrast the EGNN, whose cage is re-selected from the cluster centroid → state-DEPENDENT.)
This is what unlocks the simplest exact MTM variant below. The keystone is already regression-tested
(tests/test_ka_cluster.py); the MTM module must assert it structurally (frame recomputed with the cluster
displaced must be bit-identical).

## Kernel options compared

### K1 (PRIMARY): Barker/Tjelmeland ensemble independence-MTM — no reverse draws

Given state x (cluster block x_C, surroundings S):
1. Draw M candidates y₁..y_M ~ q(·|S) i.i.d. (ONE batched AR forward with B·M rows; log q_j returned free).
2. Form the augmented set {y₀ ≡ x_C, y₁..y_M}. Compute log-weights `ℓ_j = −β·U_clu(y_j) − log q(y_j|S)` for ALL
   M+1 members, where U_clu = the cluster-involving energy only (cluster–rest + intra-cluster; the rest–rest
   term is identical across the set and cancels — never compute it).
3. Select the next state J ∼ softmax(ℓ) over the M+1 members (log-space, logsumexp).

**Exactness (detailed balance):** for the joint move x→y (both in the set, the other M−1 candidates shared),
π(x)·q(y)·∏q(others)·[w(y)/Σw] against π(y)·q(x)·∏q(others)·[w(x)/Σw]: with w = π/q both reduce to
π(x)π(y)·∏q(others)/Σw — symmetric ⇒ balance holds. Requires (a) candidates i.i.d. from a state-independent q
(TRUE, the structural fact above), (b) full support of q wherever π>0 (TRUE for the spline head: Gaussian base +
linear tails on all of R²ᵏ — note the OLD bins head violated this outside the box (−69 ≈ hard zero); MTM must
run on spline checkpoints only, assert `head=="spline"`).

Properties: always “moves somewhere” (Barker-style, possibly staying at y₀); recycles every candidate’s
information; move probability = Σ_{j≥1}w_j / Σ_{j≥0}w_j.

### K2: classical Liu–Liang–Wong I-MTM (select ∝ w among M, then draw M−1 reverse candidates, accept
min(1, Σw_fwd/Σw_rev)) — exactly the same stationary law here; strictly more draws for no benefit when q is
state-independent. Kept as a cross-check in tests (K1 and K2 must agree in distribution), not for production.

### K3 (TRAP — documented as invalid): plain sampling-importance-resampling over the M candidates WITHOUT the
current state in the set / without an acceptance step. Not π-stationary (it targets q reweighted only within the
proposal cloud; the current state's weight never competes). The earlier session sketch "sample M, resample by
weight" is this trap — do not implement.

### K4 (extension, out of scope): MTM for the EGNN proposal. Valid but needs the full state-dependent I-MTM
(M−1 reverse draws from q(·|cage(y))) at 2× the ODE cost, and its integrator-priced log_q (self-consistency
~0.36 worst-case) enters the weights → bias risk. The AR's exact 1-forward log_q is the reason MTM belongs to
the AR arm first.

## Expected performance (set expectations honestly)

Per-candidate ratio w_j/w₀ = exp(−βΔU_j + log q(x_C) − log q(y_j)) is exactly the single-try log-α. With mean
single-try α ≈ 0.003: move probability ≈ M·ᾱ/(1 + M·ᾱ) → M=32 ≈ 9%, M=128 ≈ 28% (saturating). MTM buys ~M× the
acceptance, NOT a fix to the energy tail — its economics rest on candidates being nearly free:
- proposal: one batched forward (B·M rows) — GPU-parallel, sublinear wall-clock in M until memory-bound;
- weights: vectorized U_clu (k×(N−k) + k(k−1)/2 pair terms per candidate) — cheap;
- log q: already returned by `sample()`; log q(x_C|S) is ONE extra `log_q` call per move.

## Implementation sketch

New file `liquid_coupling_flow/ka_cluster_mtm.py`:
- `cluster_energy(xC[B,M,k,2], pos[B,N,2], cluster_idx, s, L) -> U_clu[B,M]` — vectorized cluster-involving
  shifted-KA energy (reuse SIGMA/EPS/RCUT from ka_energy; mask the cluster columns out of pos for the cross
  term; intra term over k(k−1)/2 pairs).
- `mtm_move(P, pos, s, cluster_idx, sc, L, M, gen) -> (pos_new, moved[B], info)` — expand pos to B·M rows →
  one `P.sample` → reshape; ℓ for M+1 members; Gumbel/multinomial select in log-space; write back selected
  cluster; info = {move_prob, w-ESS, clashfree_frac}.
- `mtm_sweep(...)`: seeds 0..N−1 permuted, one mtm_move per seed (cluster_slots per seed as now).
- Assertions: `P.head_mode == "spline"`; frame x_C-independence spot-check (frame with cluster zeroed == frame
  with cluster displaced) at kernel construction.

## Tests / gates

1. **Exactness-by-stationarity (the decisive test):** start B=64 chains AT the parallel-tempering reference
   (equilibrium); run ~200 MTM sweeps; ⟨U⟩/N and g_BB must stay at reference values within noise (an invalid
   kernel drifts — this catches weight/selection bugs the algebra can't). Contrast run: K3 (the trap) as a
   negative control SHOULD drift — proves the test has teeth.
2. **K1 == K2 distributional cross-check** at M=4 (small-M chi-square on move destinations from a fixed state,
   or matched move-prob estimates).
3. **M=1 reduction:** K1 at M=1 = Barker acceptance w₁/(w₀+w₁) — verify against a direct Barker implementation.
4. **Performance curve:** move-prob and wall-clock per accepted move vs M ∈ {8, 32, 128}; report saturation
   point; compare against plain-MH AR (0.31%) and EGNN-big (1.02%) economics.
5. **Follow-up gate (separate decision):** if move-prob ≥ 5% at practical wall-clock — the mixing benchmark that
   actually matters: MTM-cluster sweeps vs swap-MC baseline on ⟨U⟩ relaxation / cage-escape statistics, i.e. does
   the coordinated 7-particle move buy real decorrelation the single-site kernel lacks [[glass-smc-corrector-headstart]]
   economics question, cluster-move edition.

## Risks

- **Weight variance**: w = π/q with a diffuse q has heavy right tails (occasional dominant candidate). Valid but
  inefficient; monitor w-ESS per move. Mitigation knob if needed: temper the selection (γ<1 powers) is NOT exact —
  do not; instead raise M.
- **Memory at B·M**: transformer rows B·M×(1+n_ctx+k) tokens — B=64, M=128 ≈ 8k rows, fine; cap via chunking.
- **π-support / q-support**: spline-head only (asserted). Bins checkpoints are excluded by construction.
- **Saturation economics**: if 9% @ M=32 doesn't beat single-site swap-MC per wall-clock in gate 5, the kernel is
  a negative result — record and close (the SMC direction remains the validated lever either way).
