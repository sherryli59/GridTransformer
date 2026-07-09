# Point-to-set static length in the 2D KA glass via random pinning, with a transferable learned interior sampler — design

**Date:** 2026-07-08 · **Status:** spec (v1 = random pinning; cavity deferred to v1.5)
**Context:** liquid-coupling-flow campaign; closed ledgers (local-smc-pilot, sb-staged-upgrades) concluded the
learned components are sound-but-not-load-bearing for plain configurational sampling vs swap MC. This spec
redeploys them where the *measurement itself is conditional generation*: point-to-set (PTS) correlations.

## 1. Goal and claim

Measure the **PTS amorphous-order length ξ_PTS(T)** of the 2D KA glass (65:35-nominal, ρ=1.2) by the
**random-pinning protocol**, using the existing **N=100-trained local EGNN kernel as an exact-guarded (MH)
proposal** for re-equilibrating the mobile particles among frozen pins.

- **Physics deliverable:** the excess-overlap family Q_∞(c; T) and a threshold length ξ_pin(T) at T ∈
  {0.8, 0.65, 0.5}, testing growth of static order on cooling.
- **Method deliverable:** every learned part is N=100-trained and applied zero-shot; the **large-ℓ (small-c)
  points physically require larger boxes** (L/ℓ_c ≥ 4), so size transfer is *load-bearing for the measurement*,
  not a demo.
- **Positioning:** competes where swap MC is weak (constrained re-equilibration is itself a glassy inner loop,
  repeated over c × T × realizations) and where exactness is guaranteed by construction (learned proposal behind
  Metropolis — the campaign-proven safe role; ARM-1 taught that raw/IS use is poisoned by proposal imperfection).
- **Why pinning (not cavity) first:** field-recommended geometry — mildest constrained slowdown, cleanest
  statistics, power-law analysis; additionally, in 2D, pinning breaks translational invariance and suppresses
  Mermin–Wagner long-wavelength fluctuations that contaminate positional overlap. The closed cavity (narrow
  window, compressed-exponential decay, brutal interior slowdown) is deferred to v1.5 as the method-showcase
  where the accelerator's value is maximal.

## 2. Fixed definitions (locked; anchor-matched at protocol level)

Protocol anchored to the Berthier–Kob geometry-comparison methodology (occupancy overlap; long-time plateau;
random baseline computed, not fit). No published ξ_PTS exists for this exact 2D-KA model, so external comparison
is protocol-level and qualitative (pinning-c route per Karmakar–Parisi PNAS 2013; 2D pinning ≈ 3D qualitatively,
Chakrabarty et al. 2016).

- **Cell grid:** square cells of side a_c = 0.3σ_AA over the box; occupancy n_i ∈ {0,1} per cell. Cells
  containing a pinned particle are **excluded** from all sums.
- **Overlap:** Q(t) = Σ_i ⟨n_i(t) n_i(0)⟩ / Σ_i ⟨n_i(0)⟩ over included cells; occupancy-based (invariant under
  particle exchange), t = MCMC iterations of the constrained sampler (wall-clock recorded alongside).
- **Random baseline (computed):** Q_rand = ρ₀ a_c² = 1.2 × 0.09 = **0.108**.
- **Static overlap:** Q_∞(c;T) = long-time plateau of Q(t), extracted by a stretched-exponential fit
  Q(t) = Q_∞ + A·exp[−(t/τ)^β] on the reference-initialized (decaying) arm, cross-checked against the
  scrambled-initialized (rising) arm's tail mean (see gate G-conv).
- **Pin-spacing:** ℓ_c = (c·ρ₀)^(−1/2) (mean inter-pin distance; c = pinned fraction).
- **Length extraction:** ξ_pin(T) = the ℓ_c at which Q_∞(c;T) − Q_rand crosses a fixed threshold, by monotone
  interpolation of the measured curve. Primary threshold **0.2**; robustness thresholds **0.1 and 0.3** — the
  growth claim ξ(0.5) > ξ(0.65) > ξ(0.8) must hold at all three. Full Q_∞(c;T) curves are always reported;
  the threshold length is a summary, not a replacement.
- **Composition:** pins drawn uniformly from all particles (composition-neutral in expectation); x_B of the
  mobile set recorded per realization. References composition-matched within N; all cross-N absolute statements
  carry x_B (dU/dx_B = −2.80 lesson).

## 3. Components (isolated, testable)

1. **Frozen-mask harness** (`liquid_coupling_flow/ka_pin.py`): given x [B,N,2], s [B,N], and a per-chain boolean
   mask frozen [B,N], expose constrained energy/forces (mobile–mobile + mobile–frozen; frozen–frozen constant
   dropped) and masked kernel steps. Geometry-agnostic: pinning and (later) cavity are just mask generators.
2. **Mask generators:** `pin_mask(c, seed)` — random subset of ⌈cN⌉ particles per chain/realization.
   (`cavity_mask(center, R)` deferred to v1.5, same interface.)
3. **Constrained sampler (reused, no new training):** masked tamed-MALA + masked swap among mobile particles +
   the learned components behind exact Metropolis — the two-time joint EGNN geometry table (knn=32) for species
   block-relabel, and the β-conditioned full-cage heat-bath as learned position proposal. The AR transformer is
   not in the loop (ordering is size/geometry-specific — disqualified for transfer).
4. **Overlap module:** cell-grid occupancy, Q(t) accumulation vs the reference config, exclusion of pinned cells.
5. **Extractors:** stretched-exp fit → Q_∞ per (T, c, realization); threshold interpolation → ξ_pin(T) with
   realization-bootstrap errors.

## 4. Protocol per (T, c)

- **References:** equilibrium configs at T (N=256: PT2 dataset at T=0.5, x_B=0.371; T=0.65/0.8 generated fresh
  by swap MC — cheap at these T). One reference config per realization.
- **Realizations batched in the chain dimension:** 16 pin realizations × 2 init arms × 2 thermal replicas =
  B=64 chains in a single batched run.
- **Two-arm initialization (the convergence instrument):**
  (a) *reference-init:* mobile particles start at the reference positions → Q(t) decays from 1;
  (b) *scrambled-init:* mobile particles start from a high-T/randomized state under the same pins → Q(t) rises.
- **Sampling:** run until the G-conv gate passes or budget exhausted (budget per run recorded up front);
  post-plateau samples give the thermal average of Q_∞.
- **Incremental persistence:** per-unit saves (per (T,c): Q(t) traces, configs snapshots, masks, fit params)
  the moment each unit finishes — full trajectories, not summaries (standing directives).

## 5. Gates (blocking, in order)

- **G-corr (correctness of constrained sampling):** at one *easy, convergeable* point (T=0.8, c=0.16), the
  learned-kernel result must match a brute-force constrained sampler (masked displacement + swap only) with both
  samplers individually passing init-independence. Validates correctness only — explicitly does NOT validate
  large-ℓ convergence.
- **G-conv (convergence, the binding gate — run at EVERY c, smallest c binding):** |Q_∞(decaying arm) −
  Q_∞(rising arm)| ≤ 0.02 per (T,c). **Stated failure mode:** an unconverged large-ℓ point biases Q_∞ upward
  (interior hasn't forgotten the reference) → ξ_PTS too large, smoothly and silently. Points failing G-conv are
  reported as bounds, never fit.
- **G-anchor (protocol sanity):** Q_∞(c) curves qualitatively consistent with the established pinning
  phenomenology (monotone in c; high-T curve below low-T curve; Q_∞ → Q_rand as c → 0). Quantitative external
  comparison is out of scope (no 2D-KA literature value exists).
- **Mixing measurement (informative for v2 trigger, not blocking):** interior decorrelation time of learned
  kernel vs brute-force at the hardest passing (T,c) — if ≤ ~1.5× speedup, that is the documented trigger to
  build the v2 dedicated collective cavity/pin-region generator.

## 6. Ladder and boxes (transfer is load-bearing)

Constraint L/ℓ_c ≥ 4 (box holds ≥4 pin spacings):

| c | ℓ_c | min box |
|---|---|---|
| 0.24 | 1.86 | N=256 |
| 0.16 | 2.28 | N=256 |
| 0.12 | 2.64 | N=256 |
| 0.08 | 3.23 | N=256 |
| 0.06 | 3.73 | N=576 |
| 0.04 | 4.56 | N=576 |

- v1 ladder: c ∈ {0.24, 0.16, 0.12, 0.08} at N=256 (all T) + c ∈ {0.06, 0.04} at N=576 for T ∈ {0.8, 0.65}
  (cheap references). The T=0.5, N=576 low-c points are a stretch goal pending the crossover campaign's N=576
  equilibrium reference (in flight; per-protocol only today).
- All learned parts N=100-trained, zero-shot at N=256/576 (knn=32 cap). Kernel is c- and ℓ-agnostic by locality.

## 7. Scope and staging

- **v1 (this spec):** pinning; T ∈ {0.8, 0.65, 0.5} chosen as a subset of the eventual 5-T ladder
  {0.8, 0.65, 0.55, 0.5, 0.45}; T=0.5 runs through the full G-conv gate (the hard, scientifically-live point —
  v1 must prove the hard case, not only easy high-T points).
- **v1.5 (after v1 banks):** closed-cavity geometry off the same harness — the method-showcase ("reviving the
  hard geometry"), exponential-decay extraction, central-cells measurement.
- **v2 (only if the mixing measurement triggers):** dedicated boundary-conditioned collective proposal
  (whole-region redraw behind MH), trained on (interior | frozen surroundings) pairs.
- **Out of scope:** free-energy estimation; Franz–Parisi potential; IS/SNF weighting of any learned component.

## 8. Risks and honest framing

- **Learned-kernel value is a hypothesis here too:** on unconstrained KA the learned movers were marginal. The
  claim v1 actually needs is only that the *measurement pipeline* is correct + transferable; any interior-mixing
  speedup is upside recorded by the mixing measurement. If the learned kernel adds ~nothing over masked
  displacement+swap, v1 still delivers ξ_pin(T) and the transfer story, and the spec's method-claim narrows to
  "exact, transferable, amortized-across-(c,T,realization) constrained sampler" — still novel for PTS practice.
- **T=0.5 convergence at small c** is the known hard corner (that's where the physics lives); G-conv failure
  there yields reported bounds, and the v2 trigger.
- **2D caveat:** ξ_PTS values in 2D may be modest at T=0.5 (well above T_g≈0.43); the growth *trend* across
  three T is the claim, not a diverging length.

## 9. Artifacts and durability

Per CLAUDE.md: all runs log to `reports/logs-<date>/`; per-(T,c) trajectories + masks + fits saved to
`liquid_coupling_flow/artifacts/pts/` incrementally; every plot committed with full path printed; analysis
scripts copied to the log dir and committed.
