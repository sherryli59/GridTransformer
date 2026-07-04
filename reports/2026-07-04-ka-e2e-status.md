# KA e2e pipeline — status report (2026-07-04)

Scope: train-small (N=100) joint flow + species-block + tamed-MALA corrector → sample-big (N=256).
Backing detail: `reports/2026-07-03-ka-block-transfer-results.md` (Addenda 2–3); memory `ka-reference-underconverged`,
`ka-block-transfer`. Plots in `liquid_coupling_flow/artifacts/ka_e2e_*`.

## 1. Key conceptual questions (blocking first)

**Framing.** The program is a three-part loop: a **generator emits an imperfect proposal** (approximate positions
*and* approximate species) → a **"smart" local MC relaxation** corrects it toward equilibrium → and we want that whole
loop, **trained small, to work at larger sizes (transfer)**. "Smart" is the load-bearing word: an imperfect proposal
introduces *frustration* — a particle can sit in a good local cage yet carry the wrong species — and pure position
moves cannot undo that, so the corrector must also do *informed species relabeling* (flip labels toward what the local
geometry implies). The conceptual questions are about where the value and the limits of this decomposition actually
live.

1. **[BLOCKING] Division of labor — what must the proposal get right, and what is the corrector's job?**
   A proposal only pays off if repairing it costs less than sampling from scratch. Conceptually the generator's job is a
   *warm start* — deliver a configuration already in the right structural basin — and the corrector pays down the
   residual. The unresolved boundary: does the generator only need to nail **structure** (basin, g(r) shells) and cede
   the **energy floor** to MC, or must it also land near the floor? This is blocking because the two halves scale
   differently. Getting into the basin is a *size-independent burn-in*; grinding to the energy floor of a glass is a
   *size-dependent, collective* relaxation. If the proposal only removes the burn-in, the paradigm amortizes the cheap
   half and leaves the expensive half untouched — which is exactly the pattern we see (head-start real but shrinking
   with N; the tail slow and unaccelerated).

2. **[BLOCKING] Is proposal-induced *frustration* actually the rate-limiting defect — i.e., is "informed relabeling"
   load-bearing or decorative here?** The smart-relabeling move exists to repair species defects a local displacement
   cannot. But its value depends entirely on whether species carry *independent* frustration. On KA the species are
   **geometry-slaved**: given the cage, the correct label is essentially determined, so informed relabeling is *correct
   but rarely changes anything* — it contributes a little depth (removing a small quenched species-disorder floor), not
   speed. The conceptual point that generalizes: informed relabeling is only *essential* where the proposal's species
   errors are both common and not fixable by reading geometry — i.e., systems with genuine species/size frustration
   (additive or polydisperse mixtures), not KA. So a live question is whether KA is even the right substrate to
   *demonstrate* the smart-relabeling idea, versus merely to validate that it's exact and safe.

3. **[BLOCKING for the transfer claim] What must be invariant for the loop to transfer, and which costs are intrinsic?**
   Transfer requires every learned piece to depend only on *local, size-invariant* features. We have one clean success
   (the geometry→species readout transfers once its aggregation is made local — the degree cap) and two size-dependent
   costs (the proposal's position quality degrades with N; the relaxation time to the floor grows with N). The
   conceptual crux: the head-start amortizes a size-*independent* quantity, but the glassy tail is a size-*dependent*
   collective relaxation that no locality trick removes — so the amortized value **necessarily** shrinks with N. Is that
   shrinkage fundamental to "local proposal + local corrector," or is it an artifact of a still-weak proposal that a
   stronger generator would push back?

4. **Does "proposal + corrector" cede exactly the glassy part?** The decomposition splits the problem into a fast
   learnable proposal and an exact local corrector — but glassiness *is* slow collective relaxation, which sits in the
   crack between them. A one-shot proposal cannot emit a fully-relaxed glass (re-proposing whole configs hit an overlap
   wall), and local corrector moves grind through the tail slowly; the hard part falls precisely where the split is
   weakest. So the honest question is whether the achievable claim is only *"amortize structure + burn-in, not the
   tail,"* or whether a proposal/corrector **co-design** — the generator proposing *collective/cluster moves* the
   corrector accepts, rather than just an initial config — could reach into the tail. This is where "generator as
   proposal" would graduate from *warm start* to *transition kernel*.

## 1b. Positions on the conceptual questions (grounded in the campaign's dead-ends)

**(a) The generator only speeds burn-in — would a stronger generator push *deeper*?** Mostly no. The floor is a
*corrector* property: uniform and flow seeds reach the *same* depth (they differ only in time-to-basin). A stronger
generator lowers the *starting* energy → shorter burn-in, but does not change the corrector's floor. To make the
generator itself reach the floor, it would have to emit near-equilibrium glass configs one-shot — and that is exactly
the wall the campaign hit repeatedly: coupling-flow can't build hard-core exclusion from a uniform base, whole-config
re-proposal gives ESS=1 (overlap wall), AR sees only half the cage. Smooth one-shot generators provably struggle to
represent the sharp many-body correlations of a glass. **So: stronger generator ⇒ faster to the same floor, not a
deeper floor. "Deeper" is bought by a better corrector, not a better proposal.**

**(b) How to do position + species moves *jointly* for faster equilibration?** Today they're *sequential* (10 MALA
steps, then a species block) — which zig-zags across the correlated defect (wrong species ⇔ slightly-wrong local
geometry). The conceptual fix: a single MH move that resamples a particle's/cluster's **(x, s) jointly** from the flow's
*local joint conditional* P(x_i,s_i | neighbourhood), energy-guarded for exactness. The joint flow already models this
(shared trunk + position-velocity + species-denoiser heads, two-time). It must be MH not Gibbs — iterating the learned
conditional alone collapses (overlap→1, g_BB explodes); the exact energy is the guard. This moves *along* the
frustration direction instead of across it. Corollary: a flow-*informed local position proposal* (bigger, cage-aware
steps than MALA) is the untested lever that could grind the tail faster — the same object, used as a transition kernel
rather than a seed.

**(c) Must sampling efficiency degrade with size under zero-shot transfer?** Split efficiency into *local* mixing time
τ (per region) and *global* equilibration. For a genuinely local sampler at fixed (ρ,T) the *local* relaxation time is
size-**invariant** — and we saw it: τ_U 14 (N=100) → 20 (N=256), a mild bump, not a blow-up. What grows with N is (i)
the one-time **burn-in** and (ii) the slowest **collective/long-wavelength** mode (~L² hydrodynamics; for a glass, the
slow structural tail). **So zero-shot transfer preserves local efficiency (this is the payoff of locality) but cannot
buy size-invariant *global* equilibration.** The generator's head-start (a burn-in accelerator) therefore *necessarily*
shrinks in relative value with N; the corrector's local mixing does not degrade. Efficiency degradation is not
inevitable for the corrector — it is inevitable for the *proposal-as-seed* framing.

**(d) EGNN flow vs AR transformer?** For *transfer*, EGNN flow, decisively. The AR transformer needs a canonical order
(space-filling curve), and the campaign's firm result is that curve-conditioning is *curve-specific* → a cliff at
unseen N; AR also sees only half the cage (residual floor). EGNN is equivariant (no order ⇒ no curve ⇒ size-invariant
once aggregation is made local via the degree cap), sees the full cage, and *still* has exact likelihood (traceable
divergence). AR's only edges are generation speed and one-shot log-q — the first is irrelevant for *local* corrector
moves, the second is matched by EGNN's exact divergence. **The ordering requirement is disqualifying for zero-shot size
transfer; EGNN is the transferable architecture.**

**(e) Local cluster moves instead of global proposals?** This is the unifying answer to (a)–(d). Global proposals die
at large N because ΔE is *extensive* → acceptance → 0 (the overlap wall). A **local cluster move** (resample a small
spatial cluster's (x,s) from the flow's local conditional, accept on a *bounded* local ΔE) simultaneously fixes:
acceptance (ΔE size-independent ⇒ O(1) acceptance at any N), transfer (local features only), and frustration (a cluster
is exactly the correlated position+species defect). It graduates the generator from *warm-start seed* to *local
transition kernel* — the object that can actually reach into the tail. Caveat from the campaign: on KA the cluster/swap
lever is *marginal* (species geometry-slaved; equilibrium swap already wins), so KA validates exactness/safety but is a
poor *showcase* — the mechanism's value needs a genuinely frustrated substrate (additive/polydisperse mixtures).

**Unifying thread.** All five collapse to one move: **replace "global proposal + local corrector" with a learned
*local cluster transition kernel*.** It is the single change that fixes acceptance (bounded ΔE), transfer (locality),
and frustration (joint cluster (x,s)) at once — and it reframes the generator's job from seeding burn-in to *proposing
moves*. The recurring caveat is substrate: KA's geometry-slaved species keep hiding the species/frustration levers, so
the demonstration system may matter as much as the algorithm.

## 2. Done — with reflection

- **Corrected N=256 analysis** (PT-convergence, decorrelation, g(r); MH arms dropped). → τ_U≈20 arm-independent,
  g_AA/g_AB on the PT line. *Reflection:* the stationarity decorrelation plot is the wrong regime to judge the mover
  (corrector-not-mixer, eff→0 at equilibrium); it looked null but wasn't informative either way.
- **Diagnosed PT-reference under-convergence.** Energy = ka_energy = MC's `_u_matrix` = brute-force (machine precision);
  drift + finite-size + gate blind spots. Literature: standard 2D KA, T=0.5 reachable (Flenner–Szamel to 0.45). →
  *Reflection:* decisive and clarifying — not an energy bug; the target was soft; true N=256 eq ≈ −3.26. Reframes all
  prior gap numbers. Best single result of the day.
- **N=100 in-distribution control** (raw gen + 3 MALA arms + matched panels). → reaches true eq ≈−3.26 with visible
  head-start; g_BB (incl. r≈1.4 shoulder) reproduced. *Reflection:* localizes the N=256 g_BB under-fill to
  transfer/under-convergence, not a corrector limitation — encouraging for the *structural* claim.
- **Relaxation curves both N.** → the honest value/limit view: head-start + depth real but modest; N=256 stuck above the
  floor. *Reflection:* replaced a misleading "null" (τ_U) with the plot that shows where value and limits actually live.
- **Infra:** generalized `ka_e2e_generate.py`/`ka_e2e_mala.py` to take N; all trajectories/plots/scripts committed
  (`5503d7d`,`a1b9731`,`70874d8`,`d701a05`,`12f1cc7`). *Reflection:* reproducible; the N=256 slow-grind and soft target
  mean the numbers are trustworthy but the verdict is not yet in.

## 2c. Project arc — done + reflection (what the program has established)

- **Built the Hilbert / space-filling-curve AR transformer for size-transferable Boltzmann sampling** (turn-tokens,
  curve-rail, arc-repr). → *Reflection:* curve-conditioning is **curve-specific** → a hard cliff at unseen N. Clean
  NEGATIVE that redirected the whole program away from curve-anchored AR.
- **Built the geometry-invariant local-frame conditional.** → *Reflection:* removed the transfer cliff (the conditional
  transfers, no cliff) — but exposed the next wall: teacher-forcing ≠ free-running, a residual per-step gap.
- **Ran the exhaustive generation-gap diagnosis** (TF-vs-FR, per-index; variants deltas/dsperp/arcnorm/pair-aware/cell).
  → *Reflection:* proved the gap is **residual** (the 50%-future-neighbour / half-cage wall), universal across
  representations — so *no representation change can cross it*. Saved a long tail of chasing the wrong lever.
- **Built + gated the exact coupling flow (uniform base)**, the principled invertible arm. → *Reflection:* FAILED the
  in-dist gate — a smooth flow can't build hard-core exclusion from a uniform base (transport wall). Another clean
  negative.
- **Built the non-causal full-cage conditional as a diagnostic.** → *Reflection:* proved the floor *is* the half-cage
  wall (g_BB 1.25→2.00 with the full cage), but iterating the learned conditional alone (Gibbs) collapses → full-cage
  needs an **energy guard**. Pinpointed the mechanism and the constraint.
- **Pivoted to the SMC + swap-augmented corrector; tested easy LJ vs hard KA glass.** → *Reflection:* on easy LJ the
  flow adds ~nothing (SMC equilibrates in a few sweeps); on the hard glass the head-start is **durable** (~270–510
  sweeps). Justified the glass pivot *and* set the honest value bar (head-start, not exactness-for-free).
- **Built the KA glass substrate** (energy, scalable swap-MC, observables, PT reference). → *Reflection:* validated
  foundation — but the PT reference is **under-converged** (found today), which quietly made every absolute-energy gap a
  soft number.
- **Built the joint species-position flow** (two-time conditioning, swap-CTMC + torus ODE, traceable-EGNN exact
  divergence). → *Reflection:* exact, learns species-position correlation, denoiser-as-MH gave ~40× — but on KA species
  are **geometry-slaved**, so the species lever is marginal. Right tool, wrong substrate to showcase it.
- **Explored the cluster-move / MTM line** (AR proposal-A, MTM kernel, EGNN cluster flow). → *Reflection:* AR cluster
  proposal NO-GO — the load-bearing ingredient is **EGNN equivariant message passing**; MTM validated only for near-eq
  relaxation + heavy-tailed transport. Correctly narrowed the mover's true domain.
- **IPL44 / eRSI reproduction + lever push.** → *Reflection:* AR strictly dominated by eRSI on the stiff r⁻¹² target;
  IPL44 is *additive* → demixing (swap-unsafe), which proved the block mover's home is the **KA glass**, not IPL. A
  negative that relocated the lever.
- **KA train-small→sample-big** (degree cap, flow seeds, e2e pipeline). → *Reflection:* the **degree cap (knn=32) is THE
  size-transfer fix** (geometry table 0.50→0.99 at N=256, no retrain) — the genuine locality crux resolved; the e2e loop
  reaches PT-level *structure* at unseen N (1.83×). The clearest positive of the program.
- **Pressure-tested the PT reference (this session).** → *Reflection:* energy verified correct three ways; reference
  under-converged (true eq ≈−3.26). So the *transfer + structure* claims stand, but the *absolute-energy speedup* claims
  rest on a soft target until a converged reference exists.

## 3. Resources needed to move forward

- **Compute for a converged N=256 reference**: stronger swap-MC/PT (higher swap rate + longer equilibration) + a
  drift-aware convergence gate (block-average the *full* collection window and test for a trend, not coarse mean +
  seed-agreement). Pins Q2.
- **Compute for an extended N=256 corrector run** (~20k iters) to answer Q1, plus a **swap-MC-from-scratch baseline** at
  N=256 for the wall-time control (Q4).
- **A second converged size** (or the N=100 anchor) for a clean finite-size extrapolation of the true floor.
- *Possibly* a faster position-relaxation move if Q1 shows a hard floor — note species-swap won't help (position, not
  species, is the KA bottleneck), and flow re-proposal already hit the overlap wall (memory `joint-species-flow` A1).

## 4. Planned next steps

1. Re-run the N=256 PT reference with the stronger protocol + drift-aware gate → **pin the true floor** (Q2).
2. Extend N=256 flow+block8 to ~20k iters → **does it reach ≈−3.26, and how slowly** (Q1); report equilibration time.
3. Add the **wall-time-to-equilibrium control**: pipeline vs swap-MC-from-scratch at N=256 (Q4) — the amortization test.
4. Only if (2) reveals a hard floor: investigate a position-relaxation accelerator; otherwise the verdict is "transfers,
   slow tail" and the write-up focuses on the structural transfer + head-start with honest limits.
