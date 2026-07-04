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
