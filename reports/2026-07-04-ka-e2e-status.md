# KA e2e pipeline — status report (2026-07-04)

Scope: train-small (N=100) joint flow + species-block + tamed-MALA corrector → sample-big (N=256).
Backing detail: `reports/2026-07-03-ka-block-transfer-results.md` (Addenda 2–3); memory `ka-reference-underconverged`,
`ka-block-transfer`. Plots in `liquid_coupling_flow/artifacts/ka_e2e_*`.

## 1. Key conceptual questions (blocking first)

1. **[BLOCKING] Is the slow grind at unseen N fatal or just slow?** At N=256 the corrector sits ~0.07 *above* the
   estimated equilibrium (−3.19 vs ≈−3.26) after 6000 iters, still creeping (~0.008/1000 iters, log-slowing). Does it
   reach the floor with more iterations, or is there a floor local correction cannot cross? Every equilibration/speedup
   claim depends on this.
2. **[BLOCKING] What is the true N=256 equilibrium energy?** The PT reference is under-converged (energy fn verified
   correct 3 ways; but finite-size check: N=100 −3.26 vs N=256 −3.21 for an *intensive* quantity ⇒ N=256 relaxed less).
   Without a converged reference, every "gap to PT" number is against a moving target.
3. **What is the flow's real value proposition at unseen N?** The head-start amortizes burn-in but *shrinks with size*
   (1.7×@N100 → 1.17×@N256); the block mover fixes only the species-disorder floor (0.006@N100 → 0.026@N256, small
   because KA species are geometry-slaved). Neither accelerates the slow glassy *position* relaxation — is there a lever
   that does, or is it intrinsically local-move-limited at T*=0.5?
4. **Does "train-small→sample-big" beat the honest baseline?** The decisive control is wall-time-to-equilibrium at N=256:
   pipeline vs swap-MC-from-scratch. Not yet measured; without it the amortization claim is unproven.

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
