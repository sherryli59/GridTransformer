# Phase-1 verdict: one-shot swap-endpoint block kernel is ORACLE-LIMITED

Date: 2026-07-17. System: NBC continuously-polydisperse soft spheres, N=300, rho=1, T=0.085
(near swap-arrest onset; tau_swap ~ 10k sweeps). All measurements on CLEAN rebanked frames
(poly_bank_fixed_run3.pt, held-out run). Harness: poly_gate_swap.py (review-validated: reverse-leg
convention exact to 0.0, dU double-count cross-check 1.6e-4, classical baseline reproduced).

## The measured chain

1. **Classical direct swap acceptance vs |dsig|** (the baseline, frozen positions):
   0.44 / 0.026 / 0.0000 / 0.0000 / 0.0000 across bins (0,.1)/(.1,.2)/(.2,.3)/(.3,.45)/(.45,.9).
   Classical swap is stone dead beyond |dsig|=0.2 (0/524 attempts).

2. **Noise scan** (identity proposal, all 8 block particles displaced by w): ANY isotropic noise is
   fatal at beta=11.8 — w=0.05 costs dU med +2.4 (acc ~1e-13); w=0.35 costs +14,467.
   The r^-12 glass tolerates ~zero random smear on a block.

3. **Trained flow gates** (20k-step FM, sigma(t)-conditioned, k=8, 400 attempts, T=0.085):
   - base_w=0.35: acc 0.0000 everywhere; dU med +110 (but 130x below random-noise +14,467 —
     the flow steers error into soft modes).
   - base_w=0.05: acc 0.0000 everywhere; dU med +3.47 (q10 +1.58) + logq_gap −9.1
     (contraction asymmetry). Close to oracle but the oracle itself is the wall:

4. **ORACLE** (propose the exact 400-sweep-relaxed endpoint — the best any proposal can be):
   | bin | dU med | acc-proxy mean |
   |---|---|---|
   | (0,.1) | +0.17 | 0.42 |
   | (.1,.2) | +0.87 | 0.14 |
   | (.2,.3) | +2.46 | 0.016 |
   | (.3,.45) | +4.00 | 0.012 |
   | (.45,.9) | +4.49 | 0.000 |
   Nonzero dead-bin oracle acceptance is a TAIL effect (~1.5% of swaps have cheap accommodation
   pathways), not typical.

5. **Oracle vs k** (does unfreezing more environment pay the cost?): FLAT.
   bin (.2,.3) dU med: +2.52 (k=8) / +2.35 (16) / +2.20 (24) / +2.27 (32); bin (.3,.45) ~flat
   ~+3.4-4.3; top bin worsens (fixed 400-sweep relax under-relaxes big blocks). The insertion
   cost of a dissimilar swap is LOCAL to the swapped pair's cages and does not relax away with
   block size.

## Verdict

**The wall is thermodynamic, not a learning deficit.** A dissimilar-pair swap with (even partially)
frozen environment costs 2-4.5 energy units of local free energy (25-50 kT at T=0.085) that
block-local relaxation cannot pay, at any measured block size. A PERFECT one-shot proposal
(the oracle) is capped at ~1-2% tail-driven acceptance in the classical-dead bins; a learned flow
sits strictly below that (endpoint imprecision + contraction asymmetry logq_gap ~ −9).

One-shot learned block proposals CANNOT extend swap's reach here — consistent with (and a
sharper localization of) Ciarella 2023's global-proposal negative, the KA fine-energy-precision
wall (ka-egnn-cluster-flow), and BCY's choice of GLOBAL shrinkage deformation over local moves.

## What survives / the open path

- **Phase 2 (gradual sigma):** pay the 25-50 kT in installments — many small accepted semi-grand
  u-moves form a PATH through sigma-space (each step dU ~ 0.1-0.3). Machinery built + validated
  (semigrand.py J1, joint_flow.py J2); needs mu(sigma) calibration to pin composition (measured:
  without mu, mean sigma drifts 0.998->0.82; volume-conserving variant still shape-drifts).
- Flow-drafted MTM (cluster-MTM template) as an alternative exactness route if one-shot density
  is abandoned.
- All flow machinery (sigma(t) conditioning, exact reverse-RK4 logq, batched gate) is validated
  and reusable; the gate harness measures any future kernel drop-in.

Data: poly_gate_bw{35,05}_T0.085_400att.pt (every attempt: ds/dU/logqs/accepted + classical),
probes inline in session; trainings poly_train_swapflow_*.out; ckpts liquid_coupling_flow/artifacts/
poly_swapflow_{,bw05_}T0.085_k8_best.pt.

## Formalized instrument results (NOVELTY-2, 2026-07-18)

Instrument: `poly_oracle_instrument.py`. T=0.085, held-out FIXED bank run=3, 300 attempts/bin
(1500/protocol), N_BOOT=10000 bootstrap on the continuous acceptance-proxy min(1,e^-betadU) (never
on binarized accepts). Figure: `poly_oracle_cap_T0.085.png`. Full arrays + every number below:
`poly_oracle_cap_T0.085.pt`. This section **supersedes** the informal oracle read above (60-200
attempts/bin, fixed-400 relax at all k, no CIs) with two corrections the informal read got wrong.

**Classical curve reproduces the informal baseline almost exactly** (sanity check the instrument
is measuring the same thing): 43.8% [39.2,48.4] / 2.72% [1.24,4.48] / 0.014% [0.00,0.04] / 0.000%
[0,0] / 0.000% [0,0] across the five bins (vs. the original 0.44/0.026/0/0/0).

**Correction 1 -- the oracle cap is k-DEPENDENT, not flat.** Converged-relax acceptance-proxy in
bin (.2,.3): **0.74% [0.05,1.76] at k=8 vs 3.87% [1.93,6.07] at k=32** -- CIs do not overlap
(bootstrapped k32-k8 difference: +3.13% [+1.03,+5.45]). Bins (.1,.2) and (.45,.9) show the same
CI-separated k-effect (+11.2% [+6.1,+16.4] and +1.37% [+0.31,+2.75] respectively); bins (0,.1)
(ceiling, ~52% both k) and (.3,.45) are not significant. **Median dU stays flat across k in every
bin** (e.g. (.2,.3): +2.24 at k=8 vs +2.01 at k=32) -- the entire k-effect lives in the tail of
the acceptance-proxy distribution (more movers = more independent chances at a cheap accommodation
pathway), which is exactly why phase-1's median-only, 60-200-attempt read never saw it: mean
acceptance is tail-dominated and needs the statistics this instrument adds to resolve. Separately,
the sweep-BUDGET confound is real and independently measured: converged relax needed ~1600-2200
sweeps at k=8 and ~2500-2650 at k=32 (recorded per attempt) to plateau, i.e. k=32 needed >6x the
fixed-400 budget -- so the historical fixed-400 protocol under-relaxed k=32 in absolute terms even
though (checked directly) it still shows a similarly-sized k-effect once given enough attempts
(fixed bin (.2,.3): 1.06% [0.19,2.19] k=8 vs 6.04% [3.57,8.74] k=32). Both relax modes and all
numbers with full CIs are in the .pt (`oracle[k][relax_mode]['binned']`, `flat_in_k`).

**Correction 2 -- NCMC does NOT beat the oracle cap; v1's 4.1% was a tail fluke.** A fresh,
independent, properly-powered NCMC run at the best-known v2 config (r_loc=3.0, n_steps=800,
150 attempts/bin, distinct seeds from the standing probe) gives bin (.2,.3) = **0.037%
[0.004,0.089]** -- below BOTH oracle variants there (k=8 converged 0.74%, k=32 converged 3.87%,
bootstrapped NCMC-vs-oracle32 difference -3.83% [-6.04,-1.92], significant). NCMC is below the
oracle band in every bin from (.1,.2) up. The original v1 probe's headline 4.1% in this bin came
from only 25 attempts with a CI of [0.01%,12.12%] -- consistent with one lucky low-W draw, not a
real effect; the properly-powered number (0.037%, ~110x smaller, tight CI) settles this as a tail
fluke, not a cap-beating result. **Conclusion revised: NCMC does not escape the frozen-env oracle
cap** (it cannot -- it is solving an easier, non-frozen-env problem by construction, but at
n_steps=800/r_loc=3.0 its measured acceptance is simply lower than the oracle's, not higher).

**NCMC's genuine, verified wins are narrower than "beats the cap" but real:**
  - **Beats classical where classical is weak**, bin (.1,.2): NCMC 5.77% [3.10,8.99] vs classical
    2.72% [1.24,4.48] -- point estimate 2.1x higher; individual CIs only narrowly overlap
    (3.10-4.48%); the bootstrapped *difference* CI is +3.05% [-0.10,+6.56] -- directionally
    consistent and suggestive but does not cleanly clear 95% significance at these sample sizes
    (n=150 vs n=295). Reported honestly rather than rounded up to "significant."
  - **Resolves classical's noise-floor bin to a robust nonzero**, bin (.2,.3): classical's own CI
    [0.00%,0.04%] includes zero (statistically indistinguishable from "truly dead" at n=280), while
    NCMC's CI [0.004%,0.089%] excludes zero -- a real, if small, statistically confirmed opening
    that classical's own statistics cannot claim.
  - **Does NOT open the two deepest dead bins** (.3,.45) and (.45,.9): NCMC fresh is exactly 0.000%
    [0,0] in both, identical to classical's floor -- it is the ORACLE (not NCMC) that opens these,
    with CIs excluding zero at k=32 in both bins (0.30,.45): 1.03% [0.03,2.35]; (.45,.9): 1.38%
    [0.33,2.76]) and at k=8 in (.3,.45) (0.65% [0.01,1.59]). An earlier draft of this section
    (and the figure's first title) claimed NCMC "opens exactly-zero bins to nonzero" generally --
    checked directly against the CIs and retracted; that claim only holds for bin (.2,.3), not the
    two deepest bins.
  - **Is actually implementable, unlike the oracle**: the oracle requires ~1600-2650 block-relax
    sweeps of privileged lookahead per single proposal (it IS the equilibrium answer, not a
    proposal one could generate without already having run the relaxation) -- a real sampler could
    never afford to pay the oracle's own cost to generate each candidate. NCMC's composite
    acceptance min(1,e^-betaW) is self-contained and exact by construction (Nilmeier et al. 2011);
    it is a real, deployable kernel that happens to sit below the cap at this configuration, not a
    cap-beating one.

Figure `poly_oracle_cap_T0.085.png` (title + caption carry this corrected framing verbatim).
