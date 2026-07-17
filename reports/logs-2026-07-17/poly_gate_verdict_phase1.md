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
