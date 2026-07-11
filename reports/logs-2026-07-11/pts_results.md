# Point-to-set (PTS) via the K=4 multi-axis generator — first computation + gaps (2026-07-11)

Goal: measure the static PTS cavity overlap q(R) = <overlap(C, C0)> with C ~ P(interior | frozen boundary)
~ e^{-bU}, using the multi-axis EBM (ka3d_cavity_ebm3ax) as the generator. Overlap = collective density
overlap (frac of reference sites with a sampled particle within a=0.3), beta=2.0 (T*=0.5), boundary = shell
within R+2.5.

## Route 1 -- block-MTM as MCMC (ka3d_pts_ebm.py): FAILS to equilibrate
K=4 local moves have OK acceptance (~19%) but accepted moves only JIGGLE particles within their pockets
(<a=0.3): the retained interior + boundary pin them, so a K=4 regeneration returns ~the same positions. A
chain from the reference stays at overlap 1.00; it cannot cross basins. Not an acceptance problem (more
trials doesn't help) -- a STEP-SIZE / rearrangement problem. Needs collective moves (K>=6 fail) or
impractically many sweeps (95s/sweep observed).

## Route 2 -- one-shot importance sampling (full-regen proposal): FAILS to converge
q_full proposal (regenerate whole interior given boundary) is exact-logq, so w = e^{-bU-logq} is a valid IS
weight. But ESS% DEGRADES ~1/M (2.1%->1.0%->0.5% as M 48->96->192): eff samples stuck at ~1, estimate
drifts (0.52->0.58->0.63), never converges. Var(log w) ∝ n (per-particle mismatch compounds over the whole
interior) => heavy-tailed weights. The known one-shot-ESS=1 wall.

## Route 3 -- SMC / AIS (ka3d_pts_smc.py): WORKS
Geometric path pi_t ~ q^{1-lam}(e^{-bU})^{lam}, lam 0->1. Incremental weight = dlam*(-bU-logq_full);
mutation = block move with MH log-accept SCALED by lam (= exact block-MTM at lam=1, higher acceptance at
broad rungs where the interior CAN rearrange). All exact, reuses block_log_prob/sample_block/energy.
**PTS overlap q(R) (M=48, T=40, nmut=4, ncav=4) -- monotonic decay = the point-to-set signal:**
| R | q_PTS (SMC) | ESS | naive-IS (biased low, ESS 2.1%) |
|---|-------------|-----|---------------------------------|
| 1.6 | 0.731 | 77% | 0.507 |
| 2.0 | 0.562 | 83% | 0.432 |
| 2.4 | 0.494 | 91% | 0.325 |
q_PTS decays 0.73->0.56->0.49 toward the random floor (~0.12) as the cavity grows -- amorphous order
weakening with R. SMC ESS healthy at every R; naive-IS stuck at 2.1% and biased LOW (heavy tails miss the
typical set). ESS(lam) shows larger cavities need the anneal to work harder early (R=2.4 sits at 2% until
lam~0.4 then climbs to 59%) -> ADAPTIVE lambda schedule is the next efficiency lever. Overlap still >> floor
at R=2.4, so xi_PTS exceeds the box-limited range (R<=~2.5). Plot: pts_qR.png.

## Remaining gaps
1. COST: ~54 min/R at M48/T40/ncav4 -- M x T x (1+nmut) sequential full-context generator passes.
   FIX = batch the generator over the M-particle dimension (M x speedup); the biggest lever for a real q(R)
   curve over many R/cavities.
2. T-CONVERGENCE: q_smc still moves with T (T=8 -> 0.79, T=40 -> 0.73); needs a T-ladder / extrapolation
   or higher T to certify convergence.
3. MUTATION AT lam=1: at the target rung the mutation is the low-acceptance full block MH (the same jiggle
   limit as Route 1), so diversification there leans on resampling+broad-rung mixing. K>=6 or a swap/
   cluster move would strengthen the terminal rung.
4. EXACTNESS LEAK: sample_block inherits the base cat-head 8e-3 sample-vs-score bin-flip -> a small bias in
   the target; fix the ball-map round-trip in Cat3Head.
5. NO GOLD STANDARD: q_PTS is unvalidated vs an independent reference (long PT/MD cavity run). Need one to
   certify absolute accuracy (self-consistency across T/M/seed is necessary, not sufficient).
6. R RANGE: capped ~2.5 by box (2R+r_cut<=L=7.53); true large-xi extrapolation needs a bigger bulk.

## Verdict
The K=4 generator CANNOT compute PTS on its own (local MCMC won't rearrange; one-shot IS won't converge),
but as the base+mutation of an SMC it DOES: healthy ESS and a stable-ish q_PTS. SMC is the required wrapper,
exactly the campaign's ARM-1 conclusion. Files: ka3d_pts_ebm.py, focused_is_converge.py, ka3d_pts_smc.py.

## MATCHED to Berthier-Charbonneau-Yaida 2016 (JChemPhys 144 024501), T=0.51, their exact core G_PTS
ka3d_pts_batched.py (batched-over-M SMC, ~50-100x speedup, GPU saturated; exact vs unbatched 1.9e-5).
Observable = their core <q_c> (same-species Gaussian b=0.2, |r|<0.5 field-integrated). 8 cavities, M=48, T=48.
| R | my G_PTS(core) | paper est (A0.85,xi2.8,eta3) | q_whole | qc_pair | ESS |
|---|----------------|------------------------------|---------|---------|-----|
| 1.4 | 0.752 | ~0.81 | 0.946 | 0.862 | 87% |
| 1.7 | 0.196 | ~0.74 | 0.700 | 0.883 | 86% |
| 2.0 | 0.152 | ~0.65 | 0.631 | 0.988 | 71% |
| 2.3 | 0.186 | ~0.55 | 0.484 | 0.923 | 90% |
Plot: pts_paper_compare.png.

VERDICT: MATCH at R=1.4 (single-basin, 0.75 vs 0.81); FAIL at R>=1.7 (core crashes to ~bulk vs paper's
gentle decay). Diagnosis (from the 3 columns): qc_pair stays 0.86-0.99 (samples cluster with EACH OTHER =
degenerate, one basin) while G_PTS(vs reference) ~ bulk (samples FAR from reference) -> SMC converged to a
WRONG basin at the cavity CORE and can't reach the reference's. q_whole decays smoothly only because the
boundary-pinned shell holds; the CORE (furthest from pinning, where multiple states emerge) breaks. ROOT
CAUSE: K=4 local mutation can't cross glassy basins + the annealing path q->e^{-bU} interpolates toward the
MODEL PROPOSAL, not a MELTED high-T ensemble -> no basin-hopping. This is exactly the multi-basin regime the
paper needed temperature-PT (+ shrinkage, ~xi_PTS replicas) for. FIX = add TEMPERATURE annealing (melt-then-
cool) to the SMC path and/or basin-crossing moves (swap/cluster); local-move + constraint-anneal alone is
insufficient beyond the single-basin regime.

## Basin-crossing attempts (user: tackle multi-basin failure WITHOUT classic tempering)
Estimator note: the PAPER's G_PTS = overlap of TWO INDEPENDENT samples (= qc_pair here), NOT sample-vs-ref.
Under qc_pair the naive baseline is DEGENERATE (0.86-0.99 = under-mixed).
- lambda-scheduled LARGE random blobs: R=2.0 qc_pair 0.99->0.79 PARTIAL. Random-centered clusters BEAT a
  fixed geometric CORE (deterministic same-slot regen collapses all M samples -> qc_pair ~0.93).
- REJECTED temperature-melt: it IS the paper's classic PT (model becomes a mere proposal) -> no new science.
- MODEL-NATIVE MTM-JUMP (mtm_jump_b): batched N-trial large-block independence-MTM, generator teleports to
  best-of-N basin, exact log q filters. Equilibrates ~4 sweeps, qc_pair->0.12 -- LOOKED like it revealed
  under-mixing, but ENERGY CHECK REFUTES: MTM interior is +48/particle ABOVE reference at R=2.0 (+9 at R=1.4,
  grows with R). Large-block (K=12) regen is CLASHY (model can't generate a clean 12-particle rearrangement);
  MTM accepts among clashy proposals -> low qc_pair is a CLASH ARTIFACT, not equilibrium.
BINDING CONSTRAINT: local moves clean-but-can't-cross; large moves cross-but-clashy. Basin-crossing is
bottlenecked by the model's PROPOSAL QUALITY for large collective rearrangements (conditional gen-gap /
in-block exposure bias) -- a MODEL problem, not a sampler trick. Fix = cleaner large-block conditional
(scheduled sampling / in-block exposure-bias fix), NOT more enhanced sampling.
