# Kernels that obey the two laws — results

Spec: docs/superpowers/specs/2026-07-05-kernels-two-laws-design.md. Branch liquid-coupling-flow.

## Stage 0 — PT-ladder dataset (N=100 DONE)

Instrumented `ka_reference.main_ladder` (saves all M rungs + convergence gates). N=100, M=10 rungs (β 2.0→0.8),
n_equil=16000, n_collect=4000, 2 seeds, ~8000 configs/rung.

- **Cold (β=2, T=0.5) ⟨U⟩/N = −3.264** (seed0 −3.2641±0.0009, seed1 −3.2646±0.0010, **seed-diff 0.0005**).
  Exchange acc 0.49 (healthy). **TRUSTWORTHY N=100 reference** — the first dividend, and it CORRECTS the old
  `ka_reference_N100.pt` which was ~−3.231 (≈0.033/particle too shallow — under-equilibrated, same disease as
  [[ka-reference-underconverged]] but now quantified for N=100). Independently corroborated by Stage-A1 cSMC
  (below), which pulls a stale slice straight to −3.264.
- Gate calibration fix: the "flat" gate originally took max drift over ALL rungs; hot rungs (T=1.25) fluctuate
  far more, mislabeling the converged cold reference as non-flat (cold-drift ~0 but hot-max 0.058). Fixed to
  gate on the COLD rung (index 0). Verdict now keys on the reference target, not the ergodic hot bath.
- Dataset saved: `artifacts/pt_ladder_N100.pt` (80,000 configs over 10 rungs) — feeds Stage A2 (β-conditioned
  heat-bath training data) and Stage B (event mining). N=256 ladder running.

## Stage A1 — cSMC cluster move around ARM-FULL (NO retraining)

Module `ka_cluster_csmc.py`; byte-identical `_step` refactor of ClusterProposal (golden test); 7/7 TDD
(telescope binds the incremental energy to canonical `ka_energy`; M=1 identity; invariances; PGAS branch).

### GATE: stationarity-from-reference — **PASS (exact by construction, empirically confirmed)**
`csmc_stationarity_core.out`, exact core (M=16, resample off), 60 sweeps from the old reference slice:
- **band 0.0113, net-drift −0.0204 → STATIONARY.** g_BB peak flat 2.35–2.52 throughout (NO upward smear —
  the pure-Gibbs-collapse signature is absent). Plot: `artifacts/csmc_stationarity_N100.png`.
- The small net-drift is DEEPER, toward the Stage-0 truth −3.264 (start slice was the stale −3.231): a correct
  π-sampler fixing a stale reference, not a non-invariance. This is the strongest exactness signal in the
  campaign — an independent method (PT) and the learned kernel agree on −3.264.

### Acceptance — the single-shot wall broken
cSMC accept **3.5% (M=16)** vs ARM-FULL single-shot MH **0.3%** — ~10× from the ensemble + incremental
weighting, same frozen checkpoint. Converts "diffuse-but-calibrated" into accepted moves (LAW 1: filter, never
force; the reference always anchors selection ⇒ cannot poison like P2's forced IS-transport).

### GATE: in-SMC depth (primary criterion) — **cSMC is NOT a depth lever (clean negative, well-attributed)**
- +cSMC (n_csmc=1): −3.136 / 7899s. Baseline (n_disp=40): −3.101 / 1812s. cSMC deeper per-rung BUT 4.36× wall.
- Crucially, −3.136 = the SAME −3.137 floor P1 hit with plain displacement+SB. cSMC does NOT push past it.
- **Matched-wall control (the decider):** displacement is CHEAP — scaling n_disp 40→176 (4.4×) barely moved
  wall (1725s vs 1812s; the +cSMC 4.36× cost was ALL the 60s cSMC sweeps). That cheap 4.4×-displacement
  baseline reaches **−3.129 in 1725s** — within 0.007 of +cSMC's −3.136 at **4.6× less wall.** So cheap
  displacement captures ~all the accessible depth; cSMC's exactness + 10× acceptance buy 0.007/particle for
  4.6× the time. **The −3.13 → −3.264 gap is the glassy-tail barrier (PT-class), NOT mutation-acceptance
  limited** — a better local cluster move cannot close it. Redirects the depth lever to A2 (full-cage
  single-site, sees more of the cage) / Stage B (collective moves).

### A1 net verdict
EXACT by construction (stationarity PASS, PT cross-validated) + breaks the single-shot acceptance wall 10×
(0.3→3.5%) — the scoped deliverables. But NOT a depth lever: dominated by cheap displacement in the SMC stack.
Value: a validated exact high-acceptance cluster kernel (useful wherever cluster-scale exact moves are wanted);
the depth question is answered negative and points to A2.

## Stage 0 — N=256: DONE but UNDER-CONVERGED
Cold −3.2099 (seed0) / −3.2090 (seed1), seed-diff 0.0008, exch 0.28. BUT vs N=100 −3.264 the intensive energy
is 0.055/particle too SHALLOW — finite-size direction WRONG ⇒ still under-equilibrated at 16k sweeps (bigger
system, slower); the tight seed agreement is false confidence (both under-equilibrate identically). `finite_size_check`
(N=100 vs N=256) is the honest flag. N=256 sharp transfer gaps stay directional until a much longer run.
Configs (`pt_ladder_N256.pt`, 80k) still usable as A2 N=256 conditioning data. Consistent with [[ka-reference-underconverged]].

### RUNNING
- GA1 (utility vs MTM at β=2) — rerun after two generator/arity fixes.

## Stage A2 — beta-conditioned FULL-CAGE single-site heat-bath — THE DEPTH LEVER

Kernel `ka_heatbath.HeatBathModel` = `ClusterProposal` at k=1 + beta-FiLM. Single site => sees the FULL frozen
cage (non-causal, the sharp regime). Frozen-cage exactness INHERITED from `frame_ctx_slots([site])` config-
independence (same property that makes the cluster move exact). 5/5 TDD incl the frozen-cage DB invariant.
Trained beta-conditioned MLE on all Stage-0 rungs -> SHARP conditional (35% acceptance, NEGATIVE -logq = peaked,
the full-cage sharpness that [[full-cage-lever-needs-energy]] predicted).

### Exactness — subtle, resolved by 5 concrete tests (never-refute discipline)
Standalone stationarity DRIFTED BELOW eq (-3.30). Investigated rather than trusted/dismissed:
- single-site move vs exact grid Boltzmann: -0.015 (~grid noise)
- single-site move vs exact Gaussian-MH, frozen cage: **-0.003 => move is EXACT**
- exact displacement seeded from heat-bath -3.30 config: RELAXES UP to -3.276 (=> -3.30 over-cooled)
- **heat-bath MIXED with displacement: converges to -3.276** (the true eq; a biased move would settle between)
- ka_pair_row == ka_energy to 3e-5 (energy convention clean)
VERDICT: the move is pi-invariant (DB-exact), but the beta-conditioned proposal is SHARPER than pi(x_i|cage)
(neg -logq), so as an INDEPENDENCE proposal it rarely proposes uphill -> STANDALONE it over-cools (quasi-non-
ergodic). MIX with an ergodic kernel (displacement) => correct AND fast. Never standalone; the SMC stack always
co-mutates (documented in single_site_mh_sweep). Bonus: pins the TRUE N=100 beta=2 eq = **-3.276** (exact disp
converges from both -3.264 above and -3.30 below); PT ref -3.264 mildly under-converged.

### GATE: in-SMC depth (primary criterion) — **POSITIVE. The depth lever A1 wasn't.**
- baseline (disp+SB): -3.1007 / 1197s
- **+heat-bath (disp+SB+heat-bath): -3.1690 / 1324s**
- **delta -0.068 DEEPER at 1.11x wall** (near matched).
- vs A1 cSMC: -0.007 deeper at 4.6x wall (neutral). A2 is 10x the depth gain at 1/4 the cost premium.
- Closes ~39% of the remaining gap ((-3.169+3.101)/(-3.276+3.101)); single-site heat-bath is CHEAP (10x < cSMC)
  because no M-ensemble, so 1 sweep/mutation-round barely adds wall. STILL 0.107 above true -3.276: the deep
  glassy tail remains (Stage-B collective moves / more heat-bath budget = the next knob), but this is the FIRST
  real, cost-effective depth improvement in the campaign.

MECHANISM (why A2 works where A1 failed): the barrier is local-relaxation-limited, and the full cage (all
neighbours at once) is what a single-site conditional needs to place precisely; the AR cluster move saw only a
half-built cage. A1's clean negative localized the barrier; A2 supplies the missing cage.

### GA2 supporting gates
- Acceptance 33% @ beta=2 (2.7s/sweep) — 110x the single-shot cluster proposal (0.3%), 10x cSMC (3.5%).
- **Amortization: accept 0.318-0.366 FLAT across all 10 beta-rungs (2.0->0.8), range 0.048 << 0.15** — one
  beta-conditioned model serves ALL temperatures. The anti-init-dependence property swap-breathe LACKED
  ([[swap-breathe-kernel]] dropped 4-5x off-equilibrium). A2 is temperature-robust.
- (accept gate's displacement side-metric was invalid — per-row species not canonicalized before _disp_sweeps,
  melted to -1.46; fixed in ka_heatbath_gate.accept. Heat-bath number unaffected.)

## Campaign verdict (kernels-two-laws)
A1 (cluster cSMC): EXACT + 10x acceptance, but NOT a depth lever (glassy barrier is not cluster-acceptance-
limited) — a clean negative that LOCALIZED the barrier to "half-cage". A2 (full-cage single-site heat-bath):
supplies the full cage, and is the FIRST learned kernel to give a real cost-effective depth gain in SMC
(-0.068 @ 1.11x wall, ~39% of the remaining gap), exact (mixed), and amortized across temperatures. The learned
generator finally earns its keep. Open: the deep glassy tail (0.107 remaining) -> Stage B collective moves /
heat-bath budget; N=256 transfer -> needs the longer PT reference.

### n_heatbath BUDGET SWEEP — plateau => the residual gap is COLLECTIVE (Stage-B trigger FIRES)
Same seed/schedule, n_heatbath in {0,1,2,4,8} (heat-bath sweeps per mutation round):
| n | final U/N | gap to -3.276 | wall |
|---|---|---|---|
| 0 | -3.1007 | +0.175 | 1204s |
| 1 | -3.1690 | +0.107 | 1328s |
| 2 | -3.1298 | +0.146 | 1406s |
| 4 | -3.1442 | +0.132 | 1654s |
| 8 | -3.1143 | +0.162 | 2397s |
- All four n>=1 points are deeper than n=0 (sign 4/4) => the heat-bath gain is REAL, but the n>=1 points
  SCATTER -3.11..-3.17 with NO trend in n — a plateau at ~-3.14 +/- (single-seed noise ~+/-0.02-0.03, RNG
  stream shifts with n + adaptive-ladder interaction). CAVEAT: the headline -0.068 (n=1) is the best draw;
  the expected gain is ~-0.04; a multi-seed rerun would tighten it (cheap follow-up).
- n=8 (2x wall) is no deeper than n=1 => depth is NOT budget-limited. More single-site relaxation cannot
  close the remaining ~0.13: the tail is COLLECTIVE (multi-particle rearrangements a single-site move can't
  express, however sharp). Consistent with the spec's pre-committed trigger: "acceptance high but depth
  plateaus => the missing class is collective." **STAGE B (learned collective move, event-mined from the
  Stage-0 ladder) is the required next stage; its dataset (pt_ladder_N100.pt) is in hand.**
Plot: reports/logs-2026-07-05/heatbath_insmc_sweep.png (artifacts/heatbath_insmc_sweep.{png,pt}).

## Stage B — GB0 event mining (2026-07-06)

Miner `ka_events.py` (7/7 TDD: exchange rejection, min-image, locality; harvester displacement-only).
**PT-ladder mining is structurally unsuitable, twice over:**
1. PT exchanges contaminate 40-65% of adjacent snapshot pairs (measured; edge rungs less — one exchange
   partner instead of two; N=100 worse than N=256 because exch acc 0.49 vs 0.28).
2. DECISIVE: ladder windows (8 sweeps) are SHORTER than a cage-break's duration (tens of sweeps) — events
   fragment below the d_event threshold. Canonical N=100 cold rungs: **0** usable events; N=256: 5.
   The dedicated-harvest rate at matched beta is ~3.5x the ladder-implied rate for exactly this reason.

**Fallback harvester = strictly better data, nearly free** (displacement-only `_mc_sweep`, no swaps => no
identity-teleports, no PT => no contamination; per-chain betas {2.0, 1.81, 1.63}; 41k sweeps / 71 s):
- WINDOW CALIBRATION (measured): dt=400 over-merges (k median 6-8, max 23, mostly multi-cluster); dt=8
  fragments (see above); **dt=100 is the sweet spot** — k median 1-2, max 5-10, ~70% single-cluster localized.
- **MEASURED MOVE CLASS (the GB1 design input): cold collective events are 1-10 particle, single-cluster,
  compact (extent <~ 2 sigma) string-like hops — k_block ~ 8-12 covers the class. NOT avalanches.**
- Rates: beta=2.0: 63 events/42k pairs(dt=100); 1.81: 133; 1.63: 194. Gate needs >=200 localized at
  beta>=1.81 -> 142 at 100k sweeps -> 300k-sweep run in flight (~10 min; the harvest scales trivially).
Bank: artifacts/ka_event_bank_N100.pt (full (xa,xb) config pairs per event, [[record-simulation-data]]).

### GB0 CLOSE-OUT: **GO** (300k-sweep bank, 588s)
beta=2.00: 195 events / 147 localized; 1.81: 320/199; 1.63: 532/306 => **346 localized at beta>=1.81 (>=200)**,
652 total. Morphology STABLE across all runs: k median 1-2, max 6-8, single-cluster, compact. Bank:
artifacts/ka_event_bank_N100.pt (full config pairs). GB1 design inputs fixed: k_block=8 (covers max), betas
{2.0, 1.81, 1.63} via FiLM, two-way conditional q(x'_block|x_block, env) with A1-cSMC composition as the
acceptance fallback.

## GB1 gate 1 + the SELECTION-SYMMETRY WALL (2026-07-07)

Training: best group-held-out val 0.871 (ckpt saved at the val optimum; late-run overfit gap confirmed —
train -0.5 vs val 0.92 by step 15k — user called it; 1M-sweep harvest (4x bank) banked for a v2).

**Gate 1 (one-shot MH acceptance): 0.0000 — but the diagnosis is NOT proposal quality: abort 92-94%.**
The occupancy-symmetry check (required for DB under slot-anchored block selection) kills moves before MH
evaluates. Attribution chain (3 measurements):
1. `_curve_order` is a GLOBAL argsort of curve codes => any cell-crossing mover changes its own rank and
   shifts every rank between old/new codes. Real collective moves inherently cross cells => slot-set
   preservation ~never holds. The 93% abort is bookkeeping-structural, not physics.
2. Particle-kNN selection (fold q_sel into Hastings): only **2%** of real bank events are reverse-selectable
   — a cage-break BY DEFINITION rearranges the neighborhood any kNN rule keys on.
3. Region-anchored selection (fixed ball, occupants-as-block, symmetric abort): best case R=1.6sigma gives
   **30%** of real events covered+occupancy-stable (block ~9-11); larger R strictly worse.

**THE FINDING (new, campaign-level): the collective move class DESTROYS the neighborhood structure that any
local block-selection rule keys on.** Exact same-particle-set block kernels can express at most ~30% of real
events (region-anchored, R=1.6). The pre-committed cSMC-composition fallback does NOT apply (it fixes hit
rate; acceptance never evaluates here) — overridden by the measured diagnosis.

Options forward: (a) v2 region-anchored kernel trained on the stable ~30% subset (1M bank => ~650 such
events; variable-k blocks ~9-11; one more build-train-gate cycle); (b) accept the structural finding and
pivot (learned-augmented PT / transfer-scale crossover); (c) variable-membership block machinery (grand-
canonical-flavored) — principled but a research project of its own.

## ORACLE TEST — the collective channel is priced at ~ZERO; STAGE B CLOSES (2026-07-07)

3 arms from the same A2-stack stuck population (-3.159), matched schedule (ka_collective_oracle.py):
real-event transplant vs scrambled control vs displacement-only. Library = 2280 localized events (1M bank).
- **accept 0.0000 over ~66,000 FIRED transplant row-attempts** (fire rate 13%): a real rearrangement pattern
  transplanted onto a 0.35-matched lookalike structure is ALWAYS energetically catastrophic. The collective
  move is a dance tuned to its exact micro-cage — the precision wall's collective face.
- real (-3.1705) == scrambled (-3.1727) == disp (-3.1689): the true rearrangement structure carries NO
  transferable advantage over random kicks or nothing. No arm approaches -3.2.
- VERDICT: pattern-replay collective proposals are worthless; combined with the selection-symmetry wall
  (kNN 2% / region 30%) and A1's one-shot precedent, **v2 is NOT built. Stage B closes**: the residual
  ~0.10 to -3.276 is KINETICS (alpha-relaxation), not a missing expressible move class. (Caveat kept honest:
  the oracle upper-bounds replay, not a fully env-adaptive kernel — but every adaptive precedent (A1 0.3%)
  and the 0/66k measurement point the same way.)

## PT+A2 — PT0 PASS (A2 transfers to N=256 zero-shot) + PT1 promising
- **PT0: DB max|dlogq| 0.00e+00 at N=256 (exact); acceptance 28.6% zero-shot (vs 33% at N=100); mixed
  stationarity band 0.011.** The N=100-trained heat-bath is valid inside an N=256 reference generator.
- PT1 (N=100, 18k+2k sweeps each): plain final -3.2615 @782s; **PT+A2 final -3.2702 @1667s (2.16x wall)** —
  the deepest PT result yet, past the -3.264 plateau toward the true -3.276. Strict matched-wall verdict
  needs plain PT at the same 1667s (38k sweeps) — control queued; interpolated single-snapshot comparisons
  are +/-0.02 noisy (8-replica cold mean).

### PT1 matched-wall control — VERDICT: PT+A2 NEGATIVE at matched wall at N=100 (+ a goalpost discovery)
Plain PT at the SAME wall (38k sweeps / 1662s): **-3.2840** vs PT+A2 -3.2702 (18k+hb / 1667s). At N=100 the
heat-bath's 2.16x per-sweep overhead costs more than its mixing buys inside PT — plain sweeps are simply too
cheap (43 ms). PT+A2 @N=100: NEGATIVE at matched wall.
- **GOALPOST: plain-PT-38k reaches -3.284, DEEPER than the "-3.276 true eq" pinned by exact displacement.**
  Both displacement plateaus (-3.264 from above, -3.30 relaxing to -3.276) were KERNEL-RELATIVE; PT digs past
  them. The true N=100 eq is <= -3.284 and every "floor" so far was a protocol plateau — reference numbers
  must always carry their protocol. (Classic glass: the target descends as samplers improve.)
- N=256 OUTLOOK unchanged-to-favorable: hb cost is ~constant (model-forward-bound, local) while plain sweeps
  scale O(N^2) and equilibrate slower — overhead ratio drops from 2.16x (N=100) to ~1.3x (N=256), so the
  augmentation crossover may itself sit between N=100 and 256. PT2 (running, WITH hb) judges via its internal
  gates; its finite-size check must now use N=100 <= -3.284.

### PT2 seed-0 (N=256, 32k equil + hb): -3.2311 — deeper (+0.021 vs plain-16k) but STILL under-converged
Cold -3.2311+/-0.0006, cold-drift 0.0005 (FLAT — the plateau blind spot; only the cross-N check catches it),
vs N=100 goalpost <= -3.284: finite-size direction still wrong (0.053 shallow). Credit for the +0.021
confounded (hb AND 2x equil changed). **NEW DIAGNOSIS: exchange acc 0.27 at N=256 vs 0.49 at N=100 on the
same M=10 ladder — sigma_U ~ sqrt(N) shrinks rung overlap; N=256 wants M~16 rungs. Ladder RESOLUTION, not
just sweep count: redesign before buying more sweeps.** Seed 1 running (~6h; artifact saves only after both).

### PT2 COMPLETE (N=256, 32k equil, +hb, 12h): internal gates "CONVERGED" — read as PER-PROTOCOL only
seed0 -3.2311+/-0.0006, seed1 -3.2267+/-0.0006 (|d| 0.0044, 9x looser than N=100's 0.0005), cold-drift <=0.004,
exch 0.27. Dataset: pt_ladder_hb_N256.pt (80k configs / 10 rungs) + .bak. ROLE (downgraded per user challenge):
a deeper N=256 ladder DATASET + one confounded data point (hb + 2x equil changed together) — NOT the reference
deliverable. The -3.231-vs-N=100(-3.284) gap remains unresolved between protocol shortfall (evidence: number
moved 0.021 with budget; exch 0.27 => M=16 ladder needed) and genuine 2D finite-size physics (user hypothesis;
composition rounding + single species realization uncontrolled). DECISIVE NEXT TEST (pending approval):
converged-by-construction U/N(N) at N=64/100/144 x 2-3 species realizations -> slope vs 1/N decides, and
calibrates every cross-N gap metric in the campaign (incl transfer experiments).

### TELEPORT PRICING — CLOSES at the gate (the variable-membership direction ends here)
Teleport-swap (long-range occupant exchange via A2 frozen-cage conditionals, exact, 4/4 TDD): acceptance
**2e-5 at equilibrium, 0.0 at mid-anneal (b=1.2), 0.0 on the SMC-stuck population** (5 sweeps x 128 chains
each; abort 0.23-0.32). Even a learned smart-insertion never finds a pocket — the glass has no interstitial
room even off-equilibrium (the "stuck states have density inhomogeneities" bet: WRONG, measured). The
grand-canonical/variable-membership block direction closes at its cheapest gate, as designed. Kernel + tests
remain in the repo (correct, reusable if a future system has real pockets — e.g., lower density or mixtures
with size disparity).

### ALCHEMICAL NCMC T-SCAN — mechanism VALIDATED, economics at PARITY with SB v2 (not a win at first settings)
| T | accept | accepted-exchanges/s (B=128) | s per exchange-chain |
|---|--------|------------------------------|----------------------|
| 1 | 0.00000 | 0.00 | dead (sanity anchor = plain-swap-dead REPRODUCED) |
| 4 | 0.00000 | 0.00 | dead |
| 16 | 0.00102 | 1.48 | ~86 |
| 64 | 0.00336 | 1.23 | ~104 |
- **The pathwise hypothesis is confirmed**: the dead species channel turns ON monotonically with path length —
  the third independent "paths beat endpoint jumps" datapoint (PT temperature paths; oracle endpoint-transplant
  zero; now alchemical lambda-paths). A PURE-PHYSICS kernel (no learning) reaches within 1.5x of the learned
  SB v2's economics (~86 vs ~70 s/exchange-chain).
- Verdict vs the gate bar (beat SB v2): NOT passed at first settings => flow-drift upgrade stays LOCKED per the
  pre-agreement. Headroom unexplored: n_relax/T tuning, adaptive protocols, geometry-targeted pair selection.
  Second independent equilibrium species kernel banked (diversity value even at parity).
