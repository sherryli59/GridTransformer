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
