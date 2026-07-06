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

### RUNNING
- GA1 (utility vs MTM at β=2).
- in-SMC depth (primary criterion): local-SMC baseline vs +cSMC mutation, matched schedule.
- Stage-0 N=256 ladder.
