# KA block-mover transfer: train-small→sample-big — design

**Date:** 2026-07-03 · **Status:** approved (user: "go w your rec") · **Predecessor:**
`reports/2026-07-03-ipl44-lever-push-results.md` (all machinery built + gated there)

## 1. Goal & claim

Port the two proven IPL44 pieces — the **two-time joint flow** (geometry table 0.978) and the **exact
block-relabel mover** (3.4× sweeps, wall-clock winner) — to the **KA 2D glass**, and measure the
**train-small→sample-big** claim: a model trained at N=100 accelerates equilibration toward the
parallel-tempering reference at unseen N=256 (and N=36), with **sweeps-saved** as the metric.

KA is the right arena: strongly non-additive (σ_AB=0.8 < additive 0.94 → **swap-safe**, the IPL44 demixing
failure mode does not apply), swap-MC is the validated workhorse there, and converged PT references exist.

## 2. System & data (all verified on disk)

- Energy: `liquid_coupling_flow/ka_energy.py::ka_energy(x, s, L)` — 2D KA binary LJ, σ=[[1,.8],[.8,.88]],
  ε=[[1,1.5],[1.5,.5]], shifted, min-image. T*=0.5 (β=2), ρ=1.2 → L=√(N/1.2).
- References (`liquid_coupling_flow/artifacts/`): `ka_reference_N100.pt` x(14400,100,2) — **training set**;
  `ka_reference_N256_thin.pt` x(2560,256,2) and `ka_reference_N36_val.pt` x(5120,36,2) — **held-out transfer
  targets**. Each has ONE fixed species labeling `s [N]` (65:35) and its `L`.
- Observables: `ka_observables.partial_gr`; PT-converged ⟨U⟩/N ≈ −3.21 at N=256 (cold); the prior head-start
  experiment (`ka_localframe_smc.py`) used thresholds ⟨U⟩/N ∈ {−3.0, −3.1} — reuse them.

## 3. Components (ports, not new designs)

1. **`ka_block_smc.py`** (new, thin): KA adapter wiring `ka_energy` + `partial_gr` into the system-agnostic
   kernels of `ipl44/ipl_swap_smc.py` (pair `swap_attempt`, `block_relabel_attempt`, `position_sweep`).
   `run_chain` gets an observable-hook parameter (currently hardcodes `ipl_gr_partials`) — small generalizing
   edit, IPL tests must stay green. Position step size re-tuned for KA (target 30–60% acceptance at T*=0.5).
2. **`train_joint_flow.py`** gains a `--system ka` data path: loads the N=100 reference (positions + the fixed
   labeling broadcast per config), trains `JointSpeciesFlow(two_time=True)` at N=100, L=9.1287, nB=35.
   Same recipe as midtt: 64|4, batch 256, lr 1e-3 warmup+EMA, global-OT bank, 20k steps, best+last by
   val `acc_mismatch` (val = last 1000 configs).
3. **Transfer mechanics:** EGNN weights are N-agnostic; at eval, construct `JointSpeciesFlow` with the target
   (N, L) and load the N=100 weights. Same-density locality is the invariance being tested.

## 4. Experiments & gates

- **E0 (cheap, decisive first):** geometry-table quality across sizes — `denoiser_eval` (t_pos=1, t_spec=0)
  on N=100 val AND on the held-out N=36 / N=256 references with the N=100-trained model.
  **Gate T0:** acc_mismatch at unseen N within 10 points of train-N. (Kill → the conditional doesn't
  transfer; stop before any SMC.)
- **E1 (train-N benchmark, N=100):** seeds {uniform, flow-samples} × movers {random-pair, block4, block8},
  B=128, sweeps-to-⟨U⟩/N-threshold (−3.0 and −3.1) + g_BB sanity vs the PT reference.
  **Gate T1:** block-k beats random swaps ≥2× on sweeps at ≥1 seed type, with wall-clock also ahead.
- **E2 (the headline, N=256 unseen):** same benchmark with the N=100-trained model.
  **Gate T2 (the claim):** block-k with the transferred model saves ≥2× sweeps vs uniform+random at N=256;
  report also vs the historical local-frame head-start (~270–510 sweeps saved).
  N=36 down-transfer as a bonus row.
- **Swap-safety check (standing bug rule):** at KA the annealed and quenched ensembles should agree —
  verify chains from reference seeds do NOT drift below the PT ⟨U⟩/N under block moves (long run). Drift ⇒
  investigate before claiming anything.

## 5. Out of scope

PF (killed), IPL44 anything further, v1r pair ablation, SMC reweighting variants, N>256.

## 6. Code layout

```
liquid_coupling_flow/ka_block_smc.py      # NEW: KA adapter + E1/E2 benchmark driver
liquid_coupling_flow/ipl44/ipl_swap_smc.py  # small edit: observable hook in run_chain
liquid_coupling_flow/ipl44/train_joint_flow.py  # small edit: --system ka data path
liquid_coupling_flow/tests/test_swap_smc.py     # stays green; + KA smoke (energy wiring, count preservation at 65:35)
```

Checkpoints `jf_ka100tt_{best,last}.pt` → `liquid_coupling_flow/ipl44/data/` (keeps the jf_* convention);
figures/logs → `liquid_coupling_flow/artifacts/` (KA convention). Results → dated report.
