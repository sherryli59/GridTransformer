# Hand-off: KA3D cavity / point-to-set (PTS) sampling

**Date:** 2026-07-13 · **Branch:** `liquid-coupling-flow` · **Repo:** `/mnt/ssd/GridTransformer`
**Audience:** a fresh agent (Codex) picking up the KA3D cavity/PTS line. Assumes no session memory.
Convention notes: results/scripts live in `reports/logs-<date>/`; commit the moment a run finishes; save
full data (`.pt`), never just summary scalars; launch long jobs with the harness background runner + a
crash/stall watcher; check where a >30-min run saves BEFORE launching.

---

## 0. One-paragraph orientation

Goal: measure the point-to-set (PTS) overlap of a 2D/3D Kob-Andersen glass former inside a frozen-boundary
cavity, and (the research bet) do it with an **amortized learned sampler** that transfers across cavity
radius R and system size, instead of re-running expensive parallel tempering per cavity. The learned object
is `KA3DScaffoldEBMBatched` (`liquid_coupling_flow/ka3d_ebm_batched.py`) — an autoregressive (AR) generator
that regenerates a K-particle block (or the whole interior) given the frozen cage, with an **exact
one-forward log-density** (`sample_block_b` / `block_log_prob_b`, tempered by `pos_temp`, both round-trip
to ~1e-13 in float64). State point: N=4096, T*=0.5, ρ≈1.1486, β=2.0, from Bapst/DeepMind-derived configs
(`liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt`, 16 test-split chains; training split
`ka3d_train_N4096_T0.5_rho1.15.pt`, 112 chains — chain-level, zero leakage).

---

## 1. The load-bearing conclusion of this session

**Cold single-shot acceptance of the AR block move is structurally dead, and the wall is not fixable by any
representation change, mask, or post-hoc corrector — it is the causal half-cage limit.** Evidence, all
committed and reproducible:

- **Declash ceiling** (`reports/logs-2026-07-13/diag_declash_ceiling.*`, commit `e559b23`): even a
  *clash-free* coarse-head block sits **+1.4/particle above data = −22 nats/block** → MTM acceptance
  e^−22 ≈ 0. Declashing is necessary-not-sufficient.
- The **+1/particle floor recurs in 4 independent measurements**: clash-free coarse +1.40, frozen-cage
  relaxed AR +1.12 (`diag_relax_ar_targets`), deterministic-corrector real-basin +0.9, and the flow's
  concentration gap. It is the half-cage/exposure limit: a causal AR places each particle blind to ~half
  its final neighbourhood.
- **Masking is settled (see §4): the nested mask DOES fix clash with a complete cage, but clash was never
  the acceptance bottleneck.** The nested within-cell mask (EXACT 1.4e-14) looked ineffective (30.4→27.8%)
  only because it was **cage-starved** (mask cage = knn_pot=8, but a dense glass has 12–14 first-shell
  neighbours); with a 24–32 cage it drops clash to **10.7%** (near the future-blind floor). But directly
  measured (`diag_mtm_nested`, commit `c14b629`), adding the nested mask moves MTM acceptance by **nothing**
  (K=4 25→25%, K=8 0→0%) — because once the energy is low the residual is *structural*, not clash. Clean
  separation: **the tilt CAGE is the acceptance lever; the nested mask is an SMC accelerant only.**
- **The tilt cage (knn_pot) is the real lever, and knn=8 is too small** (`diag_mtm_inference_knn`,
  `diag_tilt_vs_mask_cage`, commits `0bdf918`,`ed01f2b`). Bumping the tilt cage 8→24 AT INFERENCE (no retrain)
  cuts K=8 trial energy **30× (+896→+28/particle)** and lifts K=4 acceptance 16.7→25% — but is *fragile*
  (knn=32 regresses; the trained-at-8 tilt goes off-distribution). So a **learned** knn=24 tilt is the
  necessary step, and its payoff should land in the K=8 regime. K=8 stays 0% even at +3.9/particle (62 nats)
  → the retrain must close the structural gap further. **THIS IS THE TOP OPEN LEVER — see §2 (running) + §5.**
- The **flow corrector** (EGNN, `ka3d_cavity_egnn.py`) is the best proposal-quality object (clash 20→10%,
  held FM 0.06) but accepts **0/96** at K=8 — NOT a likelihood bug (`diag_flow_likelihood`, commit
  `2ed463b`): it is the **concentration wall** — logZ-calibration corr(logq,−βU)=+0.11 (`c4696c7`), the
  reverse map sends 44% of data states out of the ball. A *better-trained* (more contractive) flow makes it
  *worse* as a cold proposal.

**Two live directions** (not mutually exclusive): (a) **SMC / tempering** — anneal through the −22 nat gap
in increments, never reject it in one shot; the AR is the annealed base, the nested mask + relaxation
declash the population, the corrector's pull is an asset (§2 head-to-head). (b) **The bigger-knn tilt
retrain** — the one lever with a credible path to the structural gap ITSELF (the cage, not the mask, is the
acceptance lever, and knn=8 is measurably too small) (§2 job 3). These converge: a better tilt makes a
better SMC base too.

---

## 2. What is RUNNING right now (as of hand-off)

Three GPU jobs (check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`; verify PIDs are alive
with `ps -p <pid>` — GPU has ample free memory, all three coexist). All write incrementally.
Job 3 (the tilt retrain) is the TOP lever; jobs 1+2 are the PT-vs-AR head-to-head.

1. **PT referee (gold standard)** — `reports/logs-2026-07-13/pt_gate_g1_24rep.out`, artifact
   `liquid_coupling_flow/artifacts/ka3d_pt_g1_24rep.pt`. Command:
   `python reports/logs-2026-07-10/pt_gate_g1.py --radii 2.2 2.4 --n-rep 24 --n-sweep 3000 --n-centers 2 --skip-high-t --artifact liquid_coupling_flow/artifacts/ka3d_pt_g1_24rep.pt`
   This is the **shrinkage-PT** method (Berthier–Charbonneau–Yaida): replicas along a coupled (T,λ) path
   that shrinks particle diameters (`ka3d_pt_cavity.py::build_ladder`, endpoint (T_hot=1.0, λ_min=0.8)).
   WHY 24 replicas: the earlier 12-replica run (`reports/logs-2026-07-10/pt_gate_g1_calibrate_12rep_2k.*`)
   FAILED — but the saved per-pair exchange rates diagnose it cleanly as **under-resolved**, not broken:
   healthy 0.11–0.43 exchange at R=1.6 (N_mobile≈19) vs collapsed 0.00–0.12 at R=2.2/2.4 (N_mobile 54–74).
   Textbook PT scaling (spacing ~1/√N_mobile). 24 replicas should restore two-arm convergence. Its
   `ref_samples` (saved per center) are the trustworthy multi-basin ground truth we have never had.

2. **AR island-SMC challenger** — `reports/logs-2026-07-13/island_smc_production.out`, artifact
   `island_smc_production.pt`, script `run_island_smc.py`. Command:
   `python reports/logs-2026-07-13/run_island_smc.py --T 48 --m 16 --islands 8 --radii 2.2 2.4 --centers 2 --out reports/logs-2026-07-13/island_smc_production.pt`
   Same deterministic cavities as the G1 run (copied its `frame=-(c+1)`, `particle_id=(37c+11)%N`
   construction verbatim), so P(q) compares directly.

**The head-to-head** answers the user's guiding question: *"ideally with AR assist no shrinkage-PT is
needed."* Island SMC = J independent annealed-SMC populations from the AR base, geometric path
q_AR^(1−λ)·e^(−βλU), λ-scheduled block-MTM mutations, per-island exact evidence logZ_j accumulated, islands
recombined ∝ exp(logZ_j) — the one-pass substitute for PT's round trips. Decision rule when both land:
island-reweighted P(q) matches G1 `ref_samples` at ≤ cost ⇒ PT retires to once-per-regime validation and the
AR pipeline stands alone; mismatch ⇒ the miss is diagnostic (missing basins = base coverage; wrong weights =
evidence variance) and the AR falls back to accelerator-inside-PT.

**NEXT ACTION when both finish:** write the P(q) comparison (island-reweighted vs G1 ref_samples histograms,
same cavities) and report. `run_island_smc.py` has a `--compare <g1.pt>` hook stub — wire it or write a small
compare script. CAUTION: the 24-rep G1 is still `conv=False` at R=2.2/2.4 (gap 0.29–0.49) mid-run — it may
need MORE replicas or sweeps; check its final verdict before trusting `ref_samples` as gold.

3. **Bigger-knn TILT RETRAIN (the top lever)** — `reports/logs-2026-07-13/train_knn24.out`, artifact
   `liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_knn24.pt` (+ `_best`). Command:
   `python reports/logs-2026-07-12/train_ebm3d_bigbox.py --knn-pot 24 --steps 40000 --eval-every 500 --patience 8 --warm liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt --out liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_knn24.pt`
   knn_pot is now a ctor kwarg (default 8); the tilt nets are per-pair+sum so warm-start from the knn=8 ckpt
   is clean (0 new/0 unused). WHY: the cage — not the mask — is the acceptance lever, and knn=8 is
   measurably too small (§1). STATUS at hand-off update: ~step 5000/40k, held NLL −2.84 (NOT yet beating the
   knn=8 baseline −2.88), MTM(K=4) noisy 6–25%. Warm-started tilt must UNLEARN its 8-nbr weighting → expect a
   dip before improvement; judge on CONVERGED held NLL + a proper K-ladder vs knn=8 (esp. K=8, where the
   inference sweep says the payoff should land — K=8 was 0% at +3.9/particle, the structural gap to close).
   If it doesn't beat baseline by ~step 15k, try knn=16 (the inference sweep peaked there) or a fresh
   (non-warm) train. **Evaluation harnesses ready:** `diag_mtm_nested.py` (set the ckpt) for the K-ladder,
   `diag_declash_ceiling.py` for the L-BFGS structural floor.

---

## 3. Key design facts the island-SMC arm taught us (don't re-derive)

The AR base needs THREE exactness-preserving fixes to give sane annealed evidence (all in `run_island_smc.py`):
- **Quartic warmup λ = linspace(0,1,T+1)^4**: a linear start multiplies clash-blown AR full-regen energies
  (U~1e7 at r→0) by dλ into the first increment → logZ ~ −5.5e6. Quartic makes early increments ~1e-6·U.
- **K=1 single-site AR heat-bath pass per rung** (same exact tempered-MH ratio): declashes what block moves
  can't; drove overlap q 0.30→0.61 (into the G1 ref band at R=2.2) and island logZ spread 20k→1.1k nats.
  This is the deployment-side declash lever that actually works (contrast the masks, §4).
- **`@torch.no_grad` on the island loop**: `block_log_prob_b`/`energy_b` are not internally no_grad; without
  it the logw accumulation retains the autograd graph across all rungs → 22.5 GB OOM.

Interpretation: the "no-PT" arm works only insofar as it internally rebuilds the easy→hard structure the
shrinkage ladder provides externally (warmup + declash sweeps = a home-grown ladder).

---

## 4. Dead ends — do NOT re-try these (all measured, committed)

| lever | result | commit |
|---|---|---|
| coarse-joint-then-fine head (16³×8³) | gates NEGATIVE: clash 25.3% > baseline 20.7%, NLL loses to baseline | `4e497a6` |
| per-axis c-axis min_sep mask | exact but inert (a,b-plane clashes) | `49de79b` |
| conservative coarse-cell mask | inert (coarse-leak: straddling cells) | `27575b6` |
| fine-c mask | 23.4→22.9%, a,b-plane wall (3rd confirm) | `5a9f2a0` |
| ~~nested within-cell mask~~ | NOT a dead end — cage-starved; works at knn≥24 (10.7% clash). See CORRECTION | `86544e3`,`ed01f2b` |
| top-p nucleus truncation | 16.7→14.3%, clash is in the CONFIDENT BULK not the tail; fp-fragile on GPU | `f36b872` |
| enlarge mobile set (absorb retained) | clash RISES 20→31% (frozen-correct context was helping) | `d0e660a` |
| flow corrector as cold-MTM proposal | 0/96, concentration wall (not a bug) | `2ed463b`,`c4696c7` |
| oracle heat-bath basin scan (12 sweeps) | under-equilibrated; guard flagged it; needs ~150 sweeps | `9b3203a`,`05457b1` |

**CORRECTION (commit `ed01f2b`, after the residual decomposition + knn isolation):** the nested-mask
"ineffective" verdict was WRONG — it was **cage-starved**, not fundamentally broken. The mask cage was
`knn_pot=8`-nearest, but a dense glass has 12–14 first-shell neighbours, so clash partners beyond the 8th
were invisible to the mask (residual decomposition: retained 40% + boundary 22% dominate, only 22% is
future-blind). With a complete mask cage the clash drops:
  tilt=8 mask=8 → 27.8% · tilt=8 **mask=32** → 17.0% · **tilt=32** mask=32 → **10.7%** (near the future floor).
Two separable deficits: (a) the MASK cage (fixed for free, inference-time separate cage; `mask_knn` param
now on `KA3DScaffoldEBMCoarse.sample_block_b`; scorer wiring for SMC-exactness = TODO), and (b) the TILT
cage — the learned energy embedding at knn=8 optimizes placement against ~8 of 14 neighbours → the
confident-but-wrong placements that ARE the +1/particle structural-gap signature. So masking CAN get min
separation right with a complete cage; and a **bigger-knn (~24–32) TILT RETRAIN** is the first lever with a
credible path to the +22-nat acceptance floor itself (untested: does it move the L-BFGS structural floor?).
This is the highest-value untried experiment — see §5 item 2 (fold it together with the relaxation-target retrain).

---

## 5. Open questions / candidate next steps (ranked)

0. **JUDGE THE knn=24 TILT RETRAIN** (§2 job 3, RUNNING) — the top lever. When it converges/early-stops:
   K-ladder MTM acceptance vs knn=8 (`diag_mtm_nested.py`, esp. K=8) + L-BFGS structural floor
   (`diag_declash_ceiling.py`) + held NLL. Does a *learned* fuller-context tilt close the +1/particle
   structural gap that the inference-time bump only dented (K=8 energy 30× lower but still +3.9/particle =
   62 nats = 0% accept)? If yes, this is the first thing to move the acceptance wall. If it never beats
   baseline, fall back to knn=16 or a fresh (non-warm) train, then hand to §5.1/§5.2.
1. **Finish + compare the head-to-head** (§2 NEXT ACTION). Live deliverable; caution on G1 convergence.
2. **Relaxation-target AR retrain** (training-side declash, de-risked but NOT run). `diag_relax_ar_targets`
   proved frozen-cage T=0.5 MC turns +101/particle AR blocks into +1.1/particle clash-free ones (errors are
   *slidable*), giving cheap multi-targets per cage. Attacks "confident-but-wrong placement" directly.
   **Fold together with the knn retrain** — both are "teach the conditional to place confidently AND
   correctly"; a bigger-knn tilt trained on relaxed targets is the combined bet. Caveat: relaxed targets are
   AR-basin-biased; iterate or e^{−βU}-reweight.
3. **Basin structure vs R** (the "does cross-basin matter" question, currently UNRESOLVED). Both quick MC
   probes were confounded by glassy dynamics (`05457b1`). The proper measurement is the oracle full-cage
   heat-bath run to convergence (~150 sweeps, `run_oracle_basins_long.py` is STAGED but not launched) OR
   just read it off the G1 ref_samples once they land (cluster by PTS overlap). Physics anchor: ξ_PTS≈3.8σ;
   R<ξ pinned (single basin, oracle-confirmed 2% rearr at R=2.0), R>ξ multi-basin.
4. **The AR IS a global species-reassigning proposal** (a user correction worth remembering): `sample_block_b`
   regenerates positions AND reassigns species under a count-preserving budget = the multi-swap that plain
   single-swap MC cannot do in binary KA (plain swaps are exact-0 — that's WHY BCY built shrinkage-PT). So
   the AR's real large-R niche is the easy-rung mutation inside the ladder, not a cold proposal.

---

## 6. Code map (the pieces you'll touch)

- `liquid_coupling_flow/ka3d_ebm_batched.py` — the AR generator. `sample_block_b` / `block_log_prob_b`
  (exact, tempered `pos_temp`, optional `min_sep`/`top_p`, both default-None). `_csep_allowed`, `_nucleus_mask`.
- `liquid_coupling_flow/ka3d_coarse_head.py` — `KA3DScaffoldEBMCoarse` (coarse-joint head, NEGATIVE verdict
  but kept). Has the full exact `min_sep`/`nested` machinery (`cells_allowed`, `fine_c_allowed`,
  `fine_clashfree_cube`). Reference implementation of nested masking if ever needed elsewhere.
- `liquid_coupling_flow/ka3d_cavity_egnn.py` — the EGNN flow corrector (exact CNF log-q; concentration wall).
- `liquid_coupling_flow/ka3d_pt_cavity.py` — shrinkage-PT machinery (`build_ladder`, `equilibrate_cavity`,
  `two_arm_converged`). `reports/logs-2026-07-10/pt_gate_g1.py` is the driver.
- `liquid_coupling_flow/ka_cavity_3d.py` / `ka_pmc_3d.py` — frozen-cage MC moves (`local_displacement`
  single-site + hard wall; `local_identity_swap` returns (species,U,acc) — NOT positions, a past bug;
  `parallel_mc_disp` = one batched sweep, needs positive coords so shift by +BIGL/2).
- `reports/logs-2026-07-13/` — this session's diagnostics (all the `diag_*` and `test_*` above).
- `reports/logs-2026-07-11/ka3d_pts_batched.py` — the original PTS-SMC (`smc_batched`), R=1.4 matched
  Berthier, R≥1.7 failed by *resampling collapse* onto one basin (island SMC is the fix for that).

## 7. Landmines (each cost real time this session)

- `_mic(x, center, L)`: 3rd arg is the BOX LENGTH, never the radius. Passing R folds the box → +-1e22
  energies + OOM. Bit us twice (`3196fff`, and the EGNN writeup).
- Clash/overlap self-exclusion must be **index-based** (`d[arange,arange]=inf`), never a distance threshold
  (cdist self-distances carry ~1e-4 fp error → fake clashes AND erases genuine overlaps).
- Center-relative coords are negative on one side; `torch.remainder(x, L)` wraps them → tears geometry.
  Shift to box centre (+BIGL/2) before any wrapping MC.
- GPU float64 round-trip is ~5e-3 (vs 1e-13 CPU) from reduction-order noise; run exactness bars on CPU or
  accept the nat-scale slack. SMC evidence needs the exact base density → prefer float64.
- Long runs: `local_identity_swap` returns 3 values (species,U,acc); `@torch.no_grad` the SMC loops.
