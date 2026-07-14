# Hand-off: train a block-conditional mW AR with a FREE-RUN energy objective to save expensive energy evals

**One-line goal.** Train an autoregressive (AR) block proposal for the mW (monatomic water, Stillinger-Weber)
liquid so that its proposed K-particle blocks are *low mW-energy* → high MH acceptance → **fewer expensive
mW energy evaluations to equilibrate**. Then sample two ways — (1) AR configs as **seeds** for local MC, (2)
AR blocks as a **high-acceptance MH kernel** — and report **energy-evals-saved vs plain MC**.

Why this is the right lever: the mW 3-body (Stillinger-Weber) term makes each energy eval expensive, and an
AR proposal is **energy-free to generate** (a transformer forward, no potential eval); energy is touched
only at MH acceptance. So a better proposal directly converts to fewer wasted expensive evals. This is the
same free-run energy REINFORCE lever that just worked on the KA glass (see §5), ported to mW energy.

---
## 0. What already EXISTS in the repo (read these first — do not rebuild)

- **`liquid_coupling_flow/mw/mw_energy.py`** — the potential, reduced units (sigma=eps=1):
  - `mw_energy(x, L)` and `mw_energy_chunked(x, L, chunk)` → total SW energy [B].
  - **`du_move(x, i, xi_new, L)`** → INCREMENTAL energy change for moving particle `i` to `xi_new`
    (recomputes only locally-affected 2- and 3-body terms). *This is the key to cheap MH acceptance — use
    it, never a full recompute per attempt.*
  - Constants: `T_STAR = 0.09632` (ambient 300 K), `RHO_STAR = 0.4564`, `A_CUT=1.8`, `KMAX=32` (3-body
    neighbour cap). Supercooled = LOWER T_STAR (that regime is where a collective block-MH helps most).
- **`liquid_coupling_flow/mw/mw_reference.py`** — `mc_run(N, L, beta, n_equil, n_collect, every, seed, ...,
  init_cfgs=None)`: batched single-site displacement MC (B chains), incremental via `du_move`, tracks U
  incrementally with periodic full-recompute drift checks. **`init_cfgs=[B,N,3]` seeds the chains** — this
  is the seed-injection hook. `g_r(cfgs, L, ...)` for structure. This is BOTH the ground truth and the
  training-data generator and the baseline to beat.
- **`liquid_coupling_flow/mw/mw_generator_v10.py`** — an existing GLOBAL AR (`MWV4ToroidalResidual`: v4
  MDN body + identity-init toroidal RQ-spline flow), exact density (`log_prob`/`sample`), val NLL 0.253.
  Already has a TEACHER-FORCED structure-energy finetune (`finetune_struct` / `teacher_forced_structure`,
  `_insertion_energy`) that penalizes TF SW insertion energy via reparameterized gradients. **This is a
  global AR (whole config) and a TF objective** — good as a SEED generator, but NOT the block-MH proposal
  and NOT the free-run objective this hand-off adds.
- **`liquid_coupling_flow/mw/mw_smc.py`, `mw_ersi*.py`** — the SMC / eRSI-flow path. **Do NOT use for
  equilibration** (see §6 caveat: it COSTS energy evals; it is the FREE-ENERGY tool).
- Results context: `reports/2026-07-10-mw-ersi-results.md` and memory `mw-ersi-flow-arc`.

## 1. The KA machinery to PORT (today's lever, proven on the glass)

- **`liquid_coupling_flow/ka3d_scaffold_ebm.py` + `ka3d_ebm_batched.py`** — a BLOCK-conditional scaffold-AR:
  regenerate a K-particle block given the rest, with EXACT log_q (sample==score, `sample_block_b` /
  `block_log_prob_b`), species head, learned pairwise-potential tilt, and opt-in `use_demand`
  (remaining-count embedding) + `use_future_anchors`. This is the architecture to give mW.
- **`reports/logs-2026-07-13/train_block_cond.py`** — block-conditional MLE (mixed block mask: allmask /
  blob / single / tail). The base training.
- **`reports/logs-2026-07-13/train_capacity_rl.py`** — THE FREE-RUN ENERGY REINFORCE trainer. Loss =
  TF-anchor + `lam * mean(adv.detach() * logq)`, `adv = standardize(capped_repulsive_energy(free-run
  block))`, `logq = block_log_prob_b(free-run sample)` (exact, differentiable). Per-config backward for
  memory. **The reward is REPULSIVE-only capped energy** (`clamp(u_pair, 0, cap)`) — a clash-specific,
  bounded penalty; do NOT cap only the max (that rewards over-packing — a bug we hit and fixed).
- Memories: `blockcond-ft-energy-lever`, `ar-overstretch-compounding`, `ar-basin-trapping-pts`,
  `evaluate-distributions-not-scalars`, `checkpoint-incrementally`, `record-simulation-data`,
  `auto-resume-background-runs`.

## 2. Plan (phased; each phase self-checks before the next)

**Phase A — mW block-conditional AR + exact log_q.** Port `ka3d_scaffold_ebm` to a PERIODIC box (mW has no
frozen cavity: the "boundary" of a block = the block's neighbours within a cutoff in the periodic box).
Concretely: block = K particles; context = all other particles within ~A_CUT+buffer of the block (periodic
min-image). Regenerate the K block positions AR (mW is single-species → drop the species head, or keep it
trivial). GATE: exact roundtrip `|sample_block logq − block_log_prob| < 5e-2` (float32), and allmask ==
full-config density. This is the load-bearing exactness for valid MH.

**Phase B — data.** Generate equilibrated mW configs with `mw_reference.mc_run` at the target state point
(start ambient T_STAR=0.09632, RHO_STAR=0.4564, N~64-216; then a supercooled T for the collective-move
test). Carve local blocks (a seed particle + its K-1 nearest, + the surrounding context shell) as training
examples — mirror `train_block_cond.build_pool` but periodic.

**Phase C — train.** (1) block-conditional MLE (port `train_block_cond`). (2) free-run energy REINFORCE
(port `train_capacity_rl`) with **mW energy in the reward**: `capped_mw_repulsive(free-run block)` — a
bounded, repulsive/high-energy-only surrogate (the SW 2-body core + a capped 3-body penalty for
non-tetrahedral placement; clamp per-contribution to a cap ~ few kT to avoid the r→0 blow-up). Keep the TF
anchor. Warm-start from the MLE checkpoint. Watch: free-run block mW-energy DOWN, structure (g(r)) NOT
degrading (the under-packing guard — mW's tetrahedral g(r) is the analogue of KA's density/spread).

**Phase D — sample & measure (the deliverable).** Instrument a GLOBAL energy-eval counter (wrap every
`mw_energy` / `du_move` call). Compare three arms to reach a target (autocorrelation time of g(r) first-peak
or U/N below threshold), reporting **energy-evals-to-equilibrate**:
  1. **plain MC** (`mc_run` from random/lattice init) — baseline.
  2. **AR-seeded MC** — `mc_run(init_cfgs=AR_samples)` → measures burn-in evals SAVED (the proven seed lever).
  3. **AR-block-MH** — a new MC loop: propose AR blocks, accept with the tempered/plain MH ratio using a
     BLOCK incremental energy (extend `du_move` to K particles: recompute the union of locally-affected
     2/3-body terms once, not K separate calls). Measures decorrelation per expensive eval.

## 3. Energy-eval accounting (the metric — get this right)

- The AR forward is **energy-free** — count ONLY `mw_energy`/`du_move` calls (and weight by their true cost:
  a full `mw_energy` on N particles ≈ N × a single `du_move`; a block `du_move` on K ≈ K× a single, roughly).
- Headline number: **energy-evals (cost-weighted) to reach target vs plain MC** = the saving.
- Report the full autocorrelation/relaxation curve vs energy-evals, not a single scalar (per
  `evaluate-distributions-not-scalars`).

## 4. Predictions / expected outcome (from the KA results + the mW campaign)

- **Seeds give a robust few-× saving** on burn-in (the campaign measured ~3× seed-amortization) — this is
  the safe win.
- **AR-block-MH adds meaningfully mostly in the SUPERCOOLED regime** (slow collective modes); at ambient,
  local MC already mixes and the block advantage is smaller.
- **mW should behave BETTER than KA for cold block-MH acceptance**: KA had a +1.4/particle glassy declash
  CEILING (`ar-basin-trapping-pts`) that killed cold K=8 acceptance regardless of proposal; mW liquid has no
  analogous deep glassy barrier, so the RL-improved low-energy proposal can actually raise cold block
  acceptance, not just help a tempered path.

## 5. Why the free-run objective (not the existing TF finetune)

`mw_generator_v10.finetune_struct` penalizes energy under TEACHER FORCING (true causal prefix). Today's KA
result (`ar-overstretch-compounding`) proved TF training cannot teach a proposal to survive its OWN
free-run drift — the block's clash/energy pathology only appears under free-running. The REINFORCE free-run
objective trains on the model's own rollout, which is what actually governs MH acceptance. Port
`train_capacity_rl`, not the TF finetune.

## 6. Hard caveats

- **Do NOT use the tempered island-SMC / eRSI-SMC for equilibration.** It is energy-eval-HUNGRY (energy at
  every rung × particle × mutation) and the mW campaign measured the geometric-SMC path **3.75-4.4× WORSE
  than seed-only** for sampling. The SMC is the tool for **free energy (logZ)**, not for cheap equilibration.
- **Incremental block energy is mandatory** — if you full-recompute `mw_energy` per proposed block, the
  accounting flips negative. Extend `du_move` to a block (single union recompute).
- **Reward must be repulsive-only + capped** (`clamp(u,0,cap)`) — capping only the max rewards over-packing
  (the bug fixed in the KA run: reward −3259 dominated by the attractive well).
- **Single species**: mW is monatomic — drop the KA species head and the swap-MC discussion entirely
  (species redistribution is N/A here).
- **Exactness gate before any MH use**: sample==score roundtrip, or the moves are not valid MCMC.
- Checkpoint incrementally; save full sim data (configs + per-config observables + energy-eval traces),
  not just scalars; launch long runs with the harness background runner for auto-resume.

## 7. First concrete steps for the agent

1. Read `mw_energy.py` (esp. `du_move`), `mw_reference.py` (esp. `mc_run` + `init_cfgs`), and the KA
   `ka3d_ebm_batched.py` / `train_capacity_rl.py`.
2. Generate a small equilibrated mW bank (`mw_reference.mc_run`, N=64, ambient) to use as data + the plain-MC
   baseline (record its energy-evals-to-equilibrate).
3. Phase A: stand up the periodic block-conditional AR + exactness gate (smoke).
4. Phases B-C: MLE then free-run REINFORCE with capped mW energy.
5. Phase D: the three-arm energy-evals-saved comparison; supercooled + ambient.

Success = a plotted energy-evals-to-equilibrate curve where AR-seed (and, supercooled, AR-block-MH) reach
target in materially fewer cost-weighted energy evals than plain MC, with g(r) matching the reference.
