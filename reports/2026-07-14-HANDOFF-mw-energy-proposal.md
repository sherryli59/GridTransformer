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

## 1. PORT THE EXACT ka3d TRANSFORMER ARCHITECTURE (do not re-architect)

**Reuse `KA3DScaffoldEBMBatched` (`ka3d_ebm_batched.py`, subclass of `KA3DScaffoldEBM`
`ka3d_scaffold_ebm.py`, subclass of `KA3DScaffoldCatAR` `ka3d_scaffold_ar.py`) VERBATIM.** The key
realization: ka3d already operates on **local frozen-boundary cavities carved from a PERIODIC box**
(`ka3d_cavity_carve.carve` + `_mic` min-image on the N=4096 KA config). An mW block move is the SAME object
— carve a local ball of radius R around the block from the periodic mW config; the K interior particles are
regenerated; the surrounding mW particles within R+r_ctx are the boundary context. So the architecture
transfers with NO structural change. Components reused exactly:

- `fixed_ball_scaffold` (Morton-ordered low-discrepancy anchors in |x|<R) + `label_to_scaffold`
  (assign particles to anchors) + `ball_squash`/`ball_unsquash` (ball<->unbounded coords). The interior is
  a ball; periodicity only enters via `_mic` when carving + computing boundary distances (the local cavity
  is small vs the mW box).
- `_fc_batched` — the KNN transformer context: separate interior (`knn=20`) / boundary (`knn_bnd=20`,
  `bnd_cutoff=3.0`) streams, `nbr_proj`+`sp_emb`+`kind_emb`, `query`+`slot_proj`(slot_feat), `self.tr`.
- `Cat3Head` per-axis position head (`head_a/b/c` + bin embeddings, `pos_temp`) — exact categorical => exact
  `log_q`.
- The learned pairwise-potential **tilt** (`_phi_pair`, `_V_axis_b`, `_cage_knn_b` cage `knn_pot`) — a
  generic RBF-on-distance + pair-embedding MLP; keep it (it will learn an mW-shaped local potential).
- `R_embed`, `_slot_features` (frac, |a|/R, R/2.5), opt-in `use_demand` / `use_future_anchors` / `sp_temp`.
- Exact block API `sample_block_b` / `block_log_prob_b` (sample==score) — the load-bearing MH exactness.

**The ONLY changes for mW:**
1. **Energy**: swap `ka_energy` -> `mw_energy` / `du_move` in the RL reward AND the MH acceptance. Nowhere
   else (the AR density itself never calls the energy).
2. **Species**: mW is monatomic -> drop `head_species` and the `rem`/species-budget mask (set n_species=1
   or bypass). Remove `use_future_anchors`' species and the swap-MC discussion; `use_demand` becomes just
   the remaining-COUNT scalar (no A/B split).
3. **Training data**: carve local cavities from mW `mc_run` banks instead of KA configs (Phase B).
4. **3-body context (optional refinement)**: the tilt is 2-body-cage by construction; mW is 3-body. Start
   with the 2-body tilt (it still declashes via the RL reward, which USES the true 3-body mW energy). Only
   if structure (tetrahedral g(r)) is off, add a 3-body-aware tilt feature (angles to cage pairs) — a
   localized extension, architecture otherwise unchanged.

**Trainers to reuse** (same files, energy swapped):
- `reports/logs-2026-07-13/train_block_cond.py` — block-conditional MLE (mixed mask allmask/blob/single/
  tail); `build_pool` carves the cavities. Point its data loader at mW banks; drop species from the loss.
- `reports/logs-2026-07-13/train_capacity_rl.py` — THE FREE-RUN ENERGY REINFORCE trainer. Loss =
  TF-anchor + `lam * mean(adv.detach() * logq)`, `adv = standardize(capped_repulsive_energy(free-run
  block))`, `logq = block_log_prob_b(free-run sample)` (exact, differentiable). Per-config backward for
  memory. **Reward = REPULSIVE-only capped energy** `clamp(u_pair, 0, cap)`; for mW use the capped mW
  insertion/local energy (2-body core + capped 3-body) — do NOT cap only the max (rewards over-packing — a
  bug we hit and fixed). Recent runs: `--k-rl 8 12 16`, `--lam 0.5`, `--use-demand`, per-config backward,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for large blocks.
- Memories: `blockcond-ft-energy-lever`, `ar-overstretch-compounding`, `ar-basin-trapping-pts`,
  `curve-conditioning-blocks-transfer` (train the R range you'll sample — see the R-OOD lesson),
  `evaluate-distributions-not-scalars`, `checkpoint-incrementally`, `record-simulation-data`,
  `auto-resume-background-runs`.

## 2. Plan (phased; each phase self-checks before the next)

**Phase A — stand up the ported architecture + exactness gate.** Instantiate `KA3DScaffoldEBMBatched` with
n_species=1 (species head bypassed) and `mw_energy` wired into the RL reward + MH only. Carve ONE local mW
cavity (ball radius R from an equilibrated config, min-image; K interior + boundary shell) and verify the
load-bearing gates on the UNCHANGED block API: (1) exact roundtrip `|sample_block_b logq -
block_log_prob_b| < 5e-2` (float32); (2) allmask `block_log_prob_b` == full-cavity AR density; (3)
`du_move`/mw block energy matches a full `mw_energy` recompute of the cavity. NO training yet — this proves
the architecture ported cleanly before any objective is added.

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

**Phase E — eRSI-seeded SMC (use when the proposal's one-shot ESS is inadequate; ONE q0 CORRECTION,
then a FLOW-DENSITY-FREE bath).** The AR proposal will be imperfect — do NOT lean on it as a standalone IS
proposal. Use the **eRSI flow already trained for mW** as the proposal at the initial endpoint, correct it
exactly once, and then use a thermal SMC whose kernel is LOCAL and whose per-rung bath does NOT contain the
flow density.

*Hard prior result to respect (`mw-ersi-flow-arc`, measured 2026-07-11):* the eRSI-seeded **geometric** SMC
(bath π_λ ∝ q_eRSI^{1−λ} · e^{−λβU}) was **3.75–4.4× WORSE than seed-only** for mW sampling. Root cause:
q_eRSI in every weight ⇒ a full reverse-ODE per single-site score (~5.7 s) ⇒ forced GLOBAL moves ⇒ optimal
σ ~ N^{−1/2} ⇒ ~N× worse diffusion per energy-eval, growing with N. **Keep the flow density OUT of the per-rung
bath.** The flow's proven value is (a) SEEDS (2.4–3× head-start, size-flat) and (b) exact logZ — nothing else.

*The exact procedure that avoids the trap:*
- **Seed + ONE initial q0 correction:** jointly draw `(x_i, log q_eRSI(x_i))` from the eRSI forward flow.
  For its own samples this is the forward-composed log density/Jacobian from the SAME ODE pass — no reverse
  ODE. Evaluate `U_i` once and initialize
  `log w_i = -β_hot U_mW(x_i) - log q_eRSI(x_i)` (up to a shared constant). This correction is mandatory:
  eRSI positions alone are not exact draws from `π_βhot`, and a finite π-invariant mutation does not magically
  erase their bias. Resample the corrected walkers (multinomial; reset weights) before the thermal ladder.
- **Path — thermal β-ladder, flow-density-free PER-RUNG bath:** `π_t ∝ exp(-β_t U_mW)`,
  `β_t: β_hot → β_target` (needed only for a supercooled target; at ambient the corrected initial endpoint is
  already the target). For every later rung,
  `Δ log w_i = -(β_t - β_{t-1}) U_mW(x_i)`: only `U`, never `q_eRSI`. Thus the flow density appears exactly
  once at initialization and never in a mutation acceptance or thermal-rung weight.
- **Resample** walkers when ESS < ½ (multinomial; reset weights); do one endpoint resample if an unweighted
  population is required for ordinary histograms / `g(r)`.
- **Mutate:** several sweeps of a LOCAL, π_λ-invariant kernel — single-particle displacement scored with
  `du_move` (O(neighbours), N-parallel), and/or the ported ka3d local block-MTM. Local + incremental-energy is
  the whole point (this is what the global-move version got wrong).
- **Endpoint:** the weighted population targets `π_βtarget` because the initial importance correction and all
  later thermal increments are correct; the local kernels preserve each rung target and improve mixing.
  π-invariance alone is NOT a finite-time correction for biased seeds. For exact **logZ**, continue to use the
  eRSI exact-Jacobian one-shot IS bound (its measured WIN), NOT this SMC path's logZ (systematically low across
  seeds — shared mutation bias).

**Phase E MEASURED VERDICT (2026-07-14, `mw_thermal_smc.py`, N=64 M=512 β 9.5→10.381):** implemented exactly
as specified and it is **correct but NOT an efficiency win** — endpoint U/N −1.6270 vs ref −1.6274 (KS p=0.85,
g(r) rmse 0.012), but the **initial q0 correction is degenerate: ESS 1.53/512, resample → 8 unique parents**,
so 720 target sweeps were needed to re-diversify (≈ a from-scratch relaxation; seed-only reference ~3k
site-evals/cfg vs ~46k here). This was PREDICTABLE from `mw-ersi-flow-arc`: δ ≈ 1.0 nats/particle × N=64 ⇒
weight spread ~e^64 ⇒ any M ≪ e^64 gives ESS→1; and the curse is temperature-scaled ("entry reweight collapses
at EVERY β") so **retuning β_hot cannot fix it**. The exact correction *converts the flow's one asset (512
diverse near-equilibrium seeds) into 8 near-copies* — it spends the head start to buy exactness-in-weights.
**Operational split going forward:**
- **E1 — equilibration (the efficiency arm): seed-only, NO initial q0 correction.** Uncorrected eRSI seeds +
  local `du_move` MCMC (β-ladder only if supercooled). Asymptotically exact by the π-invariant kernel alone;
  the eq-bracket result (hot arms converge onto the same distribution, KS 0.08–0.11) plus THIS run's endpoint
  agreement are the evidence the healed population is right. This is the proven 2.4–3× arm.
- **E2 — certification/logZ (the exactness arm): the one-q0-correction SMC above.** Its ESS~1/M degeneracy is
  acceptable HERE because logZ bounds/certification don't need population diversity. Run it as a *check* on a
  small M, not as the sampler.
Do not pair the exact q0 correction with the equilibration arm again — the two goals need different arms.

**Cost-regime nuance (matters if this is ported beyond mW / to pricier potentials):** the 3.75–4.4×
geometric-arm negative is denominated in ENERGY EVALS (site-evals/cfg), so a pricier energy does NOT rescue
that measured (global-move) arm — the ratio is currency-invariant. BUT the *forcing* to global moves was a
wall-clock argument (reverse-ODE ~5.7 s per logq0 score ≫ a cheap mW du_move). If energy evals dominate all
flow costs (DFT-like), **geometric entry bridge + ODE-scored LOCAL moves becomes viable**: per local move the
energy cost is still only du_move, the q0 score is energy-free — an entry path that is exact, ESS-preserving
(no one-shot collapse / 8-parent destruction), and energy-eval-competitive. Untested (was wall-clock-insane
for mW); the right corner-(2) design there. Third axis = DECOMPOSABILITY: with no du_move analogue (DFT — a
full eval per move regardless of locality), the local-move advantage collapses entirely and the design flips
to few-but-excellent LARGE learned proposals (one precious eval each; flow-in-bath then costs nothing extra
in evals). Regime table: cheap+decomposable → E1 seed-only + local sweeps (mW today); expensive+decomposable
→ geometric bridge + ODE-scored local moves; expensive+non-decomposable → learned large-block proposals,
flow density freely in the bath.

*Which arm when:*
- **Ambient / ergodic:** eRSI-seed + local-move MCMC (no bath at all) — the proven 2.4–3× winner; the β-ladder
  adds little.
- **Supercooled / slow-relaxing:** the flow-free β-ladder SMC above earns its keep (anneal past the slow
  modes), still local-move + incremental-energy.
- **Free energy:** eRSI exact-Jacobian IS bound (not the SMC path).

*Accounting:* one forward flow/Jacobian pass at initialization (not an mW energy eval), then
`M · (Σ_rungs 1 reweight-eval + n_sweep · local-move evals)`. eRSI seed ⇒ short ladder; local moves ⇒
incremental energy. The hard prohibition is **repeated/per-rung/reverse-ODE flow scoring**. The single joint
initial `log q_eRSI` is required for correctness and is not the 3.75–4.4× geometric-SMC failure mode.

*Exact implementation + measured N=64 validation (2026-07-14):* `mw_thermal_smc.py` now performs the joint
forward `(x,logq0)` draw, mandatory initial correction/resample, energy-only β increments, ESS resampling,
local `du_move` mutation, and an endpoint resample. `mw_thermal_smc_continue.py` continues an unweighted
endpoint with the same exact target-invariant kernel; `mw_thermal_smc_diagnostics.py` plots relaxation,
data-scaled energy histograms, and `g(r)`. The production check used M=512, β=9.5→10.381229 (4 rungs), then
720 target sweeps. The one-time initial correction was extremely degenerate (ESS 1.53/512; 8 unique parents),
so the first 24 target sweeps were visibly insufficient (U/N −1.6043 vs data −1.6274). After the explicit
target continuation it relaxed to U/N −1.62704 vs −1.62738, energy KS=0.0283 (p=0.851), W1=0.00092, and
`g(r)` RMSE=0.0124 / max|Δ|=0.0380. The final cost was 25,329,664 weighted energy units (49,472 per walker,
about 773 N=64 sweeps). The initialization was split into 8 memory-bounded forward/Jacobian batches, but
remained one logical q0-correction stage; per-rung flow-density evaluations=0 and reverse-ODE calls=0.

Artifacts: `liquid_coupling_flow/mw/artifacts/mw_thermal_smc_N64_exact_M512_relaxed720.pt` and
`reports/logs-2026-07-14/mw_thermal_smc_N64_exact_M512_relaxed720_energy_gr.{png,pt}`. Interpretation: the
scheme is distributionally correct after enough local healing, but this particular q0→βhot correction is
not an efficiency win because its tiny initial ESS forces a long recovery. Do not hide that recovery budget.

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

- **Do NOT use a FLOW-DENSITY-IN-BATH (geometric) SMC for equilibration.** The mW campaign measured that path
  **3.75-4.4× WORSE than seed-only** (`mw-ersi-flow-arc`) — q_eRSI in every weight forces a reverse-ODE per
  site ⇒ global moves ⇒ N× diffusion penalty. And the one-q0-correction variant is now ALSO measured
  (Phase E verdict above): exact but degenerate at initialization (ESS 1.53/512, 8 parents, 720 heal sweeps) —
  the extensive δ·N logq-mismatch collapses the entry reweight at every β. For **equilibration** use E1
  (seed-only warm-started population MCMC — asymptotically exact via the π-invariant kernel, and the efficient
  arm); reserve the exact one-correction SMC for E2 certification and the eRSI exact-Jacobian IS bound for
  **logZ** (not the SMC path's biased-low logZ).
- **Incremental block energy is mandatory** — if you full-recompute `mw_energy` per proposed block, the
  accounting flips negative. Extend `du_move` to a block (single union recompute).
- **Reward must be repulsive-only + capped** (`clamp(u,0,cap)`) — capping only the max rewards over-packing
  (the bug fixed in the KA run: reward −3259 dominated by the attractive well).
- **Single species**: mW is monatomic — drop the KA species head and the swap-MC discussion entirely
  (species redistribution is N/A here).
- **Exactness gate before any MH use**: sample==score roundtrip, or the moves are not valid MCMC.
- **Train the block/cavity-size range you will SAMPLE** (KA R-OOD lesson, 2026-07-14,
  `curve-conditioning-blocks-transfer`): the KA model was trained only to R=3.0 and the proposal degraded
  OOD past it (clash 0.79 at R=4.5 vs 0.59 after retraining on R<=4.5). If mW block moves span a range of
  local-cavity radii / K, include that whole range in `--radii` / `--k-rl`. The RL energy lever DID transfer
  to OOD size (energy stayed low), but clash/mixing degrade — so match training to deployment.
- Checkpoint incrementally; save full sim data (configs + per-config observables + energy-eval traces),
  not just scalars; launch long runs with the harness background runner for auto-resume.

## 7. First concrete steps for the agent

1. Read `mw_energy.py` (esp. `du_move`), `mw_reference.py` (esp. `mc_run` + `init_cfgs`), and — for the
   architecture to reuse VERBATIM — `ka3d_ebm_batched.py`, `ka3d_scaffold_ebm.py`, `ka3d_cavity_carve.py`,
   `train_block_cond.py`, `train_capacity_rl.py`.
2. Generate a small equilibrated mW bank (`mw_reference.mc_run`, N=64-216, ambient) to use as data + the
   plain-MC baseline (record its energy-evals-to-equilibrate).
3. **Phase A: instantiate `KA3DScaffoldEBMBatched` with n_species=1, carve ONE local mW cavity, pass the
   three exactness gates (§2 Phase A) — architecture ported, NO training yet.**
4. Phases B-C: block-conditional MLE then free-run REINFORCE (`train_block_cond` / `train_capacity_rl` with
   `mw_energy` swapped in), radii/K spanning the deployment range.
5. Phase D: the three-arm energy-evals-saved comparison; supercooled + ambient.

Success = a plotted energy-evals-to-equilibrate curve where AR-seed (and, supercooled, AR-block-MH) reach
target in materially fewer cost-weighted energy evals than plain MC, with g(r) matching the reference.
