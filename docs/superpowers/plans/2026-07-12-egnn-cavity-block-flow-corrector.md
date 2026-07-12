# EGNN Cavity Block-Flow Corrector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An exact-likelihood, full-cage, equivariant flow that corrects an AR-generated cavity block's relative structure, deployed as a block MOVE in the PT/SMC cavity ladder.

**Architecture:** AR block sample `x0` (frozen `KA3DScaffoldEBMBatched`) → cage-conditioned 3D EGNN velocity field integrated `x0→x1` (dopri5 augmented ODE) → exact composed log-q `logq_AR(x0) + ∫∇·v dt` via the analytic k-NN divergence. Trained by conditional flow-matching (AR-block→data-block), NOT composed-MLE. Species fixed.

**Tech Stack:** PyTorch, torchdiffeq (dopri5), the ipl44 `egnn_traceable` core (3D-capable), `ka_cluster_egnn` patterns (frozen-cage context + cluster-restricted divergence), `KA3DScaffoldEBMBatched`.

## Global Constraints

- **Exactness is non-negotiable:** analytic divergence must match brute-force autograd trace to ~1e-6 (double precision); composed log-q must be the true density of the returned sample (carry-the-latent, no inversion).
- **3D, isolated (non-periodic):** cavity is a ball in free space, center-relative coords; use raw Euclidean distances (pass `L` large/`None` so no wrapping), matching `ka_energy`'s `BIGL=100.0` convention.
- **Cage frozen:** boundary shell + retained interior are context (message sources), never move, never enter the log-det.
- **Species fixed** from the AR base (node feature), never flowed.
- **Chain-level zero-leakage split:** train on `ka3d_train_N4096_T0.5_rho1.15.pt` (112 chains); held = the 16 reference chains (`ka3d_dataset_N4096_T0.5_rho1.15.pt`). Radii {1.6, 2.0, 2.5, 3.0}.
- **Frozen AR base ckpt:** `ka3d_cavity_ebm3ax_rho115_best.pt` (held-NLL −2.636).

---

### Task 1: 3D non-periodic conditional EGNN velocity + exact divergence

**Files:**
- Create: `liquid_coupling_flow/ka3d_cavity_egnn.py`
- Test: `reports/logs-2026-07-12/test_egnn3d_exact.py`

**Interfaces:**
- Consumes: the ipl44 `egnn_traceable` EGNN dynamics (verify its 3D + `max_neighbors` k-NN path; `n_dimension=3`, non-periodic by large/no `L`).
- Produces: `class CavityCondEGNN(nn.Module)` with `vel_div(cloud[B,P,3], t, sp[B,P], k) -> (vel[B,k,3], div[B])` — first `k` are movers (block), rest are frozen cage; divergence is the mover-restricted trace (cage not in log-det). `n_species=2`, `hidden_nf`, `n_layers`, `r_c`, `max_neighbors=k_knn`.

- [ ] **Step 1: Write the failing exactness test** — analytic per-particle divergence vs brute-force autograd trace of the velocity Jacobian (movers only), 3D, n_species=2, double precision, random cloud (k movers + cage). Assert max abs diff < 1e-6.
- [ ] **Step 2: Run it, confirm it fails** (module absent).
- [ ] **Step 3: Implement `CavityCondEGNN`** wrapping the 3D `egnn_traceable` dynamics (mirror `ConditionalEGNN.vel_div`: forward_and_perparticle_divergence, slice `[:, :k]` for movers, sum mover div). Non-periodic (no `remainder`), raw Euclidean, `max_neighbors=k_knn`.
- [ ] **Step 4: Run test, confirm pass** (exactness ~1e-6).
- [ ] **Step 5: Commit.**

### Task 2: Augmented-ODE integrator + composed log-q (carry-the-latent)

**Files:**
- Modify: `liquid_coupling_flow/ka3d_cavity_egnn.py`
- Test: `reports/logs-2026-07-12/test_egnn3d_flow.py`

**Interfaces:**
- Produces: `class CavityBlockFlow(nn.Module)` with:
  - `flow(x0_block[B,k,3], cage_x[B,m,3], sp_block[B,k], sp_cage[B,m], R, reverse=False) -> (x1[B,k,3], logdet[B])` — dopri5 on `[x_block, logdet]`, cage frozen, exact `∫∇·v dt`.
  - `composed_logq(x0_block, logq_ar[B], cage_x, sp_block, sp_cage, R) -> (x1[B,k,3], logq[B])` where `logq = logq_ar + logdet` (carry `x0`, no inversion).

- [ ] **Step 1: Write failing test** — (a) forward `x0→x1` then reverse `x1→x0` recovers `x0` to ~rtol and logdets cancel to ~1e-5 (double, tight rtol); (b) cage positions unchanged by `flow`. Assert both.
- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement** the dopri5 augmented-ODE integrator (adapt `EGNNClusterFlow._dopri5_integrate`, drop periodic wrap; velocity from Task-1 `vel_div`) + `composed_logq`. Velocity zero-init (identity flow at init).
- [ ] **Step 4: Run test, confirm pass.**
- [ ] **Step 5: Commit.**

### Task 3: Cavity wiring — AR base + cage assembly + sampling API

**Files:**
- Modify: `liquid_coupling_flow/ka3d_cavity_egnn.py`
- Test: `reports/logs-2026-07-12/test_egnn3d_cavity.py`

**Interfaces:**
- Consumes: `KA3DScaffoldEBMBatched.sample_block_b` (AR base + `logq_ar`, with `pos_temp`), `carve`, `label_to_scaffold`, `fixed_ball_scaffold`.
- Produces: `sample_corrected_block(ar_model, xo, so, block_mask, bnd, s_bnd, R, gen, pos_temp) -> (x_full[M,n,3], s_full, logq_composed[M])` — AR-sample the block, assemble cage = boundary+retained, flow-correct the block, return corrected full config + composed log-q. Identity flow (untrained) must reproduce the AR block + `logq_ar` (composition sanity).

- [ ] **Step 1: Write failing test** — with a zero-velocity (untrained) flow, `sample_corrected_block` returns block == AR block (to ~1e-5) and `logq_composed == logq_ar` (to ~1e-5). Assert.
- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement** the wiring (reuse the cage-assembly + ordering from `frontier_threearm.py`/`ka3d_ebm_batched`; block = movers first).
- [ ] **Step 4: Run test, confirm pass.**
- [ ] **Step 5: Commit.**

### Task 4: Flow-matching data pipeline + minibatch-OT coupling

**Files:**
- Create: `reports/logs-2026-07-12/fm_data.py`
- Test: `reports/logs-2026-07-12/test_fm_data.py`

**Interfaces:**
- Produces: `fm_batch(ar_model, X, S, L, cavs, gen, pos_temp) -> {x0_block, x1_block, cage_x, sp_block, sp_cage, R, t, x_t, target_v}` — per cavity: AR block (`x0`), data block (`x1`, count/species-matched), minibatch-OT coupling `x0↔x1`, interpolant `x_t=(1−t)x0+t·x1`, target velocity `x1−x0` (rectified-flow). OT via exact small-batch assignment (Hungarian on the K-block; fall back to identity coupling if degenerate).

- [ ] **Step 1: Write failing test** — coupling is a valid permutation (each x0 mapped to one x1); species preserved along the pairing; `x_t` at t=0 == x0, t=1 == x1. Assert.
- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement** `fm_batch` (Hungarian per-block OT on centered positions).
- [ ] **Step 4: Run test, confirm pass.**
- [ ] **Step 5: Commit.**

### Task 5: Flow-matching training loop (chain-level split, early stop)

**Files:**
- Create: `reports/logs-2026-07-12/train_cavity_egnn_flow.py`
- Test: overfit smoke inside the script (`--smoke`).

**Interfaces:**
- Consumes: Tasks 1-4. Frozen AR base. Loss = FM MSE `||v_theta(x_t,t|cage) − (x1−x0)||^2`, cage frozen.
- Produces: ckpt `liquid_coupling_flow/artifacts/ka3d_cavity_egnn_flow.pt` (+ `_best`). Held = 16 reference chains; early-stop on held FM loss (patience); best-by-held + gap monitor.

- [ ] **Step 1: Write the smoke** — on a tiny fixed cavity pool the FM loss must DECREASE (assert final < first).
- [ ] **Step 2: Run smoke, confirm it can fail** (untrained → high loss).
- [ ] **Step 3: Implement** the training loop (AdamW, random SO(3) aug on cage+block together, per-eval held FM loss, early stop, incremental ckpt).
- [ ] **Step 4: Run smoke, confirm loss decreases.**
- [ ] **Step 5: Commit.** Then launch the full run (background, watched).

### Task 6: Structure-fix gate (Gate 2)

**Files:**
- Create: `reports/logs-2026-07-12/gate_structure.py`

**Interfaces:**
- Consumes: trained flow + AR base + `ka_energy` L-BFGS floor harness (reuse `test_species_vs_position.py`/`test_untangle.py`).
- Produces: per held cavity, inherent-structure floor of the FLOW-CORRECTED AR block (AR species) vs the raw AR block (+105 stuck) and the data floor (+0.9); g_BB peak of corrected vs AR vs data.

- [ ] **Step 1:** Measure corrected-block floor + g_BB over ~10 held cavities, K and whole-block. **PASS** = corrected floor << +6.9 (toward +0.9), g_BB peak toward 2.0.
- [ ] **Step 2: Commit** results + verdict (this is the make-or-break structural gate).

### Task 7: Deployment + move-quality + frontier gates (Gates 3-4)

**Files:**
- Modify: `reports/logs-2026-07-12/frontier_threearm.py` (add a `MSEED_FLOWMOVE` arm using `sample_corrected_block` as the hot-rung block move) OR a new `frontier_flow.py`.
- Create: `reports/logs-2026-07-12/mtm_flow_read.py` (Gate 3).

**Interfaces:**
- Consumes: Tasks 3+5. Composed log-q in the MTM/SMC weight.

- [ ] **Step 1 (Gate 3):** high-stat exact block-MTM acceptance of the flow-corrected move (K∈{6,8,12}, R∈{2.0,2.5,3.0}) vs the sharpened-AR baseline (currently ~0 at K≥6). **PASS** = clear improvement.
- [ ] **Step 2 (Gate 4):** the flow-move arm in the frontier benchmark vs `MSEED_MOVE` (4/12) at matched budget. **PASS** = more converged cavities.
- [ ] **Step 3: Commit** results + verdict; update `ka3d-pts-smc` memory.

## Global test/exactness notes
- Run all exactness checks in **double precision**, tight dopri5 rtol.
- The composed log-q's AR term carries the AR's ~8e-3 ball-round-trip leak — record it; it bounds the composed exactness (not the flow's fault).
