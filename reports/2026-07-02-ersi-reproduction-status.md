# eRSI reproduction status — where we are on the Grenioux ESS result

**Date:** 2026-07-02
**Branch:** liquid-coupling-flow · `liquid_coupling_flow/ipl44/`
**Target paper:** Grenioux et al. 2025, *Riemannian Stochastic Interpolants for Amorphous Particle Systems*
— local PDF **`/mnt/ssd/GridTransformer/grenioux.pdf`**; code **github.com/h2o64/learndiffeq**; data **Zenodo 17966995**.
**System:** 2D N=44 IPL / BHHP soft spheres (r⁻¹²), 22:22, σ=[[1,1.2],[1.2,1.4]], ε=1, rcut=2.5σ, ρ=0.5→L=√88≈9.38, **T=0.1 (β=10)**.

---

## 0. TL;DR — where we are

- **Transformer arm (AR generator as Boltzmann proposal): closed, negative.** The AR +A curve-flow transformer
  is strictly dominated by eRSI on N=44 IPL (discard ~0.99, ESS≈1); IS cannot rescue it. Documented in
  **`reports/2026-06-26-ipl44-vs-ersi-results.md`** (that comparison used the paper's *published* eRSI numbers,
  not a re-run).
- **eRSI arm (this report): NOT yet reproduced.** We have not reproduced the paper's ESS(R) **crossover**
  (their ESS/R deviates from the degenerate 1/R around R≈10⁴). Every eRSI model we have trained/evaluated sits
  in the **sub-crossover regime**: ESS ≈ O(1–few), ESS/R ∝ 1/R, pinned by a heavy upper weight-tail.
- **The mechanism is understood; a good-enough model is not yet in hand.** The gap is **proposal quality**, not
  the sampler, the divergence estimator, or the integrator (all ruled out below). Our best eRSI model was
  trained at lr 5e-3 (per request) and **diverged late** (epoch 684); the paper's clean recipe (lr 1e-3 + EMA)
  has not yet been run to completion.
- **One real bug found and fixed** (`log_prob` sign) — but it is **incidental**: it never touched any ESS
  number (all ESS uses the forward `sample_and_log_prob`, which is correct).

**Bottom line:** we can explain *why* our eRSI proposals don't cross over (real rare high-weight configs), but
we have not yet trained an eRSI good enough to match the paper's crossover, nor confirmed the exact claim in
the PDF we are targeting. The decisive next experiment is the **clean lr-1e-3 + EMA + OT retrain**.

---

## 1. Two arms of the project

### Arm A — Transformer as the Boltzmann proposal (prior work, closed)
- **AR +A curve-flow transformer** (`KACurveFlowModel`, exact likelihood): on N=44 IPL, discard 0.987 vs
  eRSI's 0.03, ESS≈1 vs paper ~10⁶; the causal-AR conditional's breadth fills the excluded-volume core and
  smears g(r) (worst for g_BB). **Full write-up: `reports/2026-06-26-ipl44-vs-ersi-results.md`.**
  Memory: `ipl44-ar-vs-ersi-verdict`.
- **Transformer-as-flow-base (tf2boltz):** use the AR transformer as the flow's *base* (not uniform), compose
  exact-likelihood IS (logq_transformer + flow Jacobian). Scripts `tf_bank` / `train_tf2boltz` /
  `eval_tf2boltz`. Memory: `tf2boltz-ersi`. This reuses the eRSI machinery and therefore depends on the eRSI
  arm being sound.

### Arm B — Reproduce eRSI itself (this report)
The 2026-06-26 comparison deliberately did **not** run eRSI (no released checkpoint; numbers taken from the
paper). This session set out to actually reproduce eRSI's ESS(R) behavior.

---

## 2. What we established about the eRSI recipe (this session)

Full details in memory `ipl44-ersi-paper-recipe`. Key facts (corrected several earlier misconceptions):

- **Code (two local copies):**
  - **eval / density (FIXED here):** `liquid_coupling_flow/ipl44/learndiffeq/` — used by all our ESS scripts.
  - **training:** `/mnt/ssd/flow_matching/learndiffeq-main/` — `experiments/training_rfm.py`, driver
    `experiments/scripts/ipl_precompute_and_train.sh`.
- **Data:** Zenodo `ipl44_T0.1_positions.pt` / `_species.pt` = **(10000, 44, 2)**, 22:22, **UNWRAPPED**
  (min −42/max 51.8; torus ops handle it). Pristine copy at `/mnt/ssd/GridTransformer/datasets/`.
- **Canonical README eRSI** = `--velocity_type particles_equivariant --egnn_hidden_nf 32 --egnn_n_layers 3`
  (**3|32**, 22.6k params), `--base_type uniform`, `--target_type bhhp`, `--training_algo rfm`, on the 10K
  Zenodo data. README example: batch 256, 1250 epochs, **lr 1e-3 + `--use_ema` + `--enable_ess_callback`**.
- **OT is opt-in and OFF by default.** `training_step` couples base→data only with `--ot_particles`
  (per-species minibatch OT, `ot_permutation`) or `--use_linear_assignment_particles` (Hungarian). The README
  canonical command sets neither.
- **The 64|4 `epoch=1199` checkpoint is NOT the canonical eRSI.** It came from the driver on 100K
  *self-generated MCMC* data with a precomputed global-OT base (lr 1e-4, 117k params). It is IS-poor.

---

## 3. eRSI ESS attempts and what each showed

| model / eval | ESS behavior | verdict |
|---|---|---|
| **64\|4 epoch-1199**, Hutchinson div | ESS≈1 flat, logw std ~10¹² (under-converged epoch-116) → ~670 (converged); ESS/R ∝ 1/R to 5×10⁴ | sub-crossover; Hutchinson adds a spurious +6.7 logw outlier that *artificially* pins ESS |
| **64\|4 epoch-1199**, exact div | ESS 2.26 @ 8192 (vs Hutch 1.03); max logw bounded −80 | exact div removes the Hutchinson artifact, but ESS/R **still ∝ 1/R** → still no crossover |
| **3\|32 OT+EMA+warmup `epoch=625`**, dopri5 + exact div | ESS flat ~1.5–2, drops to ~1 at 10⁴; ESS/R clean 1/R through R=16384 | **no crossover**; pinned by ONE real config (96% of weight) |

**Consistent finding:** across estimators, integrators, and both architectures, eRSI ESS/R is a clean 1/R in
the range we can reach (R ≤ ~5×10⁴) — the degenerate sub-crossover regime. We have **not** observed the
upward bend the paper reports near 10⁴.

---

## 4. Diagnostics — what we ruled OUT as the cause

1. **Divergence estimator is not it.** Hutchinson vs exact div: Hutchinson injects a spurious high-logw outlier
   (max logw +6.7 vs exact −80) that pins ESS to 1; exact div lifts ESS ~2–4×. But the **scaling is identical**
   (both ESS/R ∝ 1/R). `make_comparison_plot.py`, `ess_hutch_vs_exact.png`.
2. **Integrator is not it.** Euler-100 ≈ Euler-400 ≈ dopri5 on **logq** (agree to <0.6 nats). Euler-100 only
   occasionally creates hard-overlap **energy** outliers (lower-tail, tiny-weight, ESS-irrelevant). dopri5 does
   not rescue the crossover. `integrator_check.py`.
3. **The hard overlaps are not it.** The dopri5+exact bank's min logw (−10⁸, a severe overlap) is lower-tail;
   dropping every overlap config leaves ESS unchanged (`make_new_plot.py`, `ess_newmodel_dissect.png`).
4. **The ESS-killer is REAL, not a numerical artifact.** ESS≈1 is one config holding **96%** of the weight;
   remove top-1 → ESS 3.8, top-5 → 6.6. A per-config integrator cross-check (`diag_dominant_config.py`) was run
   to test whether that config's weight was a dopri5 artifact — it instead exposed the `log_prob` bug (§5); the
   forward logq used for ESS is physically correct (concentration: logq > log ρ₀), so the dominant config is a
   **genuine proposal gap**.

**What we ruled IN as the key training ingredient:** **OT**. Enabling `--ot_particles` drops the RFM loss
~6.5× at init (7.24 → 1.12) → far lower target-velocity variance. The canonical README command omits it; our
retrain includes it.

**What remains the leading suspect for the crossover gap:** **model/training quality.** Our best eRSI
(`epoch=625`) was trained at lr 5e-3 (per request) which was stable for 680 epochs then **diverged at
epoch 684**; `epoch=625` is pre-divergence but trained at the edge of instability. The paper's clean lr 1e-3 +
EMA recipe (no divergence) has not yet been run.

---

## 5. Bug found and fixed: `RiemannianFlowMatching.log_prob` sign

- **Root cause:** `ModuleAndDivergence` integrates `d(log_jac)/dt = −div(b)` with **no reverse-time sign
  handling**. `sample_and_log_prob` (forward, t:0→1) accumulates the correct log-Jacobian; `log_prob`→`inverse`
  (t:1→0, dt<0) accumulates its **negative**, then adds it → density off by ~250 nats (sign-flipped).
- **Fix** (`liquid_coupling_flow/ipl44/learndiffeq/learndiffeq/flow_matching/riemannian_fm.py`):
  `return ret − log_jac.flatten()`. Verified: fixed `log_prob` matches forward logq to **<0.06 nats**.
- **Impact = none on ESS.** Every ESS computation — ours and the paper's `callbacks/ess.py` — uses the forward
  `sample_and_log_prob`, which was always correct. The bug affected only `log_prob` callers: the `ml`
  training objective, density-of-data evaluation, the ESS-callback logp-vs-logp scatter. **The same bug is
  still present in the training copy** `/mnt/ssd/flow_matching/learndiffeq-main/` (harmless for `rfm` training).

---

## 6. Step back — evaluation and the crux

**What is solid:**
- The transformer (AR) arm is a clean negative and well-explained.
- We have correctly located the eRSI code, data, and canonical recipe, and corrected earlier confusion about
  which config is "the paper's."
- We understand the sub-crossover mechanism (heavy upper weight-tail; rare real dominant configs) and have
  ruled out the sampler, the div estimator, and the integrator as the cause.
- OT is confirmed as the key training ingredient.
- The likelihood path used for ESS is now bug-verified correct.

**What is unresolved — the crux:** *we have not produced an eRSI model good enough to cross over, and we have
not confirmed the exact claim we are chasing.* Three live hypotheses, in priority order:

1. **Under-trained/divergence-damaged model (most likely).** lr 5e-3 diverged; a clean **lr 1e-3 + EMA + OT**
   run (the paper's actual recipe) may have a light enough upper tail to cross over. **Untested.**
2. **Claim mismatch.** The paper's "ESS/R deviates at ~10⁴" may be for a specific figure/system/temperature/R̄
   we have not pinned down. We have **not yet read `grenioux.pdf` closely** to nail exactly which curve, which
   system (N=10 vs 44), which T (0.04/0.07/0.1), and what R̄ they state.
3. **Residual pipeline difference.** e.g. our `ipl_energy`/β vs the paper's `SoftSphere.U` in `ess.py`, or a
   sampler setting. The energy reuse was verified earlier (`reports/2026-06-26-...`) but the ESS-callback path
   specifically has not been diff-checked end-to-end.

**Recommended next steps (in order):**
1. **Read `grenioux.pdf`** — extract the exact ESS(R) claim: which system/T, the stated R̄, and how they define
   the crossover. This decides whether we are even targeting the right regime for N=44/T=0.1.
2. **Clean retrain**: lr 1e-3 + `--use_ema` + `--ot_particles`, 3|32, 10K data — run to completion (no
   divergence), then re-run the dopri5+exact ESS eval **saving samples** for config-level diagnosis.
3. If a clean model still doesn't cross over at the R the paper claims, **diff the ESS-callback path**
   (energy, β, base, solver) against `experiments/training_rfm.py --enable_ess_callback` end-to-end.

---

## 7. Artifacts & paths

**Paper / reference**
- `grenioux.pdf` (target paper) · code `github.com/h2o64/learndiffeq` · data Zenodo 17966995
- eRSI code: `liquid_coupling_flow/ipl44/learndiffeq/` (eval, `log_prob` FIXED) · `/mnt/ssd/flow_matching/learndiffeq-main/` (training)

**This session's training output** — `/mnt/ssd/GridTransformer/ipl44_ersi_reproduce/`
- `checkpoints/ipl44_rfm_5e3_ema_warmup_ot/epoch=625.ckpt` — best canonical 3|32 OT+EMA model (pre-divergence)
- `checkpoints/ipl44_rfm_5e3_ema_warmup_ot/last.ckpt` — final (diverged, epoch 999)
- `checkpoints/ipl44_rfm_5e3_k3hf32/` — first (lr 5e-3 no-EMA) run, diverged at epoch 5
- `train_ema_warmup_ot.log`, `train.log`, `config.json`

**This session's scripts / figures** (scratchpad `…/2f102e4c-…/scratchpad/`)
- `ess_R_newmodel_dopri5.py` — dopri5+exact ESS(R) bank (progressive) · `ess_newmodel_dopri5.png`
- `make_new_plot.py` — tail dissection + ESS(R) · `ess_newmodel_dissect.png`
- `integrator_check.py` — Euler vs dopri5 logq · `diag_dominant_config.py` — per-config logq cross-check (found the bug)
- `ess_R_paper.py` / `progressive_bank*.py` / `make_comparison_plot.py` — 64|4 exact-vs-Hutchinson ESS work

**Related reports / specs / memory**
- `reports/2026-06-26-ipl44-vs-ersi-results.md` — AR transformer vs eRSI (Arm A)
- `reports/transformer-architecture-and-alternatives.md`
- `docs/superpowers/specs/2026-06-26-ipl44-benchmark-vs-ersi-design.md` (+ plan)
- Memory: `ipl44-ersi-paper-recipe`, `ipl44-ar-vs-ersi-verdict`, `tf2boltz-ersi`, `ipl44-unwrapped-data-bug`,
  `traceable-egnn`, `joint-species-flow`

**Code (ours)**
- `liquid_coupling_flow/ipl44/`: `ipl_energy.py` (energy/g(r)/data adapters), `ipl_model.py` (AR model+trainer),
  `ipl_benchmark.py` (discard/ESS), `joint_flow.py` (joint species-position flow)
