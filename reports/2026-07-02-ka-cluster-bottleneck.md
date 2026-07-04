# KA cluster-flow bottleneck diagnostics

## Task 1: backend audits

Numerical audits of the traceable-EGNN periodic velocity backend
(`liquid_coupling_flow/egnn_traceable.py`), targeting the suspected frame-mixing
bug in the periodic branch of `_compute_common_terms` (lines 140-173): it sets
`source_xs_com = source_xs` (raw box coords) while `x_final` lives in the
cloud-centered frame, so `diffij = x_final - source_xs` mixes frames. This
would make `r_ij` contaminated by absolute position and give the velocity a
`-(sum_j pot_j) * x_i` term that breaks translation invariance.

Test file: `liquid_coupling_flow/tests/test_egnn_audit.py`
Run: `python -m pytest liquid_coupling_flow/tests/test_egnn_audit.py -v -s`

### Results — 5 failed, 1 passed

| Test | Verdict | Number |
|---|---|---|
| `test_velocity_translation_invariance_random_init` | **FAIL** | max diff `19.10422134399414` |
| `test_velocity_translation_invariance_trained_ckpt` | **FAIL** | max diff `0.35218334197998047` |
| `test_velocity_D4_equivariance` | **FAIL** | max diff `16.962852478027344` |
| `test_central_force_semantics_pot_one` | **FAIL** | max diff `245.72695922851562` |
| `test_no_phantom_edges_knn24` | **FAIL** | `294` phantom inner edges at kNN-24 (brief predicted PASS) |
| `test_phantom_edges_fullcloud_report` | PASS (informational) | printed `full-cloud phantom inner edges: 50150` |

Pytest tail (`-v -s`, last 25 lines):

```
        cloud, sp, L = _cloud()
        n = _phantom_count(cloud, L, 24)
>       assert n == 0, f"{n} phantom inner edges at kNN-24"
E       AssertionError: 294 phantom inner edges at kNN-24
E       assert 294 == 0

liquid_coupling_flow/tests/test_egnn_audit.py:104: AssertionError
=============================== warnings summary ===============================
liquid_coupling_flow/tests/test_egnn_audit.py::test_velocity_translation_invariance_random_init
liquid_coupling_flow/tests/test_egnn_audit.py::test_velocity_translation_invariance_trained_ckpt
liquid_coupling_flow/tests/test_egnn_audit.py::test_velocity_D4_equivariance
liquid_coupling_flow/tests/test_egnn_audit.py::test_central_force_semantics_pot_one
liquid_coupling_flow/tests/test_egnn_audit.py::test_no_phantom_edges_knn24
liquid_coupling_flow/tests/test_egnn_audit.py::test_phantom_edges_fullcloud_report
  /home/sherryli/xsli/softwares/anaconda3/envs/lightning/lib/python3.11/site-packages/torch/nn/modules/transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED liquid_coupling_flow/tests/test_egnn_audit.py::test_velocity_translation_invariance_random_init
FAILED liquid_coupling_flow/tests/test_egnn_audit.py::test_velocity_translation_invariance_trained_ckpt
FAILED liquid_coupling_flow/tests/test_egnn_audit.py::test_velocity_D4_equivariance
FAILED liquid_coupling_flow/tests/test_egnn_audit.py::test_central_force_semantics_pot_one
FAILED liquid_coupling_flow/tests/test_egnn_audit.py::test_no_phantom_edges_knn24
=================== 5 failed, 1 passed, 6 warnings in 2.74s ====================
```

### Interpretation

The frame-mixing hypothesis is **confirmed**:

- `test_central_force_semantics_pot_one` (the frame-pinning test) fails with
  max diff `245.7`, of the predicted O(10-100) order (in fact larger) — with
  `pot` forced to 1 and inner coord updates zeroed, the periodic velocity does
  NOT equal `sum_j min-image(x_j - x_i)`. The raw output magnitude (tens to
  hundreds) versus the expected central-force magnitude (O(1-10)) is
  consistent with the `-(sum_j pot_j) * x_i` absolute-position contamination
  term predicted by the brief.
- Both translation-invariance tests fail (random-init max diff `19.1`,
  trained-checkpoint max diff `0.35`) — the trained model's diff is smaller
  (consistent with the brief's counter-evidence that training can partially
  cancel the effect by driving `sum_j pot_j -> 0`) but still two orders of
  magnitude above the `1e-3` tolerance, i.e. not actually invariant.
- D4 rotational equivariance also fails (max diff `17.0`), same order as the
  translation break, as expected since both stem from the same absolute-`x_i`
  term.
- `test_no_phantom_edges_knn24` was predicted to PASS (kNN-24 cloud diameter
  ~5.1 < cutoff-margin 6.13) but instead **FAILS** with 294 phantom inner
  edges. This is an additional finding beyond the frame-mixing hypothesis: the
  kNN-restricted cloud used in practice already contains spurious min-imaged
  edges under the current `L`, not just the full-cloud reference config. This
  widens the suspected defect surface and should be carried into the next
  diagnostic task.
- `test_phantom_edges_fullcloud_report` (informational) confirms gross
  contamination at full-cloud scale (50150 phantom edges), as expected.

### Branch decision

**Any red -> Task 3.** All five substantive invariance/frame tests are RED.
Proceeding to **Task 3** (root-cause fix of the frame-mixing bug in
`_compute_common_terms`, `liquid_coupling_flow/egnn_traceable.py:140-173`),
carrying forward the additional kNN-24 phantom-edge finding as an open
question to resolve alongside the frame fix.

## Task 3: fixes

Applied both fixes verbatim from the Task-3 brief to
`liquid_coupling_flow/ipl44/learndiffeq/learndiffeq/particles/velocities/egnn_traceable.py`:

- **Step 1** (`_compute_common_terms`, was lines 140-146): the periodic branch
  now shares the same `com`/`source_xs_com` expression as the non-periodic
  branch — the central particle is expressed in the same cloud-centered frame
  as `x_final`, so `diffij` (and therefore `r_ij`) is the physical pair
  distance, translation-invariant. The `L is None` path is unchanged (same
  expression, now shared); the analytic divergence is unaffected because
  `d(com)/d(x_i) = 0`.
- **Step 1b** (`compute_edges`, was lines ~336-339): removed the box-`L`
  folding of `pvec` — the input `x` is already the min-imaged, contiguated
  neighbour cloud (non-periodic by construction), so folding here was
  fabricating phantom edges between genuinely distant cloud members whose
  `edge_attr` was then computed from the raw (unfolded) difference.

### Step 2 — audit suite rerun (`test_egnn_audit.py`)

`python -m pytest liquid_coupling_flow/tests/test_egnn_audit.py -v -s`

**6/6 pass** — audit tests corrected to probe production `compute_edges`:

```
liquid_coupling_flow/tests/test_egnn_audit.py::test_velocity_translation_invariance_random_init PASSED
liquid_coupling_flow/tests/test_egnn_audit.py::test_velocity_translation_invariance_trained_ckpt PASSED
liquid_coupling_flow/tests/test_egnn_audit.py::test_velocity_D4_equivariance PASSED
liquid_coupling_flow/tests/test_egnn_audit.py::test_central_force_semantics_pot_one PASSED
liquid_coupling_flow/tests/test_egnn_audit.py::test_no_phantom_edges_knn24 PASSED
liquid_coupling_flow/tests/test_egnn_audit.py::test_no_phantom_edges_fullcloud PASSED

======================== 6 passed, 6 warnings in 5.40s =========================
```

Frame-fix verdicts (Step 1, all confirmed): translation invariance
(random-init and trained-ckpt), D4 equivariance, and the central-force
semantics test (`v == sum_j min-image(x_j - x_i)` exactly, with `pot` forced
to 1 and inner coord updates zeroed) all now PASS at `atol=1e-3` — the
frame-mixing defect is fully fixed and verified two independent ways (an
untrained random-init model and the old trained checkpoint loaded into the
fixed architecture). Phantom-edge audits probing production `compute_edges`
also pass, confirming the box-`L` fold removal is correct.

**`test_no_phantom_edges_knn24` root-cause (FIXED):** the original test used a
**standalone reimplementation** `_phantom_count` of the old fold/raw comparison;
it never called `EGNN_dynamics.compute_edges`, so Step 1b's code fix could not
change its outcome by construction. The test helpers have now been corrected to
probe the **PRODUCTION code path** (`_production_phantom_count`): the helper
contiguates the cloud matching the production flow, then feeds it directly to
`dyn.compute_edges()` and counts edges whose RAW separation in the contiguated
frame exceeds the cutoff. This catches any future re-introduction of box-`L`
folding. With the fixed production `compute_edges` (box-`L` fold removed,
raw distance only), the audits are now **6/6 pass**.

### Step 3 — existing exactness suite rerun (`test_ka_cluster_egnn.py`)

`python -m pytest liquid_coupling_flow/tests/test_ka_cluster_egnn.py -v`

**8/8 pass** (19.69s), including the two load-bearing checks named in the
brief:

```
test_build_cloud_shape_and_cluster_first PASSED
test_base_logp_closed_form PASSED
test_sigma_b_from_data PASSED
test_sampler_equals_scorer PASSED
test_not_translation_invariant PASSED
test_perparticle_divergence_matches_bruteforce PASSED
test_ot_species_and_cost PASSED
test_train_smoke_and_load PASSED
======================== 8 passed, 7 warnings in 19.69s ========================
```

`test_perparticle_divergence_matches_bruteforce` passing unchanged confirms
the Step-1 comment's claim (`d(com)/d(x_i)=0` so `d(diffij)/d(x_i)=-I` still
holds exactly, analytic divergence unaffected by the frame fix).
`test_sampler_equals_scorer` (atol 3e-1) and `test_train_smoke_and_load`
(real 60-step GPU training run) both pass. `test_not_translation_invariant`
also correctly still passes — it tests a different invariant (moving the
cluster relative to a fixed cage must change `log_q`), orthogonal to the
global-shift invariance this fix restores.

### Step 4 — dependents of `egnn_traceable`

`grep -rn "egnn_traceable" /mnt/ssd/GridTransformer --include="*.py" -l`

```
liquid_coupling_flow/ka_cluster_egnn.py
```

Single dependent. (Checked for near-miss confusion: `liquid_coupling_flow/ipl44/joint_flow.py`
imports `EGNN_dynamics` too, but from the sibling non-traceable module
`learndiffeq.particles.velocities.egnn` — a different file, unaffected by
this fix.)

**Flag:** any periodic-branch results produced through this backend before
this fix (e.g. IPL44 exact-div ESS evaluations, if run through this same
`egnn_traceable` module) were computed under the old, frame-mixed semantics.
Checkpoints *trained* under the old semantics are incompatible with the fixed
code (weights learned to cancel the frame leak) — this is why
`ka_cluster_egnn_N100.pt` is being preserved as
`ka_cluster_egnn_N100_framebug.pt` before retraining. Not rerun here — out of
scope for Task 3.

### Step 5 — old checkpoint preserved

```
cp liquid_coupling_flow/artifacts/ka_cluster_egnn_N100.pt liquid_coupling_flow/artifacts/ka_cluster_egnn_N100_framebug.pt
```
Done; both files present in `liquid_coupling_flow/artifacts/`.

### Steps 6-7 — retrain + re-gate (deferred)

Skipped per controller instruction — these run separately (hours-long
background job). Results (loss curve vs. old 0.64 plateau, gate clash/g_BB
numbers, old-vs-new A/B split) will be appended to this section after
training completes.

## Task 2: diagnostic harness

Created `liquid_coupling_flow/ka_cluster_egnn_diag.py` with six CLI subcommands
(split | nocage | cagesens | losst | traj | overfit) for bottleneck analysis.
Harness shakeout on the framebug checkpoint to establish the diagnostic
infrastructure and record a baseline reading:

```
python -m liquid_coupling_flow.ka_cluster_egnn_diag split ka_cluster_egnn_N100_framebug.pt
```

Output (n_steps=8, B=128, 20 seeds sampled at stride 5):

```
ckpt ka_cluster_egnn_N100_framebug.pt step 15000 loss 0.642 n_steps 8
intra: clash<0.7  23.3%  mean 0.92   (TRUE 0.0% / 1.00)
cage : clash<0.7  34.4%  mean 0.75   (TRUE 0.0% / 0.88)
overall: clash 51.6%   [EGNN-15k 52 / A-sharp 44 / B 55 / data 0]
parity among intra clashes: same 43% (43% by construction)
```

**Note on old-weights-under-new-code:** The recorded framebug-checkpoint
baseline (51.6% overall clash) is numerically close to the old-code record
(52%), but this is NOT a direct reproduction. The backend fix (frame-mixing
and phantom-edge corrections in `egnn_traceable.py`) changed the model
semantics; the old checkpoint's weights learned to compensate for the old
bugs, so they produce different readings under the fixed code. The two sets
of numbers are not comparable at face value. This is expected and documented
here as the old-weights/new-code caveat per the brief. Metric
cross-validation (split's overall clash % vs. the controller's gate_measure
on the same checkpoint) happens in Task 3 Step 7.

## Task 4: nocage edits

Implemented the no-context ablation (n_cage=0) plumbing in
`liquid_coupling_flow/ka_cluster_egnn.py`, per the Task-4 brief, Steps 1-4 (code
+ smoke tests only — the real nocage training run and its evaluation, Steps
5-6, are deferred to the controller):

- **Step 1** (`build_cloud`): with `n_cage=0` the cage is empty and
  `cage_centroid` is NaN (mean over zero cage members), so the base anchor now
  falls back to the cluster centroid `ccen` when `n_cage == 0`, else the
  original `cage_centroid(cage, L)`. Note this anchor leaks the true cluster
  position, so nocage results below speak only to intra (k=7 mutual)
  exclusion, not to a fair generative test.
- **Step 2** (`train()`): added keyword-only args `tag="", n_configs=None,
  fix_seed=None, use_augment=True, resume=False` at the end of the signature
  (all defaulted, existing calls unchanged). Four body edits: (a) optional
  `data = data[:n_configs]` truncation for the overfit probe; (b) checkpoint
  path now `ka_cluster_egnn{tag}_N{train_N}.pt` with an optional `resume`
  load of an existing checkpoint's `state_dict` before training starts; (c)
  the per-step batch now honors `use_augment` (skips `augment()` when False)
  and `fix_seed` (pins the cluster seed instead of randomizing it); (d) the
  final save print now reports the actual resolved `ckpt` path instead of a
  hardcoded `ka_cluster_egnn_N{train_N}.pt` string.
- **Step 3** (`__main__`): added `nocage` mode (`n_cage=0, max_neighbors=None,
  batch=256, tag="_nocage"`, 15000 steps) and `overfit` mode (`n_configs=4,
  fix_seed=5, use_augment=False, tag="_overfit"`, 6000 steps, batch=64),
  alongside the existing `gate` and default-train dispatch.

No deviations from the brief were needed — both smoke-test steps passed
against the brief's code as given, no minimal fixes required.

### Step 4 — smoke tests

**60-step nocage train (`save=False`)**, exercising the NaN-anchor path:

```
python -c "from liquid_coupling_flow.ka_cluster_egnn import train; train(steps=60, n_cage=0, max_neighbors=None, batch=32, save=False)"
```

```
EGNN-FLOW train N=100 steps=60 k=7 n_cage=0 batch=32 sigma_b=0.944 params 0.13M
  step     0 fm-loss 1.3307 0s
```

`sigma_b=0.944` (matches the expected ≈0.94), 60 steps ran to completion,
loss finite (1.3307), no NaN. Confirms the `ccen` fallback anchor is wired
correctly for `n_cage=0`.

**Existing exactness suite** (unchanged defaults, so all should still pass):

```
python -m pytest liquid_coupling_flow/tests/test_ka_cluster_egnn.py -v 2>&1 | tail -5
```

```
liquid_coupling_flow/tests/test_ka_cluster_egnn.py::test_ot_species_and_cost PASSED [ 87%]
liquid_coupling_flow/tests/test_ka_cluster_egnn.py::test_train_smoke_and_load PASSED [100%]
======================== 8 passed, 7 warnings in 34.68s ========================
```

**8/8 pass.** `test_train_smoke_and_load` (a real 60-step GPU train+load
round trip under the unmodified default `train()` path) confirms the new
keyword-only args are fully backward compatible.

### Steps 5-6 — nocage training + evaluation (deferred)

Skipped per controller instruction — the real `nocage` training run
(`python -u -m liquid_coupling_flow.ka_cluster_egnn nocage`) and its
evaluation (`python -m liquid_coupling_flow.ka_cluster_egnn_diag nocage`,
intra-clash/mean-minr decision rule) run separately and will be appended to
this section by the controller after completion.

## Task 3 Steps 6-7: retrain + re-gate (controller, 2026-07-02)

Retrain under fixed semantics, identical budget (15k, batch 64, kNN-24): FM loss 1.39 -> 0.46-0.53 band
(framebug run plateaued 0.64 => ~28% lower floor: capacity freed from cancelling the position leak).

Gate + split on the retrained ckpt (figure: liquid_coupling_flow/artifacts/ka_cluster_egnn_gate_N100.png):
- overall single-cluster clash **18.1%** (framebug 52 / A-sharp 44 / B 55 / data 0)
- intra 24 -> **7.0%** (mean 0.95, TRUE 1.00); cage 35 -> **12.8%** (mean 0.83, TRUE 0.88); parity clean 42%~43%
- harness cross-validation: gate_measure clash_pct 18.3% == split overall 18.1% (within noise) => harness validated
- 3.3b g_BB "peak" 12.14 = **r->0 core collapse under iterated self-conditioned resampling** (pure-Gibbs collapse,
  the known full-cage-lever-needs-energy mechanism, now EXPRESSED because the conditional is sharp enough to
  contract; one-step true-cage histogram sits on data). MH energy acceptance is the guard by construction.

VERDICT: the frame bug was LOAD-BEARING for the precision wall. Decision rule "clash well below 44%" met.

## Task 6: conditioning probes (retrained ckpt, controller, 2026-07-02)

- cagesens: |dv|(nearest cage) / |dv|(farthest) ~ 8-10x at t=1 (0.42-0.46 vs 0.04-0.06; |v|~0.55) and ~8x at
  t=0.5 => cage conditioning ALIVE and strongest late — conditioning pathway is NOT the bottleneck.
- losst (FM sq-err, t x dist-to-cage): U-shaped in t; minimum mid-flow (~0.2-0.4 at t=0.55-0.75), sharp rise at
  t=0.95 in EVERY distance bin (0.85/0.90/1.55) => residual deficit = LATE-TIME SHARPENING, global not near-cage.
- traj (clash vs t, n_steps=32): monotonic resolution from base (intra 51%, cage 70%) to (6.6%, 12.7%) at t=1,
  no overshoot; endpoint == n_steps=8 split values => integrator resolution NOT the limiter, the field is.

READING: remaining 18% is a late-t field-sharpness deficit with healthy conditioning. Targeted lever = Task 8
analytic repulsive prior (supplies the steep short-r term at t->1); softer lever = Task 7 longer training
(FM loss still descending at 15k).

## Task 4: no-context ablation RESULT (controller, 2026-07-02)

nocage flow (n_cage=0, base at TRUE cluster centroid, full intra attention, 15k/batch256; FM loss noisy 1.2-1.8):
- intra-clash 28.1/27.6/27.4% at n_steps 8/32/64 (flat), mean-minr 0.95 (TRUE 0.0%/1.00)
- WITH-CAGE retrained model: intra-clash 7.0% => removing context made intra-exclusion 4x WORSE
- pair-distance histograms (liquid_coupling_flow/artifacts/ka_egnn_nocage_intra.png): data's sharp AA/AB/BB
  first-shell peaks smeared into broad humps

INTERPRETATION (inverts the plan's naive decision rule): no-context is NOT the easy capability baseline — a
free k=7 cluster has unbroken rotational multimodality, so the OT-CFM marginal velocity field averages over
modes and blurs (loss can't converge: irreducible conditional entropy). The CAGE is symmetry-BREAKING
information that collapses the conditional to near-unimodal, which is exactly where the with-cage flow
sharpens to 7%. Conclusion: the CNF CAN carve exclusion when the conditional is well-posed; context is not
the obstacle but the enabler. Bottleneck localization stands: late-t field sharpness (Task 6), levers = Task 8
physics prior / Task 7 longer training.

## Task 5: overfit probe RESULT (controller, 2026-07-02)

4 configs x seed-5 cluster, no augment, 6k steps: intra 8.7% / cage 11.9% / overall 17.5% / |sample-true(OT)| 0.399.
OVERFIT (17.5%) == GENERALIZATION (18.1%): the residual is NOT a generalization gap — the field class + OT-CFM
objective floors at ~18% even when memorizing 4 fixed cages (a perfectly overfit flow would map every base draw
onto the single observed cluster: clash->0, |sample-true|->0). Confirms Task 6's late-t sharpness deficit as a
STRUCTURAL floor of the learned field. DECISION: Task 7 (longer training) de-prioritized (cannot beat the floor
memorization already hits); Task 8 (analytic species-pair repulsive prior, exactness-preserving) is the lever.

## Task 8: repulsive prior (code)

Added an OPTIONAL analytic species-pair repulsive prior to the `pot` central-force scalar in
`egnn_traceable.py`: `pot <- pot - softplus(A[sp_i,sp_j]) * (sig_pair[sp_i,sp_j]/r_ij)^12`, with
`sig_pair` fixed to `ka_energy.SIGMA = [[1.0,0.8],[0.8,0.88]]` and `A` a learnable `n_species x n_species`
matrix initialized at `-4.0` (`softplus(-4)≈0.018`, a gentle start). The term is applied INSIDE the same
`torch.enable_grad()` block on the same `rij` tensor that `dpotdr = torch.autograd.grad(pot.sum(), rij, ...)`
differentiates, and BEFORE the `[B,P,nb,1]` reshape, in all three pot sites (`forward`,
`forward_and_divergence`, `forward_and_perparticle_divergence`) — autograd picks up the prior's analytic
derivative for free, so the central-force divergence identity is preserved exactly.

**Backend changes** (`liquid_coupling_flow/ipl44/learndiffeq/learndiffeq/particles/velocities/egnn_traceable.py`):
- `EGNN_dynamics.__init__`: `self.rep_prior = bool(kwargs.get("rep_prior", False))`; when True, creates
  `self.rep_scale` (learnable `[n_species,n_species]`, init -4.0) and registers buffer `self.sig_pair`.
  When False, NEITHER is created (guards `load_state_dict` strict-mode compatibility with existing checkpoints).
- `_compute_common_terms`: returns two new flat `LongTensor` entries, `a_central_idx` / `a_nbr_idx`
  (`None` when `n_species == 1`, matching the existing `a_nbr`/`a_central` None-guard pattern).
- New helper `_apply_rep_prior(self, pot_flat, rij, common)`: no-ops (returns `pot_flat` unchanged) when
  `self.rep_prior` is False OR either species-index tensor is None; otherwise gathers `sig`/`amp` by
  `(a_central_idx, a_nbr_idx)` and subtracts the repulsive term. [SUPERSEDED by 3c8ad51 + c97299c: the shipped
  form is `amp * t^8 * (sig / torch.maximum(rij, 0.6*sig))**12` — per-pair 0.6σ clamp (the original absolute
  0.05 clamp bombed FM training) and a g(t)=t^8 late-time gate; see the retrain sections below.]
- Call sites: `forward` (split the inline `pot_model(...).reshape(...)` into `pot = pot_model(...)`;
  `pot = self._apply_rep_prior(pot, rij, common)`; `pot = pot.reshape(...)`), `forward_and_divergence`
  (parity), `forward_and_perparticle_divergence` (the one actually used by `ConditionalEGNN.vel_div`) —
  all applied before their respective reshape, inside `enable_grad()` where applicable.

**Plumbing** (`liquid_coupling_flow/ka_cluster_egnn.py`): `rep_prior=False` added to
`ConditionalEGNN.__init__` (passed to `EGNN_dynamics`), `EGNNClusterFlow.__init__` (passed to
`ConditionalEGNN`), `train()` (passed to `EGNNClusterFlow` and included in the saved `arch` dict); added
`"rep_prior"` to `_ARCH`. `load_flow` builds `arch` as `{kk: ck[kk] for kk in _ARCH if kk in ck}`, so old
checkpoints (no `rep_prior` key) silently default to `False` — verified below.

**Test evidence:**

1. `python -m pytest liquid_coupling_flow/tests/test_ka_cluster_egnn.py -v` — **8/8 passed** (defaults-off
   regression; includes a real GPU `train()` round trip in `test_train_smoke_and_load`, confirming the new
   keyword-only `rep_prior` arg is fully backward compatible).

2. Added `test_divergence_exact_with_rep_prior` (brief Step 3) to `test_egnn_audit.py` and ran the full file:
   `python -m pytest liquid_coupling_flow/tests/test_egnn_audit.py -v` — **7/7 passed**, including the new
   test: brute-force Jacobian trace vs analytic `vel_div` divergence agree to `< 1e-4` with `rep_prior=True`
   (double precision, random-init weights, `n_layers=2`, `max_neighbors=24`) — the prior's analytic `-12/r`
   derivative is correctly picked up by autograd; exactness is preserved with the prior ON.

3. Old-checkpoint load check: loaded `liquid_coupling_flow/artifacts/ka_cluster_egnn_N100.pt` (no
   `rep_prior` key in the saved dict) via `load_flow` — `rep_prior` defaults to `False`, no `rep_scale`/
   `sig_pair` are instantiated, `load_state_dict` (strict) succeeds, and a `sample_fast` call on real
   scaffold data produces a finite `[4,7,2]` tensor. Confirms the flag is fully additive: existing
   checkpoints are unaffected.

Also spot-checked that with `rep_prior=False` no new parameters/buffers appear on a fresh
`ConditionalEGNN` (`state_dict()` key count unchanged, `hasattr(egnn, "rep_scale")` / `"sig_pair"` both
False) — the defaults-off path is byte-identical to pre-Task-8 behavior.

Step 4 (the `rep_prior=True` retrain + `diag split`/`diag nocage` decision) is deferred to the controller
per the task scope; this section covers code + exactness evidence only.

Commit: `dde6de8` (follow-ups: `3c8ad51` 0.6σ clamp, `c97299c` t-gate).

## Task 8: STATIC prior retrain RESULT (controller, 2026-07-02) — NEGATIVE

Bounded static prior (0.6*sig clamp, amp init 0.018), 15k/batch64: FM plateau ~0.88-1.03 (no-prior: 0.46-0.53);
split overall 50.3% / intra 20.9 / cage 35.2 (no-prior: 18.1/7.0/12.8); learned amps CRUSHED to 0.006-0.009.
The optimizer rejects a t-STATIC repulsion: the true FM field is transport-dominated for most of t, so an
always-on repulsive term contaminates the trajectory everywhere while helping only at t>0.85 (losst). Design
violated the tree's own localization. NEXT (single further iteration): t-gated amplitude g(t)=t^8 so the prior
fires only in the late sharpening regime. Note first attempt with absolute 0.05 clamp bombed FM outright
(step-0 loss 1.3e22, fixed in 3c8ad51).

## Task 8 FINAL: t-gated prior RESULT (controller, 2026-07-02) — MILDLY POSITIVE, adopted as best model

g(t)=t^8 gate (c97299c), 15k/batch64: FM plateau 0.476 (== no-prior 0.46 — optimizer conflict GONE);
split overall **16.8%** / intra 6.1 / cage 12.3 (no-prior 18.1/7.0/12.8) — consistent ~1pp gains across all
metrics. Learned amps: AA 0.013, AB 0.010/0.020, BB **0.026** (GREW from 0.018 init) — the prior is used most
for B-B, the campaign's historically weakest exclusion. CONCLUSION: t-gating turned the prior from harmful to
mildly helpful; the residual ~17% floor is NOT missing short-range repulsion but marginal-field smearing (the
per-config conditional fields differ; the FM marginal blurs them). Best checkpoint: ka_cluster_egnn_prior_N100.pt.

## EXIT (plan criterion 1+2): bug-driven breakthrough + located wall + measurably-moved lever
52% -> 16.8% overall clash (data 0%). MH cluster kernel GO discussion is open (exact log_q + energy acceptance
also cures the 3.3b Gibbs collapse). Remaining gap = late-t marginal smearing; further levers (beyond scope):
per-config conditioning capacity, MLE fine-tune atop FM, or accept-and-let-MH-filter at ~17%.

## Follow-up: prior exactness coverage-hardening (2026-07-02)

Two tests added (16/16 pass):
- test_divergence_exact_with_rep_prior HARDENED: now checks analytic==brute-force at t=0.37 AND t=0.90
  (gate g(t)=t^8 open, prior active), + a liveness assert (prior changes the divergence at late t) so the
  exactness check is non-vacuous. rep_scale fixed to 0.0 (amp 0.69) for an unambiguous prior.
- test_sampler_equals_scorer_with_rep_prior NEW: full RK4 sample->log_q round-trip with the prior on at the
  deployed amplitude (0.019).
FINDING (recorded in the test docstring): the prior DEGRADES the RK4 log_q round-trip, residual ~1.0 (prior on,
deployed amp) vs ~0.1-0.3 (prior off) — and NON-monotonic in n_steps (64~=32). The divergence is exact pointwise;
the r^-12 field is just stiff for fixed-step RK4. Gross log-det/base/sign bug scale is O(10-300) (15x amp -> ~300).
IMPLICATION: an EXACT MH/IS kernel with the prior ON needs an adaptive/reversible integrator, OR use the prior-OFF
model (18.1% clash, log_q round-trips to ~0.2) as the exact proposal. The g(r) gate is unaffected (sample_fast, no
log_q). So the "best model" split: prior-ON = best structural sampler (16.8%), prior-OFF = best EXACT-log_q proposal.

## Follow-up: reversible integrator option (2026-07-02)

Added EGNNClusterFlow integrator="midpoint" (implicit midpoint, time-reversible) alongside the default "rk4".
It REUSES the existing analytic divergence (vel_div) — the augmented [x, logdet] is stepped by the reversible
scheme; div is evaluated at the shared MIN-IMAGE midpoint so forward/reverse log-dets cancel exactly. It does
NOT use a discrete-map Jacobian (that would abandon the cheap divergence).

MEASURED sample==log_q round-trip (prior-off warmed flow, B=4):
  DOUBLE n_steps=32:  midpoint 4.3e-14   vs   rk4 4.1e-3   (11 orders tighter — reversible win confirmed)
  DOUBLE n_steps=16:  midpoint 3.3e-2 (WORSE) — Picard does not contract at h=1/16
  FLOAT32 n_steps=32: midpoint 6.4e-3 (~= rk4) — float32 floors the Picard solve at ~1e-7
REQUIREMENTS: (1) double precision; (2) enough steps for Picard contraction (h*Lip(v)<1, n_steps>=32 here).
NOT for the stiff rep_prior-on field (Picard diverges: midpoint 21 vs rk4 1.6) -> that needs implicit/Newton.
USE: exact-log_q proposal for the MH/IS kernel on the prior-OFF model (ka_cluster_egnn_N100.pt), run in double.
Default integrator stays rk4 (checkpoints/behavior unchanged). Test: test_reversible_midpoint_exact_roundtrip.

## 2026-07-03: USER CHALLENGE -> "structural floor" OVERTURNED (positive control + capacity probe)

User challenged the residual-clash conclusions ("EGNN should learn LJ7; suspect a bug"). Two controls:
- **E2 LJ7 positive control** (scratchpad lj7_control.py, ckpt artifacts/lj7_control.pt): exact Boltzmann LJ7
  data (2D, T=0.2, 102k configs via vectorized Metropolis, ka_energy; data min-r 1.105, 0.00%<0.85) trained with
  the IDENTICAL production core (ConditionalEGNN backend, ot_assign, OT-CFM loop, RK4) at the production size
  (0.13M): generated min-r 1.084, **P(<0.85) 0.66-0.88%** (flat n_steps 8/32/64). CORE EXONERATED — same
  machinery learns LJ7 essentially cleanly at the same capacity that fails the glass task (nocage 27%).
- **E3 capacity probe**: overfit-4-configs with hidden 192 / 6 layers (1.64M params, batch 16, 6k steps):
  overall clash **17.5% -> 7.1%** (intra 8.7->3.6, cage 11.9->4.5), |sample-true(OT)| 0.399->0.168, loss
  0.31->0.148 — STILL DESCENDING. The small model's overfit==generalization coincidence was because CAPACITY was
  the binding constraint in both regimes, NOT a task wall.

CORRECTED DIAGNOSIS: no core bug; the dense-glass conditional velocity field needs far more capacity than 0.13M
(task-complexity-dependent); the late-t sharpness deficit is what under-capacity looks like. Residual FM tail is
never exactly 0 (LJ7 shows ~0.7% at 0.13M) but scales down with capacity/task. LEVER = SCALE. Production big run
(hidden 192 / 6 layers, full data, 20k steps, tag=_big) launched 2026-07-03.

## 2026-07-03: BIG production model (1.64M, hidden 192/6 layers, full task, 20k steps batch 16)

Split (single-cluster resample given TRUE cage, n_steps 8): intra 6.8% / cage **7.8%** / overall **13.3%**
(small 0.13M: 7.0/12.8/18.1; +prior 16.8). Gain concentrated in CAGE exclusion (mean min-r 0.85 vs data 0.88) —
the component once called the "fundamental wall". Trajectory 52 -> 18.1 -> 16.8 -> 13.3, still descending with
capacity (batch-16 run, likely under-trained). Ckpt artifacts/ka_cluster_egnn_big_N100.pt.

## 2026-07-03: MH acceptance with the BIG model (fp32, rk4-64, 640 moves)

small 0.13M -> big 1.64M: mean acceptance **0.64% -> 1.02%**; no-clash fraction 35.9 -> **50.0%**;
acceptance|no-clash 1.79 -> 2.03%; logq_rev-logq_fwd 1.12 -> 0.20; P(dU<0) 0.8 -> 2.7%.
Self-consistency (fp32): big-model max 0.36 vs small 1.49 — the higher-capacity field is also SMOOTHER to
integrate (further under-capacity corroboration). READING: capacity fixed hard-core avoidance (50% clash-free
proposals) but acceptance-given-no-clash barely moved — the binding constraint is now FINE ENERGY PRECISION
among valid placements (clash-free proposals still land a few kT above the basin; beta=2 exponentially
unforgiving over a 7-particle simultaneous move). Plain-MH viability would need acc|no-clash to rise ~5-10x;
levers: more capacity/training (curve still descending), smaller k, or (validated) SMC/annealed correction
where single-shot acceptance does not gate. NOTE fp64-on-GeForce lesson: the estimator ran 1:64-throughput
double precision unnecessarily (acceptance is energy-dominated); fp32 rerun ~20x faster, same conclusion.

## 2026-07-03: ARM-H (spline head only, old scale) — NULL, attribution secured

ka_cluster_flow_armH_N100.pt (spline head, d192/L4/ctx32, 1.92M, 15k): clash 53.3% (bins-40k: 44%; bins-15k-small:
70%), seed spread 0.79-0.82 ~ nn-dist, g_BB(3.3b) 1.26. The head swap alone does NOT unlock the AR generator —
REPLICATES the 2026-06-26 full-config null in the cluster setting: the transformer-context conditional is diffuse
regardless of head expressiveness. Attribution: A's wall = context/representation pathway (+possibly scale), NOT
the output head. ARM-FULL (pair features + d256/L6 + 40k) is the arm that tests exactly that lever.
Figure: liquid_coupling_flow/artifacts/ka_cluster_armH_gate.png

## 2026-07-03: ARM-FULL verdict — AR line CLOSED with full attribution

ka_cluster_flow_full_N100.pt (spline + per-step radial/LJ pair features + d256/L6/ctx32, ~6.5M, 40k):
**clash 32.7%** (best AR ever: bins-40k 44, ARM-H 53.3), -logq/k plateau ~-1.8 to -2.2 (2 nats/particle sharper
than ARM-H — the pair-feature/scale lever is real). BUT: misses tier 1 (<30), far from tier 2 (18); EGNN-big is
13.3% at 1/4 the params. NOTE: the printed seed-spread (0.74-0.78) is from gate_measure's aux probe which does
NOT apply pair features (known Minor) — discount it; the clash metric uses P.sample() and is correct.
ATTRIBUTION COMPLETE: order sound; head not the constraint (ARM-H null); context is the constraint, and explicit
distance features + scale through a transformer close only part of the gap — the EGNN's equivariant distance-based
message passing is the load-bearing ingredient for precision placement in the glass. Per spec gates: no MH
acceptance rerun. AR's exact-1-forward log_q remains attractive ONLY if some future context pathway reaches
EGNN-level clash. Figure: liquid_coupling_flow/artifacts/ka_cluster_full_gate.png

## 2026-07-03: ARM-FULL MH acceptance (user follow-up)

640-move protocol: mean **0.308%** (EGNN small 0.64 / EGNN big 1.02). Decomposition: clash-free 10.3% (vs 50%
EGNN-big — pure per-particle compounding, (0.673)^7) BUT acceptance|no-clash 2.99% > EGNN-big 2.03%, and
logq_rev-logq_fwd = +4.26: the AR is diffuse-but-CALIBRATED (assigns the true cluster far higher density than
its own samples; clash-free draws are energetically slightly better than the EGNN's). Bottleneck = hit rate,
not energy quality or density calibration.

## 2026-07-03: MTM cluster kernel — BUILT, VALIDATED, economics measured

K1 Barker/Tjelmeland ensemble independence-MTM over the ARM-FULL spline proposal (ka_cluster_mtm.py, b033f1d;
spec 2026-07-03-cluster-mtm-design.md; review: detailed-balance arithmetic verified airtight).
VALIDATION (exactness-by-stationarity, B=64 chains):
- from the FIRST-64 reference slice (found to be 0.04/N SHALLOW — early PT frames): kernel relaxed it ONTO the
  true ensemble mean (-3.2614 vs file U_per_N -3.2600) holding g_BB — correct behavior initially misread as drift;
- from a RANDOM (representative) slice: non-monotone HOLD, max excursion 0.024 ~ reference plateau_drift 0.0155,
  ending -0.013 and shrinking; g_BB flat 2.21-2.29 (data 2.30);
- K3 negative control (SIR w/o current state, M=2): U/N explodes to ~1e10 within 5 sweeps — test sensitivity proven.
ECONOMICS: move_prob 2.6/5.5/16.2% at M=8/32/128 on the shallow slice; ~26s per ACCEPTED move roughly FLAT in M
(batched AR forward scales linearly — no free lunch, MTM converts compute->acceptance at fixed rate). At TRUE
equilibrium move_prob is lower (M=8: ~0.8-1.5% settled) — deeper minima are harder to beat.
OPEN (the decisive follow-up): mixing-vs-swap-MC benchmark — does a ~26s accepted 7-particle coordinated move buy
more cage-escape/decorrelation than the same wall-clock of single-site swap-MC. Lesson logged: stationarity gates
need REPRESENTATIVE initial slices (first-frames of a PT reference are shallow).

## 2026-07-03: MTM-vs-swap MIXING BENCHMARK (the decisive economics gate) — QUALIFIED POSITIVE

3 arms, matched 900s wall-clock, B=64 chains from the identical shallow slice (U/N -3.2194 -> eq -3.2600),
positions-only metrics (fair to species swaps). v1 was INVALID (slot_order gives per-row slot->species maps;
vectorized swap displacement used row-0's -> U/N went POSITIVE) — v2 keeps all state in ORIGINAL ordering
(single exact species vector) and slot-orders per MTM move on the fly, scattering the cluster back.

RESULTS (final): swap U/N -3.2696 / Q 0.277 / MSD 0.507 | pure-MTM -3.2452 / 0.623 / 1.135 |
HYBRID -3.2764 / 0.262 / **0.927**.
- Pure MTM alone: loses energy+overlap (too few moves/second) but its MSD is 2.2x swap's — accepted cluster
  moves are genuine coordinated hops, not rattle.
- HYBRID: matches-or-edges swap on U/N and Q (differences ~ noise) while carrying **1.8x swap's MSD** — the
  cluster channel relocates small subsets FAR (Q barely changes, MSD doubles = rare large coordinated
  displacements, exactly the transport mode swap lacks positionally).
- SYNERGY: MTM move_prob in the hybrid holds ~8% vs ~3% in pure MTM at the same M=32 — interleaved swap sweeps
  keep refreshing local environments so cluster candidates land better.
VERDICT: as a REPLACEMENT kernel MTM is uneconomical; as a SUPPLEMENT inside swap-MC it adds a real mobility/
transport channel at no energy cost at matched wall-clock. Worth carrying into the SMC/equilibration stack
wherever positional decorrelation (not just energy) is the bottleneck. Caveats: single run/arm; N=100; energy
and Q edges within noise — the MSD factor is the robust finding.

## 2026-07-04: EQUILIBRIUM DECORRELATION from the full trajectories — verdict REFINED

Time-origin-averaged self-overlap Q(dt) + MSD(dt), origins t>300s, vs WALL-CLOCK lag (figure
artifacts/mtm_decorrelation.png; analysis reports/logs-2026-07-03/decorr_time.py):
- tau_alpha > 560s for ALL arms (no 1/e crossing in-window; T*=0.5 glass).
- Q at 560s: swap 0.50 < hybrid 0.66 < mtm 0.80 — AT EQUILIBRIUM plain swap decorrelates the typical particle
  FASTEST per wall-clock (hybrid's MTM moves, acceptance ~3%, do not repay their ~50% budget on this metric).
- MSD at 560s INVERTS: mtm 0.50 (3x swap) > hybrid 0.27 > swap 0.16 (visible caging plateau) — MTM adds a
  heavy-tailed displacement channel (rare 7-particle hops), i.e., strongly heterogeneous dynamics: Q counts the
  majority, MSD the tail.
REFINED VERDICT: hybrid's edge lives in the RELAXATION regime (matched energy + 2x transport + elevated 8%
acceptance); at EQUILIBRIUM, swap wins standard Q-decorrelation while MTM/hybrid win long-range transport 3x/1.7x
(basin-hopping channel; matches the deep-energy-tail hint in the final distributions). Use the hybrid for
equilibration + basin exploration; plain swap for equilibrium Q-throughput. Caveats: mtm window not fully
equilibrated (-3.24 vs -3.26); absolute tau needs multi-hour runs.

## 2026-07-04: AR-SEEDED RELAXATION + TWO-TIME AGING ANALYSIS (figures aging_two_time.png)

Arms (900s matched wall-clock, B=64): rand+swap, rand+hybrid, ar+{swap,mtm,hybrid} (AR = flowhead full-config
samples, start clash 30.2%, U/N ~1e14; rand: 84.6%, ~1e22). Full trajectories artifacts/arstart_traj_*.pt.
- CLASH REPAIR IS FREE: swap alone removes ALL clashes in seconds (astronomical dE => near-certain accepts) —
  the cluster-move defect-repair niche does not exist. Hybrid slightly BEHIND swap on energy from BOTH starts
  (rand -2.89 vs -2.93; ar -2.99 vs -3.02) => the earlier "hybrid = relaxation tool" holds only NEAR equilibrium
  (shallow-slice regime), not for deep quenches. MTM niche narrows to: near-eq relaxation + heavy-tailed transport.
- AR SEEDING: REAL head start in BOTH energy (-3.016 vs -2.931 at 900s, ~0.09/N durable) and DYNAMICAL AGE
  (tau_0.7 at t_w=30s: ar+swap 143s vs rand+swap 71s ~ 2x older at matched wall-clock age).
- AGING QUANTIFIED: tau_0.7(t_w) seconds -> >300s within ~100s of age for all quenched starts; equilibrium
  anchor tau > 560s. ar+hybrid ages fastest (298s @ t_w=30) while LESS deep => freezing-without-settling.
- None of the arms reaches equilibrium (-3.26) within 900s from any quench (deep glassy tail is the wall; the
  PT reference needed 12k swap-sweeps + tempering).
Caveats: single run/arm; one origin per (arm,t_w); eq+swap tau@600s=223s is single-origin noise.

## 2026-07-04: SWAP-AND-BREATHE — the equilibrium species channel is OPEN (G1+G2 PASS)

Kernel ka_swap_breathe.py (241e422; spec 2026-07-04-swap-and-breathe-design.md; review: DB math verified incl.
transposition-symmetry + abort-multiset argument; "every accepted move is an exchange" proven a theorem).
- **G1: 1563/1,024,000 = 0.153% equilibrium species EXCHANGES vs measured baseline EXACTLY 0/102,400** for
  position-preserving swaps. Steady per-sweep counts (11-24/12,800). mean_dlogq +5.5 (diffuse-but-calibrated);
  huge mean_dU from clash-rejected candidates (the proposal's 32.7% clash tax). v1 single-try, uniform seeds —
  I-MTM (M-candidates) and denoiser-guided seeding are staged multipliers.
- **G2: stationarity + species observables HOLD** (50 rounds x [100 disp sweeps + 1 SB sweep], B=64 random
  slice): U/N non-monotone band 0.021; g_BB 2.22-2.41 ~ data 2.30; g_AB 6.13-6.32; BB-contacts 17.3-18.2 flat
  => no species-composition bias. SB acceptance steady 0.09-0.28%.
Logs: reports/logs-2026-07-04/sb_g1.out, sb_g2.out. G3 (tau_s) next.

## 2026-07-04: LOCAL-SMC PILOT — P1/P2 verdicts

Module ka_local_smc.py (2539283 + 3 measured scheduler fixes: warm-init [exact generator-IS weights gave ESS=1
at init — A1's wall at the seed stage], absolute ESS target [relative target let a degenerate population jump
to beta1], resample-on-<=target [strict < deadlocked at a 1e-4 crawl]).
- **P1 (ARM-0, mutation-only annealed SMC, N=100 B=256): MACHINERY PASS, depth partial.** beta 0.8->2.0 in
  23 rungs / 17 min, ESS held at target throughout (A1 fixed). Final: U/N -3.137±0.053, g_BB 2.21 (ref -3.260 /
  2.30). Strict depth bar NOT met (mutation-limited glassy tail) BUT deepest non-tempering result in the
  campaign at this wall-clock (direct AR-seeded: -3.016 @ 15 min).
- **P2 (ARM-1, always-apply IS transport, 50 AR-cluster moves/rung): CATASTROPHIC NEGATIVE, mechanistic.**
  Forced application of a ~90%-clash-per-move proposal poisons all chains (U/N +5.6e9 at rung 0); resampler
  recovers ESS among garbage; ladder completes meaninglessly (final -2.48). VERDICT: IS-style learned transport
  demands proposal quality far beyond MH-grade (MH filters the clash tax; IS weights must absorb it). Learned
  components belong behind accept/reject or as seeds; per-rung-trained transport (AFT/CRAFT) is the only path
  for ARM-1 and is out of pilot scope. Artifacts smc_pilot_arm{0,1}_N100.pt (+ _v1exactinit failure record).

## 2026-07-04: FINAL CONSOLIDATION — SMC P3 + SB upgrades U1/U2 + init dependence

**SMC P3 (ARM-0, N=256 ZERO-SHOT, B=128, all learned parts N=100-trained): TRANSFER PASS.** 33 rungs / 74 min,
ESS at target throughout, no cliff in any component. Final U/N -3.089±0.040, g_BB 2.24. The gap to true
equilibrium is SIZE-INVARIANT (0.12/N at both N=100 and N=256) — "the sampler transfers," including its
mutation-limited depth. Strict depth bar (soft ref -3.211) not met; same mutation-budget knob as P1.
Artifact smc_pilot_arm0_N256.pt.

**SB upgrades (v2 I-MTM + v3 denoiser-guidance, 026d952):**
- U1 @ equilibrium, M=16 uniform: **1.897% = 12.4x v1's 0.153%**, AND cheaper per exchange (~70s vs ~98s
  per exchange/chain) -> the multi-try upgrade converts SB from correctness-tool to cheap.
- U1 M=16 denoiser-scored: 1.369% — v3 NEGATIVE (28% below uniform): at equilibrium the geometry table is
  0.999-confident, "ambiguous pairs" too rare; concentration costs more diversity than it buys.
- U2-short (upgraded kernel): stationarity + BB-contacts HOLD (band 0.016).
- INIT DEPENDENCE (prediction refuted on record): exchange rate equilibrium 1.90% >> flow-init 0.40% ~
  rand-init 0.45% — proposal-OOD dominates off-equilibrium (cold-shaped candidates vs self-consistent
  disordered incumbents). SB = EQUILIBRIUM SPECIALIST: peaks exactly where the channel is uniquely needed.
Logs: reports/logs-2026-07-04/ (sb_u1u2, sb_u2short, sb_u1_inits, smc_p1c/p2/p3).
