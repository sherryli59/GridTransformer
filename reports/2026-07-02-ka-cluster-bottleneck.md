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
