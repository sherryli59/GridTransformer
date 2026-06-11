# GridTransformer code review — branch `hilbert-arc-repr` (2026-06-10)

**Scope:** `git diff main...HEAD` (~10.5k insertions, 55 files) plus uncommitted working-tree
changes (the in-progress arc-length representation, ~270 lines). Reviewed for correctness,
train/sample parity, removed behavior, reuse/simplification/efficiency, and architecture
altitude — then assessed against the project goal: **a size-transferable autoregressive
transformer whose samples have small OT gap to the Boltzmann distribution, so they can be
refined by the EGNN Riemannian Flow Matching repo at `/mnt/ssd/flow_matching/learndiffeq-main`.**

**Method:** 7 independent finder passes (3 correctness angles, reuse, simplification,
efficiency, altitude) producing 38 raw candidates (Appendix A), then per-candidate
verification against the working tree. §3 lists the verified findings ranked by severity;
2 candidates were refuted outright and 2 downgraded.

---

## 1. Executive summary

The branch contains a methodical, well-documented research arc: diagnose why fixed-density
size transfer fails → prove the model cannot learn the Hilbert curve → give it the curve as a
scaffold ("rail") → validate the rail end-to-end on a toy → discover and fix two generation
pathologies (exposure bias from relative-delta targets; discrete input-token bin-crossing) →
run a disciplined two-stage zero-shot K=1 rail transfer study (verdict: **NOT FEASIBLE
zero-shot**, OTgap 1.44/1.63 > 1.0 = worse than uniform noise) → begin the arc-length
representation (`arc_repr`, uncommitted) as the next, principled attempt.

The science is in good shape; the **code is now the bottleneck**. The review confirmed
**10 correctness findings** (§3), and the critical ones cluster in the *uncommitted
arc_repr path* — the very experiment meant to answer the size-transfer question next:
particle 0 is never placed (3.5), a Gaussian-tail fine offset can silently teleport the
chain along the Hilbert curve (3.4), the saved log-prob lacks the arc→position Jacobian
(3.6), and three independent wiring guards are dead or bypassable (3.1, 3.2, 3.9). **If
arc_repr is evaluated before these are fixed, a negative result is uninterpretable** — it
could be any of these bugs rather than the representation. Two further confirmed traps
make default-CLI sampling of rail/arc checkpoints fail or fall back to 2D (3.7, 3.8).

Structurally, the experimental flags (discrete/binned/continuous/MDN-fullcov heads ×
cartesian/polar × factorized × rail{lookahead,fixed_template} × residual-target ×
continuous-input × arc_repr) are threaded as ~10 loose scalars through 6 files with two
parallel model classes, and the highest-risk defect class — train-time vs sample-time
geometry skew — is guarded only by convention (the pow2 resolution rule alone exists in 5
hand-maintained copies). `CURVE_RAIL_REFACTOR_TODO.md` already names the right fix (a
single `CurveRailConfig` carried in the checkpoint); it should be done **before** the
arc_repr experiments multiply the variant matrix again.

**Where the goal stands:** in-distribution N=27 is essentially solved for refinement
(best OTgap 0.12, well inside the flow's green band ≲0.6). Size transfer is not: every
zero-shot attempt scores OTgap > 1 at N=64/125. The audit-validated conclusion is that the
*marginal* features transfer but the learned *conditional* does not — driven by the
absolute-reference rail extrapolating out of the training coordinate range, AR error
compounding over 2.4–4.6× longer sequences, and the pow2 cell-size shift. The roadmap in
§5 follows directly from that diagnosis.

---

## 2. What has been done (the research arc)

### 2.1 Base model and in-distribution success
`GraphormerAR` (grid_transformer/models/transformer.py): a causal transformer over particles
sorted by a 3D Hilbert curve at grid resolution R (default 64 over L=3, cell ≈ 0.047σ).
Per-step target = displacement to the next particle (`RelativeDeltaTokenizer`, binned
discrete) or a continuous MDN head (optionally full-covariance, optionally polar/factorized —
both of those variants underperform; polar_fullcov is at uniform-noise level, OTgap ~1.0).
Edge-bias attention injects pairwise-distance structure; `density_emb` conditions on box
density. **Best in-distribution result: N=27 binned-discrete, OTgap 0.12, gr_L1 0.045**
(`lj_ckpts_lj27_pbc_0423`) — comfortably refinable by the flow.

### 2.2 The refinability benchmark
`benchmark_lj27.py` scores samples for *EGNN flow readiness*, not raw quality: core-clamped
LJ energy (bare r⁻¹² explodes on fixable overlaps), and `OTgap = (cand − self)/(uniform −
self)` so 0 = matches the MCMC target, 1 = as far as uniform noise (the flow's base
distribution). OTgap < 1 is the hard refinability line; empirical bands green ≲0.6 / yellow
0.6–0.8 / red ≳0.8.

### 2.3 Size-transfer failure diagnosed (tests T0–T5)
Zero-shot at fixed density ρ=1 collapses: L3→L4→L5 OTgap 0.09 → 2.34 → 8.77. Two
independent mechanisms (scripts: `generalization_tests.py`, `t5*_*.py`):
1. **Turn tokens unlearned.** At L3, window W=1.5=L/2 means min-image caps every delta, so
   0% of training steps are "turns"; at L4/L5 ~8% are. The lossy `long_jump` token decodes
   to ~0, dropping a particle onto its predecessor.
2. **Local conditional overfits to the L3 system.** Local-token NLL degrades 13× even with
   true coordinates, while the local physics is provably near-identical across L —
   a generalization failure, not a physics wall.

Multi-size training ({L3,L5} hold out L4) fixed the *learning* (turn ppl 1e8→3.5, local ppl
1949→236) but not *generation* (OTgap ~2.26 even on the train size L5): the model knows when
to turn but cannot **express** the turn through the lossy token.

### 2.4 Design pivot: give the curve, don't learn it
Toy A showed a transformer cannot length-generalize the Hilbert recursion (G4→G8 acc 0.43).
Decision: supply the curve as an a-priori scaffold and predict only the L-invariant local
residual; hold **cell size constant** across boxes (R = next_pow2(L/cell)) so traversal
statistics are L-invariant.

### 2.5 Curve rail (D2) built and validated
The rail — K upcoming curve cell-centers, cross-attended per step — is wired through
dataset → cache metadata → model hparams → sampler, with the geometry
(`curve_rail_offsets`/`hilbert_resolution`/`cell_size`) riding in the checkpoint. Two modes:
`lookahead` (sample-dependent) and `fixed_template` (sample-independent waypoints
`decode(j·X)`, X=R³//N; verified identical across configs). Toy validation is decisive:
rail-ON NLL −131 vs rail-OFF +30 (~160 nats).

Two real generation pathologies were isolated on the toy and fixed:
- **Exposure bias from the relative-delta target:** teacher forcing lets `delta=const` fit
  without using the rail; generated chains drift. Fix: `curve_rail_residual_target`
  (predict pos − anchor) — drift 2.2→0.28.
- **Discrete input-token bin-crossing (the acute trigger):** the AR model leans on the
  quantized feedback token; ~0.04 drift flips a bin → under-trained adjacent token →
  cascade at step 2. Fix: `continuous_input` (linear projection of the continuous previous
  displacement) + `continuous_input_noise` scheduled-sampling noise to teach
  self-correction.
- Also fixed: the cache builder *always* applied a torus shift + 90° rotation, silently
  breaking rail==target exactness (now gated by `--no_augment_shift`).

### 2.6 K=1 rail zero-shot transfer study (this branch's headline)
A two-stage gated feasibility study (spec + plan in `docs/superpowers/`, results in
`reports/k1_rail_transfer/`):
- **Stage 1 (audit):** the K=1 fixed-template rail input is ~scale-invariant under pow2 R
  (KS 0.30 @N=64, 0.163 @N=125) and the cartesian-delta target transfers almost exactly
  (KS < 0.024). The `exact` (non-pow2 R) strategy is **disqualified outright** — the Hilbert
  curve fills a 2^bits cube, so R=85/107 leaks 44%/35% of waypoints outside the grid
  (`raw_cell_inbox_fraction` diagnostic, a genuinely subtle catch). Marginal GO.
- **Stage 2 (zero-shot generation):** **NOT FEASIBLE.** OTgap 1.44 (N=64) / 1.63 (N=125),
  worse than uniform noise; gr_L1 at uniform-noise level. N=125 failed despite a *better*
  rail KS than N=64 — the key scientific lesson: **matching marginal feature distributions
  is necessary but not sufficient; the learned conditional p(delta | rail, context) is what
  fails to transfer.** Named causes: (a) `reference="absolute"` rail extrapolates outside
  the L=3 coordinate range, (b) AR error compounds over longer sequences, (c) pow2 cell
  shrinks (0.031/0.039 vs training 0.047).

### 2.7 In progress (uncommitted): arc-length representation (D3)
`hilbert_arc_delta()` produces `(Δs, fine_x, fine_y, fine_z)` targets — Δs = Hilbert-code
step / X (≈1 for local steps, size-invariant by construction) and `fine` = offset from the
cell center / cell_size (dimensionless, ≈[−0.5, 0.5], size-invariant at fixed density+cell).
Threaded as `arc_repr` through cached dataset → datamodule → model (`_delta_dim=4`, MDN head
switched to 4D, torus wrap disabled for the 4D target) → sampler. This is the principled
endpoint of the design memo: every turn is absorbed into the fixed s↔xyz map, so the model
never has to express a turn at all. It is the right next experiment given §2.6's diagnosis.

---

## 3. Verified code-review findings

Findings are listed most-severe first; each was independently verified against the working
tree (verdicts: CONFIRMED = reproduced from the code; PLAUSIBLE = realistic, not refutable;
refuted candidates are listed at the end of this section).

> **Status update (2026-06-11):** findings **3.1–3.5 and 3.9 are FIXED**, test-first, in
> `tests/test_arc_repr_fixes.py` (6 tests; full suite 69 passing). Fixes: fine-offset
> min-image at decode (3.4 — note: with the wrap, the next step's position re-bin recovers
> `c_next` exactly, so carrying the code forward is now only an optimization);
> `_arc_initial_positions` helper + sampler wiring for particle 0 (3.5);
> `_validate_arc_repr_flags` in train.py reading `lj_transfer_ordering` (3.1);
> `_validate_arc_repr_cache` checking `metadata["ordering"]` (3.3); init-time datamodule
> guard (3.2); `--arc_repr` un-nested in `train_sample_lj27_pbc.sh` (3.9). Still open:
> 3.6 (logp Jacobian), 3.7 (`--use_ida` default), 3.8 (coord_dim key), 3.10 (non-pow2
> guard), 3.11 (saved-deltas min-image), 3.12 (particle_length assert).

### 3.1 CONFIRMED — `train.py:590`: the arc_repr ordering guard can never fire
The guard reads `getattr(args, "ordering", "hilbert")`, but the only ordering flags argparse
defines are `--lj_transfer_ordering` and `--lj_abs_ordering`; `args.ordering` does not
exist, so the getattr always returns the default and the ValueError is unreachable.
Combined with 3.3, `--arc_repr 1` on a spectral-ordered cache trains on meaningless Δs
targets with no error anywhere. **Fix:** check `args.lj_transfer_ordering` (and prefer
validating in the dataset, where the cache's actual `metadata["ordering"]` is known — see 3.3).

### 3.2 CONFIRMED — `lightning_module.py:495`: arc_repr silently dropped on the on-the-fly dataset path
`arc_repr=self.arc_repr` is passed only to `LJTransferableCachedDataset` (line 459); the
on-the-fly `LJTransferableDataset` constructor omits it and that class has no arc support at
all. No setup-time guard requires a preprocessed cache, so `--arc_repr 1` without
`--lj_transfer_preprocessed_path` builds the model, loads the full dataset, then dies on the
first training step with `KeyError: continuous_input=True requires batch['arc_delta']`.
**Fix:** raise in `LJTransferableDataModule.__init__`/`setup` when `arc_repr` is requested
without a cache (or implement it in the shared dataset).

### 3.3 CONFIRMED — `lj_transferable.py:1678`: arc_repr never checks the cache's ordering
The arc validation checks only `absolute_coords` presence and `periodic`; the cache build
writes `metadata["ordering"]` (line 2115) and supports `ordering="spectral"`, but the
`__getitem__` recompute assumes Hilbert-monotone codes (`Δs = (c_{i+1}−c_i)/X`). A spectral
cache yields non-monotone codes → garbage Δs, silently. **Fix:** require
`metadata["ordering"] == "hilbert"` in the arc_repr validation block.

### 3.4 CONFIRMED — `sample_lj.py:448`: unclamped fine offset can teleport the chain along the curve
`_arc_decode_positions` returns `cell_centers + fine·cell_size` with no clamp, while the
training-side invariant (`hilbert_arc_delta`, lj_transferable.py:342) min-images fine into
±cell/2. The GMM sample has unbounded Gaussian tails, so |fine| > 0.5 is reachable; the
next step re-bins the realized position to get `c_curr` (lines 429–432), and a *spatially*
adjacent cell can sit arbitrarily far along the Hilbert curve. After that,
`c_next = c_curr + round(Δs·X)` continues from the wrong arc location — one tail sample
silently relocates the rest of the chain. **Fix:** wrap fine into ±0.5 cell at decode
(mirror the encoder's min-image), and/or carry `c_next` forward instead of re-binning.
*This is the arc-mode analogue of the discrete bin-crossing cascade already diagnosed on
the toy — same failure shape, new representation.*

### 3.5 CONFIRMED — `sample_lj.py:561`: particle 0 never initialized in arc mode
The only `x_base[:,0,:]` assignment is in the `rail_residual` branch (line 627, "Start
particle 0 at its anchor"); the arc branch has none, so particle 0 stays at the exact box
corner (0,0,0) in every generated configuration. In plain-delta mode that's a harmless
translation gauge, but arc decode places particles 1..N−1 at *absolute* box-frame cell
positions, so the corner particle is a real, systematic misplacement — and step 0's
`c_curr` is forced to Hilbert code 0, unlike training where the first sorted particle sits
at the (nonzero) lowest occupied code, with a cell-interior position. **Fix:** initialize
particle 0 the way training sees it (e.g. sample a position consistent with the first-code
distribution, or at minimum the code-0 cell center).

### 3.6 CONFIRMED — `sample_lj.py:758`: arc-mode `logp_continuous` lacks the change-of-variables Jacobian
`logp_continuous += _gmm_log_prob(..., delta_model_t)` accumulates density of the 4D
(Δs, fine/cell) variable. Converting to a position density needs −log Π cell_size per step
(≈ +9.2 nats/particle at cell = 3/64), plus the `round(Δs·X)` discretization is ignored.
Saved log-probs are therefore incomparable across representations and across box sizes —
exactly the quantity `eval_mcmc_logprob.py`-style ESS/importance diagnostics consume when
judging size transfer. **Fix:** add the Jacobian at save time (and document Δs's discrete
component), or store the arc-space value under a distinct key.

### 3.7 CONFIRMED — `sample_lj.py:1058`: `--use_ida` defaults True, so default CLI can't load standard/rail/arc checkpoints
`--use_ida` is `BooleanOptionalAction, default=True` and `main` always passes a non-None
value, so `resolve_ar_arch` never returns "auto" and the checkpoint-metadata inference is
unreachable. Rail/arc checkpoints are standard-arch by construction (train.py requires it),
and `GraphormerARIDA.__init__` accepts none of their hparams — plain
`python sample_lj.py --ckpt best.ckpt` fails to load unless the user knows to pass
`--ar_arch standard` (the memory-recorded gotcha is this bug). **Fix:** default
`--use_ida` to None so arch resolution actually falls through to checkpoint inference.

### 3.8 CONFIRMED — `sample_lj.py:1156`: coord_dim inference reads the wrong hparams key
`getattr(model.hparams, "spatial_dim", None)` — but the standard arch stores
`ida_spatial_dim` (transformer.py constructor), so inference returns None and coord_dim
silently defaults to **2** for every rail/arc checkpoint; the run then errors confusingly
on tokenizer-vocab/MDN-dim mismatch (or worse, samples 2D if vocab happens to fit).
**Fix:** try `ida_spatial_dim` then `spatial_dim`; better, validate against the
checkpoint's MDN head dimension.

### 3.9 CONFIRMED — `train_sample_lj27_pbc.sh:771`: `--arc_repr` only emitted inside the continuous-head branch
`ARC_REPR=1` with `USE_CONTINUOUS_HEAD=0` silently drops the flag — and because the flag
never reaches train.py, it *bypasses* train.py's loud `--arc_repr requires
--continuous_input` ValueError (train.py:589). The whole "arc ablation" trains and samples
the baseline with no error anywhere. **Fix:** move the ARC_REPR append out of the nesting
and let train.py do the validation it already has.

### 3.10 CONFIRMED (latent) — `lj_transferable.py:208`: rail code clip assumes power-of-two R
`np.clip(base + offs, 0, R**3 − 1)` is only valid when R is a power of two; with
`bits=ceil(log2(R))` and non-pow2 R, `_hilbert3d_encode` legitimately returns codes up to
2^(3·bits)−1 > R³−1 for in-grid cells, so valid codes get rewritten and decode to unrelated
cells. Currently only reachable via the (already-disqualified) `exact` transfer strategy or
a manual resolution override — but it is a second, independent reason non-pow2 R can never
work, worth a guard (`raise` on non-pow2 R) so future transfer experiments fail loudly.

### 3.11 Downgraded — `sample_lj.py:744`: missing min-image on the arc context delta
The mechanics are real (`delta_cart_t = raw_pos − x_base[:,t,:]` between two wrapped
positions, no min-image), **but it does not affect the sampling distribution**: arc_repr
requires `continuous_input`, and in that mode the forward pass discards every post-SOS
token embedding (`x = cat([sos_e, cont[:,1:,:]])`, transformer.py:487) — the model is
conditioned on the raw 4D arc delta buffer, not the tokens. The corruption lands only in
the **saved npz `deltas` and `token_ids` arrays** (box-magnitude outliers at every boundary
crossing), which will mislead any downstream jump/displacement analysis of generated
samples. Fix is one `min_image` call before storing/encoding.

### 3.12 PLAUSIBLE (edge case) — `lj_transferable.py:1783`: particle_length −1 fallback feeds padding into arc targets
Caches lacking `particle_length` load as −1 (line 1573) and the arc branch falls back to
the padded width, feeding zero-padded rows (Hilbert code 0) into `hilbert_arc_delta`.
Modern cache builds write `particle_length` and `absolute_coords` together, so this needs
an intermediate-vintage or hand-edited cache — low likelihood, but the failure would be
silent for mixed-N data. A cheap `assert particle_len > 0` closes it.

### Refuted along the way
- *Arc sampling wrap consistency:* every placement branch stores wrapped positions, and
  `_arc_decode_positions` independently re-wraps its input — the round-trip is
  self-consistent mod box. REFUTED.
- *Arc `__getitem__` vs torus-shift augmentation:* the claim that the Hilbert recompute
  could disagree with the cache-build grid is REFUTED — the shift is applied *before*
  sorting and the cache stores post-shift coordinates (`_build_batch_3d` lines 1382–1460),
  and the recompute (`floor(mod(coords,box)/cell)`) is mathematically identical to
  `_grid_coords_periodic` at the same resolution. (The *efficiency* half of that finding
  stands: the recompute is per-fetch wasted work; see §4.)

---

## 4. Architecture / maintainability assessment (verified)

A dedicated fact-check pass confirmed 19 of 21 cleanup candidates (two sub-claims
corrected, noted below). The unifying theme: **the train/sample contract is duplicated by
hand everywhere it matters**, which is exactly the defect class that keeps producing this
project's silent-wrongness bugs. Grouped by what to do about it:

**Single-source the geometry (highest value).** The pow2 resolution rule exists in **5
copies** (`sample_lj.py:361`, `lj_transferable.py:1074` and `:1690`,
`sample_k1_transfer.py:26`, `analyze_k1_rail_transfer.py:39`) — `sample_k1_transfer.py`'s
monkey-patch exists *because* one copy disagreed. The rail waypoint pipeline
(`_compute_rail_waypoints_batch`) reproduces the dataset "step-for-step" per its own
docstring; the 2D Hilbert encoder is byte-copied across 3 files; `_hilbert_d2xy` exists
twice with different loop styles; the polar conversion exists in numpy and torch; the
analysis scripts re-implement the dataset Hilbert sort (correction from the finder:
`generalization_tests.py` *does* have the box epsilon — it's `analyze_k1_rail_transfer.py`
that lacks it). The planned `CurveRailConfig` (refactor TODO Task 1) plus moving these
helpers to module level in `lj_transferable.py`/`utils/spatial.py` collapses all of it.

**Put run config in the checkpoint, not the CLI.** Sampling-time tokenizer settings are
guessed from vocab size while `--relative_window/--relative_bins` must happen to match
training (a wrong-but-vocab-compatible combination samples silently mis-binned context);
`_encode_relative_delta_context` re-implements `RelativeDeltaTokenizer.encode` in torch;
`CELL_TRAIN = 3.0/64` is hardcoded in 3 files (the test file imports it — correction to the
finder) and the transfer wrapper monkey-patches a private function because the checkpoint
stores `cell_size=None`. The rail geometry already rides in hparams — extend the same
pattern to tokenizer config and effective cell_size.

**Delete dead/duplicated model code.** `transformer_ida.py` is a ~600-line near-fork that
receives none of the new features (`--ar_arch ida` + `--arc_repr/--continuous_input/rail`
silently no-ops); RoPE params are accepted, stored in hparams, and never applied
(`--ar_use_rope` is never even read by train.py); `training_step` has two byte-identical
cross-entropy blocks; `ar_registry.py:104` does checkpoint compat by string-matching an
error message and retrying `strict=False` (which would silently zero-init *any* mismatched
key). The arc sampling path adds a `_arc_raw_pos` sentinel carried ~40 lines and a third
copy of the periodic-wrap placement logic.

**Efficiency (matters at N=125, the regime this branch targets).** All confirmed: the
sampling loop re-forwards the full prefix with no KV cache while EdgeBias rebuilds the full
[B,nH,t,t] bias (incl. a dir_mlp over t² pairs) every step — O(T³) total, ~40× extra FLOPs
at N=125; `_arc_decode_positions` does ~5 host↔device syncs per step and re-derives
loop-invariant constants; the lookahead-rail path round-trips through numpy per step;
arc targets are recomputed in `__getitem__` every epoch though fully deterministic;
`SAMPLE_SAVE_EACH_BATCH=1` rewrites the entire accumulated npz per chunk (O(C²));
per-step defensive `.clone()` of the coordinate prefix. None of these block correctness;
the KV-cache/EdgeBias item is the one worth real effort before large-N sampling runs.

**Scripts encode semantics Python should own.** `--arc_repr` emitted only inside the
continuous-head branch of the shell script (finding 3.9); `sample_from_run_dir.sh` replays
runs by sed-scraping `reproduce.sh` and its `OVERRIDE_VARS` allowlist already misses
`ARC_REPR`; `arc_repr` is wired only into the cached-dataset branch of the datamodule
(finding 3.2).

---

## 5. What's needed toward the goal (roadmap)

The goal decomposes into: (G1) a representation whose *conditional* transfers across box
size, (G2) a generation procedure that doesn't compound error over longer sequences, and
(G3) OTgap ≲ 0.6 at unseen sizes so `learndiffeq` can refine. Ordered plan:

### Near term (days)
0. **Fix the confirmed arc_repr bugs before any arc run** (else results are
   uninterpretable): particle-0 initialization (3.5), fine-offset wrap at decode +
   carry `c_next` forward instead of re-binning (3.4), the dead ordering guard and cache
   ordering check (3.1, 3.3), the datamodule/scripting wiring traps (3.2, 3.9), the
   log-prob Jacobian (3.6), and min-image on the saved context delta (3.11). Also fix the
   two CLI traps that have already cost debugging time per the project notes:
   `--use_ida` default (3.7) and coord_dim inference (3.8). All are small, localized fixes.
1. **Finish and evaluate `arc_repr` (D3).** It directly removes cause (a) of the Stage-2
   failure: both Δs and fine are translation- and size-invariant by construction. Gate it
   the same way as the K=1 study: teacher-forced NLL parity across L first, then zero-shot
   generation, then `benchmark_lj27.py` vs size-matched MCMC. One design caution: Δs is
   *not* turn-free — a Hilbert turn still appears as a large Δs outlier the MDN must
   express; check the Δs distribution across L (the same analysis as
   `analyze_hilbert_lj_jumps.py`) before assuming the representation removed the
   turn-expression problem rather than moved it into the Δs tail.
2. **Do the `CurveRailConfig` consolidation now** (Task 1 of the refactor TODO), bumping
   cache version, before arc_repr experiments fork the config matrix further. Train/sample
   geometry skew is the #1 silent-wrongness risk in this codebase and it has already
   produced real bugs (always-on augmentation shift; cell_size=None pinning R=64).
3. **Add the rail/arc parity tests** (Task 2 of the TODO): sampler-computed rail ==
   dataset `curve_waypoints` for the same positions; `hilbert_arc_delta` round-trip
   (encode → decode == original positions, including across a turn and across the periodic
   boundary); `continuous_input` shift alignment. These are cheap and convert convention
   into contract.

### Mid term (1–2 weeks)
4. **Translation-invariant rail reference.** Even with arc_repr, the rail conditioning
   should switch `reference="absolute"` → `"prev_step"` (or be dropped in favor of the arc
   frame) so no input feature encodes an absolute box position that extrapolates at larger L.
5. **Multi-size training of the arc_repr model** (e.g. {L3,L5} hold out L4, reusing
   `eval_multisize.sh`). The §2.3 diagnosis showed both fixes are needed: representation
   (express turns / invariant features) AND multi-size data (fix the overfit local
   conditional). Zero-shot from single-size training has now failed twice; expecting it to
   work a third time without multi-size data contradicts the project's own evidence.
6. **Attack AR error compounding directly** (cause (b), unaddressed by representation):
   scheduled-sampling/input-noise is already implemented (`continuous_input_noise`) — tune
   it at LJ scale, not just the toy; consider per-step re-anchoring (the fixed-template
   anchor already provides an absolute reference that cannot drift).
7. **Fine-tune-at-target-size control run.** Cheap, and it cleanly separates "conditional
   can't transfer zero-shot" from "conditional can't be represented at this size at all."
   If a short fine-tune at N=64 reaches OTgap ≲0.6, multi-size training is guaranteed to
   work and the question becomes pure data efficiency.

### Hand-off criterion (G3)
8. **Define the done-line against `learndiffeq` explicitly:** held-out size OTgap ≲ 0.6
   (green band), gr_L1 ≲ 0.1, clamped U/N near the target's, clash% irrelevant (uniform
   noise is 100% and the flow heals it). When an arc_repr/multi-size model crosses that
   line at a held-out L, wire its `.npz` output into the flow's OT-precompute and run one
   end-to-end refinement as the true acceptance test.

### Hygiene (whenever, but soon)
9. Repo hygiene (Task 3 of the TODO): stop tracking `.pyc`/binary artifacts (this branch
   commits `.pyc` files and 1.4 MB of `.npz` into git history), add `.gitignore` entries,
   move toy plot scripts into `scripts/`, and consider whether `reports/` artifacts beyond
   the two FINDINGS.md belong in git.
10. Fix the ablation wart (Task 4): the datamodule force-sets `use_curve_rail` from the
    cache, so a rail cache can't be A/B'd rail-off.

---

## Appendix A — raw finder candidates (as collected, pre-verification)

*Verdicts live in §3 (correctness) and §4 (cleanup; 19/21 confirmed). Corrections found
during verification: in #12 the epsilon claim is reversed (`generalization_tests.py` has
the box epsilon, `analyze_k1_rail_transfer.py` lacks it); in the CELL_TRAIN item the test
file imports the constant rather than re-declaring it; #5's grid-shift sub-claim and #27's
"corrupts model conditioning" framing were refuted (see §3.11 and §3 refutations).*

### Simplification angle (received)
1. `transformer.py:114` — dead RoPE plumbing (`--ar_use_rope` recorded but never applied).
2. `transformer.py:656` — duplicated byte-identical CE loss blocks in `training_step`.
3. `transformer_ida.py:1` — 73%-identical fork of GraphormerAR; new flags silently dropped
   under `--ar_arch ida`.
4. `lightning_module.py:595` — eight near-identical `_eff` forwarding properties.
5. `lj_transferable.py:1782` — arc_repr `__getitem__` re-derives Hilbert codes inline
   (third copy of the quantize+encode pipeline; ignores grid shift → potential wrong Δs
   targets on shifted caches; per-sample recompute in the dataloader hot path).
6. `sample_lj.py:736` — `_arc_raw_pos` sentinel carried ~40 lines; periodic-wrap logic
   triplicated across placement branches.

### Reuse angle (received)
7. `sample_lj.py:361` — `_rail_resolution_for_box` is the **fifth** copy of the
   next-pow2(L/cell_size) rule (also in `lj_transferable.py:1074` and `:1690`,
   `sample_k1_transfer.py:26`, `analyze_k1_rail_transfer.py:39`). The monkey-patch in
   `sample_k1_transfer.py` exists precisely because one copy disagreed — the drift is live.
8. `sample_lj.py:380` — `_compute_rail_waypoints_batch` re-implements the dataset rail
   pipeline "step-for-step" (its own docstring); train/sample rail parity is by convention.
9. `lj_abs_dataset.py:16` — `_hilbert_rot`/`_hilbert_xy2d`/`_is_power_of_two` byte-copied
   from `lj_transferable.py` (and a third copy in `scripts/report_adjacent_sorted_displacements.py`).
10. `utils/spatial.py:326` — new `_hilbert_d2xy` duplicates `sample_lj.py:31`'s decoder with
    a different loop style; curve-orientation agreement is untested.
11. `lj_transferable.py:352` — `_cartesian_to_spherical_np` duplicates
    `utils/spatial.py:43` `cartesian_to_spherical` (numpy vs torch); polar encode/decode
    conventions live in two places.
12. `analyze_k1_rail_transfer.py:93` — `hilbert_sorted_codes` re-implements the dataset
    Hilbert sort; `generalization_tests.py:54` is a third copy missing the dataset's box
    epsilon — audit orderings can diverge from training orderings.

### Altitude angle (received)
13. `sample_k1_transfer.py:49` — size transfer via monkey-patching a private function,
    keyed to hardcoded `CELL_TRAIN = 3.0/64` re-declared in 4 files; should be a
    `--cell_size`/`--rail_resolution` override on `sample_lj.py` (or persist effective
    cell_size in hparams instead of None).
14. `ar_registry.py:107` — checkpoint compat by string-matching the RuntimeError message
    for a missing `edge_bias.far_bias` key, then retrying with `strict=False` (which would
    silently zero-init *any* mismatched key at that point).
15. `sample_lj.py:214` — sampling-time tokenizer settings guessed from vocab size +
    CLI defaults (`--relative_window/--relative_bins` must happen to match training), and
    `_encode_relative_delta_context` re-implements `RelativeDeltaTokenizer.encode` in torch.
    Tokenizer config belongs in checkpoint hparams like the rail geometry already does.
16. `sample_lj.py:409` — arc codec split across layers: encoder in the package
    (`hilbert_arc_delta`), decoder only in the top-level script (`_arc_decode_positions`);
    other entry points can't decode arc predictions without importing the script.
17. `lightning_module.py:495` — `arc_repr` wired only into the cached-dataset branch; the
    on-the-fly dataset path never receives it and there's no setup-time validation
    (fails mid-first-batch with a KeyError, or worse if requirements are later relaxed).
18. `sample_from_run_dir.sh:82` — run replay by sed-scraping `reproduce.sh` (undeclared
    format contract; `OVERRIDE_VARS` allowlist already missing `ARC_REPR`).

### Efficiency angle (received)
19. `sample_lj.py:427` — `_arc_decode_positions` does ~5 host↔device syncs per AR step and
    re-derives loop-invariant constants (box, cell_size, bits, X) plus the previous step's
    Hilbert code every call.
20. `sample_lj.py:655` — lookahead-rail path: CPU round-trip + numpy waypoint math per step
    inside the sampling loop.
21. `sample_lj.py:667` — full-prefix re-forward each step with no KV cache, and EdgeBias
    rebuilds the entire [B,nH,t,t] pairwise bias (incl. dir_mlp over t² pairs) per step —
    O(T³) total; ~40× extra FLOPs at the N=125 sizes this branch targets.
22. `lj_transferable.py:1782` — arc_repr targets recomputed in `__getitem__` every epoch
    though deterministic functions of immutable cached tensors (also reuse finding #5).
23. `sample_lj.py:1433` — `SAMPLE_SAVE_EACH_BATCH=1` rewrites the full accumulated .npz
    after every chunk (O(C²) work).
24. `sample_lj.py:646` — defensive `.clone()` of the growing coordinate prefix every step;
    forward never mutates coords.

### Correctness — removed-behavior audit (received)
*(note: the agent confirmed the big `ar.py` rewrite is a clean relocation of GraphormerAR
into `models/transformer.py` — validation, SOS shifting, pad masking, `nll` intact)*
25. `train.py:591` — the `--arc_repr` ordering guard reads `getattr(args, "ordering", ...)`
    but the flag is `--lj_transfer_ordering`; the "arc_repr requires hilbert ordering"
    validation can never fire.
26. `lightning_module.py:495` — arc_repr forwarded only to the cached dataset; on-the-fly
    path drops it (KeyError mid-first-batch; duplicate of #17).
27. `sample_lj.py:744` — arc path computes `delta_cart_t = raw_pos − x_base[:,t,:]` from
    the already-wrapped position **without min-image wrap**; boundary crossings feed the
    tokenizer a ≈±L-magnitude context delta the model never saw in training.
28. `sample_lj.py:1059` — `--use_ida` defaults True and `resolve_ar_arch` returns "ida"
    whenever use_ida is not None, so checkpoint-metadata arch inference is unreachable;
    plain `python sample_lj.py --ckpt best.ckpt` on a standard/rail/arc checkpoint fails to
    load (matches the memory gotcha "force --ar_arch standard").
29. `lj_transferable.py:1796` — arc_repr `__getitem__` assumes the cached order is Hilbert
    order but never checks `metadata['ordering']`; a spectral-ordered cache trains on
    garbage Δs targets with no error.
30. `sample_lj.py:409` — `_arc_decode_positions` never clamps the predicted fine offset to
    ±0.5 cell and re-bins the realized position for the next step's code; one Gaussian-tail
    sample (|fine|>0.5) lands in a neighboring cell whose Hilbert code is arbitrarily far,
    teleporting the rest of the chain along the curve.

### Correctness — cross-file tracer (received)
31. `sample_lj.py:744` — independently re-found #27 (arc context delta not min-imaged;
    also corrupts the saved `deltas` npz field).
32. `sample_lj.py:740` — arc_repr sampling never initializes `x_base[:,0,:]`: particle 0
    stays pinned at the exact box corner (0,0,0). Unlike plain-delta mode (translation
    gauge), arc decode anchors particles 1..N−1 absolutely, so the corner particle is a
    real, systematic misplacement (the rail_residual branch *does* initialize particle 0).
33. `lightning_module.py:495` — re-found #17/#26 (arc_repr dropped on the on-the-fly
    dataset path).
34. `train_sample_lj27_pbc.sh:771` — `--arc_repr` appended only inside the
    `USE_CONTINUOUS_HEAD==1` branch; `ARC_REPR=1 USE_CONTINUOUS_HEAD=0` silently runs the
    baseline experiment end-to-end (checkpoint stores arc_repr=False, sampler follows).
35. `sample_lj.py:757` — arc-mode `logp_continuous` accumulates MDN density in
    (Δs, fine/cell) space with **no Jacobian correction** (Π cell_size per particle) and no
    account of the Δs discretization; cross-checkpoint NLL/ESS comparisons are off by a
    size-dependent constant — exactly the quantity used to judge size transfer.
36. `lj_transferable.py:208` — `_rail_relative_from_codes` clips codes to [0, R³−1], only
    valid for power-of-two R; with non-pow2 R (the `exact` transfer strategy) valid codes
    exceed R³−1 and get rewritten → garbage rail features with no crash. (Consistent with —
    and a mechanistic deepening of — the Stage-1 `exact`-strategy disqualification.)

### Correctness — line-by-line scan (received; all 7 angles now in)
*(re-found #25 train.py ordering guard, #27/#31 min-image, #35 Jacobian, #26/#33
datamodule arc_repr drop — independent rediscovery by 2–3 angles each is strong signal)*
37. `sample_lj.py:1156` — coord_dim auto-inference reads hparams key `spatial_dim`, but
    the standard arch (all rail/arc checkpoints) stores `ida_spatial_dim`; inference
    silently falls back to 2D without an explicit `--coord_dim 3`.
38. `lj_transferable.py:1784` — arc_repr `__getitem__`: missing/−1 `particle_length` falls
    back to the padded width, feeding zero-padded rows (Hilbert code 0) into
    `hilbert_arc_delta` → shape mismatch or corrupted Δs targets on mixed-N/old caches.

Angle A also explicitly cleared: MDN full-covariance log-prob + torus wrap in `mdn_loss`,
the double `_prepare_attention_coords` call (idempotent), rail train/sample recomputation
parity for the committed paths, the `sample_k1_transfer.py` monkey-patch mechanics, the LJ
cutoff zeroing in `physics/energy.py`, and BucketedBatchSampler collation of `arc_delta`.
