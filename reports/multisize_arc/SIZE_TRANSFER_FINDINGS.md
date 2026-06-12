# Size-transferable arc_repr transformer — consolidated findings (2026-06-12)

Canonical writeup of the multi-size arc_repr investigation: what the trained model does, why
held-out generation fails, and what the candidate fix ("predict deviation from the Hilbert
center") does and does not buy. Detail docs: [`DELTA_MARGINALS.md`](DELTA_MARGINALS.md),
[`HILBERT_CENTER_ANCHOR_SCOPING.md`](HILBERT_CENTER_ANCHOR_SCOPING.md), `EP{18,38,54}_STATUS.md`.
Figures in `figs/`. Memory: `arc-cell-discontinuity`.

## 0. Setup

Autoregressive transformer over Hilbert-ordered LJ particles (ρ=1, T=1, PBC). arc_repr target
per step = `(Δs, fine_x, fine_y, fine_z)`:
- **Δs** = `(c_{j} − c_{j−1}) / X`, X = R³//N — coarse along-curve step, *relative to the
  previous particle* (cumulative decode). Size-invariant target (mean≈1).
- **fine** = `(pos_j − cellcenter(c_j)) / cell_size` — within-cell offset, *absolutely anchored*
  to the actual cell center. Size-invariant (uniform on [−0.5,0.5]).

Run: `lj_ckpts_multisize_arc_L3L5/multisize_arc_fullcov`, train {L3 N=27, L5 N=125}, **hold out
L4 N=64**, 100 epochs, dim 512 / depth 4 / heads 4, full-cov continuous head (64 mixtures),
`continuous_input`, cell_size 0.046875. Refinability metric OTgap (benchmark_lj27.py):
0 = matches Boltzmann target, 1 = uniform noise; green ≲0.6.

## 1. Multi-size result (epoch 100, 256-sample OTgap)

| size | role | OTgap final | zero-shot baseline | teacher-forced NLL |
|------|------|------------:|-------------------:|-------------------:|
| L3 N=27   | train    | **0.555** ✓ | 0.524 | 0.174 |
| L4 N=64   | held out | **0.892** ✗ | 0.795 | 0.259 |
| L5 N=125  | train    | **0.450** ✓ | 0.911 | 0.164 |

- **Trained sizes became refinable.** L5 crashed from a zero-shot 0.911 to a green 0.450;
  L3 sits at 0.555. Multi-size training works *at the trained sizes*.
- **The held-out size did not.** L4 stayed at 0.892 — worse than its own 0.795 zero-shot
  baseline. Multi-size training did **not** deliver held-out transfer at the generation level.
- **The conditional interpolated fine.** Held-out L4 teacher-forced NLL (0.259) sits between
  the train sizes (0.16–0.17), i.e. given clean data context the per-step prediction at L4 is
  good. The failure is specifically in *free-running generation*.

## 2. Where the failure lives: Δs drifts, fine transfers

Per-step marginals, generated vs data, both pushed through the exact training recipe (recompute
Hilbert codes from positions → `hilbert_arc_delta`). Figures `figs/arc_delta_*`.

| component | L3 (train) | L5 (train) | L4 (held out) | L10 (held out) |
|-----------|-----------:|-----------:|--------------:|---------------:|
| **Δs** W1 | 0.029 | 0.014 | **0.107** | **0.129** |
| **Δs** gen mean | 0.991 | 0.988 | 0.889 | 0.882 |
| **fine** W1 | ~0.011 | ~0.011 | ~0.011 | ~0.011 |

- **The absolutely-anchored coordinate (fine) transfers everywhere** (W1 ≈ 0.011 at every
  size, trained or held-out). The within-cell offset is uniform in data and the model
  reproduces it up to a mild central peak (tempering at T=0.9).
- **The cumulative coordinate (Δs) transfers only at trained sizes.** At held-out sizes the
  generated Δs *mode collapses* (data band ≈0.6 → generated ≈0.15; heatmaps
  `figs/arc_delta_ds_heatmap_*`) with mean dragged to ~0.88. A small per-step downward bias
  compounds over the 64–1000-step rollout. Fine never accumulates, so it is immune.
- **Monotonicity** mostly holds: 84% of generated L5 configs are fully Hilbert-monotone; only
  0.14% of steps go backward (Δs<0) vs 0% in data.

This is the controlled A/B that pins the mechanism: same model, same training, same held-out
sizes — the absolutely-anchored part is robust, the relative/cumulative part drifts.

## 3. Hypothesis tested and REFUTED: it is not the cell discontinuity

First hypothesis: the "constant cell 0.046875" is pow2-rounded, so actual cells differ
(L3=0.0469, L5=0.0391, **L4=0.03125 — finest, off-ladder**), and Δs (coupled to cell scale
X=(R/L)³) fails to extrapolate at the L4 outlier.

**Test (cheap, decisive):** generate L=10 / N=1000 from the same checkpoint. L10 shares L5's
*exact* cell (0.0391) and X (16777), and the model has `use_pos_emb=False` (no positional
ceiling). The cell hypothesis predicts L10 transfers like L5.

**Result:** L10 Δs fails identically to L4 (W1 0.129). The 2×2:

| | cell 0.0469 | cell 0.0391 | cell 0.0313 |
|---|---|---|---|
| **trained**  | L3 W1 0.029 ✓ | L5 W1 0.014 ✓ | — |
| **held out** | — | L10 W1 0.129 ✗ | L4 W1 0.107 ✗ |

Two trained sizes have *different* cells and both succeed; L10 has the *same* cell as the
successful L5 and fails. **Cell is orthogonal — the splitter is trained-vs-held-out.** The real
cause is exposure bias / AR drift at any unseen size (the conditional generalizes; the
free-running rollout does not). The constant-cell pow2 ladder will NOT fix transfer.

## 4. Candidate fix: "predict deviation from the Hilbert center"

Idea (recalled): anchor each particle to its index's expected curve position `decode(j·X)` (the
"Hilbert center", the mean s at index j) and predict `residual_j = pos_j − decode(j·X)`, instead
of the cumulative `pos_j − pos_{j−1}`. Absolute anchor ⇒ no error accumulation ⇒ no drift.

**Already implemented** end to end as `curve_rail_residual_target` + `fixed_template` (anchor =
`fixed_template_anchors`, lj_transferable.py:290; target baked at cache build L1297/L1432;
standard-arch `rail_attn` cross-attention; sampler re-adds the anchor, sample_lj.py:952;
flags on `preprocess_lj_transferable.py` and `train.py`). It has **never been run multi-size**
— prior rail work was single-size and judged infeasible (`lj_ckpts_lj27_pbc_fixedrail_k1_*`),
but that was zero-shot from one size, the regime multi-size exists to fix.

## 5. Why the literal version likely won't transfer (measurement)

Measured the target the model would have to predict, on real MCMC data, per particle index.
Figures `figs/anchor_residual_perindex_{L3,L5,L10}.png`, `figs/anchor_residual_std_vs_size.png`.

Per-axis std of the target vs system size:

| size | **absolute** anchor (`pos_j − decode(j·X)`) | **prev-particle** (`pos_j − pos_{j−1}`) | abs/prev |
|------|-------------------------------------------:|----------------------------------------:|---------:|
| L3 N=27    | 0.636 | 0.673 | 0.94 |
| L5 N=125   | 0.826 | 0.699 | 1.18 |
| L10 N=1000 | 1.127 | 0.702 | 1.61 |

- The absolute residual is **structured, not noise** (L10 std 1.13 vs uniform 2.89; the
  per-index heatmaps show a centered, roughly stationary blob) — so it is learnable.
- **But it is size-dependent:** std grows 0.64 → 0.83 → 1.13 with box size and bulges
  mid-curve (order-statistic random walk — liquids lack the long-range positional order that
  would make an absolute index template a sharp predictor of pos_j).
- **The prev-particle target is the opposite:** std pinned at ~0.70 for *all* sizes
  (size-invariant) and tighter — but it is the one that drifts. In
  `anchor_residual_std_vs_size.png` the prev-particle curves overlap; the absolute curves
  separate.

### The load-bearing tension

**Tight + size-invariant + non-drifting cannot all hold at once.**
- Tight & size-invariant ⇒ a *local* reference (prev particle — sharp via short-range order).
- Non-drifting ⇒ an *absolute* reference (index template) — which liquids make loose and
  size-scaling.

arc Δs lives at the tight+invariant+drifts corner; the Hilbert-center anchor lives at the
non-drift+loose+size-scaling corner. Trading drift for a size-scaling target reintroduces the
exact 2-point box-size extrapolation that has been the project's recurring wall.

## 6. Options

- **B — exposure-bias mitigation on the existing arc rep (recommended).** Keep the proven
  tight, size-invariant arc target (the reason fine transfers); retrain {L3,L5} with
  `continuous_input_noise` / scheduled sampling so generation learns to recover from its own
  drift. No new cache; attacks the diagnosed cause (drift) directly.
- **A — bounded-local k-step anchor** (`pos_j − pos_{j−k}`, k≈4–8). Keeps a tight,
  size-invariant target; bounds drift accumulation to ~N/k hops instead of N steps. Partial,
  tunable. Representation change + rebuild + train.
- **C — run the global-template residual multi-size anyway** as a data point (it's built).
  Interpolated L4 might still benefit even if extrapolated L10 won't. Costs rebuild + train.
- **Open probe before committing:** the abs-anchor heatmaps are roughly stationary in index, so
  the size-dependence is mostly an overall *scale*. Worth testing whether a **normalization**
  (divide by a size-derived scale, or anchor in cell units like fine does) collapses the
  abs-anchor std curves the way the prev-particle curves already collapse. If one does, the
  absolute anchor becomes non-drifting *and* size-invariant — the actual win.

## 7. One-line status

Multi-size arc_repr makes **trained** sizes refinable (L5 0.91→0.45) but **not** held-out sizes
(L4 0.89); the cause is Δs exposure-bias drift, not the cell discontinuity (L10 refutes it); the
"Hilbert-center anchor" fix is built but its target is measured to be size-dependent, so the
next move is either exposure-bias mitigation on the existing rep (B) or finding a normalization
that makes the absolute anchor size-invariant.
