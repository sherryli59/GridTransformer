# D1 verdict — plain-MH block acceptance forecast (mW, N=64)

Script: `reports/logs-2026-07-14/d1_block_acc_forecast.py` (verbatim from the plan).
Controls: `reports/logs-2026-07-14/d1_controls.py` (identity + jitter, same bank/code path).
Raw logs: `d1_ambient.out`, `d1_controls.out`. Data: `d1_forecast_T0.0963.pt`.
Plot: `/mnt/ssd/GridTransformer/reports/logs-2026-07-14/d1_forecast_T0.0963.png`

## Bank source (ambient column)

`mw_bedrock.pt` holds no config stack (only `{N2, N3, L, beta}` scalars). The ambient bank
is **`mw_ref_N64_ext.pt`** — the 64,000-config extension bank that `mw_generator_v4.py`'s
`train()` (and `mw_block_train`, i.e. the very model under test) consume at ambient
T\*=0.09632. Normalized copy saved once (exists-guarded) to the canonical path
`liquid_coupling_flow/mw/artifacts/mw_bank_ambient_N64.pt` as `{"cfgs","U","tstar","L"}`,
configs defensively wrapped into [0,L) with `torch.remainder` (source was already in-box:
min 4.3e-7, max 5.19531 < L=5.19531) and `U` recomputed with the current `mw_energy`
(mean/std −104.18/1.17, consistent with the source bank's stored U).

## Harness controls (must pass before believing the forecast) — BOTH PASS

| control | K=1 | K=4 | K=8 | verdict |
|---|---|---|---|---|
| identity (proposal = current block) max\|dU\| | 0.000e+00 | 0.000e+00 | 0.000e+00 | **PASS** (exactly 0, over all 64 trials/K) |
| jitter 0.05σ, median βΔU | +0.54 | +3.94 | +7.35 | **PASS** (+K·O(1) small positive, as expected) |

Identity dU is exactly zero through the same clone/overwrite/wrap/`mw_energy` path the
forecast uses, and a physical 0.05σ jitter costs ~+0.9 βΔU per moved particle — so the
+120…+132 βΔU that the *model's* K=1 proposals cost is not a wrapping/indexing/dtype
artifact. It is the model.

## Per-K acceptance forecast, ambient T\*=0.09632 (256 proposals per K per selector)

acc bound = E[min(1, e^{−βΔU} · q_rev/q_fwd)] computed exactly per proposal, no chains.

| K | random: mean acc / median βΔU | blob: mean acc / median βΔU |
|---|---|---|
| 1 | 3.526e-04 / +132.5 | 9.085e-08 / +120.6 |
| 2 | 8.054e-19 / +326.4 | 2.208e-09 / +293.1 |
| 3 | 0 / +634.1 | 1.856e-08 / +495.8 |
| 4 | 0 / +1120.6 | 1.825e-32 / +698.5 |
| 6 | **0 / +2133.2** | **0 / +1195.9** |
| 8 | 0 / +4423.4 | 0 / +1682.3 |
| 12 | 0 / +8566.5 | 0 / +4103.3 |
| 16 | 0 / +14146.1 | 0 / +8246.3 |

The full βΔU distributions (histograms in the plot, per-proposal rows in the .pt) show no
usable left tail: the distributions sit entirely at βΔU ≫ 0 and shift right roughly
linearly in K. Median βΔU ≈ +120–132 for ONE regenerated particle is ~13 ε of excess
energy — core-overlap scale: the MLE block model routinely proposes positions inside
neighbours' cores. The K=1 random-vs-blob mean-acceptance inversion (3.5e-4 vs 9e-8
despite a *lower* blob median) is a tail artifact — at these levels the mean is set by a
handful of lucky proposals among 256, not by the bulk, and the medians agree both
selectors are dead; not a meaningful signal at this sample size.

## K\* verdicts

| bank / T\* | K\* @ acc ≥ 0.1 | K\* @ acc ≥ 0.01 |
|---|---|---|
| ambient, T\*=0.09632 | **NONE** (best K=1: 3.5e-4) | **NONE** |
| supercooled, T\*_work | **PENDING** — D0 ladder still producing the bank | PENDING |

Supercooling only makes this worse (larger β multiplies the same ΔU), so the pending
column cannot rescue the gate; it is retained for the record and for calibrating the RL
fine-tune target.

## Gate outcomes

- **(a) Heat-bath row lives? NO** — K\* does not exist even at the 0.01 threshold, at the
  *easy* (ambient) temperature. Plain-MH block moves with the current MLE proposal are
  fully dead at all K ≥ 1.
- **(b) Task 7 RL fine-tune needed? YES — gate FIRES.** acc at K=6 is exactly 0 in 256
  trials (both selectors), ≪ the 1e-3 trigger.

## Implications

- **SMC bridge:** two-blob/block moves cannot rely on target-temperature acceptance at
  all; they will do their work entirely on the low-λ rungs of the bridge where the
  tempered target tolerates the proposal's energy excess. Kernel-mixture scheduling
  should front-load block moves at low λ and hand off to single-site/local moves as λ→1.
- **What Task 7 must fix: proposal ENERGY, not clash count.** The failure is a smooth
  +O(100) βΔU per particle (core placement), the exact failure mode the KA glass line's
  block-conditional fine-tune fixed — dE per particle 742→14 (~50×, see
  blockcond-ft-energy-lever, ckpt `ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt`).
  That precedent sets the expected magnitude: a ~50× energy drop would bring K=1 median
  βΔU from +130 to ~+2.6, i.e. from acc ~e^-130 to a viable ~e^-3 — so the RL/energy
  fine-tune is plausibly sufficient for small K, and is the only path to a live
  heat-bath row.
