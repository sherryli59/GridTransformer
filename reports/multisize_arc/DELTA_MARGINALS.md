# Did the model learn the arc-delta marginals? (N=125 / L=5, a trained size)

Compares the per-step arc_repr targets — Δs and the fine x/y/z offsets — between the
final-epoch model's samples and the L5 MCMC data. Both sides go through the **exact training
recipe** (`LJTransferableCachedDataset.__getitem__`): recompute Hilbert codes from positions
(`grid = floor(mod(pos,box)/cell) → _hilbert3d_encode`), then `hilbert_arc_delta`. 2048
generated configs (GPU, seed 1, T=0.9) vs 2048 random L5 cache configs.

Figures in `reports/multisize_arc/figs/arc_delta_*_N125_L5.png`:
`aggregate` (pooled marginals), `perindex_meanstd`, `ds_heatmap` (Δs vs index), `examples`
(per-index histograms at t=1,30,62,100,123).

## Verdict: marginals are learned well; two small systematic gaps.

| component | data mean/std | gen mean/std | W1 | KS |
|-----------|--------------:|-------------:|---:|---:|
| Δs        | 0.997 / 0.645 | 0.988 / 0.656 | 0.0139 | 0.016 |
| fine_x    | -0.001 / 0.289 | -0.002 / 0.277 | 0.0144 | 0.023 |
| fine_y    | -0.001 / 0.289 |  0.000 / 0.278 | 0.0134 | 0.023 |
| fine_z    |  0.000 / 0.288 |  0.001 / 0.278 | 0.0123 | 0.020 |

1. **Δs (the Hilbert arc-length jump): learned essentially perfectly.** Right-skewed,
   peaked at ~0.5, mean ≈ 1 (the size-invariant local-step value), reproduced shape-for-shape
   (aggregate W1 0.014). The heatmap shows it is **stationary across particle index** in both
   data and model — no drift along the sequence. Per-index agreement is tight and only mildly
   degrades for late particles (per-index Δs W1 0.033 at t=1 → 0.079 at t=123).

2. **Fine offsets: variance matched, shape mildly off.** The data's within-cell offset is
   **uniform** on [−0.5, 0.5] (std 0.289 = 1/√12, the uniform value) — expected, since the
   R=128 cell (≈0.039 σ) is far below any physical correlation length. The model adds a **mild
   central peak** (over-concentration near the cell center), systematic across all three axes;
   std comes out 0.277 (~4% narrower). This is the mode-seeking signature of the tempered
   (T=0.9) full-covariance continuous head. KS ≈ 0.02 — small but structured.

3. **Monotonicity mostly respected.** 84% of generated configs (1721/2048) are fully
   Hilbert-monotone; only **0.14% of individual steps go backward (Δs<0)** vs **0% in data**.
   A rare generation artifact — the AR head occasionally samples a downward Hilbert jump the
   representation never contains.

## What this does and does not tell us

These are **1-D marginals**. They are near-perfect for a *trained* size (L5, whose generation
OTgap is the green 0.45), which is consistent. The important corollary for the **held-out**
size (L4, OTgap 0.89): since the held-out per-step conditional is also well-fit (NLL ≈ 0.26),
the held-out generation failure is **not** a marginal problem — it lives in the *joint /
correlation* structure (relative particle arrangement) that marginals cannot see, i.e. AR
error compounding over the rollout. Next diagnostic to localize that: run this same script on
the held-out N=64 samples (`--gen samples_N64_*.npz --cache lj64_L4_R128_cache.pt
--tag N64_L4`) and compare pair-correlation g(r), not just per-step marginals.

## Repro

```bash
python sample_lj.py --ckpt lj_ckpts_multisize_arc_L3L5/multisize_arc_fullcov/best.ckpt \
  --mode relative --coord_dim 3 --ar_arch standard --use_continuous_head --full_covariance \
  --temperature 0.9 --relative_window 3.0 --relative_bins 64 --nsamples 2048 --periodic \
  --seed 1 --density 1.0 --device cuda --Lx 5 --Ly 5 --Lz 5 \
  --save reports/multisize_arc/samples_N125_gpu2k.npz
python analyze_arc_delta_marginals.py            # writes figs/arc_delta_*_N125_L5.png
```
