# Did the model learn the arc-delta marginals? (L=5 trained vs L=4 held out)

**TL;DR:** trained size L5 — yes, near-perfect. Held-out size L4 — fine offsets transfer but
**Δs does not** (mode collapses small), and the cause is the pow2-rounding cell discontinuity
that makes L4 the finest-cell outlier. See the L=4 section.

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

## Held-out L=4 (N=64): Δs does NOT transfer (figs `*_N64_L4.png`)

Same protocol, 2048 GPU configs at L=4 vs 2048 L4 cache configs.

| component | data mean/std | gen mean/std | W1 | KS | vs L5 W1 |
|-----------|--------------:|-------------:|---:|---:|---------:|
| **Δs**    | 0.995 / 0.609 | **0.889** / 0.610 | **0.1067** | **0.130** | 7.7× worse |
| fine_x    | 0.000 / 0.288 | -0.001 / 0.280 | 0.0105 | 0.020 | ~same |
| fine_y    | 0.001 / 0.289 | -0.001 / 0.280 | 0.0111 | 0.021 | ~same |
| fine_z    | -0.001 / 0.289 | 0.000 / 0.280 | 0.0111 | 0.018 | ~same |

**The fine offsets transfer; Δs does not.** At the held-out size the model over-produces
*small* arc-length jumps: the Δs heatmap shows the data band stationary at ≈0.6 while the
**generated band collapses to ≈0.15** — a large mode shift. The aggregate mean only falls to
0.889 because a heavy right tail compensates the wrong mode. Backward (Δs<0) steps also double
(0.29% vs 0.14% at L5). The fine x/y/z offsets, by contrast, are as good as the trained size
(W1 ≈ 0.011) — they even slightly beat L5.

### Why Δs specifically, and why L4
This is the **pow2-rounding cell discontinuity**. With cell held "constant" at 0.046875 but
R forced to a power of two, the *actual* cells are: **L3 = 3/64 = 0.0469, L5 = 5/128 = 0.0391,
L4 = 4/128 = 0.03125.** L4 has the **finest** cell — an outlier *below* both training sizes
(4 sits just past the 85.3→128 pow2 jump). Equivalently the Δs scale X = (R/L)³ is
L3≈9.7k, L5≈16.8k, **L4≈32.8k — the largest, outside the trained range.** Δs is the one target
coupled to that pow2 cell scale, so it is exactly the component that fails to extrapolate;
the cell-normalized fine offset is scale-free and transfers. This is a concrete,
marginal-level mechanism for the L4 OTgap (0.89) staying red while L5 (0.45) is green.

## What this does and does not tell us

For the **trained** L5 the per-step marginals are near-perfect (consistent with its green
0.45 OTgap). For the **held-out** L4 there is now a clear **marginal-level** culprit — the Δs
mode collapse driven by the pow2 cell discontinuity — *plus* whatever joint/correlation error
the rollout compounds on top. Two concrete follow-ups this points to:
1. **Hold out an interpolated size instead** (one whose pow2 cell falls between the trained
   cells), or condition the head explicitly on `cell_size` so Δs can extrapolate its scaling.
2. Compare pair-correlation g(r) at L4 to separate the Δs-marginal error from residual
   joint-structure error.

## Repro

```bash
python sample_lj.py --ckpt lj_ckpts_multisize_arc_L3L5/multisize_arc_fullcov/best.ckpt \
  --mode relative --coord_dim 3 --ar_arch standard --use_continuous_head --full_covariance \
  --temperature 0.9 --relative_window 3.0 --relative_bins 64 --nsamples 2048 --periodic \
  --seed 1 --density 1.0 --device cuda --Lx 5 --Ly 5 --Lz 5 \
  --save reports/multisize_arc/samples_N125_gpu2k.npz
python analyze_arc_delta_marginals.py            # writes figs/arc_delta_*_N125_L5.png
```
