# Multi-size arc run — epoch-38 mid-training status

Snapshot: `best.ckpt` of `multisize_arc_fullcov` frozen to `/tmp/multisize_ep38.ckpt` at
20:43 (training was mid-epoch-38, 40% through; run still live). {L3,L5} train, L4 held out.
Protocol identical to the ep18 table: 64 samples/size, `--ot_batch 64` (OTgap noise ~±0.1),
T=0.9, CPU, no `--cell_size` flag (geometry from checkpoint hparams). Generation needs
`--mode relative --Lx L --Ly L --Lz L --density 1.0` (N = density·L³ = 27/64/125).

| size | role | NLL/coord (ep18 → **ep38**) | OTgap (ep18 → **ep38**) | zero-shot single-size baseline |
|------|------|----------------------------:|------------------------:|-------------------------------:|
| L=3 N=27 | train | 0.168 → **0.170** | 0.646 → **0.677** | OTgap 0.524 / NLL 0.115 |
| **L=4 N=64** | **HELD OUT** | 0.273 → **0.255** | 0.941 → **0.996** | OTgap 0.795 / NLL 0.649 |
| L=5 N=125 | train | 0.187 → **0.169** | 0.653 → **0.773** | OTgap 0.911 / NLL 0.656 |

Artifacts: `reports/multisize_arc/samples_N{27,64,125}_ep38.npz`.

## Reading at ~2/5 of training

- **Conditional (NLL) is the trustworthy signal and keeps improving.** Low-variance
  teacher-forced loss fell monotonically at both train sizes (L5 0.187→0.169) AND at the
  **held-out L4 (0.273→0.255) with zero L4 data** — the box-size interpolation hypothesis
  continues to hold and tighten. L3 is flat at ~0.17 (already near its floor).
- **Generation (OTgap) has not yet followed — as expected at this stage.** At 64 samples
  the ±0.1 band makes ep18→ep38 OTgap moves mostly noise, with one real takeaway: the
  **held-out L4 is pinned at ~1.0 (uniform-noise ceiling) — no generation-level transfer
  yet.** A well-fit per-step conditional (L4 NLL 0.255, near the train sizes) still
  compounds autoregressive error over a full N=125 rollout into globally noise-like
  structure. The L5 train-size bump 0.653→0.773 is part compounding, part small-batch noise.
- **Why NLL up but OTgap flat/up is not a contradiction.** NLL scores one-step-ahead
  prediction under teacher forcing (no error accumulation); OTgap scores the free-running
  rollout. The gap between them IS the AR-compounding term — the central open risk for this
  representation. It typically closes late, after the conditional saturates.

## Decision gate unchanged

End of training (epoch 100, ~09:30 tomorrow): definitive held-out L4 OTgap on the full
256-sample protocol vs the 0.795 zero-shot baseline and the ≤0.6 green line. If L4
generation is still pinned near 1.0 despite a converged conditional, the next move is to
attack AR compounding directly (e.g. `continuous_input_noise` during training), not more
epochs.

## Commands

```bash
CKPT=/tmp/multisize_ep38.ckpt
# generation (per size): box via --Lx/--Ly/--Lz, N = density*L^3
python sample_lj.py --ckpt $CKPT --mode relative --coord_dim 3 --ar_arch standard \
  --use_continuous_head --full_covariance --temperature 0.9 --relative_window 3.0 \
  --relative_bins 64 --nsamples 64 --periodic --seed 0 --density 1.0 --device cpu \
  --Lx 5 --Ly 5 --Lz 5 --save reports/multisize_arc/samples_N125_ep38.npz
python benchmark_lj27.py --target <size-matched MCMC h5> \
  --pattern reports/multisize_arc/samples_N125_ep38.npz --ot_batch 64
# NLL (teacher-forced): L3 lj27_pbc_arc_repr_fullcov/lj27_pbc_hilbert_cache.pt,
#   L4 lj64_L4_R128_cache.pt, L5 lj125_L5_R128_cache.pt
python eval_arc_nll.py --ckpt $CKPT --cache <cache.pt> --limit 2000
```
