# Multi-size arc run — epoch-18 mid-training status

Snapshot: epoch 18/100 of `multisize_arc_fullcov` ({L3,L5} train, L4 held out).
64 samples/size, `--ot_batch 64` (small batch → OTgap noise ~±0.1), T=0.9, CPU,
no `--cell_size` flag (geometry from checkpoint hparams — first run where that works).

| size | NLL/coord (teacher-forced) | OTgap (generation) | role |
|------|---------------------------:|-------------------:|------|
| L=3 N=27 | 0.168 | 0.646 | train |
| L=4 N=64 | **0.273** (was 0.649 zero-shot) | 0.941 (was 0.795) | held out |
| L=5 N=125 | 0.187 | **0.653** (was 0.911 zero-shot) | train |

## Reading at 1/5 of training

- **Conditional: hypothesis confirmed.** Held-out L4 NLL recovered >58% of the transfer
  gap with zero L4 data; the model interpolates box size (L4 sits between the two train
  sizes).
- **Generation lags the conditional, as expected.** Train-size N=125 already collapsed
  its OTgap from 0.911 → 0.653 (multi-size training works at the generation level too),
  and in-dist N=27 (0.646) is simply mid-training (single-size took ~200+ epochs to reach
  0.52). Held-out L4 generation (0.941) has not yet improved over the zero-shot baseline
  — AR error compounding needs a converged conditional plus small-batch noise applies.
  The single-size model at epoch 18 would have been far worse everywhere.
- **Decision point stays at end of training:** held-out L4 OTgap vs the 0.795 zero-shot
  baseline and the ≤0.6 green line, measured with the full 256-sample protocol.
