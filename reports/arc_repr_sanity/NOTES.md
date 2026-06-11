# arc_repr mid-training sanity check (epoch 210/400)

Checkpoint: snapshot of `lj_ckpts_lj27_pbc_arc_repr_norm/lj27_pbc_arc_repr_norm_fullcov/best.ckpt`
at epoch 210 (training still running). Sampled with the FIXED sampler (post-review:
particle-0 init at code-0 cell center, fine-offset min-image at decode), 256 samples,
T=0.9, CPU. `samples_N27_ep210.npz`.

Purpose: (1) end-to-end integration test of the review fixes with a real arc checkpoint —
no arc checkpoint existed when the fixes were written; (2) early in-distribution
refinability read; (3) replaces the tainted `samples_arc_ep25.npz` (old sampler:
corner-pinned particle 0, unwrapped fine).

Benchmark vs N=27 MCMC (`benchmark_lj27.py`):

| metric | arc_repr ep210 | best-ever (binned discrete 0423) | K=1 rail final | uniform |
|--------|----------------|----------------------------------|----------------|---------|
| OTgap  | **0.510**      | 0.12                             | 0.51           | 1.0     |
| gr_L1  | 0.269          | 0.045                            | 0.19           | 0.38    |
| Uc/N   | 27.8           | 5.1                              | 12.2           | 39.2    |

Read: sampler runs cleanly; the half-trained arc model already matches the K=1 rail's
FINAL in-distribution OTgap and is inside the refinable band (<1). Energies/clash are
still high (expected mid-training for a fullcov MDN). No red flags — let the run finish,
then redo this at the final checkpoint before the zero-shot transfer evaluation.
