# Step-4a: teacher-forced NLL parity across box sizes (arc_repr)

Script: `eval_arc_nll.py` (reuses `training_step` normalization; validated at L=3 against
the run's own `train/loss_epoch` ≈ 0.133 → eval 0.115, consistent since eval mode drops
training noise and best.ckpt is the best epoch). Checkpoint: **epoch-210 snapshot** of the
still-running `lj27_pbc_arc_repr_norm_fullcov` run. 2000 configs per size, constant-cell
pow2 caches (R=128 for L4/L5).

| size | cache | NLL/coord | Δ vs train | ≈ nats/particle |
|------|-------|-----------|-----------|------------------|
| L=3 N=27 (train) | R=64 (training cache) | 0.115 | — | — |
| L=4 N=64 zero-shot | R=128 | 0.649 | +0.53 | +2.1 |
| L=5 N=125 zero-shot | R=128 | *(pending cache build)* | | |

## Interpretation (epoch 210; re-run at the final checkpoint)

- **The catastrophic failure mode is gone.** The old discrete model's zero-shot L3→L4
  breakdown had two parts: turn tokens at ppl ~1e8 (unbounded) and a 13× local-conditional
  degradation (~2.6 nats/token). arc_repr shows NO unbounded component — the Δs/fine
  representation absorbed the turns, exactly as the Step-3 audit predicted.
- **What remains is the known second mechanism.** +2.1 nats/particle is strikingly close
  to the local-conditional overfit measured in the original diagnosis (the part that
  multi-size training previously recovered: ppl 1949→236 ≈ 2.1 nats). The size-invariant
  representation fixed mechanism #1 (expressing turns); mechanism #2 (the conditional
  overfits to the single training size — context length, edge-bias distance range
  L/2·√3: 2.6 at L3 vs 3.5 at L4) was always expected to need multi-size data.
- **Prediction for Stage-2 zero-shot generation:** degraded-but-structured samples
  (OTgap likely yellow/red but, unlike K=1's >1.0, not necessarily worse than noise,
  since there is no absolute-position extrapolation in the conditioning). If that holds,
  the pre-registered move is multi-size training ({L3,L5} hold out L4) on the arc
  representation — the combination the project's own evidence says is required.

## Commands

```bash
python eval_arc_nll.py --ckpt <ckpt> --cache <cache.pt> --limit 2000
# caches: lj_caches_arc_transfer/lj64_L4_R128_cache.pt, lj125_L5_R128_cache.pt
```
